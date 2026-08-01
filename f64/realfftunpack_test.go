package f64

import (
	"fmt"
	"math"
	"testing"
)

// =============================================================================
// REAL FFT UNPACK TESTS (float64)
// =============================================================================

// realFFTUnpackRef is a test-local scalar reference, transcribed independently
// from the unpack formula. It is an oracle independent of realFFTUnpack64Go, so a
// mistake copied into both the kernel and its Go reference would still be caught.
// It matches realFFTUnpack64Go numerically (both use separate multiply and add,
// no FMA); the ~1 ULP FMA difference shows up only against the SIMD kernels. For
// k in [1, n-1] with nk = n-k it computes X[k].
func realFFTUnpackRef(outRe, outIm, zRe, zIm, twRe, twIm []float64, n int) {
	for k := 1; k < n; k++ {
		nk := n - k

		// Load Z[k] and conj(Z[n-k])
		zkRe, zkIm := zRe[k], zIm[k]
		znkRe, znkIm := zRe[nk], -zIm[nk] // Conjugate

		// even = 0.5 * (Z[k] + conj(Z[n-k]))
		evenRe := 0.5 * (zkRe + znkRe)
		evenIm := 0.5 * (zkIm + znkIm)

		// diff = Z[k] - conj(Z[n-k])
		diffRe := zkRe - znkRe
		diffIm := zkIm - znkIm

		// odd = W[k] * (-0.5i) * diff
		wr, wi := twRe[k-1], twIm[k-1]
		oddRe := 0.5 * (wr*diffIm + wi*diffRe)
		oddIm := 0.5 * (wi*diffIm - wr*diffRe)

		// X[k] = even + odd
		outRe[k] = evenRe + oddRe
		outIm[k] = evenIm + oddIm
	}
}

// Tolerances for realFFTUnpackClose.
//
// MEASURED over sizes 2 to 1000: the worst absolute divergence between the
// dispatched kernel and realFFTUnpackRef is 3.553e-15 on an AVX+FMA amd64 core
// and 1.776e-15 on a NEON Cortex-A76, so realFFTUnpackAbsTol sits about 28x above
// the worst case. The absolute floor has to exist because the unpack's sum and
// difference can cancel to near zero while the rounding that produced them does
// not shrink with them; the relative band only takes over above |want| == 0.01,
// and the worst relative divergence measured is 3.757e-14 against it.
//
// The absolute floor was 1e-9 until #211, four orders of magnitude looser than
// this, while the comment above it already named 1e-13 as the divergence it
// needed to admit.
const (
	realFFTUnpackAbsTol = 1e-13
	realFFTUnpackRelTol = 1e-11
)

// realFFTUnpackClose reports whether got is within the float64 tolerance of want.
// The dispatched kernel and realFFTUnpackRef differ only in how their
// multiply-adds fuse, so they agree to within rounding rather than exactly.
func realFFTUnpackClose(got, want float64) bool {
	diff := math.Abs(got - want)
	return diff <= realFFTUnpackAbsTol || diff <= math.Abs(want)*realFFTUnpackRelTol
}

func TestRealFFTUnpack(t *testing.T) {
	// 6 and 10 give (n-1) % 4 == 1, the only AVX scalar-tail length not otherwise hit.
	sizes := []int{2, 3, 4, 5, 6, 8, 9, 10, 16, 17, 31, 32, 33, 63, 64, 65, 128, 256, 512, 1000}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			// Allocate arrays
			zRe := make([]float64, n)
			zIm := make([]float64, n)
			twRe := make([]float64, n-1)
			twIm := make([]float64, n-1)
			outRe := make([]float64, n)
			outIm := make([]float64, n)
			outReRef := make([]float64, n)
			outImRef := make([]float64, n)

			// Initialize with test data
			for i := range n {
				zRe[i] = float64(i+1) * 0.1
				zIm[i] = float64(i+2) * 0.2
			}

			// Generate twiddle factors
			for k := 1; k < n; k++ {
				angle := -2 * math.Pi * float64(k) / float64(2*n)
				twRe[k-1] = math.Cos(angle)
				twIm[k-1] = math.Sin(angle)
			}

			// Apply SIMD version
			RealFFTUnpack(outRe, outIm, zRe, zIm, twRe, twIm)

			// Apply reference
			realFFTUnpackRef(outReRef, outImRef, zRe, zIm, twRe, twIm, n)

			// Compare results
			for k := 1; k < n; k++ {
				if !realFFTUnpackClose(outRe[k], outReRef[k]) {
					t.Errorf("outRe[%d] = %v, want %v, diff=%v", k, outRe[k], outReRef[k], outRe[k]-outReRef[k])
				}
				if !realFFTUnpackClose(outIm[k], outImRef[k]) {
					t.Errorf("outIm[%d] = %v, want %v, diff=%v", k, outIm[k], outImRef[k], outIm[k]-outImRef[k])
				}
			}
		})
	}
}

func TestRealFFTUnpack_EdgeCases(t *testing.T) {
	// Test with n < 2 (should return without doing anything)
	t.Run("n=0", func(_ *testing.T) {
		RealFFTUnpack(nil, nil, nil, nil, nil, nil)
	})

	t.Run("n=1", func(t *testing.T) {
		outRe := []float64{7}
		outIm := []float64{8}
		zRe := []float64{1}
		zIm := []float64{0}
		// No twiddles needed for n=1; below realFFTUnpackMinN so it must return.
		RealFFTUnpack(outRe, outIm, zRe, zIm, nil, nil)
		// Should return without modifying anything since n < 2.
		if outRe[0] != 7 || outIm[0] != 8 {
			t.Errorf("n=1 modified output: outRe=%v outIm=%v, want unchanged", outRe[0], outIm[0])
		}
	})
}

func TestRealFFTUnpack_GoVsSIMD(t *testing.T) {
	// Test that Go and SIMD implementations produce identical results.
	// n=3 exercises the arm64 NEON path (n>2); the larger sizes exercise the
	// amd64 AVX path (n>4).
	sizes := []int{3, 5, 9, 16, 17, 32, 64, 128, 256, 512}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			zRe := make([]float64, n)
			zIm := make([]float64, n)
			twRe := make([]float64, n-1)
			twIm := make([]float64, n-1)
			outReSIMD := make([]float64, n)
			outImSIMD := make([]float64, n)
			outReGo := make([]float64, n)
			outImGo := make([]float64, n)

			// Use varied test data
			for i := range n {
				zRe[i] = math.Sin(float64(i)*0.7) * 10
				zIm[i] = math.Cos(float64(i)*0.9) * 10
			}

			for k := 1; k < n; k++ {
				angle := -2 * math.Pi * float64(k) / float64(2*n)
				twRe[k-1] = math.Cos(angle)
				twIm[k-1] = math.Sin(angle)
			}

			// Run both implementations
			RealFFTUnpack(outReSIMD, outImSIMD, zRe, zIm, twRe, twIm)
			realFFTUnpack64Go(outReGo, outImGo, zRe, zIm, twRe, twIm, n)

			// Compare with tight tolerance for FMA precision differences
			for k := 1; k < n; k++ {
				if !realFFTUnpackClose(outReSIMD[k], outReGo[k]) {
					t.Errorf("k=%d outRe: SIMD=%v, Go=%v, diff=%v", k, outReSIMD[k], outReGo[k], outReSIMD[k]-outReGo[k])
				}
				if !realFFTUnpackClose(outImSIMD[k], outImGo[k]) {
					t.Errorf("k=%d outIm: SIMD=%v, Go=%v, diff=%v", k, outImSIMD[k], outImGo[k], outImSIMD[k]-outImGo[k])
				}
			}
		})
	}
}

func TestRealFFTUnpack_KnownValues(t *testing.T) {
	// Test with known values for a small case.
	// n=4: process k=1,2,3
	n := 4
	zRe := []float64{1, 2, 3, 4}
	zIm := []float64{0.1, 0.2, 0.3, 0.4}

	// Twiddles for k=1,2,3: W[k] = exp(-i*2*pi*k/(2*4)) = exp(-i*pi*k/4)
	twRe := []float64{
		math.Cos(-math.Pi / 4),     // k=1
		math.Cos(-math.Pi / 2),     // k=2
		math.Cos(-3 * math.Pi / 4), // k=3
	}
	twIm := []float64{
		math.Sin(-math.Pi / 4),     // k=1
		math.Sin(-math.Pi / 2),     // k=2
		math.Sin(-3 * math.Pi / 4), // k=3
	}

	outRe := make([]float64, n)
	outIm := make([]float64, n)
	outReRef := make([]float64, n)
	outImRef := make([]float64, n)

	RealFFTUnpack(outRe, outIm, zRe, zIm, twRe, twIm)
	realFFTUnpackRef(outReRef, outImRef, zRe, zIm, twRe, twIm, n)

	for k := 1; k < n; k++ {
		if math.Abs(outRe[k]-outReRef[k]) > 1e-12 {
			t.Errorf("k=%d outRe = %v, want %v", k, outRe[k], outReRef[k])
		}
		if math.Abs(outIm[k]-outImRef[k]) > 1e-12 {
			t.Errorf("k=%d outIm = %v, want %v", k, outIm[k], outImRef[k])
		}
	}
}

// TestRealFFTUnpack_AllocFree pins the zero-allocation guarantee. The direct
// cpu-flag dispatch keeps //go:noescape effective; routing through an init-time
// function-pointer table would reintroduce heap allocations here.
func TestRealFFTUnpack_AllocFree(t *testing.T) {
	const n = 256
	zRe := make([]float64, n)
	zIm := make([]float64, n)
	twRe := make([]float64, n-1)
	twIm := make([]float64, n-1)
	outRe := make([]float64, n)
	outIm := make([]float64, n)
	for i := range n {
		zRe[i] = float64(i+1) * 0.1
		zIm[i] = float64(i+2) * 0.2
	}
	for k := 1; k < n; k++ {
		angle := -2 * math.Pi * float64(k) / float64(2*n)
		twRe[k-1] = math.Cos(angle)
		twIm[k-1] = math.Sin(angle)
	}
	fn := func() { RealFFTUnpack(outRe, outIm, zRe, zIm, twRe, twIm) }
	if a := testing.AllocsPerRun(50, fn); a != 0 {
		t.Errorf("RealFFTUnpack allocated %v times per run, want 0", a)
	}
}

// TestRealFFTUnpack_OverRead guards the reversed mirror load. The kernel reads
// Z[n-k] descending against Z[k] ascending; a reverse pointer sized one block too
// far (or a forward load past the end) would read outside [0,n). zRe/zIm are
// backed by padded arrays whose guard bands hold NaN (two AVX blocks = 8 float64
// on each side, comfortably over one widest-kernel vector), then sliced to exactly
// length n. Any out-of-range read pulls a NaN into an output lane, which the parity
// check catches deterministically instead of a SIGSEGV or a coincidental match.
// NaN is used because it is not an algebraic identity for the add/mul in the
// unpack and can never be swallowed by the tolerance.
func TestRealFFTUnpack_OverRead(t *testing.T) {
	const pad = 8 // two AVX blocks of NaN guard band on each side of z
	// Sizes vary (n-1)%4 and (n-1)%2 so both the AVX and NEON reverse paths, at
	// their minimum reverse index and every tail length, are stressed.
	for _, n := range []int{5, 6, 16, 17, 18, 31, 32, 33, 64, 65} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			nan := math.NaN()
			zReBack := make([]float64, pad+n+pad)
			zImBack := make([]float64, pad+n+pad)
			for i := range zReBack {
				zReBack[i], zImBack[i] = nan, nan
			}
			zRe := zReBack[pad : pad+n]
			zIm := zImBack[pad : pad+n]
			for i := range n {
				zRe[i] = math.Sin(float64(i)*0.7) * 10
				zIm[i] = math.Cos(float64(i)*0.9) * 10
			}
			twRe := make([]float64, n-1)
			twIm := make([]float64, n-1)
			for k := 1; k < n; k++ {
				angle := -2 * math.Pi * float64(k) / float64(2*n)
				twRe[k-1] = math.Cos(angle)
				twIm[k-1] = math.Sin(angle)
			}
			outRe := make([]float64, n)
			outIm := make([]float64, n)
			outReRef := make([]float64, n)
			outImRef := make([]float64, n)

			RealFFTUnpack(outRe, outIm, zRe, zIm, twRe, twIm)
			realFFTUnpackRef(outReRef, outImRef, zRe, zIm, twRe, twIm, n)

			for k := 1; k < n; k++ {
				if math.IsNaN(outRe[k]) || math.IsNaN(outIm[k]) {
					t.Fatalf("k=%d: NaN in output -> kernel over-read the zRe/zIm guard band", k)
				}
				if !realFFTUnpackClose(outRe[k], outReRef[k]) {
					t.Errorf("k=%d outRe = %v, want %v", k, outRe[k], outReRef[k])
				}
				if !realFFTUnpackClose(outIm[k], outImRef[k]) {
					t.Errorf("k=%d outIm = %v, want %v", k, outIm[k], outImRef[k])
				}
			}
		})
	}
}

// TestRealFFTUnpack_ShortSlices exercises the length-validation guards: with n
// taken from len(zRe), any operand shorter than required (zIm/outRe/outIm < n, or
// twRe/twIm < n-1) must make the call return without writing output or panicking.
func TestRealFFTUnpack_ShortSlices(t *testing.T) {
	const n = 8
	makeZ := func() ([]float64, []float64) {
		zRe := make([]float64, n)
		zIm := make([]float64, n)
		for i := range n {
			zRe[i], zIm[i] = float64(i+1)*0.1, float64(i+2)*0.2
		}
		return zRe, zIm
	}
	makeTw := func(m int) ([]float64, []float64) {
		a, b := make([]float64, m), make([]float64, m)
		for i := range m {
			a[i], b[i] = 0.5, -0.5
		}
		return a, b
	}
	allZero := func(t *testing.T, name string, s []float64) {
		t.Helper()
		for i, v := range s {
			if v != 0 {
				t.Errorf("%s written at [%d]=%v, want untouched (guard should have returned)", name, i, v)
			}
		}
	}

	cases := []struct {
		name                              string
		zImLen, outReLen, outImLen, twLen int
	}{
		{"shortZIm", n - 1, n, n, n - 1},
		{"shortOutRe", n, n - 1, n, n - 1},
		{"shortOutIm", n, n, n - 1, n - 1},
		{"shortTwiddles", n, n, n, n - 2},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			zRe, zImFull := makeZ()
			zIm := zImFull[:c.zImLen]
			twRe, twIm := makeTw(c.twLen)
			outRe := make([]float64, c.outReLen)
			outIm := make([]float64, c.outImLen)
			RealFFTUnpack(outRe, outIm, zRe, zIm, twRe, twIm)
			allZero(t, "outRe", outRe)
			allZero(t, "outIm", outIm)
		})
	}
}

// TestRealFFTUnpack_SignedZero pins the sign of a zero result. Every input is a
// signed zero and every twiddle is +1 or -1, so every intermediate is an exact
// zero and nothing rounds: the comparison can be bit-for-bit. The AVX scalar
// tail used to negate for the conjugate with 0 - x, which returns +0 where -x
// returns -0, so it disagreed with both its own 4-wide body and the Go
// reference on these inputs (#208).
//
// Both comparisons are needed and neither covers the other. Dispatched-vs-Go
// pins the kernel, but wherever the dispatcher declines SIMD it CALLS
// realFFTUnpack64Go, so that arm degenerates to comparing a value with itself.
// Go-vs-realFFTUnpackRef pins the fallback against an independently written
// oracle and holds on every tier. Swapping the first arm to the oracle instead
// of adding the second would just move the blind spot onto the AVX path, where
// realFFTUnpack64Go stops being reachable from RealFFTUnpack.
func TestRealFFTUnpack_SignedZero(t *testing.T) {
	// (n-1)%4 runs 0..3 over 5..8, so the AVX scalar tail is exercised at every
	// length it can have; the NEON tail is (n-1)%2 and this list sweeps that too.
	// 16, 64 and 65 add sizes with several full 4-wide blocks ahead of the tail,
	// including one with no tail at all.
	sizes := []int{5, 6, 7, 8, 16, 64, 65}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			zRe := make([]float64, n)
			zIm := make([]float64, n)
			twRe := make([]float64, n-1)
			twIm := make([]float64, n-1)
			outRe := make([]float64, n)
			outIm := make([]float64, n)
			outReGo := make([]float64, n)
			outImGo := make([]float64, n)
			outReRef := make([]float64, n)
			outImRef := make([]float64, n)

			// Six independent sign bits. zRe and zIm each get one sign for the low
			// half of the input and one for the high half: a tail element has k in
			// the high half and its mirror index nk = n-k in the low half, so the
			// split lets the sweep drive the two ends of a mirrored pair apart.
			// The two twiddles take one sign each. 64 patterns covers every
			// combination.
			const signBits = 6
			for pat := range 1 << signBits {
				sign := func(bit uint) float64 {
					if pat&(1<<bit) != 0 {
						return -1
					}
					return 1
				}
				half := n / 2
				for i := range n {
					reBit, imBit := uint(0), uint(1)
					if i >= half {
						reBit, imBit = 2, 3
					}
					zRe[i] = math.Copysign(0, sign(reBit))
					zIm[i] = math.Copysign(0, sign(imBit))
				}
				for k := 1; k < n; k++ {
					twRe[k-1] = sign(4)
					twIm[k-1] = sign(5)
				}

				RealFFTUnpack(outRe, outIm, zRe, zIm, twRe, twIm)
				realFFTUnpack64Go(outReGo, outImGo, zRe, zIm, twRe, twIm, n)
				realFFTUnpackRef(outReRef, outImRef, zRe, zIm, twRe, twIm, n)

				for k := 1; k < n; k++ {
					if math.Float64bits(outRe[k]) != math.Float64bits(outReGo[k]) {
						t.Errorf("pat=%#02x k=%d: outRe = %v (%#016x), Go = %v (%#016x)",
							pat, k, outRe[k], math.Float64bits(outRe[k]),
							outReGo[k], math.Float64bits(outReGo[k]))
					}
					if math.Float64bits(outIm[k]) != math.Float64bits(outImGo[k]) {
						t.Errorf("pat=%#02x k=%d: outIm = %v (%#016x), Go = %v (%#016x)",
							pat, k, outIm[k], math.Float64bits(outIm[k]),
							outImGo[k], math.Float64bits(outImGo[k]))
					}
					if math.Float64bits(outReGo[k]) != math.Float64bits(outReRef[k]) {
						t.Errorf("pat=%#02x k=%d: Go outRe = %v (%#016x), ref = %v (%#016x)",
							pat, k, outReGo[k], math.Float64bits(outReGo[k]),
							outReRef[k], math.Float64bits(outReRef[k]))
					}
					if math.Float64bits(outImGo[k]) != math.Float64bits(outImRef[k]) {
						t.Errorf("pat=%#02x k=%d: Go outIm = %v (%#016x), ref = %v (%#016x)",
							pat, k, outImGo[k], math.Float64bits(outImGo[k]),
							outImRef[k], math.Float64bits(outImRef[k]))
					}
				}
			}
		})
	}
}

func BenchmarkRealFFTUnpack(b *testing.B) {
	// 513 sits next to 512 because every power of two in this list has
	// (n-1)%4 == 3, the longest AVX scalar tail (and (n-1)%2 == 1, the longest
	// NEON one). Without a size where the tail is empty its cost is charged to
	// every row and reads as the kernel's baseline.
	sizes := []int{64, 128, 256, 512, 513, 1024, 4096}

	benchFn := func(b *testing.B, n int, fn func(outRe, outIm, zRe, zIm, twRe, twIm []float64, n int)) {
		b.Helper()
		zRe := make([]float64, n)
		zIm := make([]float64, n)
		twRe := make([]float64, n-1)
		twIm := make([]float64, n-1)
		outRe := make([]float64, n)
		outIm := make([]float64, n)

		for i := range n {
			zRe[i] = float64(i) * 0.1
			zIm[i] = float64(i) * 0.2
		}
		for k := 1; k < n; k++ {
			angle := -2 * math.Pi * float64(k) / float64(2*n)
			twRe[k-1] = math.Cos(angle)
			twIm[k-1] = math.Sin(angle)
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8 * 6)) // 6 float64 slices touched (in/out re/im + twiddles)

		for range b.N {
			fn(outRe, outIm, zRe, zIm, twRe, twIm, n)
		}
	}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("SIMD_%d", n), func(b *testing.B) {
			benchFn(b, n, func(outRe, outIm, zRe, zIm, twRe, twIm []float64, _ int) {
				RealFFTUnpack(outRe, outIm, zRe, zIm, twRe, twIm)
			})
		})
		b.Run(fmt.Sprintf("Go_%d", n), func(b *testing.B) {
			benchFn(b, n, realFFTUnpack64Go)
		})
	}
}
