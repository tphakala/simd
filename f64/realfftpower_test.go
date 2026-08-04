package f64

import (
	"fmt"
	"math"
	"testing"
)

// =============================================================================
// REAL FFT POWER TESTS (float64)
// =============================================================================
//
// RealFFTPower is the fused, power-writing counterpart of RealFFTUnpack: it emits
// dst[k] = |X[k]|^2 for k in [1, n-1] in a single pass, unpacking each bin exactly
// as RealFFTUnpack does and squaring in registers. The SIMD kernels fuse the
// magnitude-squared with an FMA, so they agree with the pure-Go reference only to
// within rounding; the Go reference is bit-for-bit equal to squaring
// realFFTUnpack64Go's output. See RealFFTPower and realFFTPower64Go.

// realFFTPowerRef is a test-local oracle for RealFFTPower, transcribed
// independently of realFFTPower64Go: it unpacks every bin through the independently
// written realFFTUnpackRef and squares. A bug copied into both realFFTPower64Go and
// its derivation would still be caught here. For k in [1, n-1] it computes |X[k]|^2.
func realFFTPowerRef(dst, zRe, zIm, twRe, twIm []float64, n int) {
	outRe := make([]float64, n)
	outIm := make([]float64, n)
	realFFTUnpackRef(outRe, outIm, zRe, zIm, twRe, twIm, n)
	for k := 1; k < n; k++ {
		dst[k] = outRe[k]*outRe[k] + outIm[k]*outIm[k]
	}
}

// Tolerances for realFFTPowerClose. The dispatched kernel fuses the odd term and
// the magnitude-squared with FMA while the reference uses separate multiply-adds,
// so the two agree to within rounding rather than exactly.
//
// MEASURED by instrumenting the dispatched kernel against realFFTPowerRef over the
// ramp and sinusoid fixtures this suite uses (sizes 2..2048, powers up to ~1e5):
// worst relative divergence 1.218e-14 and worst absolute divergence 5.821e-11 (the
// absolute one on a large-magnitude bin, which the relative arm covers). So
// realFFTPowerRelTol sits about 800x above the worst relative case, matching the
// unpack predicate's rel 1e-11 (see realfftunpack_test.go). The absolute floor only
// guards outputs that cancel to near zero, which these smooth fixtures do not
// produce, so it is effectively unexercised; the relative arm decides every case.
const (
	realFFTPowerAbsTol = 1e-10
	realFFTPowerRelTol = 1e-11
)

// realFFTPowerClose reports whether got is within the float64 tolerance of want.
func realFFTPowerClose(got, want float64) bool {
	diff := math.Abs(got - want)
	return diff <= realFFTPowerAbsTol || diff <= math.Abs(want)*realFFTPowerRelTol
}

// makePowerTwiddles fills a valid unpack twiddle table W[k] = exp(-i*pi*k/n) at
// index k-1, matching realfftunpack_test.go.
func makePowerTwiddles(twRe, twIm []float64, n int) {
	for k := 1; k < n; k++ {
		angle := -2 * math.Pi * float64(k) / float64(2*n)
		twRe[k-1] = math.Cos(angle)
		twIm[k-1] = math.Sin(angle)
	}
}

func TestRealFFTPower(t *testing.T) {
	// 6 and 10 give (n-1) % 4 == 1, the only AVX remainder length not otherwise hit.
	// n=2 exercises the smallest even case; the odd sizes exercise the odd-n mirror.
	sizes := []int{2, 3, 4, 5, 6, 8, 9, 10, 16, 17, 31, 32, 33, 63, 64, 65, 128, 256, 512, 1000}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			zRe := make([]float64, n)
			zIm := make([]float64, n)
			twRe := make([]float64, n-1)
			twIm := make([]float64, n-1)
			dst := make([]float64, n)
			ref := make([]float64, n)

			for i := range n {
				zRe[i] = float64(i+1) * 0.1
				zIm[i] = float64(i+2) * 0.2
			}
			makePowerTwiddles(twRe, twIm, n)

			RealFFTPower(dst, zRe, zIm, twRe, twIm)
			realFFTPowerRef(ref, zRe, zIm, twRe, twIm, n)

			for k := 1; k < n; k++ {
				if !realFFTPowerClose(dst[k], ref[k]) {
					t.Errorf("dst[%d] = %v, want %v, diff=%v", k, dst[k], ref[k], dst[k]-ref[k])
				}
				if dst[k] < 0 {
					t.Errorf("dst[%d] = %v is negative; power must be >= 0", k, dst[k])
				}
			}
		})
	}
}

// TestRealFFTPower_GoVsSIMD checks the dispatched kernel against the pure-Go
// reference. n=3 exercises the arm64 NEON path (n>2); the larger sizes exercise the
// amd64 AVX path (n>4). 5 through 8 is one full (n-1)%4 cycle, so every AVX
// remainder length is compared. Where the dispatcher declines SIMD this arm goes
// inert (it then calls realFFTPower64Go on both sides); TestRealFFTPower, which
// checks the dispatched kernel against the independently transcribed realFFTPowerRef
// oracle, is the arm that holds on every tier.
func TestRealFFTPower_GoVsSIMD(t *testing.T) {
	sizes := []int{3, 5, 6, 7, 8, 9, 16, 17, 32, 64, 128, 256, 512}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			zRe := make([]float64, n)
			zIm := make([]float64, n)
			twRe := make([]float64, n-1)
			twIm := make([]float64, n-1)
			dstSIMD := make([]float64, n)
			dstGo := make([]float64, n)

			for i := range n {
				zRe[i] = math.Sin(float64(i)*0.7) * 10
				zIm[i] = math.Cos(float64(i)*0.9) * 10
			}
			makePowerTwiddles(twRe, twIm, n)

			RealFFTPower(dstSIMD, zRe, zIm, twRe, twIm)
			realFFTPower64Go(dstGo, zRe, zIm, twRe, twIm, n)

			for k := 1; k < n; k++ {
				if !realFFTPowerClose(dstSIMD[k], dstGo[k]) {
					t.Errorf("k=%d: SIMD=%v, Go=%v, diff=%v", k, dstSIMD[k], dstGo[k], dstSIMD[k]-dstGo[k])
				}
			}
		})
	}
}

// TestRealFFTPower_GoMatchesUnpackSquare pins the pure-Go reference against the
// documented unpack math: realFFTPower64Go must match squaring realFFTUnpack64Go's
// X[k] on every bin. The comparison is tolerance-based rather than bit-for-bit: the
// Go compiler contracts the reference's multiply-adds into hardware FMAs on some
// architectures (arm64) and not others (amd64), so the last bit of dst[k] is not
// portable, though the value is always |X[k]|^2 to within rounding. A gross error
// (a wrong index, a dropped 0.5, a swapped even/odd) moves the result well outside
// tolerance and is still caught. It holds on every tier since it never touches the
// dispatched kernel.
func TestRealFFTPower_GoMatchesUnpackSquare(t *testing.T) {
	sizes := []int{2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 32, 33, 64, 65, 128, 256, 511, 512}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			zRe := make([]float64, n)
			zIm := make([]float64, n)
			twRe := make([]float64, n-1)
			twIm := make([]float64, n-1)
			dst := make([]float64, n)
			outRe := make([]float64, n)
			outIm := make([]float64, n)

			for i := range n {
				zRe[i] = math.Sin(float64(i)*0.7) * 10
				zIm[i] = math.Cos(float64(i)*0.9) * 10
			}
			makePowerTwiddles(twRe, twIm, n)

			realFFTPower64Go(dst, zRe, zIm, twRe, twIm, n)
			realFFTUnpack64Go(outRe, outIm, zRe, zIm, twRe, twIm, n)

			for k := 1; k < n; k++ {
				want := outRe[k]*outRe[k] + outIm[k]*outIm[k]
				if !realFFTPowerClose(dst[k], want) {
					t.Errorf("k=%d dst = %v, unpack-square = %v, diff=%v",
						k, dst[k], want, dst[k]-want)
				}
			}
		})
	}
}

func TestRealFFTPower_EdgeCases(t *testing.T) {
	// n < 2 must return without doing anything.
	t.Run("n=0", func(_ *testing.T) {
		RealFFTPower(nil, nil, nil, nil, nil)
	})

	t.Run("n=1", func(t *testing.T) {
		dst := []float64{7}
		zRe := []float64{1}
		zIm := []float64{0}
		// Below realFFTUnpackMinN, so it must return untouched.
		RealFFTPower(dst, zRe, zIm, nil, nil)
		if dst[0] != 7 {
			t.Errorf("n=1 modified output: dst=%v, want unchanged", dst[0])
		}
	})

	// The Go reference has its own n < realFFTUnpackMinN guard (the public entry
	// point rejects small n first, so it is otherwise unreachable). Call it directly
	// with n=1 to pin that guard against a future direct caller.
	t.Run("go_ref_small_n", func(t *testing.T) {
		dst := []float64{7}
		realFFTPower64Go(dst, []float64{1}, []float64{0}, nil, nil, 1)
		if dst[0] != 7 {
			t.Errorf("realFFTPower64Go n=1 modified output: dst=%v, want unchanged", dst[0])
		}
	})

	// DC (dst[0]) is the caller's job, so RealFFTPower must never write it.
	t.Run("dc_untouched", func(t *testing.T) {
		const n = 8
		zRe := make([]float64, n)
		zIm := make([]float64, n)
		twRe := make([]float64, n-1)
		twIm := make([]float64, n-1)
		dst := make([]float64, n)
		for i := range n {
			zRe[i], zIm[i] = float64(i+1)*0.1, float64(i+2)*0.2
		}
		makePowerTwiddles(twRe, twIm, n)
		const sentinel = -1.5
		dst[0] = sentinel
		RealFFTPower(dst, zRe, zIm, twRe, twIm)
		if dst[0] != sentinel {
			t.Errorf("DC bin dst[0] = %v, want untouched %v", dst[0], sentinel)
		}
	})
}

// TestRealFFTPower_KnownValues checks a small case against an independently unpacked
// and squared spectrum. For n=4 the bins are k=1,2,3.
func TestRealFFTPower_KnownValues(t *testing.T) {
	const n = 4
	zRe := []float64{1, 2, 3, 4}
	zIm := []float64{0.1, 0.2, 0.3, 0.4}
	twRe := make([]float64, n-1)
	twIm := make([]float64, n-1)
	makePowerTwiddles(twRe, twIm, n)

	dst := make([]float64, n)
	RealFFTPower(dst, zRe, zIm, twRe, twIm)

	outRe := make([]float64, n)
	outIm := make([]float64, n)
	realFFTUnpackRef(outRe, outIm, zRe, zIm, twRe, twIm, n)
	for k := 1; k < n; k++ {
		want := outRe[k]*outRe[k] + outIm[k]*outIm[k]
		if !realFFTPowerClose(dst[k], want) {
			t.Errorf("k=%d dst = %v, want %v", k, dst[k], want)
		}
	}
}

// TestRealFFTPower_AllocFree pins the zero-allocation guarantee. The direct cpu-flag
// dispatch keeps //go:noescape effective; routing through an init-time function
// pointer would reintroduce heap allocations here.
func TestRealFFTPower_AllocFree(t *testing.T) {
	const n = 256
	zRe := make([]float64, n)
	zIm := make([]float64, n)
	twRe := make([]float64, n-1)
	twIm := make([]float64, n-1)
	dst := make([]float64, n)
	for i := range n {
		zRe[i] = float64(i+1) * 0.1
		zIm[i] = float64(i+2) * 0.2
	}
	makePowerTwiddles(twRe, twIm, n)
	fn := func() { RealFFTPower(dst, zRe, zIm, twRe, twIm) }
	if a := testing.AllocsPerRun(50, fn); a != 0 {
		t.Errorf("RealFFTPower allocated %v times per run, want 0", a)
	}
}

// TestRealFFTPower_ShortSlices exercises the length-validation guards: with n taken
// from len(zRe), any operand shorter than required (zIm/dst < n, or twRe/twIm <
// n-1) must make the call return without writing output or panicking. Each case
// shortens exactly one operand so every clause of the guard is pinned on its own.
func TestRealFFTPower_ShortSlices(t *testing.T) {
	const n = 8
	makeZ := func() ([]float64, []float64) {
		zRe := make([]float64, n)
		zIm := make([]float64, n)
		for i := range n {
			zRe[i], zIm[i] = float64(i+1)*0.1, float64(i+2)*0.2
		}
		return zRe, zIm
	}
	makeTw := func(reLen, imLen int) ([]float64, []float64) {
		a, b := make([]float64, reLen), make([]float64, imLen)
		for i := range a {
			a[i] = 0.5
		}
		for i := range b {
			b[i] = -0.5
		}
		return a, b
	}
	const sentinel = -1.5
	makeDst := func(dstLen int) []float64 {
		s := make([]float64, dstLen)
		for i := range s {
			s[i] = sentinel
		}
		return s
	}
	untouched := func(t *testing.T, name string, s []float64) {
		t.Helper()
		for i, v := range s {
			if v != sentinel {
				t.Errorf("%s written at [%d]=%v, want untouched (guard should have returned)", name, i, v)
			}
		}
	}

	cases := []struct {
		name                             string
		zImLen, dstLen, twReLen, twImLen int
	}{
		{"shortZIm", n - 1, n, n - 1, n - 1},
		{"shortDst", n, n - 1, n - 1, n - 1},
		{"shortTwRe", n, n, n - 2, n - 1},
		{"shortTwIm", n, n, n - 1, n - 2},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			zRe, zImFull := makeZ()
			zIm := zImFull[:c.zImLen]
			twRe, twIm := makeTw(c.twReLen, c.twImLen)
			dst := makeDst(c.dstLen)
			RealFFTPower(dst, zRe, zIm, twRe, twIm)
			untouched(t, "dst", dst)
		})
	}
}

// TestRealFFTPower_OverRead guards the reversed mirror load. The kernel reads
// Z[n-k] descending against Z[k] ascending; a reverse pointer sized one block too
// far (or a forward load past the end) would read outside [0,n). zRe/zIm are backed
// by padded arrays whose guard bands hold NaN (two AVX blocks = 8 float64 on each
// side), then sliced to exactly length n. Any out-of-range read pulls a NaN into
// an output lane, which the parity check catches deterministically instead of a
// SIGSEGV or a coincidental match. NaN squared is NaN, so it survives the square.
func TestRealFFTPower_OverRead(t *testing.T) {
	const pad = 8 // two AVX blocks of NaN guard band on each side of z
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
			makePowerTwiddles(twRe, twIm, n)
			dst := make([]float64, n)
			ref := make([]float64, n)

			RealFFTPower(dst, zRe, zIm, twRe, twIm)
			realFFTPowerRef(ref, zRe, zIm, twRe, twIm, n)

			for k := 1; k < n; k++ {
				if math.IsNaN(dst[k]) {
					t.Fatalf("k=%d: NaN in output -> kernel over-read the zRe/zIm guard band", k)
				}
				if !realFFTPowerClose(dst[k], ref[k]) {
					t.Errorf("k=%d dst = %v, want %v", k, dst[k], ref[k])
				}
			}
		})
	}
}

// TestRealFFTPower_SignedZero pins the even/odd sign handling. Every input is a
// signed zero and every twiddle is +1 or -1, so every intermediate is an exact
// zero: X[k] is zero and the power is exactly +0 on every bin (squaring collapses
// the sign of the operands), on both the dispatched and Go paths. A masking or
// negation bug in the reversed-load conjugate would surface here as a NaN or a
// non-zero.
func TestRealFFTPower_SignedZero(t *testing.T) {
	// (n-1)%4 runs 0..3 over 5..8, so the AVX remainder path is exercised at every
	// length; 16, 64, 65 add sizes with several full blocks.
	sizes := []int{5, 6, 7, 8, 16, 64, 65}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			zRe := make([]float64, n)
			zIm := make([]float64, n)
			twRe := make([]float64, n-1)
			twIm := make([]float64, n-1)
			dst := make([]float64, n)
			dstGo := make([]float64, n)

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

				RealFFTPower(dst, zRe, zIm, twRe, twIm)
				realFFTPower64Go(dstGo, zRe, zIm, twRe, twIm, n)

				for k := 1; k < n; k++ {
					if math.Float64bits(dst[k]) != 0 {
						t.Errorf("pat=%#02x k=%d: SIMD power = %v (%#016x), want +0",
							pat, k, dst[k], math.Float64bits(dst[k]))
					}
					if math.Float64bits(dstGo[k]) != 0 {
						t.Errorf("pat=%#02x k=%d: Go power = %v (%#016x), want +0",
							pat, k, dstGo[k], math.Float64bits(dstGo[k]))
					}
				}
			}
		})
	}
}

// benchRealFFTPower64 runs fn at size n over freshly built inputs.
func benchRealFFTPower64(b *testing.B, n int, fn func(dst, zRe, zIm, twRe, twIm []float64, n int)) {
	b.Helper()
	zRe := make([]float64, n)
	zIm := make([]float64, n)
	twRe := make([]float64, n-1)
	twIm := make([]float64, n-1)
	dst := make([]float64, n)

	for i := range n {
		zRe[i] = float64(i) * 0.1
		zIm[i] = float64(i) * 0.2
	}
	makePowerTwiddles(twRe, twIm, n)

	b.ResetTimer()
	b.SetBytes(int64(n * 8 * 5)) // 5 float64 slices touched (z re/im, tw re/im, dst)

	for range b.N {
		fn(dst, zRe, zIm, twRe, twIm, n)
	}
}

// realFFTPowerDispatched adapts the public entry point to the shared helper's
// signature.
func realFFTPowerDispatched(dst, zRe, zIm, twRe, twIm []float64, _ int) {
	RealFFTPower(dst, zRe, zIm, twRe, twIm)
}

// realFFTPowerBaseline is the three-pass unpack + Mul + FMA the fused kernel
// replaces: RealFFTUnpack writes the complex half-spectrum, then the power is
// squared and summed in two more passes. It carries its own scratch so the
// benchmark measures the extra traffic, not an allocation.
type realFFTPowerBaseline struct {
	outRe, outIm []float64
}

func (s *realFFTPowerBaseline) run(dst, zRe, zIm, twRe, twIm []float64, _ int) {
	RealFFTUnpack(s.outRe, s.outIm, zRe, zIm, twRe, twIm)
	Mul(dst, s.outRe, s.outRe)      // dst = outRe^2
	FMA(dst, s.outIm, s.outIm, dst) // dst = outIm^2 + outRe^2
}

// BenchmarkRealFFTPower compares the fused RealFFTPower against the three-pass
// RealFFTUnpack + Mul + FMA baseline it replaces. n is the half-size (nfft/2), so
// the rows correspond to nfft 256/512/1024/2048 from issue #198.
func BenchmarkRealFFTPower(b *testing.B) {
	sizes := []int{128, 256, 512, 1024}

	for _, n := range sizes {
		nfft := 2 * n
		b.Run(fmt.Sprintf("Fused_nfft%d", nfft), func(b *testing.B) {
			benchRealFFTPower64(b, n, realFFTPowerDispatched)
		})
		b.Run(fmt.Sprintf("UnpackMulFMA_nfft%d", nfft), func(b *testing.B) {
			base := &realFFTPowerBaseline{outRe: make([]float64, n), outIm: make([]float64, n)}
			benchRealFFTPower64(b, n, base.run)
		})
	}
}
