package f32

import (
	"fmt"
	"math"
	"testing"
)

// =============================================================================
// REAL FFT POWER TESTS (float32)
// =============================================================================
//
// RealFFTPower is the fused, power-writing counterpart of RealFFTUnpack: it emits
// dst[k] = |X[k]|^2 for k in [1, n-1] in a single pass, unpacking each bin exactly
// as RealFFTUnpack does and squaring in registers. The SIMD kernels fuse the
// magnitude-squared with an FMA, so they agree with the pure-Go reference only to
// within rounding; the Go reference squares realFFTUnpack32Go's output with a
// separate multiply and add. See RealFFTPower and realFFTPower32Go.

// realFFTPowerRef is a test-local oracle for RealFFTPower, transcribed independently
// of realFFTPower32Go: it unpacks every bin through the independently written
// realFFTUnpackRef and squares. A bug copied into both realFFTPower32Go and its
// derivation would still be caught here. For k in [1, n-1] it computes |X[k]|^2.
func realFFTPowerRef(dst, zRe, zIm, twRe, twIm []float32, n int) {
	outRe := make([]float32, n)
	outIm := make([]float32, n)
	realFFTUnpackRef(outRe, outIm, zRe, zIm, twRe, twIm, n)
	for k := 1; k < n; k++ {
		dst[k] = outRe[k]*outRe[k] + outIm[k]*outIm[k]
	}
}

// Tolerances for realFFTPowerClose. The dispatched kernel fuses the odd term and
// the magnitude-squared with FMA while the reference uses separate multiply-adds,
// and squaring a value with relative error e produces relative error ~2e, so the
// two agree to within float32 rounding rather than exactly.
//
// MEASURED by instrumenting the dispatched kernel against realFFTPowerRef over the
// ramp and sinusoid fixtures this suite uses (sizes 2..1000): worst relative
// divergence 3.353e-6 on amd64. On arm64 the divergence is exactly 0, because the
// Go compiler contracts the reference's separate multiply-adds into the same FMAs
// the NEON kernel uses, so the two match bit-for-bit there; the amd64 figure is the
// cross-arch worst. realFFTPowerRelTol sits about 30x above it and matches the
// unpack predicate's rel 1e-4 (see f32_test.go), which already covers the odd-term
// FMA that dominates here. The absolute floor only guards outputs that cancel to
// near zero, which these smooth fixtures do not produce, so it is effectively
// unexercised; the relative arm decides every case.
const (
	realFFTPowerAbsTol = 1e-3
	realFFTPowerRelTol = 1e-4
)

// realFFTPowerClose reports whether got is within the float32 tolerance of want.
func realFFTPowerClose(got, want float32) bool {
	diff := math.Abs(float64(got - want))
	return diff <= realFFTPowerAbsTol || diff <= math.Abs(float64(want))*realFFTPowerRelTol
}

// makePowerTwiddles fills a valid unpack twiddle table W[k] = exp(-i*pi*k/n) at
// index k-1, matching the twiddles the f32 unpack tests use.
func makePowerTwiddles(twRe, twIm []float32, n int) {
	for k := 1; k < n; k++ {
		angle := -2 * math.Pi * float64(k) / float64(2*n)
		twRe[k-1] = float32(math.Cos(angle))
		twIm[k-1] = float32(math.Sin(angle))
	}
}

func TestRealFFTPower(t *testing.T) {
	// The odd sizes exercise the odd-n mirror; 9 and 10 give the first AVX remainder
	// lengths (n>8), 5..8 the NEON remainder lengths (n>4). n=2 is the smallest case.
	sizes := []int{2, 3, 4, 5, 6, 8, 9, 10, 16, 17, 31, 32, 33, 63, 64, 65, 128, 256, 512, 1000}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			zRe := make([]float32, n)
			zIm := make([]float32, n)
			twRe := make([]float32, n-1)
			twIm := make([]float32, n-1)
			dst := make([]float32, n)
			ref := make([]float32, n)
			// Seed every output bin with NaN so a kernel that accumulated into
			// dst[k] (instead of assigning the power) is caught: it would leave NaN,
			// which realFFTPowerClose rejects. dst[0] is the caller's DC bin.
			for k := 1; k < n; k++ {
				dst[k] = float32(math.NaN())
			}

			for i := range n {
				zRe[i] = float32(i+1) * 0.1
				zIm[i] = float32(i+2) * 0.2
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
// reference. On amd64 the AVX path needs n>8, on arm64 the NEON path needs n>4;
// 5..8 exercise one NEON (n-1)%4 cycle and 9..16 one AVX (n-1)%8 cycle. Where the
// dispatcher declines SIMD this arm goes inert (it then calls realFFTPower32Go on
// both sides); TestRealFFTPower, which checks the dispatched kernel against the
// independently transcribed realFFTPowerRef oracle, is the arm that holds on every
// tier.
func TestRealFFTPower_GoVsSIMD(t *testing.T) {
	sizes := []int{5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 32, 64, 128, 256, 512}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			zRe := make([]float32, n)
			zIm := make([]float32, n)
			twRe := make([]float32, n-1)
			twIm := make([]float32, n-1)
			dstSIMD := make([]float32, n)
			dstGo := make([]float32, n)

			for i := range n {
				zRe[i] = float32(math.Sin(float64(i)*0.7) * 10)
				zIm[i] = float32(math.Cos(float64(i)*0.9) * 10)
			}
			makePowerTwiddles(twRe, twIm, n)

			RealFFTPower(dstSIMD, zRe, zIm, twRe, twIm)
			realFFTPower32Go(dstGo, zRe, zIm, twRe, twIm, n)

			for k := 1; k < n; k++ {
				if !realFFTPowerClose(dstSIMD[k], dstGo[k]) {
					t.Errorf("k=%d: SIMD=%v, Go=%v, diff=%v", k, dstSIMD[k], dstGo[k], dstSIMD[k]-dstGo[k])
				}
			}
		})
	}
}

// TestRealFFTPower_GoMatchesUnpackSquare pins the pure-Go reference against the
// documented unpack math: realFFTPower32Go must match squaring realFFTUnpack32Go's
// X[k] on every bin. The comparison is tolerance-based rather than bit-for-bit: the
// Go compiler contracts the reference's multiply-adds into hardware FMAs on some
// architectures (arm64) and not others (amd64), so the last bits of dst[k] are not
// portable, though the value is always |X[k]|^2 to within rounding. A gross error
// (a wrong index, a dropped 0.5, a swapped even/odd) moves the result well outside
// tolerance and is still caught. It holds on every tier since it never touches the
// dispatched kernel.
func TestRealFFTPower_GoMatchesUnpackSquare(t *testing.T) {
	sizes := []int{2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 32, 33, 64, 65, 128, 256, 511, 512}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			zRe := make([]float32, n)
			zIm := make([]float32, n)
			twRe := make([]float32, n-1)
			twIm := make([]float32, n-1)
			dst := make([]float32, n)
			outRe := make([]float32, n)
			outIm := make([]float32, n)
			// Seed with NaN so an accumulating Go reference is caught (see TestRealFFTPower).
			for k := 1; k < n; k++ {
				dst[k] = float32(math.NaN())
			}

			for i := range n {
				zRe[i] = float32(math.Sin(float64(i)*0.7) * 10)
				zIm[i] = float32(math.Cos(float64(i)*0.9) * 10)
			}
			makePowerTwiddles(twRe, twIm, n)

			realFFTPower32Go(dst, zRe, zIm, twRe, twIm, n)
			realFFTUnpack32Go(outRe, outIm, zRe, zIm, twRe, twIm, n)

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
		dst := []float32{7}
		zRe := []float32{1}
		zIm := []float32{0}
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
		dst := []float32{7}
		realFFTPower32Go(dst, []float32{1}, []float32{0}, nil, nil, 1)
		if dst[0] != 7 {
			t.Errorf("realFFTPower32Go n=1 modified output: dst=%v, want unchanged", dst[0])
		}
	})

	// DC (dst[0]) is the caller's job, so RealFFTPower must never write it.
	t.Run("dc_untouched", func(t *testing.T) {
		const n = 16
		zRe := make([]float32, n)
		zIm := make([]float32, n)
		twRe := make([]float32, n-1)
		twIm := make([]float32, n-1)
		dst := make([]float32, n)
		for i := range n {
			zRe[i], zIm[i] = float32(i+1)*0.1, float32(i+2)*0.2
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
	zRe := []float32{1, 2, 3, 4}
	zIm := []float32{0.1, 0.2, 0.3, 0.4}
	twRe := make([]float32, n-1)
	twIm := make([]float32, n-1)
	makePowerTwiddles(twRe, twIm, n)

	dst := make([]float32, n)
	RealFFTPower(dst, zRe, zIm, twRe, twIm)

	outRe := make([]float32, n)
	outIm := make([]float32, n)
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
	zRe := make([]float32, n)
	zIm := make([]float32, n)
	twRe := make([]float32, n-1)
	twIm := make([]float32, n-1)
	dst := make([]float32, n)
	for i := range n {
		zRe[i] = float32(i+1) * 0.1
		zIm[i] = float32(i+2) * 0.2
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
	const n = 16
	makeZ := func() ([]float32, []float32) {
		zRe := make([]float32, n)
		zIm := make([]float32, n)
		for i := range n {
			zRe[i], zIm[i] = float32(i+1)*0.1, float32(i+2)*0.2
		}
		return zRe, zIm
	}
	makeTw := func(reLen, imLen int) ([]float32, []float32) {
		a, b := make([]float32, reLen), make([]float32, imLen)
		for i := range a {
			a[i] = 0.5
		}
		for i := range b {
			b[i] = -0.5
		}
		return a, b
	}
	const sentinel = -1.5
	makeDst := func(dstLen int) []float32 {
		s := make([]float32, dstLen)
		for i := range s {
			s[i] = sentinel
		}
		return s
	}
	untouched := func(t *testing.T, name string, s []float32) {
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

// TestRealFFTPower_OverRead guards the reversed mirror load. The kernel reads Z[n-k]
// descending against Z[k] ascending; a reverse pointer sized one block too far (or a
// forward load past the end) would read outside [0,n). zRe/zIm are backed by padded
// arrays whose guard bands hold NaN (two AVX blocks = 16 float32 on each side), then
// sliced to exactly length n. Any out-of-range read pulls a NaN into an output lane,
// which the parity check catches deterministically instead of a SIGSEGV or a
// coincidental match. NaN squared is NaN, so it survives the square.
func TestRealFFTPower_OverRead(t *testing.T) {
	const pad = 16 // two AVX blocks of NaN guard band on each side of z
	for _, n := range []int{5, 6, 9, 10, 16, 17, 18, 31, 32, 33, 64, 65} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			nan := float32(math.NaN())
			zReBack := make([]float32, pad+n+pad)
			zImBack := make([]float32, pad+n+pad)
			for i := range zReBack {
				zReBack[i], zImBack[i] = nan, nan
			}
			zRe := zReBack[pad : pad+n]
			zIm := zImBack[pad : pad+n]
			for i := range n {
				zRe[i] = float32(math.Sin(float64(i)*0.7) * 10)
				zIm[i] = float32(math.Cos(float64(i)*0.9) * 10)
			}
			twRe := make([]float32, n-1)
			twIm := make([]float32, n-1)
			makePowerTwiddles(twRe, twIm, n)
			dst := make([]float32, n)
			ref := make([]float32, n)

			RealFFTPower(dst, zRe, zIm, twRe, twIm)
			realFFTPowerRef(ref, zRe, zIm, twRe, twIm, n)

			for k := 1; k < n; k++ {
				if math.IsNaN(float64(dst[k])) {
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
// signed zero and every twiddle is +1 or -1, so every intermediate is an exact zero:
// X[k] is zero and the power is exactly +0 on every bin (squaring collapses the sign
// of the operands), on both the dispatched and Go paths. A masking or negation bug in
// the reversed-load conjugate would surface here as a NaN or a non-zero.
func TestRealFFTPower_SignedZero(t *testing.T) {
	// 5..8 is one NEON (n-1)%4 cycle; 9..16 one AVX (n-1)%8 cycle; 64,65 add sizes
	// with several full blocks.
	sizes := []int{5, 6, 7, 8, 9, 12, 16, 17, 64, 65}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			zRe := make([]float32, n)
			zIm := make([]float32, n)
			twRe := make([]float32, n-1)
			twIm := make([]float32, n-1)
			dst := make([]float32, n)
			dstGo := make([]float32, n)

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
					zRe[i] = float32(math.Copysign(0, sign(reBit)))
					zIm[i] = float32(math.Copysign(0, sign(imBit)))
				}
				for k := 1; k < n; k++ {
					twRe[k-1] = float32(sign(4))
					twIm[k-1] = float32(sign(5))
				}

				RealFFTPower(dst, zRe, zIm, twRe, twIm)
				realFFTPower32Go(dstGo, zRe, zIm, twRe, twIm, n)

				for k := 1; k < n; k++ {
					if math.Float32bits(dst[k]) != 0 {
						t.Errorf("pat=%#02x k=%d: SIMD power = %v (%#08x), want +0",
							pat, k, dst[k], math.Float32bits(dst[k]))
					}
					if math.Float32bits(dstGo[k]) != 0 {
						t.Errorf("pat=%#02x k=%d: Go power = %v (%#08x), want +0",
							pat, k, dstGo[k], math.Float32bits(dstGo[k]))
					}
				}
			}
		})
	}
}

// Per-arm byte accounting for SetBytes, counting the distinct length-n float32
// slices each arm touches. This is a working-set proxy, not a full multi-pass
// traffic count: the baseline re-reads its outRe/outIm scratch across the Mul and
// FMA passes, so its real memory traffic is higher than the count implies. The
// point is only to stop charging both arms the same bytes; ns/op stays the
// apples-to-apples fused-vs-baseline comparison.
const (
	// fusedSlicesTouched is what the fused RealFFTPower touches per call: it reads
	// zRe, zIm, twRe, twIm and writes dst (5 distinct slices).
	fusedSlicesTouched = 5
	// baselineSlicesTouched is what the unpack+Mul+FMA baseline touches per call:
	// the same five plus the outRe/outIm scratch RealFFTUnpack writes and the
	// following Mul/FMA passes re-read (7 distinct slices).
	baselineSlicesTouched = 7
)

// benchRealFFTPower32 runs fn at size n over freshly built inputs. slicesTouched is
// how many length-n float32 slices this arm moves through memory per call, so each
// arm's reported MB/s reflects its own traffic instead of a shared count.
func benchRealFFTPower32(b *testing.B, n int, slicesTouched int64, fn func(dst, zRe, zIm, twRe, twIm []float32, n int)) {
	b.Helper()
	zRe := make([]float32, n)
	zIm := make([]float32, n)
	twRe := make([]float32, n-1)
	twIm := make([]float32, n-1)
	dst := make([]float32, n)

	for i := range n {
		zRe[i] = float32(i) * 0.1
		zIm[i] = float32(i) * 0.2
	}
	makePowerTwiddles(twRe, twIm, n)

	b.ResetTimer()
	b.SetBytes(int64(n*4) * slicesTouched) // 4 bytes per float32 across slicesTouched slices

	for range b.N {
		fn(dst, zRe, zIm, twRe, twIm, n)
	}
}

// realFFTPowerDispatched adapts the public entry point to the shared helper's
// signature.
func realFFTPowerDispatched(dst, zRe, zIm, twRe, twIm []float32, _ int) {
	RealFFTPower(dst, zRe, zIm, twRe, twIm)
}

// realFFTPowerBaseline is the three-pass unpack + Mul + FMA the fused kernel
// replaces: RealFFTUnpack writes the complex half-spectrum, then the power is
// squared and summed in two more passes. It carries its own scratch so the
// benchmark measures the extra traffic, not an allocation.
type realFFTPowerBaseline struct {
	outRe, outIm []float32
}

func (s *realFFTPowerBaseline) run(dst, zRe, zIm, twRe, twIm []float32, _ int) {
	RealFFTUnpack(s.outRe, s.outIm, zRe, zIm, twRe, twIm)
	Mul(dst, s.outRe, s.outRe)      // dst = outRe^2
	FMA(dst, s.outIm, s.outIm, dst) // dst = outIm^2 + outRe^2
}

// BenchmarkRealFFTPower compares the fused RealFFTPower against the three-pass
// RealFFTUnpack + Mul + FMA baseline it replaces. n is the half-size (nfft/2), so
// the rows correspond to nfft 256/512/1024/2048.
func BenchmarkRealFFTPower(b *testing.B) {
	sizes := []int{128, 256, 512, 1024}

	for _, n := range sizes {
		nfft := 2 * n
		b.Run(fmt.Sprintf("Fused_nfft%d", nfft), func(b *testing.B) {
			benchRealFFTPower32(b, n, fusedSlicesTouched, realFFTPowerDispatched)
		})
		b.Run(fmt.Sprintf("UnpackMulFMA_nfft%d", nfft), func(b *testing.B) {
			base := &realFFTPowerBaseline{outRe: make([]float32, n), outIm: make([]float32, n)}
			benchRealFFTPower32(b, n, baselineSlicesTouched, base.run)
		})
	}
}
