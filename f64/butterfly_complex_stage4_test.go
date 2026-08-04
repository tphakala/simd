package f64

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// =============================================================================
// BUTTERFLY COMPLEX STAGE4 TESTS (float64, radix-4)
// =============================================================================
//
// These reuse closeEnough, stageAbsTol/stageRelTol and stageFFTTolPerN2 from
// butterfly_complex_stage_test.go: both stages combine the same butterflies and
// round the same way, so one tolerance policy covers both.

// stage4FillTwiddles writes the three radix-4 twiddle powers for one span into
// caller-provided buffers of length span:
//
//	w = exp(-2*pi*i/(4*span)); tw1 = w^(2j), tw2 = w^j, tw3 = w^(3j).
//
// math.Sincos returns (sin, cos); the twiddle real part is cos, the imaginary
// part is sin.
func stage4FillTwiddles(t1r, t1i, t2r, t2i, t3r, t3i []float64, span int) {
	base := -math.Pi / float64(2*span) // = -2*pi/(4*span)
	for j := range span {
		s1, c1 := math.Sincos(base * float64(2*j))
		t1r[j], t1i[j] = c1, s1
		s2, c2 := math.Sincos(base * float64(j))
		t2r[j], t2i[j] = c2, s2
		s3, c3 := math.Sincos(base * float64(3*j))
		t3r[j], t3i[j] = c3, s3
	}
}

// stage4Twiddles allocates and returns the three radix-4 twiddle powers for one
// span. See stage4FillTwiddles for the values.
//
//nolint:gocritic // the six twiddle-power arrays are one group, matching the six-slice kernel signature under test
func stage4Twiddles(span int) (t1r, t1i, t2r, t2i, t3r, t3i []float64) {
	t1r = make([]float64, span)
	t1i = make([]float64, span)
	t2r = make([]float64, span)
	t2i = make([]float64, span)
	t3r = make([]float64, span)
	t3i = make([]float64, span)
	stage4FillTwiddles(t1r, t1i, t2r, t2i, t3r, t3i, span)
	return t1r, t1i, t2r, t2i, t3r, t3i
}

// stage4FillRadix2Twiddle writes the radix-2 twiddle table exp(-i*pi*j/span) for
// one span into caller-provided buffers of length span. This is the twiddle the
// existing ButterflyComplexStage consumes, used to build the two-radix-2 arm that
// a radix-4 stage must reproduce.
func stage4FillRadix2Twiddle(twRe, twIm []float64, span int) {
	for j := range span {
		sin, cos := math.Sincos(-math.Pi * float64(j) / float64(span))
		twRe[j], twIm[j] = cos, sin
	}
}

// stage4Data fills split-complex arrays with deterministic pseudo-random values.
// A seeded generator keeps the run reproducible while covering a wider spread of
// magnitudes and signs than a fixed trig filler.
func stage4Data(n int, seed uint64) (re, im []float64) {
	r := rand.New(rand.NewPCG(seed, 0x9E3779B97F4A7C15))
	re = make([]float64, n)
	im = make([]float64, n)
	for i := range n {
		re[i] = r.NormFloat64()
		im[i] = r.NormFloat64()
	}
	return re, im
}

// butterflyComplexStage4Ref is an independent reference for one radix-4
// decimation-in-time stage. It works in complex128 and indexes re/im directly,
// so it shares no code with butterflyComplexStage4x64Go. The -1i and +1i factors
// are the forward-DFT cross-adds on the second and fourth outputs.
func butterflyComplexStage4Ref(re, im []float64, span int, t1r, t1i, t2r, t2i, t3r, t3i []float64) {
	if span <= 0 ||
		len(t1r) < span || len(t1i) < span ||
		len(t2r) < span || len(t2i) < span ||
		len(t3r) < span || len(t3i) < span {
		return
	}
	n := min(len(re), len(im))
	for k := 0; k+butterflyStage4Radix*span <= n; k += butterflyStage4Radix * span {
		for j := range span {
			i0 := k + j
			i1 := i0 + span
			i2 := i1 + span
			i3 := i2 + span

			x0 := complex(re[i0], im[i0])
			x1 := complex(re[i1], im[i1])
			x2 := complex(re[i2], im[i2])
			x3 := complex(re[i3], im[i3])

			t1 := x1 * complex(t1r[j], t1i[j])
			t2 := x2 * complex(t2r[j], t2i[j])
			t3 := x3 * complex(t3r[j], t3i[j])

			a := x0 + t1
			b := x0 - t1
			c := t2 + t3
			d := t2 - t3

			y0 := a + c
			y1 := b - 1i*d
			y2 := a - c
			y3 := b + 1i*d

			re[i0], im[i0] = real(y0), imag(y0)
			re[i1], im[i1] = real(y1), imag(y1)
			re[i2], im[i2] = real(y2), imag(y2)
			re[i3], im[i3] = real(y3), imag(y3)
		}
	}
}

// stage4Spans covers every dispatch path: span 1 is the block-axis path, span 2
// and 3 fill neither vector axis and run the j-axis scalar tail (span 2 has no
// dedicated vector path because a radix-4 Cooley-Tukey schedule never lands on
// it), and 4..64 walk the j-axis vector path with and without a tail (span%4 != 0).
var stage4Spans = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 32, 64}

// stage4BlockCounts exercise the block-axis remainder tail: the span-1 path
// consumes 4 blocks per iteration on AMD64 and 2 on NEON, so these counts cover
// every block-remainder case for both (mod 4: 1,2,3,4,5,8; mod 2: 5,7).
var stage4BlockCounts = []int{1, 2, 3, 4, 5, 7, 8}

// TestButterflyComplexStage4_MatchesTwoRadix2Stages is the pinning test. One
// radix-4 stage at span s with the canonical twiddles must reproduce two radix-2
// stages, first at span s then at span 2*s, over the same data. This is the
// mathematical identity the whole primitive rests on; if it holds, the twiddle
// powers and the -i/+i cross-adds are all correct.
//
// Tolerance, not exact equality: the radix-4 combine and the two-pass form group
// and fuse their multiply-adds differently, so they agree only to within rounding.
func TestButterflyComplexStage4_MatchesTwoRadix2Stages(t *testing.T) {
	for _, span := range stage4Spans {
		for _, blocks := range stage4BlockCounts {
			t.Run(fmt.Sprintf("span=%d/blocks=%d", span, blocks), func(t *testing.T) {
				n := butterflyStage4Radix * span * blocks
				re, im := stage4Data(n, uint64(span*1000+blocks))

				// Arm A: one radix-4 stage with the canonical twiddles.
				reA := append([]float64(nil), re...)
				imA := append([]float64(nil), im...)
				t1r, t1i, t2r, t2i, t3r, t3i := stage4Twiddles(span)
				ButterflyComplexStage4(reA, imA, span, t1r, t1i, t2r, t2i, t3r, t3i)

				// Arm B: radix-2 at span, then radix-2 at 2*span.
				reB := append([]float64(nil), re...)
				imB := append([]float64(nil), im...)
				twRe1 := make([]float64, span)
				twIm1 := make([]float64, span)
				stage4FillRadix2Twiddle(twRe1, twIm1, span)
				ButterflyComplexStage(reB, imB, span, twRe1, twIm1)
				twRe2 := make([]float64, 2*span)
				twIm2 := make([]float64, 2*span)
				stage4FillRadix2Twiddle(twRe2, twIm2, 2*span)
				ButterflyComplexStage(reB, imB, 2*span, twRe2, twIm2)

				for i := range n {
					if !closeEnough(reA[i], reB[i]) {
						t.Errorf("re[%d]: radix4=%v, two-radix2=%v", i, reA[i], reB[i])
					}
					if !closeEnough(imA[i], imB[i]) {
						t.Errorf("im[%d]: radix4=%v, two-radix2=%v", i, imA[i], imB[i])
					}
				}
			})
		}
	}
}

// TestButterflyComplexStage4 checks the dispatched stage against the independent
// complex128 reference over the full span x blocks grid. The reference indexes
// re/im directly rather than delegating, so it does not share code with the
// implementation under test.
func TestButterflyComplexStage4(t *testing.T) {
	for _, span := range stage4Spans {
		for _, blocks := range stage4BlockCounts {
			t.Run(fmt.Sprintf("span=%d/blocks=%d", span, blocks), func(t *testing.T) {
				n := butterflyStage4Radix * span * blocks
				re, im := stage4Data(n, uint64(span*7919+blocks))
				t1r, t1i, t2r, t2i, t3r, t3i := stage4Twiddles(span)

				reRef := append([]float64(nil), re...)
				imRef := append([]float64(nil), im...)

				ButterflyComplexStage4(re, im, span, t1r, t1i, t2r, t2i, t3r, t3i)
				butterflyComplexStage4Ref(reRef, imRef, span, t1r, t1i, t2r, t2i, t3r, t3i)

				for i := range n {
					if !closeEnough(re[i], reRef[i]) {
						t.Errorf("re[%d] = %v, want %v", i, re[i], reRef[i])
					}
					if !closeEnough(im[i], imRef[i]) {
						t.Errorf("im[%d] = %v, want %v", i, im[i], imRef[i])
					}
				}
			})
		}
	}
}

// TestButterflyComplexStage4_SIMDvsGo compares the dispatched kernel against the
// Go reference over an explicit (span, blocks) grid, entering below the public
// wrapper so blocks is fixed rather than derived from the slice lengths.
func TestButterflyComplexStage4_SIMDvsGo(t *testing.T) {
	for _, span := range stage4Spans {
		for _, blocks := range stage4BlockCounts {
			t.Run(fmt.Sprintf("span=%d/blocks=%d", span, blocks), func(t *testing.T) {
				n := butterflyStage4Radix * span * blocks
				re, im := stage4Data(n, uint64(span*104729+blocks))
				t1r, t1i, t2r, t2i, t3r, t3i := stage4Twiddles(span)

				reGo := append([]float64(nil), re...)
				imGo := append([]float64(nil), im...)

				butterflyComplexStage4x64(re, im, span, blocks, t1r, t1i, t2r, t2i, t3r, t3i)
				butterflyComplexStage4x64Go(reGo, imGo, span, blocks, t1r, t1i, t2r, t2i, t3r, t3i)

				for i := range n {
					if !closeEnough(re[i], reGo[i]) {
						t.Errorf("re[%d]: SIMD=%v, Go=%v", i, re[i], reGo[i])
					}
					if !closeEnough(im[i], imGo[i]) {
						t.Errorf("im[%d]: SIMD=%v, Go=%v", i, im[i], imGo[i])
					}
				}
			})
		}
	}
}

// TestButterflyComplexStage4_Span1SyntheticTwiddle makes span==1 twiddle-multiply
// sign errors observable. At span==1 the only twiddle index is j==0, so the
// canonical powers are all (1, 0); with a zero imaginary part every term that a
// sign flip in the twiddle multiply would touch (VSUBPD vs VADDPD on AMD64, FMLS
// vs FMLA on NEON) evaluates to the same bits, and the block-axis kernels, which
// are the whole point of the primitive, run their twiddle multiply unpinned.
//
// The kernel is documented value-agnostic, so it must agree with the Go reference
// for an arbitrary twiddle. Feeding a synthetic twiddle whose real and imaginary
// parts are both non-trivial exposes the masked mutation. This is the radix-4
// analogue of the radix-2 stageTwiddlePhase defense.
func TestButterflyComplexStage4_Span1SyntheticTwiddle(t *testing.T) {
	const span = 1
	// Non-identity twiddles, both components well away from zero so a dropped or
	// sign-flipped cross-term changes the result. Magnitudes need not be unit.
	t1r, t1i := []float64{0.5}, []float64{0.7}
	t2r, t2i := []float64{-0.3}, []float64{0.9}
	t3r, t3i := []float64{0.8}, []float64{-0.4}

	// Block counts that exercise both the block-axis vector loop (4 blocks/iter on
	// AMD64, 2 on NEON) and its leftover scalar tail.
	for _, blocks := range []int{2, 4, 5, 7, 8} {
		t.Run(fmt.Sprintf("blocks=%d", blocks), func(t *testing.T) {
			n := butterflyStage4Radix * span * blocks
			re, im := stage4Data(n, uint64(blocks*99991))

			reSIMD := append([]float64(nil), re...)
			imSIMD := append([]float64(nil), im...)
			reGo := append([]float64(nil), re...)
			imGo := append([]float64(nil), im...)
			reRef := append([]float64(nil), re...)
			imRef := append([]float64(nil), im...)

			ButterflyComplexStage4(reSIMD, imSIMD, span, t1r, t1i, t2r, t2i, t3r, t3i)
			butterflyComplexStage4x64Go(reGo, imGo, span, blocks, t1r, t1i, t2r, t2i, t3r, t3i)
			butterflyComplexStage4Ref(reRef, imRef, span, t1r, t1i, t2r, t2i, t3r, t3i)

			for i := range n {
				if !closeEnough(reSIMD[i], reGo[i]) {
					t.Errorf("re[%d]: SIMD=%v, Go=%v", i, reSIMD[i], reGo[i])
				}
				if !closeEnough(imSIMD[i], imGo[i]) {
					t.Errorf("im[%d]: SIMD=%v, Go=%v", i, imSIMD[i], imGo[i])
				}
				if !closeEnough(reGo[i], reRef[i]) {
					t.Errorf("re[%d]: Go=%v, complex128 ref=%v", i, reGo[i], reRef[i])
				}
				if !closeEnough(imGo[i], imRef[i]) {
					t.Errorf("im[%d]: Go=%v, complex128 ref=%v", i, imGo[i], imRef[i])
				}
			}
		})
	}
}

// TestButterflyComplexStage4_FullFFT drives a complete iterative Cooley-Tukey
// transform through ButterflyComplexStage4 and checks it against a naive DFT. A
// radix-4 schedule runs spans 1, 4, 16, ... and, when n is not a power of 4,
// finishes with a single radix-2 stage at span n/2. One run therefore walks every
// radix-4 path in sequence plus the trailing radix-2 stage, which catches a path
// that is individually self-consistent but wrong relative to the others.
func TestButterflyComplexStage4_FullFFT(t *testing.T) {
	for _, n := range []int{8, 16, 32, 64, 128, 256, 1024, 4096} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			srcRe := make([]float64, n)
			srcIm := make([]float64, n)
			for i := range n {
				srcRe[i] = math.Sin(float64(i)*0.21) + 0.5*math.Cos(float64(i)*1.7)
				srcIm[i] = math.Cos(float64(i) * 0.13)
			}

			// Bit-reversal permutation into the working buffers.
			re := make([]float64, n)
			im := make([]float64, n)
			bits := 0
			for 1<<bits < n {
				bits++
			}
			for i := range n {
				r := 0
				for b := range bits {
					if i&(1<<b) != 0 {
						r |= 1 << (bits - 1 - b)
					}
				}
				re[r] = srcRe[i]
				im[r] = srcIm[i]
			}

			// Radix-4 stages while a full 4*span block fits, then one radix-2 stage
			// at span n/2 when n is not a power of 4.
			t1r := make([]float64, n/butterflyStage4Radix)
			t1i := make([]float64, n/butterflyStage4Radix)
			t2r := make([]float64, n/butterflyStage4Radix)
			t2i := make([]float64, n/butterflyStage4Radix)
			t3r := make([]float64, n/butterflyStage4Radix)
			t3i := make([]float64, n/butterflyStage4Radix)
			span := 1
			for butterflyStage4Radix*span <= n {
				stage4FillTwiddles(t1r, t1i, t2r, t2i, t3r, t3i, span)
				ButterflyComplexStage4(re, im, span,
					t1r[:span], t1i[:span], t2r[:span], t2i[:span], t3r[:span], t3i[:span])
				span *= butterflyStage4Radix
			}
			if span < n {
				// span == n/2 here; one radix-2 stage completes the transform.
				twRe := make([]float64, span)
				twIm := make([]float64, span)
				stage4FillRadix2Twiddle(twRe, twIm, span)
				ButterflyComplexStage(re, im, span, twRe, twIm)
			}

			// Naive DFT reference.
			for k := range n {
				var wantRe, wantIm float64
				for i := range n {
					ang := -2 * math.Pi * float64(k) * float64(i) / float64(n)
					c, s := math.Cos(ang), math.Sin(ang)
					wantRe += srcRe[i]*c - srcIm[i]*s
					wantIm += srcRe[i]*s + srcIm[i]*c
				}
				// Accumulated rounding, dominated by the naive DFT reference, grows
				// as n^2; see stageFFTTolPerN2.
				tol := stageFFTTolPerN2 * float64(n) * float64(n)
				if math.Abs(re[k]-wantRe) > tol || math.Abs(im[k]-wantIm) > tol {
					t.Fatalf("bin %d = (%v, %v), want (%v, %v), tol %v",
						k, re[k], im[k], wantRe, wantIm, tol)
				}
			}
		})
	}
}

// TestButterflyComplexStage4_PartialBlockUntouched pins the documented boundary:
// blocks are processed while k+4*span <= n, so a trailing run shorter than a full
// block must be left exactly as it was, and the complete blocks must have been
// transformed.
func TestButterflyComplexStage4_PartialBlockUntouched(t *testing.T) {
	const sentinel = -12345.5

	for _, span := range stage4Spans {
		blockLen := butterflyStage4Radix * span
		seen := map[int]bool{}
		extras := make([]int, 0, 4)
		for _, extra := range []int{1, span, blockLen - 1} {
			if extra <= 0 || extra >= blockLen || seen[extra] {
				continue
			}
			seen[extra] = true
			extras = append(extras, extra)
		}

		// blocks 4 as well as 3: the span-1 path consumes four blocks per vector
		// iteration, so at blocks == 3 it never runs its vector loop at all and a
		// trailing partial block is only ever seen by the scalar tail.
		for _, blocks := range []int{3, 4} {
			for _, extra := range extras {
				t.Run(fmt.Sprintf("span=%d/blocks=%d/extra=%d", span, blocks, extra), func(t *testing.T) {
					checkStage4PartialUntouched(t, span, blocks, extra, sentinel)
				})
			}
		}
	}
}

func checkStage4PartialUntouched(t *testing.T, span, blocks, extra int, sentinel float64) {
	t.Helper()

	used := butterflyStage4Radix * span * blocks
	n := used + extra
	re, im := stage4Data(n, uint64(span*31+blocks))
	t1r, t1i, t2r, t2i, t3r, t3i := stage4Twiddles(span)
	beforeRe := append([]float64(nil), re...)
	beforeIm := append([]float64(nil), im...)
	for i := used; i < n; i++ {
		re[i] = sentinel
		im[i] = sentinel
	}

	ButterflyComplexStage4(re, im, span, t1r, t1i, t2r, t2i, t3r, t3i)

	for i := used; i < n; i++ {
		if re[i] != sentinel {
			t.Errorf("re[%d] = %v, want untouched sentinel %v", i, re[i], sentinel)
		}
		if im[i] != sentinel {
			t.Errorf("im[%d] = %v, want untouched sentinel %v", i, im[i], sentinel)
		}
	}

	for i := range used {
		if re[i] != beforeRe[i] || im[i] != beforeIm[i] {
			return
		}
	}
	t.Errorf("no element in the %d complete-block elements changed; the stage did nothing", used)
}

// TestButterflyComplexStage4_Degenerate covers every input the public wrapper is
// documented to reject and asserts a no-op rather than a panic. Each twiddle slice
// gets its own short-slice row: the asm kernels derive every address from span and
// blocks, so a short twiddle that slipped past the guard would read out of bounds
// silently instead of panicking the way the Go fallback would.
func TestButterflyComplexStage4_Degenerate(t *testing.T) {
	const span = 4
	n := butterflyStage4Radix * span * 2

	cases := []struct {
		name string
		span int
		nRe  int
		nIm  int
		// twLen[k] is the length of twiddle slice k in {t1r,t1i,t2r,t2i,t3r,t3i};
		// -1 means "use span".
		twLen [6]int
	}{
		{"span=0", 0, n, n, [6]int{-1, -1, -1, -1, -1, -1}},
		{"span=-1", -1, n, n, [6]int{-1, -1, -1, -1, -1, -1}},
		{"nil slices", span, 0, 0, [6]int{-1, -1, -1, -1, -1, -1}},
		{"tw1Re short", span, n, n, [6]int{span - 1, -1, -1, -1, -1, -1}},
		{"tw1Im short", span, n, n, [6]int{-1, span - 1, -1, -1, -1, -1}},
		{"tw2Re short", span, n, n, [6]int{-1, -1, span - 1, -1, -1, -1}},
		{"tw2Im short", span, n, n, [6]int{-1, -1, -1, span - 1, -1, -1}},
		{"tw3Re short", span, n, n, [6]int{-1, -1, -1, -1, span - 1, -1}},
		{"tw3Im short", span, n, n, [6]int{-1, -1, -1, -1, -1, span - 1}},
		{"all twiddles nil", span, n, n, [6]int{0, 0, 0, 0, 0, 0}},
		{"re shorter than one block", span, butterflyStage4Radix*span - 1, n, [6]int{-1, -1, -1, -1, -1, -1}},
		{"im shorter than one block", span, n, butterflyStage4Radix*span - 1, [6]int{-1, -1, -1, -1, -1, -1}},
		{"exactly one element", span, 1, 1, [6]int{-1, -1, -1, -1, -1, -1}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			full, fullIm := stage4Data(n, 12345)
			re := append([]float64(nil), full[:tc.nRe]...)
			im := append([]float64(nil), fullIm[:tc.nIm]...)
			wantRe := append([]float64(nil), re...)
			wantIm := append([]float64(nil), im...)

			t1r, t1i, t2r, t2i, t3r, t3i := stage4Twiddles(span)
			tws := [6][]float64{t1r, t1i, t2r, t2i, t3r, t3i}
			for k := range tws {
				if tc.twLen[k] >= 0 {
					tws[k] = tws[k][:tc.twLen[k]]
				}
			}

			// Must not panic.
			ButterflyComplexStage4(re, im, tc.span, tws[0], tws[1], tws[2], tws[3], tws[4], tws[5])

			for i := range re {
				if re[i] != wantRe[i] {
					t.Errorf("re[%d] = %v, want unchanged %v", i, re[i], wantRe[i])
				}
			}
			for i := range im {
				if im[i] != wantIm[i] {
					t.Errorf("im[%d] = %v, want unchanged %v", i, im[i], wantIm[i])
				}
			}
		})
	}
}

func TestButterflyComplexStage4_AllocFree(t *testing.T) {
	// One case per vectorization path, so an allocation introduced in any of them
	// fails here rather than only in the one shape a single case happens to take.
	for _, span := range []int{1, 2, 3, 8} {
		t.Run(fmt.Sprintf("span=%d", span), func(t *testing.T) {
			n := butterflyStage4Radix * span * 16
			re, im := stage4Data(n, uint64(span))
			t1r, t1i, t2r, t2i, t3r, t3i := stage4Twiddles(span)
			fn := func() {
				ButterflyComplexStage4(re, im, span, t1r, t1i, t2r, t2i, t3r, t3i)
			}
			if a := testing.AllocsPerRun(50, fn); a != 0 {
				t.Errorf("ButterflyComplexStage4 allocated %v times per run, want 0", a)
			}
		})
	}
}

// FuzzButterflyComplexStage4 checks two contracts at once for arbitrary data: the
// dispatched kernel agrees with the Go reference, and the Go reference agrees with
// the two-radix-2 composition it is defined to reproduce.
func FuzzButterflyComplexStage4(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		u := f64sUnit(raw)
		if len(u) < 2 {
			return
		}
		// Pick a small span from the first byte, then split the rest into re/im.
		spanChoices := []int{1, 2, 3, 4, 5, 8}
		span := spanChoices[int(raw[0])%len(spanChoices)]
		m := len(u) / 2
		blockLen := butterflyStage4Radix * span
		blocks := m / blockLen
		if blocks == 0 {
			return
		}
		used := blocks * blockLen
		re := u[:m]
		im := u[m : 2*m]

		t1r, t1i, t2r, t2i, t3r, t3i := stage4Twiddles(span)

		// Dispatched kernel vs Go reference.
		reDisp := append([]float64(nil), re...)
		imDisp := append([]float64(nil), im...)
		reGo := append([]float64(nil), re...)
		imGo := append([]float64(nil), im...)
		ButterflyComplexStage4(reDisp, imDisp, span, t1r, t1i, t2r, t2i, t3r, t3i)
		butterflyComplexStage4x64Go(reGo[:used], imGo[:used], span, blocks, t1r, t1i, t2r, t2i, t3r, t3i)
		for i := range used {
			if !closeEnough(reDisp[i], reGo[i]) {
				t.Fatalf("dispatched vs Go re[%d] (span=%d, blocks=%d): %v vs %v", i, span, blocks, reDisp[i], reGo[i])
			}
			if !closeEnough(imDisp[i], imGo[i]) {
				t.Fatalf("dispatched vs Go im[%d] (span=%d, blocks=%d): %v vs %v", i, span, blocks, imDisp[i], imGo[i])
			}
		}

		// Go reference vs two-radix-2 composition.
		reTwo := append([]float64(nil), re[:used]...)
		imTwo := append([]float64(nil), im[:used]...)
		twRe1 := make([]float64, span)
		twIm1 := make([]float64, span)
		stage4FillRadix2Twiddle(twRe1, twIm1, span)
		ButterflyComplexStage(reTwo, imTwo, span, twRe1, twIm1)
		twRe2 := make([]float64, 2*span)
		twIm2 := make([]float64, 2*span)
		stage4FillRadix2Twiddle(twRe2, twIm2, 2*span)
		ButterflyComplexStage(reTwo, imTwo, 2*span, twRe2, twIm2)
		for i := range used {
			if !closeEnough(reGo[i], reTwo[i]) {
				t.Fatalf("Go vs two-radix2 re[%d] (span=%d, blocks=%d): %v vs %v", i, span, blocks, reGo[i], reTwo[i])
			}
			if !closeEnough(imGo[i], imTwo[i]) {
				t.Fatalf("Go vs two-radix2 im[%d] (span=%d, blocks=%d): %v vs %v", i, span, blocks, imGo[i], imTwo[i])
			}
		}
	})
}

// stage4TwSet holds one span's three radix-4 twiddle-power pairs, so a full
// transform's schedule can be precomputed once outside a timed benchmark loop.
type stage4TwSet struct {
	span                         int
	t1r, t1i, t2r, t2i, t3r, t3i []float64
}

// radix2TwSet holds one span's radix-2 twiddle pair for the same purpose.
type radix2TwSet struct {
	span       int
	twRe, twIm []float64
}

// buildStage4Schedule returns the radix-4 stage schedule for an n-point transform:
// spans 1, 4, 16, ... while a full 4*span block fits.
func buildStage4Schedule(n int) []*stage4TwSet {
	var sched []*stage4TwSet
	for span := 1; butterflyStage4Radix*span <= n; span *= butterflyStage4Radix {
		t1r, t1i, t2r, t2i, t3r, t3i := stage4Twiddles(span)
		sched = append(sched, &stage4TwSet{span, t1r, t1i, t2r, t2i, t3r, t3i})
	}
	return sched
}

// buildRadix2Schedule returns the radix-2 stage schedule for an n-point transform:
// spans 1, 2, 4, ... up to n/2.
func buildRadix2Schedule(n int) []*radix2TwSet {
	var sched []*radix2TwSet
	for span := 1; span < n; span *= 2 {
		twRe := make([]float64, span)
		twIm := make([]float64, span)
		stage4FillRadix2Twiddle(twRe, twIm, span)
		sched = append(sched, &radix2TwSet{span, twRe, twIm})
	}
	return sched
}

// BenchmarkButterflyComplexStage4 measures the decision behind issue #198: does one
// radix-4 stage beat the two radix-2 stages it replaces? The per-span sweep holds a
// fixed 1024-point transform's worth of work (blocks*span near 256) and varies only
// how short the runs are; the three arms are one radix-4 stage, the two radix-2
// stages at span then 2*span, and the pure-Go radix-4 reference. Methodology matches
// BenchmarkButterflyComplexStage: idle host, one sweep per process at -count=1,
// aggregate across rounds with benchstat and read the Stage4-vs-TwoRadix2 ratio.
//
// The in-place butterfly grows its operands to non-finite within a few thousand
// iterations, which costs nothing (Inf/NaN run at full rate) and hits all three arms
// identically, so the ratio is unaffected.
func BenchmarkButterflyComplexStage4(b *testing.B) {
	const n = 1024

	// 6 is in the sweep because it leaves span%4 == 2, the only shape that measures
	// the j-axis scalar tail; every power-of-two span skips it.
	for _, span := range []int{1, 2, 3, 4, 6, 16, 64, 256} {
		blocks := n / (butterflyStage4Radix * span)

		b.Run(fmt.Sprintf("Stage4/span=%d", span), func(b *testing.B) {
			re, im := stage4Data(n, uint64(span))
			t1r, t1i, t2r, t2i, t3r, t3i := stage4Twiddles(span)
			for b.Loop() {
				ButterflyComplexStage4(re, im, span, t1r, t1i, t2r, t2i, t3r, t3i)
			}
		})

		b.Run(fmt.Sprintf("TwoRadix2/span=%d", span), func(b *testing.B) {
			re, im := stage4Data(n, uint64(span))
			twRe1 := make([]float64, span)
			twIm1 := make([]float64, span)
			stage4FillRadix2Twiddle(twRe1, twIm1, span)
			twRe2 := make([]float64, 2*span)
			twIm2 := make([]float64, 2*span)
			stage4FillRadix2Twiddle(twRe2, twIm2, 2*span)
			for b.Loop() {
				ButterflyComplexStage(re, im, span, twRe1, twIm1)
				ButterflyComplexStage(re, im, 2*span, twRe2, twIm2)
			}
		})

		b.Run(fmt.Sprintf("Go/span=%d", span), func(b *testing.B) {
			re, im := stage4Data(n, uint64(span))
			t1r, t1i, t2r, t2i, t3r, t3i := stage4Twiddles(span)
			for b.Loop() {
				butterflyComplexStage4x64Go(re, im, span, blocks, t1r, t1i, t2r, t2i, t3r, t3i)
			}
		})
	}

	// FullCore1024 is a complete power-of-4 transform: 5 radix-4 stages vs the 10
	// radix-2 stages that do the same work.
	benchStage4FullCore(b, "FullCore1024", 1024, false)
	// FullCore2048 is 2*4^5: 5 radix-4 stages plus one trailing radix-2 stage at
	// span n/2, vs 11 radix-2 stages. It exercises the trailing-radix-2 shape.
	benchStage4FullCore(b, "FullCore2048", 2048, true)
}

// benchStage4FullCore runs the radix-4 and radix-2 full-transform arms for one size.
// When trailingRadix2 is set (n is not a power of 4), the radix-4 arm finishes with a
// single radix-2 stage at span n/2, the schedule a Cooley-Tukey radix-4 transform of
// such an n uses.
func benchStage4FullCore(b *testing.B, name string, n int, trailingRadix2 bool) {
	b.Helper()
	sched4 := buildStage4Schedule(n)
	sched2 := buildRadix2Schedule(n)
	trailRe := make([]float64, n/2)
	trailIm := make([]float64, n/2)
	if trailingRadix2 {
		stage4FillRadix2Twiddle(trailRe, trailIm, n/2)
	}

	b.Run(name+"/Radix4", func(b *testing.B) {
		re, im := stage4Data(n, 1)
		for b.Loop() {
			for _, s := range sched4 {
				ButterflyComplexStage4(re, im, s.span, s.t1r, s.t1i, s.t2r, s.t2i, s.t3r, s.t3i)
			}
			if trailingRadix2 {
				ButterflyComplexStage(re, im, n/2, trailRe, trailIm)
			}
		}
	})

	b.Run(name+"/Radix2", func(b *testing.B) {
		re, im := stage4Data(n, 1)
		for b.Loop() {
			for _, s := range sched2 {
				ButterflyComplexStage(re, im, s.span, s.twRe, s.twIm)
			}
		}
	})
}
