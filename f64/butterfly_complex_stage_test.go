package f64

import (
	"fmt"
	"math"
	"testing"
)

// =============================================================================
// BUTTERFLY COMPLEX STAGE TESTS (float64)
// =============================================================================

// butterflyComplexStageRef is an independent reference for one radix-2
// decimation-in-time stage. It indexes re/im directly rather than delegating to
// butterflyComplex64Go, so it does not share code with the implementation under
// test.
//
// It is not an unfused baseline: whether these expressions fuse is the Go
// compiler's choice, and it makes different choices per architecture (it
// contracts them on ARM64, not on AMD64). Comparisons against it are therefore
// tolerance-based, never bit-exact.
func butterflyComplexStageRef(re, im []float64, span int, twRe, twIm []float64) {
	if span <= 0 || len(twRe) < span || len(twIm) < span {
		return
	}
	n := min(len(re), len(im))
	for k := 0; k+2*span <= n; k += 2 * span {
		for j := range span {
			u, l := k+j, k+span+j

			lr, li := re[l], im[l]
			tr, ti := twRe[j], twIm[j]
			tempRe := lr*tr - li*ti
			tempIm := lr*ti + li*tr

			ur, ui := re[u], im[u]
			re[u] = ur + tempRe
			im[u] = ui + tempIm
			re[l] = ur - tempRe
			im[l] = ui - tempIm
		}
	}
}

// stageTestData fills split-complex arrays and a twiddle pair with varied,
// non-degenerate values. n is the element count, span the twiddle count.
func stageTestData(n, span int) (re, im, twRe, twIm []float64) {
	re = make([]float64, n)
	im = make([]float64, n)
	twRe = make([]float64, span)
	twIm = make([]float64, span)
	for i := range n {
		re[i] = math.Sin(float64(i)*0.317) * float64(i%7+1)
		im[i] = math.Cos(float64(i)*0.523) * float64(i%5+1)
	}
	for j := range span {
		ang := -math.Pi * (float64(j) + stageTwiddlePhase) / float64(span)
		sin, cos := math.Sincos(ang)
		twRe[j] = cos
		twIm[j] = sin
	}
	return re, im, twRe, twIm
}

// stageTwiddlePhase offsets the twiddle angle so that j == 0 is not the identity
// twiddle (1, 0).
//
// This is load-bearing, not cosmetic. At span == 1 there is exactly one twiddle,
// so an unoffset generator hands every span == 1 test tw = (1, 0), under which the
// complex multiply collapses to a pass-through: temp_re = lr*1 - li*0 computes the
// same bits as lr*1 + li*0. Any sign error in the span == 1 block-axis kernels is
// then unobservable, and those two kernels are the whole point of the primitive.
// Verified: before this offset existed, flipping VSUBPD to VADDPD in the AMD64
// span == 1 path and FMLS to FMLA in the ARM64 one both passed the entire suite on
// real hardware. A real radix-2 stage-1 twiddle IS 1+0i, so TestButterflyComplexStage_FullFFT
// cannot cover this either; only a synthetic twiddle can.
//
// The phase must be non-integral, or sin lands on zero at j == 0; and 2*(j+phase)
// must never be an odd multiple of span, or cos lands on zero. 0.4 satisfies both
// for every span in stageSpans, keeps |tw| == 1, and leaves both components above
// 1e-3 in magnitude.
const stageTwiddlePhase = 0.4

// Tolerances for closeEnough. The absolute floor carries values near zero, where a
// relative bound is meaningless because the butterfly's sum and difference can
// cancel to nearly nothing; the relative bound carries everything else. Both match
// TestButterflyComplex's relTol.
const (
	stageAbsTol = 1e-9
	stageRelTol = 1e-11
)

// closeEnough is the tolerance used throughout: the vector and scalar paths fuse
// their multiply-adds differently, so they agree to within rounding rather than
// exactly.
func closeEnough(got, want float64) bool {
	diff := math.Abs(got - want)
	return diff <= stageAbsTol || diff <= math.Abs(want)*stageRelTol
}

// stageSpans covers every dispatch path: span 1 and 2 are the block-axis AVX
// paths, span 3 is the only span that fills neither axis and so runs entirely in
// the j-axis path's scalar tail, and 4..64 walk that path with and without a
// tail (span%4 != 0).
var stageSpans = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 32, 64}

// stageBlockCounts exercise the block-axis remainder tails: span 1 consumes 4
// blocks per iteration and span 2 consumes 2, so counts 1..7 cover every
// leftover case for both.
var stageBlockCounts = []int{1, 2, 3, 4, 5, 6, 7, 8, 16, 33}

func TestButterflyComplexStage(t *testing.T) {
	for _, span := range stageSpans {
		for _, blocks := range stageBlockCounts {
			t.Run(fmt.Sprintf("span=%d/blocks=%d", span, blocks), func(t *testing.T) {
				n := 2 * span * blocks
				re, im, twRe, twIm := stageTestData(n, span)

				reRef := make([]float64, n)
				imRef := make([]float64, n)
				copy(reRef, re)
				copy(imRef, im)

				ButterflyComplexStage(re, im, span, twRe, twIm)
				butterflyComplexStageRef(reRef, imRef, span, twRe, twIm)

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

// TestButterflyComplexStage_MatchesPerBlockLoop pins the migration contract that
// motivates the API: a consumer replacing its per-block ButterflyComplex loop
// with one ButterflyComplexStage call must get the same spectrum back.
//
// Exact equality is not required. The stage and the per-block wrapper can reach
// different code for the same span (below its SIMD threshold ButterflyComplex
// runs the scalar fallback while the stage still vectorizes), and the two do not
// round identically. Which side is fused is not uniform either: the Go compiler
// contracts the fallback's multiply-add on ARM64 but not on AMD64, so the
// direction of the difference is arch-dependent and only the tolerance is
// portable.
func TestButterflyComplexStage_MatchesPerBlockLoop(t *testing.T) {
	for _, span := range stageSpans {
		for _, blocks := range []int{1, 3, 8} {
			t.Run(fmt.Sprintf("span=%d/blocks=%d", span, blocks), func(t *testing.T) {
				n := 2 * span * blocks
				re, im, twRe, twIm := stageTestData(n, span)

				reLoop := make([]float64, n)
				imLoop := make([]float64, n)
				copy(reLoop, re)
				copy(imLoop, im)

				ButterflyComplexStage(re, im, span, twRe, twIm)

				for k := 0; k+2*span <= n; k += 2 * span {
					ButterflyComplex(
						reLoop[k:k+span], imLoop[k:k+span],
						reLoop[k+span:k+2*span], imLoop[k+span:k+2*span],
						twRe, twIm,
					)
				}

				for i := range n {
					if !closeEnough(re[i], reLoop[i]) {
						t.Errorf("re[%d]: stage=%v, per-block loop=%v", i, re[i], reLoop[i])
					}
					if !closeEnough(im[i], imLoop[i]) {
						t.Errorf("im[%d]: stage=%v, per-block loop=%v", i, im[i], imLoop[i])
					}
				}
			})
		}
	}
}

// stageSIMDvsGoTol bounds the SIMD-vs-Go difference. Both sides compute the same
// stage over the same data and differ only in how their multiply-adds fuse, so this
// is a tighter absolute bound than closeEnough, which compares against a separate
// scalar reference. On stageTestData's inputs (magnitudes of order 10) the observed
// difference is a few ULP, far under this.
const stageSIMDvsGoTol = 1e-12

func TestButterflyComplexStage_SIMDvsGo(t *testing.T) {
	for _, span := range stageSpans {
		for _, blocks := range stageBlockCounts {
			t.Run(fmt.Sprintf("span=%d/blocks=%d", span, blocks), func(t *testing.T) {
				n := 2 * span * blocks
				re, im, twRe, twIm := stageTestData(n, span)

				reGo := make([]float64, n)
				imGo := make([]float64, n)
				copy(reGo, re)
				copy(imGo, im)

				butterflyComplexStage64(re, im, span, blocks, twRe, twIm)
				butterflyComplexStage64Go(reGo, imGo, span, blocks, twRe, twIm)

				for i := range n {
					if math.Abs(re[i]-reGo[i]) > stageSIMDvsGoTol {
						t.Errorf("re[%d]: SIMD=%v, Go=%v", i, re[i], reGo[i])
					}
					if math.Abs(im[i]-imGo[i]) > stageSIMDvsGoTol {
						t.Errorf("im[%d]: SIMD=%v, Go=%v", i, im[i], imGo[i])
					}
				}
			})
		}
	}
}

// TestButterflyComplexStage_FullFFT drives a complete iterative Cooley-Tukey
// transform through ButterflyComplexStage and checks it against a naive DFT.
// Every stage of an n-point transform uses a different span (1, 2, 4, ... n/2),
// so one run walks every vectorization path in sequence and would catch a path
// that is individually self-consistent but wrong relative to the others.
func TestButterflyComplexStage_FullFFT(t *testing.T) {
	for _, n := range []int{8, 16, 32, 64, 256, 1024} {
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

			twRe := make([]float64, n/2)
			twIm := make([]float64, n/2)
			for span := 1; span < n; span *= 2 {
				for j := range span {
					ang := -math.Pi * float64(j) / float64(span)
					twRe[j] = math.Cos(ang)
					twIm[j] = math.Sin(ang)
				}
				ButterflyComplexStage(re, im, span, twRe[:span], twIm[:span])
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
				// Accumulated FFT/DFT rounding grows with n; scale the bound by
				// the transform size rather than using a flat epsilon.
				tol := 1e-10 * float64(n)
				if math.Abs(re[k]-wantRe) > tol || math.Abs(im[k]-wantIm) > tol {
					t.Fatalf("bin %d = (%v, %v), want (%v, %v), tol %v",
						k, re[k], im[k], wantRe, wantIm, tol)
				}
			}
		})
	}
}

// TestButterflyComplexStage_PartialBlockUntouched pins the documented boundary:
// blocks are processed while k+2*span <= n, so a trailing run shorter than a
// full block must be left exactly as it was.
func TestButterflyComplexStage_PartialBlockUntouched(t *testing.T) {
	const sentinel = -12345.5

	for _, span := range stageSpans {
		// Dedupe: for small spans the candidates collapse onto the same value, and
		// Go silently disambiguates the duplicate subtest names as #01/#02, which
		// reads as coverage that is not there.
		seen := map[int]bool{}
		extras := make([]int, 0, 4)
		for _, extra := range []int{1, 2, span, 2*span - 1} {
			if extra <= 0 || extra >= 2*span || seen[extra] {
				continue
			}
			seen[extra] = true
			extras = append(extras, extra)
		}

		// blocks 4 as well as 3: the AMD64 span == 1 path consumes four blocks per
		// vector iteration, so at blocks == 3 it never runs its vector loop at all
		// and a trailing partial block would only ever be seen by the scalar tail.
		for _, blocks := range []int{3, 4} {
			for _, extra := range extras {
				t.Run(fmt.Sprintf("span=%d/blocks=%d/extra=%d", span, blocks, extra), func(t *testing.T) {
					checkPartialBlockUntouched(t, span, blocks, extra, sentinel)
				})
			}
		}
	}
}

// checkPartialBlockUntouched runs one shape of the partial-block boundary check:
// the trailing `extra` elements must be untouched, and the complete blocks must
// have been transformed.
func checkPartialBlockUntouched(t *testing.T, span, blocks, extra int, sentinel float64) {
	t.Helper()

	used := 2 * span * blocks
	n := used + extra
	re, im, twRe, twIm := stageTestData(n, span)
	beforeRe := append([]float64(nil), re...)
	beforeIm := append([]float64(nil), im...)
	for i := used; i < n; i++ {
		re[i] = sentinel
		im[i] = sentinel
	}

	ButterflyComplexStage(re, im, span, twRe, twIm)

	for i := used; i < n; i++ {
		if re[i] != sentinel {
			t.Errorf("re[%d] = %v, want untouched sentinel %v", i, re[i], sentinel)
		}
		if im[i] != sentinel {
			t.Errorf("im[%d] = %v, want untouched sentinel %v", i, im[i], sentinel)
		}
	}

	// Assert the complete blocks WERE transformed. Without this the test passes
	// against a stage that does nothing at all, which is exactly what its name
	// promises to rule out.
	for i := range used {
		if re[i] != beforeRe[i] || im[i] != beforeIm[i] {
			return
		}
	}
	t.Errorf("no element in the %d complete-block elements changed; the stage did nothing", used)
}

// TestButterflyComplexStage_Degenerate covers every input the public wrapper is
// documented to reject, and asserts it is a no-op rather than a panic.
func TestButterflyComplexStage_Degenerate(t *testing.T) {
	const span = 4
	n := 2 * span * 2
	_, _, twRe, twIm := stageTestData(n, span)

	// nTwRe and nTwIm are separate on purpose. Driving both twiddle slices from a
	// single field makes len(twRe) < span short-circuit first in every row, so the
	// len(twIm) < span clause is never the one that fires and deleting it from the
	// wrapper leaves the suite green. What it costs a caller passing a short twIm
	// is worse than a lost no-op: the Go fallback would panic slicing twIm[:span],
	// but the asm kernels derive every address from span and blocks, so they would
	// read out of bounds silently. Each clause needs its own row.
	cases := []struct {
		name  string
		span  int
		nRe   int
		nIm   int
		nTwRe int
		nTwIm int
	}{
		{"span=0", 0, n, n, span, span},
		{"span=-1", -1, n, n, span, span},
		{"nil slices", span, 0, 0, span, span},
		{"nil twiddles", span, n, n, 0, 0},
		{"twRe short", span, n, n, span - 1, span},
		{"twIm short", span, n, n, span, span - 1},
		{"twRe nil, twIm ok", span, n, n, 0, span},
		{"twIm nil, twRe ok", span, n, n, span, 0},
		{"re shorter than one block", span, 2*span - 1, n, span, span},
		{"im shorter than one block", span, n, 2*span - 1, span, span},
		{"exactly one element", span, 1, 1, span, span},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Build re and im at exactly the requested lengths, independently,
			// so "re shorter" and "im shorter" really are distinct cases.
			full, _, _, _ := stageTestData(n, span)
			re := append([]float64(nil), full[:tc.nRe]...)
			im := append([]float64(nil), full[:tc.nIm]...)

			wantRe := append([]float64(nil), re...)
			wantIm := append([]float64(nil), im...)

			// Must not panic.
			ButterflyComplexStage(re, im, tc.span, twRe[:tc.nTwRe], twIm[:tc.nTwIm])

			// Every case here is a no-op: either the guard rejects it outright, or
			// it clears the guard but contains no complete 2*span block.
			for i := range im {
				if im[i] != wantIm[i] {
					t.Errorf("im[%d] = %v, want unchanged %v", i, im[i], wantIm[i])
				}
			}
			for i := range re {
				if re[i] != wantRe[i] {
					t.Errorf("re[%d] = %v, want unchanged %v", i, re[i], wantRe[i])
				}
			}
		})
	}
}

func TestButterflyComplexStage_AllocFree(t *testing.T) {
	// One case per vectorization path, so an allocation introduced in any of
	// them fails here rather than only in the one shape a single case happens
	// to take.
	for _, span := range []int{1, 2, 3, 8} {
		t.Run(fmt.Sprintf("span=%d", span), func(t *testing.T) {
			n := 2 * span * 16
			re, im, twRe, twIm := stageTestData(n, span)
			fn := func() { ButterflyComplexStage(re, im, span, twRe, twIm) }
			if a := testing.AllocsPerRun(50, fn); a != 0 {
				t.Errorf("ButterflyComplexStage allocated %v times per run, want 0", a)
			}
		})
	}
}

func BenchmarkButterflyComplexStage(b *testing.B) {
	// A fixed 1024-point transform's worth of butterflies per stage, swept over
	// span. Work is identical in every row: blocks*span is constant at 512, so the
	// only variable is how short the runs are. This is the shape issue #198 measured.
	//
	// Measure on an IDLE host, and interleave by running one full sweep per process
	// invocation at -test.count=1 rather than using -test.count=N. Two reasons:
	// -count=N repeats each sub-benchmark N times consecutively rather than
	// alternating the arms, and sustained benchmarking on a laptop-class part drops
	// the sustained-load P-state, which has been observed to halve the reported
	// rate for a whole run and to recover only after the machine idles. Both inflate
	// absolutes without touching the Stage-vs-PerBlock ratio, so prefer the ratio.
	//
	// Two things the PerBlock row is NOT. At span 1 it does not reach the per-block
	// SIMD kernel on either architecture, and at span 2 it does not on AMD64, because
	// ButterflyComplex vectorizes only from len >= 4 there (len >= 2 on ARM64); those
	// rows compare the stage against a scalar loop plus call overhead rather than
	// against vector code, so they measure a new vector path as well as removed call
	// overhead. And the in-place butterfly grows its operands, so every variant
	// saturates to non-finite within a few thousand iterations and spends almost the
	// whole run there; that costs nothing (Inf and NaN are full rate, and since the
	// butterfly doubles energy per application the data grows away from the denormal
	// range rather than into it), and it hits all three variants identically.
	const n = 1024

	// 6 and 10 are in the sweep because both leave span%4 == 2, so they are the only
	// rows that measure the j-axis scalar tail; every power-of-two span skips it.
	// They also give non-power-of-two block counts (85 and 51). The two block-axis
	// remainders stay unmeasured here: they need blocks%4 != 0 at span 1 or
	// blocks%2 != 0 at span 2, which n = 1024 cannot produce. Those are covered by
	// stageBlockCounts in the correctness tests, not by this benchmark.
	for _, span := range []int{1, 2, 4, 6, 8, 10, 16, 32, 64, 128, 256, 512} {
		blocks := n / (2 * span)

		// Fresh buffers per sub-benchmark so every variant starts from the same
		// data rather than from whatever the previous one left behind.
		b.Run(fmt.Sprintf("Stage/span=%d", span), func(b *testing.B) {
			re, im, twRe, twIm := stageTestData(n, span)
			for b.Loop() {
				ButterflyComplexStage(re, im, span, twRe, twIm)
			}
		})

		b.Run(fmt.Sprintf("PerBlock/span=%d", span), func(b *testing.B) {
			re, im, twRe, twIm := stageTestData(n, span)
			for b.Loop() {
				for k := 0; k < blocks*2*span; k += 2 * span {
					ButterflyComplex(
						re[k:k+span], im[k:k+span],
						re[k+span:k+2*span], im[k+span:k+2*span],
						twRe, twIm,
					)
				}
			}
		})

		b.Run(fmt.Sprintf("Go/span=%d", span), func(b *testing.B) {
			re, im, twRe, twIm := stageTestData(n, span)
			for b.Loop() {
				butterflyComplexStage64Go(re, im, span, blocks, twRe, twIm)
			}
		})
	}
}
