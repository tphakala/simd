package f64

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// =============================================================================
// BUTTERFLY COMPLEX STAGE TESTS (float64)
// =============================================================================

// butterflyComplexStageRef is the separately rounded oracle for one radix-2
// decimation-in-time stage: every product rounds to float64 on its own, then the
// sub/add and the butterfly's upper/lower sums follow, which is the contract
// ButterflyComplexStage documents. It indexes re/im directly, models the public
// API's shape rules (partial trailing block untouched, no-op on a short twiddle
// slice), and shares no code with the implementation.
//
// It computes in two passes per block, the four products stored into
// test-allocated arrays first and combined second, so it is written
// independently of the implementation. Each stored product is wrapped in
// float64(), the fusion barrier the Go spec guarantees; a store and reload alone
// is not one. TestButterflyComplexStage_KnownAnswers also pins constants that
// depend on no compiled reference.
func butterflyComplexStageRef(re, im []float64, span int, twRe, twIm []float64) {
	if span <= 0 || len(twRe) < span || len(twIm) < span {
		return
	}
	n := min(len(re), len(im))
	scratch := make([]float64, 4*span)
	p1 := scratch[:span]           // lr*tr
	p2 := scratch[span : 2*span]   // li*ti
	p3 := scratch[2*span : 3*span] // lr*ti
	p4 := scratch[3*span:]         // li*tr
	for k := 0; k+2*span <= n; k += 2 * span {
		for j := range span {
			l := k + span + j
			p1[j] = float64(re[l] * twRe[j])
			p2[j] = float64(im[l] * twIm[j])
			p3[j] = float64(re[l] * twIm[j])
			p4[j] = float64(im[l] * twRe[j])
		}
		for j := range span {
			u, l := k+j, k+span+j
			tempRe := p1[j]
			tempRe -= p2[j]
			tempIm := p3[j]
			tempIm += p4[j]

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

// Tolerances for closeEnough.
//
// ButterflyComplexStage itself is bit-exact and its tests compare bits. These
// bounds serve the comparisons against the fused primitives: the stage versus a
// per-block ButterflyComplex loop (ButterflyComplex fuses on AVX+FMA and NEON,
// the stage never does), and the Stage4 tests. Over the full stageSpans x
// stageBlockCounts sweep the largest difference any of them produces is 1.78e-15,
// on both an AVX+FMA amd64 core and a NEON Cortex-A76.
//
// The bound has to be absolute rather than relative, because the butterfly's
// sum and difference can cancel to near zero while the rounding that produced
// them does not shrink with them. On amd64 the worst RELATIVE difference over
// the same sweep is 1.46e-13, and it occurs on exactly such a cancelled element.
//
// So stageAbsTol carries every row, at 56x the measured worst case. stageRelTol
// is a safety valve for data larger than this fixture's; it only takes over
// above |want| == 10 and no current row depends on it.
const (
	stageAbsTol = 1e-13
	stageRelTol = 1e-14
)

// closeEnough is the tolerance used where a fused primitive is compared with the
// stage: they agree to within rounding rather than exactly.
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

// stageRandData fills a stage's operands with random mantissas scaled by 2^e,
// e uniform in [-30, 30]. The wide exponent spread is what separates fused from
// unfused arithmetic: with it a fused kernel differs from the separately rounded
// oracle on about 4% of the elements.
func stageRandData(rng *rand.Rand, n, span int) (re, im, twRe, twIm []float64) {
	draw := func() float64 {
		return math.Ldexp(rng.Float64()*2-1, rng.IntN(61)-30)
	}
	re = make([]float64, n)
	im = make([]float64, n)
	twRe = make([]float64, span)
	twIm = make([]float64, span)
	for i := range n {
		re[i], im[i] = draw(), draw()
	}
	for j := range span {
		twRe[j], twIm[j] = draw(), draw()
	}
	return re, im, twRe, twIm
}

// checkStageExact runs ButterflyComplexStage and the oracle on copies of the
// operands and requires every element, partial trailing block included, to match
// bit for bit.
func checkStageExact(t *testing.T, re, im []float64, span int, twRe, twIm []float64) {
	t.Helper()
	reGot := append([]float64(nil), re...)
	imGot := append([]float64(nil), im...)
	reWant := append([]float64(nil), re...)
	imWant := append([]float64(nil), im...)

	ButterflyComplexStage(reGot, imGot, span, twRe, twIm)
	butterflyComplexStageRef(reWant, imWant, span, twRe, twIm)

	for _, part := range []struct {
		name      string
		got, want []float64
	}{{"re", reGot, reWant}, {"im", imGot, imWant}} {
		for i := range part.got {
			if !bitsEqF64(part.got[i], part.want[i]) {
				t.Fatalf("span=%d len(re)=%d len(im)=%d: %s[%d] = %v (%#x), want %v (%#x)",
					span, len(re), len(im), part.name, i, part.got[i], math.Float64bits(part.got[i]),
					part.want[i], math.Float64bits(part.want[i]))
			}
		}
	}
}

// TestButterflyComplexStage pins the stage's contract: bit-identical to the
// separately rounded oracle on every tier forTiers can force, for every span and
// block count. Under forTiers each tier, the Go tier and the below-threshold
// shapes that always take Go included, is held to the same oracle.
//
// Three shape families: every span of a 256 and a 1024 point transform (non
// divisor spans leave a trailing partial block that must stay untouched, and is
// compared as part of the whole slice), the stageSpans x stageBlockCounts grid
// (block-axis remainders 1..7 that n = 256 and 1024 cannot produce at span 1 and
// 2), and two odd shapes: a longer im than re, and twiddle slices longer than span.
func TestButterflyComplexStage(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		rng := rand.New(rand.NewPCG(0xb7f1, 0x57a6e))

		t.Run("transform-spans", func(t *testing.T) {
			for _, n := range []int{256, 1024} {
				for span := 1; span <= n/2; span++ {
					re, im, twRe, twIm := stageRandData(rng, n, span)
					checkStageExact(t, re, im, span, twRe, twIm)
				}
			}
		})

		t.Run("grid", func(t *testing.T) {
			for _, span := range stageSpans {
				for _, blocks := range stageBlockCounts {
					re, im, twRe, twIm := stageRandData(rng, 2*span*blocks, span)
					checkStageExact(t, re, im, span, twRe, twIm)
				}
			}
		})

		t.Run("odd-shapes", func(t *testing.T) {
			for _, span := range []int{1, 2, 3, 4, 7, 16} {
				const blocks = 5
				n := 2 * span * blocks
				re, im, twRe, twIm := stageRandData(rng, n, span)
				checkStageExact(t, re, append(im, 0, 0, 0), span, twRe, twIm)
				checkStageExact(t, append(re, 0, 0, 0), im, span, twRe, twIm)

				_, _, twRe, twIm = stageRandData(rng, 0, span+3)
				checkStageExact(t, re, im, span, twRe, twIm)
			}
		})
	})
}

// stageKnownAnswers rows are (lower re, lower im, twiddle re, twiddle im, upper
// re, upper im) -> (new upper re, new upper im, new lower re, new lower im),
// broadcast to every butterfly of a stage. They are hand-derived, so they depend
// on no compiled reference.
//
// With a = 1+2^-27 and c = 1+2^-26, a*a rounds to c and c*a ties to even at
// 1+2^-26+2^-27. Each detector row makes one product's rounding cancel exactly:
// unfused arithmetic gives exactly 0 in the detected component, while fusing that
// product leaves +-2^-54.
var stageKnownAnswers = []struct {
	name string
	in   [6]float64
	want [4]float64
}{
	{"fuse-lr*wr", [6]float64{a27, c26, a27, 1, 0, 0}, [4]float64{0, s25, 0, -s25}},
	{"fuse-li*wi", [6]float64{c26, a27, 1, a27, 0, 0}, [4]float64{0, s25, 0, -s25}},
	{"fuse-lr*wi", [6]float64{a27, -c26, 1, a27, 0, 0}, [4]float64{s25, 0, -s25, 0}},
	{"fuse-li*wr", [6]float64{-c26, a27, a27, 1, 0, 0}, [4]float64{-s25, 0, s25, 0}},
	{"signed-zero", [6]float64{negZero, 0, 1, 1, negZero, negZero}, [4]float64{negZero, 0, 0, negZero}},
	{"subnormal-product-ties-to-even", [6]float64{3 * 0x1p-1074, 0, 0.5, 0, 0, 0}, [4]float64{0x1p-1073, 0, -0x1p-1073, 0}},
	{"subnormal-cancel", [6]float64{0x1p-1074, 0x1p-1074, 1, 1, 0x1p-1073, 0}, [4]float64{0x1p-1073, 0x1p-1073, 0x1p-1073, -0x1p-1073}},
	{"inf-times-zero-re", [6]float64{math.Inf(1), 0, 0, 1, 1, 1}, [4]float64{math.NaN(), math.Inf(1), math.NaN(), math.Inf(-1)}},
	{"inf-times-zero-im", [6]float64{math.Inf(1), 0, 1, 0, 1, 1}, [4]float64{math.Inf(1), math.NaN(), math.Inf(-1), math.NaN()}},
	{"nan", [6]float64{math.NaN(), 1, 1, 1, 1, 1}, [4]float64{math.NaN(), math.NaN(), math.NaN(), math.NaN()}},
}

const (
	a27 = 1 + 0x1p-27
	c26 = 1 + 0x1p-26
	s25 = 2 + 0x1p-25
)

var negZero = math.Copysign(0, -1)

// TestButterflyComplexStage_KnownAnswers checks the rows of stageKnownAnswers on
// every forceable tier, for spans and block counts that reach every vector body,
// remainder and scalar tail.
//
// Regression it catches that the random sweep cannot: the sweep's oracle is
// compiled by the same Go compiler as the fallback, so a toolchain that fused
// past a float64() conversion would make both agree. These constants do not move. Under the Go tier on
// arm64 the four fuse-* rows are the proof that the fallback is not contracted.
// The subnormal rows catch a flush-to-zero change, signed-zero an operand order
// change that keeps values but flips zero signs.
func TestButterflyComplexStage_KnownAnswers(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		for _, row := range stageKnownAnswers {
			t.Run(row.name, func(t *testing.T) {
				for _, span := range []int{1, 2, 3, 4, 5, 8} {
					for blocks := 1; blocks <= 5; blocks++ {
						checkStageKnownAnswer(t, row.in, row.want, span, blocks)
					}
				}
			})
		}
	})
}

// checkStageKnownAnswer broadcasts one known-answer row to every butterfly of a
// span x blocks stage and compares all four outputs of each bitwise.
func checkStageKnownAnswer(t *testing.T, in [6]float64, want [4]float64, span, blocks int) {
	t.Helper()
	n := 2 * span * blocks
	re := make([]float64, n)
	im := make([]float64, n)
	twRe := make([]float64, span)
	twIm := make([]float64, span)
	for j := range span {
		twRe[j], twIm[j] = in[2], in[3]
	}
	for k := 0; k < n; k += 2 * span {
		for j := range span {
			re[k+j], im[k+j] = in[4], in[5]
			re[k+span+j], im[k+span+j] = in[0], in[1]
		}
	}

	ButterflyComplexStage(re, im, span, twRe, twIm)

	for k := 0; k < n; k += 2 * span {
		for j := range span {
			got := [4]float64{re[k+j], im[k+j], re[k+span+j], im[k+span+j]}
			for c := range got {
				if !bitsEqF64(got[c], want[c]) {
					t.Fatalf("span=%d blocks=%d block start %d j=%d component %d = %v (%#x), want %v (%#x)",
						span, blocks, k, j, c, got[c], math.Float64bits(got[c]),
						want[c], math.Float64bits(want[c]))
				}
			}
		}
	}
}

// FuzzButterflyComplexStage compares the stage with the oracle on arbitrary bit
// patterns (NaN, Inf, subnormals, exponent extremes) and shapes: span, a re/im
// length skew and extra twiddle length all come from the input. It runs on the
// host's default tier; the sweep above covers the forced tiers.
func FuzzButterflyComplexStage(f *testing.F) {
	addByteLenSeeds(f)
	// One seed per known-answer row, laid out so the decoder below sees span 1,
	// four blocks (the dispatcher's threshold for leaving the Go path) and no
	// skew: re = (upper, lower) per block, likewise im, and the twiddle repeated
	// to fill its quarter.
	for _, row := range stageKnownAnswers {
		in := row.in // lower re/im, twiddle re/im, upper re/im
		raw := []byte{0, 0, 0}
		var quarters [4][]float64
		for range 4 {
			quarters[0] = append(quarters[0], in[4], in[0])
			quarters[1] = append(quarters[1], in[5], in[1])
			quarters[2] = append(quarters[2], in[2], in[2])
			quarters[3] = append(quarters[3], in[3], in[3])
		}
		for _, q := range quarters {
			for _, v := range q {
				raw = binary.LittleEndian.AppendUint64(raw, math.Float64bits(v))
			}
		}
		f.Add(raw)
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		if len(raw) < 3 {
			return
		}
		spans := [...]int{128, 255, 512}
		span := int(raw[0])%64 + 1
		if raw[0] >= 192 {
			span = spans[int(raw[0])%len(spans)]
		}
		skew, twExtra := int(raw[1]%4), int(raw[2]%4)
		v := f64sBits(raw[3:])

		// Split into re, im, twRe, twIm in equal quarters, so every length is
		// reachable from the byte count.
		q := len(v) / 4
		re, im := v[:q], v[q:2*q]
		twRe, twIm := v[2*q:3*q], v[3*q:4*q]
		if skew > 0 && len(im) > skew {
			im = im[:len(im)-skew]
		}
		if len(twRe) > span+twExtra {
			twRe, twIm = twRe[:span+twExtra], twIm[:span+twExtra]
		}
		checkStageExact(t, re, im, span, twRe, twIm)
	})
}

// TestButterflyComplexStage_MatchesPerBlockLoop pins the migration contract that
// motivates the API: a consumer replacing its per-block ButterflyComplex loop
// with one ButterflyComplexStage call must get the same spectrum back.
//
// Exact equality is not required. The stage never fuses, while ButterflyComplex
// fuses its multiply-adds on AVX+FMA and NEON (and the Go compiler contracts its
// fallback on ARM64), so the two agree to within rounding and only the tolerance
// is portable.
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

// stageFFTTolPerN2 bounds TestButterflyComplexStage_FullFFT's per-bin error,
// which grows as n^2 rather than as n.
//
// Measured over n in [8, 8192] on a NEON Cortex-A76 and an AVX+FMA amd64 core,
// the error grows as roughly n^2 (fitted exponent about 2, the exact figure
// depending on the fit) and this bound holds with at least 65x
// headroom at every size measured (the minimum is 69.7x at n = 128), on arm64
// and on amd64 under both GOAMD64=v1 and v3. The mechanism sits on
// the reference side, not in the kernels: the naive DFT accumulates n terms per
// bin, and |X[k]| for this input itself grows like O(n).
//
// It replaces a 1e-10*n bound, which grew as n^1 against an n^2 error, so its
// headroom shrank with size. Scaling by n^2 keeps it roughly flat instead, so
// adding a larger row later does not silently spend the margin.
const stageFFTTolPerN2 = 1e-14

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
				// Accumulated rounding, dominated by the naive DFT reference,
				// grows as n^2; see stageFFTTolPerN2.
				tol := stageFFTTolPerN2 * float64(n) * float64(n)
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

// stageLocalArrays calls the stage on slices of its own local arrays, the shape
// go-mp3 uses. A dispatch through a function value would force both arrays to the
// heap here (2 allocations per call), which the heap-slice cases below cannot see.
//
//go:noinline
func stageLocalArrays(span int, twRe, twIm []float64) {
	var re, im [64]float64
	ButterflyComplexStage(re[:], im[:], span, twRe, twIm)
}

//go:noinline
func stageLocalStruct(span int, twRe, twIm []float64) {
	var buf struct{ re, im [64]float64 }
	ButterflyComplexStage(buf.re[:], buf.im[:], span, twRe, twIm)
}

func TestButterflyComplexStage_AllocFree(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
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

				if a := testing.AllocsPerRun(50, func() { stageLocalArrays(span, twRe, twIm) }); a != 0 {
					t.Errorf("local arrays: %v allocs per run, want 0", a)
				}
				if a := testing.AllocsPerRun(50, func() { stageLocalStruct(span, twRe, twIm) }); a != 0 {
					t.Errorf("local struct: %v allocs per run, want 0", a)
				}
			})
		}
	})
}

func BenchmarkButterflyComplexStage(b *testing.B) {
	// A fixed 256- or 1024-point transform's worth of butterflies per stage, swept
	// over span. Work is identical within an n: blocks*span is constant at n/2, so
	// the only variable is how short the runs are. This is the shape issue #198 measured.
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
	for _, n := range []int{256, 1024} {
		benchStageSweep(b, n)
	}
}

func benchStageSweep(b *testing.B, n int) {
	b.Helper()

	// 6 and 10 are in the sweep because both leave span%4 == 2, so they are the only
	// rows that measure the j-axis scalar tail; every power-of-two span skips it.
	// They also give non-power-of-two block counts (85 and 51). The two block-axis
	// remainders stay unmeasured here: they need blocks%4 != 0 at span 1 or
	// blocks%2 != 0 at span 2, which n = 1024 cannot produce. Those are covered by
	// stageBlockCounts in the correctness tests, not by this benchmark.
	for _, span := range []int{1, 2, 4, 6, 8, 10, 16, 32, 64, 128, 256, 512} {
		if span > n/2 {
			continue
		}
		blocks := n / (2 * span)

		// Fresh buffers per sub-benchmark so every variant starts from the same
		// data rather than from whatever the previous one left behind.
		b.Run(fmt.Sprintf("Stage/n=%d/span=%d", n, span), func(b *testing.B) {
			re, im, twRe, twIm := stageTestData(n, span)
			for b.Loop() {
				ButterflyComplexStage(re, im, span, twRe, twIm)
			}
		})

		b.Run(fmt.Sprintf("PerBlock/n=%d/span=%d", n, span), func(b *testing.B) {
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

		b.Run(fmt.Sprintf("Go/n=%d/span=%d", n, span), func(b *testing.B) {
			re, im, twRe, twIm := stageTestData(n, span)
			for b.Loop() {
				butterflyComplexStage64Go(re, im, span, blocks, twRe, twIm)
			}
		})
	}
}
