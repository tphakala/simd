//go:build amd64

package i32

import "github.com/tphakala/simd/cpu"

// Minimum number of int32 pairs before the AVX kernel beats the scalar loop.
// AVX processes 8 pairs (8 int32 per 256-bit register) per iteration.
const minAVXElements = 8

// hasAVX gates the SIMD kernels. The interleave kernels use only AVX1
// instructions (VUNPCKLPS / VPERM2F128 / VSHUFPS), so AVX without AVX2 is
// sufficient. This checks the CPU feature explicitly rather than relying on
// length alone, so the package is safe on the (now rare) AVX-less amd64
// baseline.
var hasAVX = cpu.X86.AVX

func interleave2I32(dst, a, b []int32) {
	if hasAVX && len(a) >= minAVXElements {
		interleave2AVX(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2I32(a, b, src []int32) {
	if hasAVX && len(a) >= minAVXElements {
		deinterleave2AVX(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

//go:noescape
func interleave2AVX(dst, a, b []int32)

//go:noescape
func deinterleave2AVX(a, b, src []int32)

// The element-wise arithmetic and min/max reductions operate on 256-bit integer
// lanes (VPADDD / VPSUBD / VPMINSD / VPMAXSD), which require AVX2 rather than the
// AVX1 that suffices for the float-shuffle interleave kernels above. They gate on
// AVX2 explicitly and fall back to the pure-Go reference otherwise.
var hasAVX2 = cpu.X86.AVX2

func addI32(dst, a, b []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		addAVX2(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func subI32(dst, a, b []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		subAVX2(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

//go:noescape
func addAVX2(dst, a, b []int32)

//go:noescape
func subAVX2(dst, a, b []int32)

// Tier-3 thresholds: one 8-wide (256-bit) vector block each, independent
// literals rather than aliases of minAVXElements, which they happen to equal,
// so retuning the interleave kernels cannot silently move these. Both kernels
// gate on AVX2 (VPADDD/VPABSD are 256-bit integer ops) and are correct at any
// length (each falls through to a scalar tail), so these are performance cuts
// only, never a safety requirement.
const (
	minAVX2Sum = 8
	minAVX2Abs = 8
)

func sumI32(a []int32) int32 {
	if hasAVX2 && len(a) >= minAVX2Sum {
		return sumAVX2(a)
	}
	return sumGo(a)
}

func absI32(dst, a []int32) {
	if hasAVX2 && len(dst) >= minAVX2Abs {
		absAVX2(dst, a)
		return
	}
	absGo(dst, a)
}

//go:noescape
func sumAVX2(a []int32) int32

//go:noescape
func absAVX2(dst, a []int32)

// minAVX2NegWhereNeg is one 8-wide (256-bit) block, an independent literal like
// the tier-3 thresholds above. The kernel is correct at any length (it falls
// through to a scalar tail), so this is a performance cut only, never a safety
// requirement. It gates on AVX2 because VPSRAD/VPXOR/VPSUBD are 256-bit integer
// ops.
const minAVX2NegWhereNeg = 8

func negWhereNegI32(dst, mag []int32, sign []float32) {
	if hasAVX2 && len(dst) >= minAVX2NegWhereNeg {
		negWhereNegAVX2(dst, mag, sign)
		return
	}
	negWhereNegGo(dst, mag, sign)
}

//go:noescape
func negWhereNegAVX2(dst, mag []int32, sign []float32)

// minAVX2ScaleQ31 and minAVX2ScaleQ15 are one 8-wide (256-bit) block each,
// independent literals like the tier-3 thresholds above. Both kernels are correct
// at any length (each falls through to a scalar tail), so these are performance
// cuts only, never a safety requirement. They gate on AVX2 because VPMULDQ/VPSRLQ/
// VPSLLQ/VPBLENDD are 256-bit integer ops.
const (
	minAVX2ScaleQ31 = 8
	minAVX2ScaleQ15 = 8
)

func scaleQ31I32(dst, a []int32, k int32) {
	if hasAVX2 && len(dst) >= minAVX2ScaleQ31 {
		scaleQ31AVX2(dst, a, k)
		return
	}
	scaleQ31Go(dst, a, k)
}

func scaleQ15I32(dst, a []int32, k int16) {
	if hasAVX2 && len(dst) >= minAVX2ScaleQ15 {
		scaleQ15AVX2(dst, a, k)
		return
	}
	scaleQ15Go(dst, a, k)
}

//go:noescape
func scaleQ31AVX2(dst, a []int32, k int32)

//go:noescape
func scaleQ15AVX2(dst, a []int32, k int16)

// minAVX2GainQ31 is one 8-wide (256-bit) block. The kernel is correct at any length
// (it falls through to a scalar tail), so this is a performance cut only, never a
// safety requirement. It gates on AVX2 because the VPMULDQ Q31 core plus the
// VPSLLD/VPADDD/VPSRAD pre- and post-shift stages are all 256-bit integer ops.
//
// This sat at 160 until #268. The kernel loaded its two runtime shift counts into XMM
// with legacy MOVQ after the VEX gain broadcast had already dirtied the upper YMM
// state, so every call paid two AVX-SSE transition assists, MEASURED as ~80-90 ns flat
// across n = 8..128 (the earlier note blaming a post-VZEROUPPER warmup was wrong: every
// AVX2 kernel here ends in VZEROUPPER and only this one carried the cost). Loading the
// counts with VEX VMOVD instead (#268) removes both assists, and the kernel now wins
// from the first 8-wide block like sumSqShiftedQ31AVX2. MEASURED on the i7-1260P (AVX2,
// taskset P-core, 20 process rounds, benchstat medians): the AVX2 kernel beats
// gainQ31Go at every size from n=8 up (n=8: 3.5 vs 6.4 ns, 1.8x; n=16: 4.0 vs 11.3;
// n=64: 8.4 vs 46.6, 5.6x; direct before/after: n=64 fell from 98.9 to 8.4 ns), so the
// cut drops to one block, like the scale kernels and minAVX2SumSqShiftedQ31 = 8. The
// arm64 NEON path stays at minNEONGainQ31 = 4. See #268 and #251.
const minAVX2GainQ31 = 8

func gainQ31I32(dst, a []int32, gain int32, preShift, postShift int) {
	if hasAVX2 && len(dst) >= minAVX2GainQ31 {
		gainQ31AVX2(dst, a, gain, preShift, postShift)
		return
	}
	gainQ31Go(dst, a, gain, preShift, postShift)
}

//go:noescape
func gainQ31AVX2(dst, a []int32, gain int32, preShift, postShift int)

// minAVX2Butterfly is one 8-wide (256-bit) block, an independent literal like
// the tier-3 thresholds above. The kernel is correct at any length (it falls
// through to a scalar tail), so this is a performance cut only, never a safety
// requirement. It gates on AVX2 because VPADDD/VPSUBD are 256-bit integer ops.
const minAVX2Butterfly = 8

func butterflyI32(lo, hi []int32) {
	if hasAVX2 && len(lo) >= minAVX2Butterfly {
		butterflyAVX2(lo, hi)
		return
	}
	butterflyGo(lo, hi)
}

//go:noescape
func butterflyAVX2(lo, hi []int32)

// minMaxI32 dispatches the signed int32 min/max reduction. The AVX2 kernel does
// the reduction in 8-wide VPMINSD/VPMAXSD lanes with a scalar tail, so it gates
// on AVX2 and at least one full 8-element block; shorter slices use the pure-Go
// reference. res is non-empty (the public MinMax guards the empty case).
func minMaxI32(res []int32) (minVal, maxVal int32) {
	if hasAVX2 && len(res) >= minAVXElements {
		return minMaxAVX2(res)
	}
	return minMaxGo(res)
}

//go:noescape
func minMaxAVX2(res []int32) (minVal, maxVal int32)

// minAVX2MaxAbs is one 8-wide (256-bit) block, an independent literal like the
// tier-3 thresholds above. The kernel does the signed min/max reduction in
// 8-wide VPMINSD/VPMAXSD lanes with a scalar tail before combining to
// max(maxVal, -minVal), so it gates on AVX2 and at least one full 8-element
// block; shorter slices use the pure-Go reference. a is non-empty (the public
// MaxAbs guards the empty case).
const minAVX2MaxAbs = 8

func maxAbsI32(a []int32) int32 {
	if hasAVX2 && len(a) >= minAVX2MaxAbs {
		return maxAbsAVX2(a)
	}
	return maxAbsGo(a)
}

//go:noescape
func maxAbsAVX2(a []int32) int32

// minAVX2SumSqShiftedQ31 is one 8-wide (256-bit) block. The kernel is correct at
// any length (it falls through to a scalar tail), so this is a performance cut
// only, never a safety requirement. It gates on AVX2 because the VPSLLD pre-shift,
// the VPMULDQ Q31 square and the VPADDD accumulate are all 256-bit integer ops.
//
// This kernel is MEASURED to win from the first 8-wide block, so the cut stays at one
// block, matching Sum and MaxAbs. On the amd64 A/B host (i7-1260P, AVX2, taskset
// P-core) it beats sumSqShiftedQ31Go at every size from n=8 up (n=8: 2.4 vs 4.3 ns,
// 1.8x; n=16: 2.8 vs 7.7; n=64: 6.8 vs 34.6, 5x), with the Go side scaling ~4x
// steeper, so there is no small-n regime where SIMD loses. Like gainQ31AVX2 since
// #268, it loads its runtime shift count with VEX VMOVD (not legacy MOVQ), so it pays
// no AVX-SSE transition assist and has no fixed per-call cost. The arm64 NEON path
// likewise stays at minNEONSumSqShiftedQ31 = 4.
const minAVX2SumSqShiftedQ31 = 8

func sumSqShiftedQ31I32(a []int32, shift int) int32 {
	if hasAVX2 && len(a) >= minAVX2SumSqShiftedQ31 {
		return sumSqShiftedQ31AVX2(a, shift)
	}
	return sumSqShiftedQ31Go(a, shift)
}

//go:noescape
func sumSqShiftedQ31AVX2(a []int32, shift int) int32

// minAVX2FIR is one 8-wide (256-bit) output block: FIRValidQ15's AVX2 kernel
// vectorizes over 8 outputs per iteration with a scalar-output tail, so it gates
// on AVX2 and at least one full 8-output block; shorter outputs use the pure-Go
// reference. The kernel is correct at any output length via the scalar-output
// tail, so this is a performance cut only, never a safety requirement. It gates
// on AVX2 because VPMULDQ/VPSRLQ/VPSLLQ/VPBLENDD/VPADDD are 256-bit integer ops.
// len(dst) here is already the clamped output count n from the public FIRValidQ15.
const minAVX2FIR = 8

func firValidQ15I32(dst, x []int32, taps []int16) {
	if hasAVX2 && len(dst) >= minAVX2FIR {
		firValidQ15AVX2(dst, x, taps)
		return
	}
	firValidQ15Go(dst, x, taps)
}

//go:noescape
func firValidQ15AVX2(dst, x []int32, taps []int16)

// minAVX2SymFIR is one 8-wide (256-bit) output block: FIRSymValidQ15's AVX2
// kernel vectorizes over 8 outputs per iteration with a scalar-output tail, so it
// gates on AVX2 and at least one full 8-output block; shorter outputs use the
// pure-Go reference. The kernel is correct at any output length via the
// scalar-output tail, so this is a performance cut only, never a safety
// requirement. It gates on AVX2 because VPMULDQ/VPSRLQ/VPSLLQ/VPBLENDD/VPADDD are
// 256-bit integer ops. len(dst) here is already the clamped output count n from
// the public FIRSymValidQ15.
const minAVX2SymFIR = 8

func firSymValidQ15I32(dst, x []int32, center int16, pairs []int16) {
	if hasAVX2 && len(dst) >= minAVX2SymFIR {
		firSymValidQ15AVX2(dst, x, center, pairs)
		return
	}
	firSymValidQ15Go(dst, x, center, pairs)
}

//go:noescape
func firSymValidQ15AVX2(dst, x []int32, center int16, pairs []int16)
