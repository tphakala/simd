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

// minAVX2GainQ31 gates the AVX2 GainQ31 kernel on total length. The kernel is
// correct at any length (it falls through to a scalar tail), so this is a
// performance cut only, never a safety requirement. It gates on AVX2 because the
// VPMULDQ Q31 core plus the VPSLLD/VPADDD/VPSRAD pre- and post-shift stages are
// all 256-bit integer ops.
//
// Unlike ScaleQ31 (whose single-broadcast prologue lets it win from one block),
// the AVX2 GainQ31 path carries a large fixed per-call cost that dwarfs its
// per-element work at small n: MEASURED as ~80-90 ns flat across n = 8..128 on two
// amd64 hosts (an i7-class and a Xeon-class part), while gainQ31Go scales linearly
// from ~6 ns (n=8). The kernel's own per-element work is cheap (~0.2 ns/elem vs Go's
// ~0.75), so the crossover is set entirely by that fixed cost: gainQ31Go is faster
// through n=144, the two are at parity near n=160, and the kernel pulls ahead from
// n=176 (reaching ~2x by n=320). The arm64 NEON path shows no such fixed cost and
// stays at minNEONGainQ31 = 4; the amd64-only gap is consistent with the 256-bit
// AVX transition/warmup penalty each call pays after its closing VZEROUPPER (a
// caller that keeps the upper YMM state warm would cross lower, so 160 is the
// conservative crossover for a kernel invoked from scalar Go). See #251; the #250
// masking speedup to gainQ31Go pushed the crossover out further still.
const minAVX2GainQ31 = 160

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
