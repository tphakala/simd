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
