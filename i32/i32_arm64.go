//go:build arm64

package i32

import "github.com/tphakala/simd/cpu"

// NEON processes 4 int32 pairs (one .4S register) per iteration.
const minNEONElements = 4

var hasNEON = cpu.ARM64.NEON

func interleave2I32(dst, a, b []int32) {
	if hasNEON && len(a) >= minNEONElements {
		interleave2NEON(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2I32(a, b, src []int32) {
	if hasNEON && len(a) >= minNEONElements {
		deinterleave2NEON(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

//go:noescape
func interleave2NEON(dst, a, b []int32)

//go:noescape
func deinterleave2NEON(a, b, src []int32)

func addI32(dst, a, b []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		addNEON(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func subI32(dst, a, b []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		subNEON(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

//go:noescape
func addNEON(dst, a, b []int32)

//go:noescape
func subNEON(dst, a, b []int32)

// Tier-3 thresholds: one 4-wide (.4S) vector block each, independent literals
// rather than aliases of minNEONElements, which they happen to equal, so
// retuning the interleave kernels cannot silently move these. Both kernels are
// correct at any length (each falls through to a scalar tail), so these are
// performance cuts only, never a safety requirement.
const (
	minNEONSum = 4
	minNEONAbs = 4
)

func sumI32(a []int32) int32 {
	if hasNEON && len(a) >= minNEONSum {
		return sumNEON(a)
	}
	return sumGo(a)
}

func absI32(dst, a []int32) {
	if hasNEON && len(dst) >= minNEONAbs {
		absNEON(dst, a)
		return
	}
	absGo(dst, a)
}

//go:noescape
func sumNEON(a []int32) int32

//go:noescape
func absNEON(dst, a []int32)

// minNEONNegWhereNeg is one 4-wide (.4S) block, an independent literal like the
// tier-3 thresholds above. The kernel is correct at any length (it falls through
// to a scalar tail), so this is a performance cut only, never a safety
// requirement.
const minNEONNegWhereNeg = 4

func negWhereNegI32(dst, mag []int32, sign []float32) {
	if hasNEON && len(dst) >= minNEONNegWhereNeg {
		negWhereNegNEON(dst, mag, sign)
		return
	}
	negWhereNegGo(dst, mag, sign)
}

//go:noescape
func negWhereNegNEON(dst, mag []int32, sign []float32)

// minNEONScaleQ31 and minNEONScaleQ15 are one 4-wide (.4S) block each,
// independent literals like the tier-3 thresholds above. Both kernels are correct
// at any length (each falls through to a scalar tail), so these are performance
// cuts only, never a safety requirement.
const (
	minNEONScaleQ31 = 4
	minNEONScaleQ15 = 4
)

func scaleQ31I32(dst, a []int32, k int32) {
	if hasNEON && len(dst) >= minNEONScaleQ31 {
		scaleQ31NEON(dst, a, k)
		return
	}
	scaleQ31Go(dst, a, k)
}

func scaleQ15I32(dst, a []int32, k int16) {
	if hasNEON && len(dst) >= minNEONScaleQ15 {
		scaleQ15NEON(dst, a, k)
		return
	}
	scaleQ15Go(dst, a, k)
}

//go:noescape
func scaleQ31NEON(dst, a []int32, k int32)

//go:noescape
func scaleQ15NEON(dst, a []int32, k int16)

// minMaxI32 dispatches the signed int32 min/max reduction. The NEON kernel does
// the reduction in 4-wide SMIN/SMAX lanes with a single-instruction SMINV/SMAXV
// across-vector fold and a scalar tail, so it gates on NEON and at least one full
// 4-element block; shorter slices use the pure-Go reference. res is non-empty
// (the public MinMax guards the empty case).
func minMaxI32(res []int32) (minVal, maxVal int32) {
	if hasNEON && len(res) >= minNEONElements {
		return minMaxNEON(res)
	}
	return minMaxGo(res)
}

//go:noescape
func minMaxNEON(res []int32) (minVal, maxVal int32)
