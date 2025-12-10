//go:build arm64

package f32

import "github.com/tphakala/simd/cpu"

var (
	hasNEON = cpu.ARM64.NEON
)

func dotProduct(a, b []float32) float32 {
	if hasNEON && len(a) >= 4 {
		return dotProductNEON(a, b)
	}
	return dotProductGo(a, b)
}

func add(dst, a, b []float32) {
	if hasNEON && len(dst) >= 4 {
		addNEON(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func sub(dst, a, b []float32) {
	if hasNEON && len(dst) >= 4 {
		subNEON(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

func mul(dst, a, b []float32) {
	if hasNEON && len(dst) >= 4 {
		mulNEON(dst, a, b)
		return
	}
	mulGo(dst, a, b)
}

func div(dst, a, b []float32) {
	if hasNEON && len(dst) >= 4 {
		divNEON(dst, a, b)
		return
	}
	divGo(dst, a, b)
}

func scale(dst, a []float32, s float32) {
	if hasNEON && len(dst) >= 4 {
		scaleNEON(dst, a, s)
		return
	}
	scaleGo(dst, a, s)
}

func addScalar(dst, a []float32, s float32) {
	if hasNEON && len(dst) >= 4 {
		addScalarNEON(dst, a, s)
		return
	}
	addScalarGo(dst, a, s)
}

func sum(a []float32) float32 {
	if hasNEON && len(a) >= 4 {
		return sumNEON(a)
	}
	return sumGo(a)
}

func min32(a []float32) float32 {
	if hasNEON && len(a) >= 4 {
		return minNEON(a)
	}
	return minGo(a)
}

func max32(a []float32) float32 {
	if hasNEON && len(a) >= 4 {
		return maxNEON(a)
	}
	return maxGo(a)
}

func abs32(dst, a []float32) {
	if hasNEON && len(dst) >= 4 {
		absNEON(dst, a)
		return
	}
	absGo(dst, a)
}

func neg32(dst, a []float32) {
	if hasNEON && len(dst) >= 4 {
		negNEON(dst, a)
		return
	}
	negGo(dst, a)
}

func fma32(dst, a, b, c []float32) {
	if hasNEON && len(dst) >= 4 {
		fmaNEON(dst, a, b, c)
		return
	}
	fmaGo(dst, a, b, c)
}

func clamp32(dst, a []float32, minVal, maxVal float32) {
	if hasNEON && len(dst) >= 4 {
		clampNEON(dst, a, minVal, maxVal)
		return
	}
	clampGo(dst, a, minVal, maxVal)
}

func dotProductBatch32(results []float32, rows [][]float32, vec []float32) {
	vecLen := len(vec)
	for i, row := range rows {
		n := min(len(row), vecLen)
		if n == 0 {
			results[i] = 0
			continue
		}
		results[i] = dotProduct(row[:n], vec[:n])
	}
}

func convolveValid32(dst, signal, kernel []float32) {
	kLen := len(kernel)
	for i := range dst {
		dst[i] = dotProduct(signal[i:i+kLen], kernel)
	}
}

func accumulateAdd32(dst, src []float32) {
	// AccumulateAdd is dst += src, use add with dst as both operands
	if hasNEON && len(dst) >= 4 {
		addNEON(dst, dst, src)
		return
	}
	accumulateAdd32Go(dst, src)
}

//go:noescape
func dotProductNEON(a, b []float32) float32

//go:noescape
func addNEON(dst, a, b []float32)

//go:noescape
func subNEON(dst, a, b []float32)

//go:noescape
func mulNEON(dst, a, b []float32)

//go:noescape
func divNEON(dst, a, b []float32)

//go:noescape
func scaleNEON(dst, a []float32, s float32)

//go:noescape
func addScalarNEON(dst, a []float32, s float32)

//go:noescape
func sumNEON(a []float32) float32

//go:noescape
func minNEON(a []float32) float32

//go:noescape
func maxNEON(a []float32) float32

//go:noescape
func absNEON(dst, a []float32)

//go:noescape
func negNEON(dst, a []float32)

//go:noescape
func fmaNEON(dst, a, b, c []float32)

//go:noescape
func clampNEON(dst, a []float32, minVal, maxVal float32)

func interleave2_32(dst, a, b []float32) {
	if hasNEON && len(a) >= 4 {
		interleave2NEON(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2_32(a, b, src []float32) {
	if hasNEON && len(a) >= 4 {
		deinterleave2NEON(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

func convolveValidMulti32(dsts [][]float32, signal []float32, kernels [][]float32, n, _ int) {
	// Kernel-major loop order: each kernel stays hot in cache for entire signal pass
	for k, kernel := range kernels {
		convolveValid32(dsts[k][:n], signal, kernel)
	}
}

//go:noescape
func interleave2NEON(dst, a, b []float32)

//go:noescape
func deinterleave2NEON(a, b, src []float32)

func sqrt32(dst, a []float32) {
	if hasNEON && len(dst) >= 4 {
		sqrtNEON(dst, a)
		return
	}
	sqrt32Go(dst, a)
}

func reciprocal32(dst, a []float32) {
	if hasNEON && len(dst) >= 4 {
		reciprocalNEON(dst, a)
		return
	}
	reciprocal32Go(dst, a)
}

func minIdx32(a []float32) int {
	return minIdxGo(a)
}

func maxIdx32(a []float32) int {
	return maxIdxGo(a)
}

func addScaled32(dst []float32, alpha float32, s []float32) {
	if hasNEON && len(dst) >= 4 {
		addScaledNEON(dst, alpha, s)
		return
	}
	addScaledGo(dst, alpha, s)
}

func cumulativeSum32(dst, a []float32) {
	cumulativeSum32Go(dst, a)
}

//go:noescape
func sqrtNEON(dst, a []float32)

//go:noescape
func reciprocalNEON(dst, a []float32)

//go:noescape
func addScaledNEON(dst []float32, alpha float32, s []float32)

func variance32(a []float32, mean float32) float32 {
	if hasNEON && len(a) >= 4 {
		return varianceNEON32(a, mean)
	}
	return variance32Go(a, mean)
}

func euclideanDistance32(a, b []float32) float32 {
	if hasNEON && len(a) >= 4 {
		return euclideanDistanceNEON32(a, b)
	}
	return euclideanDistance32Go(a, b)
}

//go:noescape
func varianceNEON32(a []float32, mean float32) float32

//go:noescape
func euclideanDistanceNEON32(a, b []float32) float32

func cubicInterpDot32(hist, a, b, c, d []float32, x float32) float32 {
	if hasNEON && len(hist) >= 4 {
		return cubicInterpDotNEON(hist, a, b, c, d, x)
	}
	return cubicInterpDotGo(hist, a, b, c, d, x)
}

//go:noescape
func cubicInterpDotNEON(hist, a, b, c, d []float32, x float32) float32

func sigmoid32(dst, src []float32) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 4 {
		sigmoidNEON(dst, src)
		return
	}
	sigmoid32Go(dst, src)
}

//go:noescape
func sigmoidNEON(dst, src []float32)

func relu32(dst, src []float32) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 4 {
		reluNEON(dst, src)
		return
	}
	relu32Go(dst, src)
}

//go:noescape
func reluNEON(dst, src []float32)

func clampScale32(dst, src []float32, minVal, maxVal, scale float32) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 4 {
		clampScaleNEON(dst, src, minVal, maxVal, scale)
		return
	}
	clampScale32Go(dst, src, minVal, maxVal, scale)
}

//go:noescape
func clampScaleNEON(dst, src []float32, minVal, maxVal, scale float32)

func tanh32(dst, src []float32) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 4 {
		tanhNEON(dst, src)
		return
	}
	tanh32Go(dst, src)
}

//go:noescape
func tanhNEON(dst, src []float32)

func exp32(dst, src []float32) {
	exp32Go(dst, src)
}

func int32ToFloat32Scale(dst []float32, src []int32, scale float32) {
	if hasNEON && len(dst) >= 4 {
		int32ToFloat32ScaleNEON(dst, src, scale)
		return
	}
	int32ToFloat32ScaleGo(dst, src, scale)
}

//go:noescape
func int32ToFloat32ScaleNEON(dst []float32, src []int32, scale float32)

// ============================================================================
// SPLIT-FORMAT COMPLEX OPERATIONS
// ============================================================================

func mulComplex32(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	if hasNEON && len(dstRe) >= 4 {
		mulComplexNEON(dstRe, dstIm, aRe, aIm, bRe, bIm)
		return
	}
	mulComplex32Go(dstRe, dstIm, aRe, aIm, bRe, bIm)
}

func mulConjComplex32(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	if hasNEON && len(dstRe) >= 4 {
		mulConjComplexNEON(dstRe, dstIm, aRe, aIm, bRe, bIm)
		return
	}
	mulConjComplex32Go(dstRe, dstIm, aRe, aIm, bRe, bIm)
}

func absSqComplex32(dst, aRe, aIm []float32) {
	if hasNEON && len(dst) >= 4 {
		absSqComplexNEON(dst, aRe, aIm)
		return
	}
	absSqComplex32Go(dst, aRe, aIm)
}

//go:noescape
func mulComplexNEON(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)

//go:noescape
func mulConjComplexNEON(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)

//go:noescape
func absSqComplexNEON(dst, aRe, aIm []float32)
