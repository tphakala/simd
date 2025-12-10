package f32

import "math"

// Loop unroll factors for SIMD-width matching
const (
	unrollFactor = 8 // Match AVX 256-bit = 8 x float32
	unrollMask   = unrollFactor - 1
)

// Numerical stability thresholds
const (
	sigmoidClampThreshold = 20.0 // sigmoid(±20) ≈ 1.0 - 2e-9 (float precision limit)
	tanhClampThreshold    = 2.5  // fast approximation threshold: tanh(±2.5) saturates to ±1
	expOverflowThreshold  = 88.0 // exp(88.72) = max float32; clamp to prevent overflow
)

// Pure Go implementations

func dotProductGo(a, b []float32) float32 {
	var sum float32
	n := min(len(a), len(b))
	n8 := n &^ unrollMask

	for i := 0; i < n8; i += 8 {
		sum += a[i] * b[i]
		sum += a[i+1] * b[i+1]
		sum += a[i+2] * b[i+2]
		sum += a[i+3] * b[i+3]
		sum += a[i+4] * b[i+4]
		sum += a[i+5] * b[i+5]
		sum += a[i+6] * b[i+6]
		sum += a[i+7] * b[i+7]
	}

	for i := n8; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func addGo(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func subGo(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func mulGo(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func divGo(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func scaleGo(dst, a []float32, s float32) {
	for i := range dst {
		dst[i] = a[i] * s
	}
}

func addScalarGo(dst, a []float32, s float32) {
	for i := range dst {
		dst[i] = a[i] + s
	}
}

func sumGo(a []float32) float32 {
	var sum float32
	for _, v := range a {
		sum += v
	}
	return sum
}

func minGo(a []float32) float32 {
	m := a[0]
	for _, v := range a[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func maxGo(a []float32) float32 {
	m := a[0]
	for _, v := range a[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func absGo(dst, a []float32) {
	for i := range dst {
		dst[i] = float32(math.Abs(float64(a[i])))
	}
}

func negGo(dst, a []float32) {
	for i := range dst {
		dst[i] = -a[i]
	}
}

func fmaGo(dst, a, b, c []float32) {
	for i := range dst {
		dst[i] = a[i]*b[i] + c[i]
	}
}

func clampGo(dst, a []float32, minVal, maxVal float32) {
	for i := range dst {
		v := a[i]
		if v < minVal {
			v = minVal
		} else if v > maxVal {
			v = maxVal
		}
		dst[i] = v
	}
}

func dotProductBatch32Go(results []float32, rows [][]float32, vec []float32) {
	vecLen := len(vec)
	for i, row := range rows {
		n := min(len(row), vecLen)
		if n == 0 {
			results[i] = 0
			continue
		}
		results[i] = dotProductGo(row[:n], vec[:n])
	}
}

func convolveValid32Go(dst, signal, kernel []float32) {
	kLen := len(kernel)
	for i := range dst {
		dst[i] = dotProductGo(signal[i:i+kLen], kernel)
	}
}

func accumulateAdd32Go(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func interleave2Go(dst, a, b []float32) {
	for i := range a {
		dst[i*2] = a[i]
		dst[i*2+1] = b[i]
	}
}

func deinterleave2Go(a, b, src []float32) {
	for i := range a {
		a[i] = src[i*2]
		b[i] = src[i*2+1]
	}
}

func convolveValidMultiGo(dsts [][]float32, signal []float32, kernels [][]float32, n, _ int) {
	// Kernel-major loop order: each kernel stays hot in cache for entire signal pass
	for k, kernel := range kernels {
		convolveValid32Go(dsts[k][:n], signal, kernel)
	}
}

func sqrt32Go(dst, a []float32) {
	for i := range dst {
		dst[i] = float32(math.Sqrt(float64(a[i])))
	}
}

func reciprocal32Go(dst, a []float32) {
	for i := range dst {
		dst[i] = 1.0 / a[i]
	}
}

func minIdxGo(a []float32) int {
	if len(a) == 0 {
		return -1
	}
	idx := 0
	m := a[0]
	for i, v := range a[1:] {
		if v < m {
			m = v
			idx = i + 1
		}
	}
	return idx
}

func maxIdxGo(a []float32) int {
	if len(a) == 0 {
		return -1
	}
	idx := 0
	m := a[0]
	for i, v := range a[1:] {
		if v > m {
			m = v
			idx = i + 1
		}
	}
	return idx
}

func addScaledGo(dst []float32, alpha float32, s []float32) {
	for i := range dst {
		dst[i] += alpha * s[i]
	}
}

func cumulativeSum32Go(dst, a []float32) {
	if len(dst) == 0 {
		return
	}
	sum := float32(0)
	for i := range dst {
		sum += a[i]
		dst[i] = sum
	}
}

func variance32Go(a []float32, mean float32) float32 {
	var sum float32
	for _, v := range a {
		diff := v - mean
		sum += diff * diff
	}
	return sum / float32(len(a))
}

func euclideanDistance32Go(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}

// cubicInterpDotGo computes: Σ hist[i] * (a[i] + x*(b[i] + x*(c[i] + x*d[i])))
// Uses Horner's method for numerical stability.
func cubicInterpDotGo(hist, a, b, c, d []float32, x float32) float32 {
	var sum float32
	n := len(hist)
	n8 := n &^ unrollMask // Round down to multiple of 8

	// Unrolled loop: 8 elements per iteration (match AVX width)
	for i := 0; i < n8; i += 8 {
		// Horner's method: coef = a + x*(b + x*(c + x*d))
		coef0 := a[i] + x*(b[i]+x*(c[i]+x*d[i]))
		coef1 := a[i+1] + x*(b[i+1]+x*(c[i+1]+x*d[i+1]))
		coef2 := a[i+2] + x*(b[i+2]+x*(c[i+2]+x*d[i+2]))
		coef3 := a[i+3] + x*(b[i+3]+x*(c[i+3]+x*d[i+3]))
		coef4 := a[i+4] + x*(b[i+4]+x*(c[i+4]+x*d[i+4]))
		coef5 := a[i+5] + x*(b[i+5]+x*(c[i+5]+x*d[i+5]))
		coef6 := a[i+6] + x*(b[i+6]+x*(c[i+6]+x*d[i+6]))
		coef7 := a[i+7] + x*(b[i+7]+x*(c[i+7]+x*d[i+7]))

		sum += hist[i]*coef0 + hist[i+1]*coef1 + hist[i+2]*coef2 + hist[i+3]*coef3
		sum += hist[i+4]*coef4 + hist[i+5]*coef5 + hist[i+6]*coef6 + hist[i+7]*coef7
	}

	// Handle remainder
	for i := n8; i < n; i++ {
		coef := a[i] + x*(b[i]+x*(c[i]+x*d[i]))
		sum += hist[i] * coef
	}

	return sum
}

// sigmoid32Go computes sigmoid(x) = 1 / (1 + e^(-x)) using math.Exp.
// This is accurate but slower than SIMD approximations.
func sigmoid32Go(dst, src []float32) {
	for i := range dst {
		x := src[i]
		// Clamp extreme values for numerical stability
		switch {
		case x > sigmoidClampThreshold:
			dst[i] = 1.0
		case x < -sigmoidClampThreshold:
			dst[i] = 0.0
		default:
			dst[i] = float32(1.0 / (1.0 + math.Exp(float64(-x))))
		}
	}
}

// relu32Go computes ReLU(x) = max(0, x).
func relu32Go(dst, src []float32) {
	for i := range dst {
		if src[i] > 0 {
			dst[i] = src[i]
		} else {
			dst[i] = 0
		}
	}
}

// clampScale32Go performs fused clamp and scale operation.
func clampScale32Go(dst, src []float32, minVal, maxVal, scale float32) {
	for i := range dst {
		v := src[i]
		if v < minVal {
			v = minVal
		} else if v > maxVal {
			v = maxVal
		}
		dst[i] = (v - minVal) * scale
	}
}

// tanh32Go computes hyperbolic tangent using math.Tanh for accuracy.
// This is the accurate implementation used as a fallback when SIMD is unavailable.
func tanh32Go(dst, src []float32) {
	for i := range dst {
		dst[i] = float32(math.Tanh(float64(src[i])))
	}
}

// exp32Go computes e^x using math.Exp.
func exp32Go(dst, src []float32) {
	for i := range dst {
		x := src[i]
		// Clamp extreme values
		switch {
		case x > expOverflowThreshold:
			dst[i] = float32(math.Exp(expOverflowThreshold)) // Prevent overflow
		case x < -expOverflowThreshold:
			dst[i] = 0.0 // Prevent underflow
		default:
			dst[i] = float32(math.Exp(float64(x)))
		}
	}
}

// int32ToFloat32ScaleGo converts int32 samples to float32 and scales.
// dst[i] = float32(src[i]) * scale
// Uses loop unrolling for better performance.
func int32ToFloat32ScaleGo(dst []float32, src []int32, scale float32) {
	n := len(src)
	n8 := n &^ unrollMask // Round down to multiple of 8

	// Unrolled loop: 8 elements per iteration
	for i := 0; i < n8; i += 8 {
		dst[i] = float32(src[i]) * scale
		dst[i+1] = float32(src[i+1]) * scale
		dst[i+2] = float32(src[i+2]) * scale
		dst[i+3] = float32(src[i+3]) * scale
		dst[i+4] = float32(src[i+4]) * scale
		dst[i+5] = float32(src[i+5]) * scale
		dst[i+6] = float32(src[i+6]) * scale
		dst[i+7] = float32(src[i+7]) * scale
	}

	// Handle remainder
	for i := n8; i < n; i++ {
		dst[i] = float32(src[i]) * scale
	}
}

// ============================================================================
// SPLIT-FORMAT COMPLEX OPERATIONS (Pure Go)
// ============================================================================

// mulComplex32Go computes element-wise complex multiplication:
//
//	dstRe[i] = aRe[i]*bRe[i] - aIm[i]*bIm[i]
//	dstIm[i] = aRe[i]*bIm[i] + aIm[i]*bRe[i]
func mulComplex32Go(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	for i := range dstRe {
		ar, ai := aRe[i], aIm[i]
		br, bi := bRe[i], bIm[i]
		dstRe[i] = ar*br - ai*bi
		dstIm[i] = ar*bi + ai*br
	}
}

// mulConjComplex32Go computes element-wise multiplication by conjugate:
//
//	dstRe[i] = aRe[i]*bRe[i] + aIm[i]*bIm[i]
//	dstIm[i] = aIm[i]*bRe[i] - aRe[i]*bIm[i]
func mulConjComplex32Go(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	for i := range dstRe {
		ar, ai := aRe[i], aIm[i]
		br, bi := bRe[i], bIm[i]
		dstRe[i] = ar*br + ai*bi
		dstIm[i] = ai*br - ar*bi
	}
}

// absSqComplex32Go computes element-wise magnitude squared:
//
//	dst[i] = aRe[i]^2 + aIm[i]^2
func absSqComplex32Go(dst, aRe, aIm []float32) {
	for i := range dst {
		r, im := aRe[i], aIm[i]
		dst[i] = r*r + im*im
	}
}

// butterflyComplex32Go performs FFT butterfly with twiddle multiply:
//
//	temp = lower * twiddle (complex multiply)
//	upper, lower = upper + temp, upper - temp
func butterflyComplex32Go(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm []float32) {
	for i := range upperRe {
		// Complex multiply: temp = lower * twiddle
		lr, li := lowerRe[i], lowerIm[i]
		tr, ti := twRe[i], twIm[i]
		tempRe := lr*tr - li*ti
		tempIm := lr*ti + li*tr

		// Butterfly: upper' = upper + temp, lower' = upper - temp
		ur, ui := upperRe[i], upperIm[i]
		upperRe[i] = ur + tempRe
		upperIm[i] = ui + tempIm
		lowerRe[i] = ur - tempRe
		lowerIm[i] = ui - tempIm
	}
}
