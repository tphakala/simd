package f32

import "math"

// Loop unroll factors for SIMD-width matching
const (
	unrollFactor      = 8 // Match AVX 256-bit = 8 x float32
	unrollMask        = unrollFactor - 1
	float32SignBitPos = 31 // IEEE 754 float32 sign bit position
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
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func subGo(dst, a, b []float32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func mulGo(dst, a, b []float32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func divGo(dst, a, b []float32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func scaleGo(dst, a []float32, s float32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] * s
	}
}

func addScalarGo(dst, a []float32, s float32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] + s
	}
}

func subFromScalarGo(dst, a []float32, s float32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = s - a[i]
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
	if len(a) == 0 {
		return posInf
	}
	m := a[0]
	for _, v := range a[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func maxGo(a []float32) float32 {
	if len(a) == 0 {
		return negInf
	}
	m := a[0]
	for _, v := range a[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

// maxAbsGo returns max_i |a[i]| (the infinity norm), 0 for an empty slice.
// It is the bit-exact source of truth for the MaxAbs kernels.
func maxAbsGo(a []float32) float32 {
	var m float32
	for _, v := range a {
		if v < 0 {
			v = -v
		}
		if v > m {
			m = v
		}
	}
	return m
}

func absGo(dst, a []float32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = math.Float32frombits(math.Float32bits(a[i]) &^ (1 << float32SignBitPos))
	}
}

func negGo(dst, a []float32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = -a[i]
	}
}

func fmaGo(dst, a, b, c []float32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	_ = c[len(dst)-1]
	for i := range dst {
		dst[i] = a[i]*b[i] + c[i]
	}
}

func clampGo(dst, a []float32, minVal, maxVal float32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
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

func dotProductIndexedGo(dst, base, query []float32, rowIDs []uint32, dims int) {
	n := min(len(dst), len(rowIDs))
	if n == 0 {
		return
	}
	if dims <= 0 || len(query) == 0 {
		clear(dst[:n])
		return
	}
	for i := range n {
		dst[i] = dotProductIndexedOneGo(base, query, rowIDs[i], dims)
	}
}

func dotProductIndexedOneGo(base, query []float32, rowID uint32, dims int) float32 {
	if dims <= 0 || len(query) == 0 {
		return 0
	}
	offset, ok := rowOffsetUint32InBase(rowID, dims, len(base))
	if !ok {
		return 0
	}
	n := min(dims, len(query))
	if remaining := len(base) - offset; remaining < n {
		n = remaining
	}
	if n <= 0 {
		return 0
	}
	return dotProduct(base[offset:offset+n], query[:n])
}

func dotProductStridedGo(dst, base, query []float32, rowCount, dims, stride int) {
	if rowCount <= 0 || len(dst) == 0 {
		return
	}
	n := min(len(dst), rowCount)
	if dims <= 0 || stride <= 0 || len(query) == 0 {
		clear(dst[:n])
		return
	}
	for i := range n {
		dst[i] = dotProductStridedOneGo(base, query, i, dims, stride)
	}
}

func dotProductStridedOneGo(base, query []float32, row, dims, stride int) float32 {
	if row < 0 || dims <= 0 || stride <= 0 || len(query) == 0 {
		return 0
	}
	offset, ok := rowOffsetStrideInBase(row, stride, len(base))
	if !ok {
		return 0
	}
	n := min(dims, len(query))
	if remaining := len(base) - offset; remaining < n {
		n = remaining
	}
	if n <= 0 {
		return 0
	}
	return dotProduct(base[offset:offset+n], query[:n])
}

func rowOffsetUint32InBase(rowID uint32, stride, baseLen int) (int, bool) {
	if stride <= 0 || baseLen <= 0 {
		return 0, false
	}
	offset, ok := mulUint64(uint64(rowID), uint64(stride))
	if !ok || offset >= uint64(baseLen) {
		return 0, false
	}
	return int(offset), true
}

func rowOffsetStrideInBase(row, stride, baseLen int) (int, bool) {
	if row < 0 || stride <= 0 || baseLen <= 0 {
		return 0, false
	}
	offset, ok := mulUint64(uint64(row), uint64(stride))
	if !ok || offset >= uint64(baseLen) {
		return 0, false
	}
	return int(offset), true
}

func mulUint64(a, b uint64) (uint64, bool) {
	if a != 0 && b > ^uint64(0)/a {
		return 0, false
	}
	return a * b, true
}

func convolveValid32Go(dst, signal, kernel []float32) {
	kLen := len(kernel)
	for i := range dst {
		dst[i] = dotProductGo(signal[i:i+kLen], kernel)
	}
}

// convolveValidMaxAbsGo is the Go-level fused fallback for ConvolveValidMaxAbs:
// it loops over the dispatched dotProduct (matching convolveValid32) and folds
// the abs-max in, so it is bit-identical to max|ConvolveValid output| on every
// backend. The dedicated convolveValidMaxAbs* kernels supersede it where present.
func convolveValidMaxAbsGo(signal, kernel []float32) float32 {
	kLen := len(kernel)
	validLen := len(signal) - kLen + 1
	var m float32 // abs values are >= 0, so 0 is the correct identity
	for i := range validLen {
		v := dotProduct(signal[i:i+kLen], kernel)
		if v < 0 {
			v = -v
		}
		if v > m {
			m = v
		}
	}
	return m
}

// convolveDecimate32Go is the pure-Go decimating valid convolution. It is the
// correctness oracle for the SIMD paths and the fallback on non-asm platforms.
func convolveDecimate32Go(dst, signal, kernel []float32, factor, phase int) {
	kLen := len(kernel)
	pos := phase
	for k := range dst {
		dst[k] = dotProductGo(signal[pos:pos+kLen], kernel)
		pos += factor
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

// interleaveNGo is the generic strided interleave: dst[i*nc+c] = srcs[c][i] for
// nc = len(srcs) streams and n frames. It is the portable fallback and the
// correctness reference for the SIMD specializations. The caller guarantees
// len(dst) >= n*nc and len(srcs[c]) >= n for every c.
func interleaveNGo(dst []float32, srcs [][]float32, n int) {
	nc := len(srcs)
	dst = dst[:n*nc]
	for c := range nc {
		s := srcs[c][:n]
		di := c
		for i := range n {
			dst[di] = s[i]
			di += nc
		}
	}
}

// deinterleaveNGo is the generic strided deinterleave: dsts[c][i] = src[i*nc+c]
// for nc = len(dsts) streams and n frames. The caller guarantees
// len(src) >= n*nc and len(dsts[c]) >= n for every c.
func deinterleaveNGo(dsts [][]float32, src []float32, n int) {
	nc := len(dsts)
	src = src[:n*nc]
	for c := range nc {
		d := dsts[c][:n]
		si := c
		for i := range n {
			d[i] = src[si]
			si += nc
		}
	}
}

func convolveValidMultiGo(dsts [][]float32, signal []float32, kernels [][]float32, n, _ int) {
	// Kernel-major loop order: each kernel stays hot in cache for entire signal pass
	for k, kernel := range kernels {
		convolveValid32Go(dsts[k][:n], signal, kernel)
	}
}

func sqrt32Go(dst, a []float32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = float32(math.Sqrt(float64(a[i])))
	}
}

// round32Go rounds each element to the nearest integer, half away from zero,
// matching math.Round. This is the trusted reference and the path that runs on
// architectures without SIMD.
func round32Go(dst, src []float32) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		dst[i] = float32(math.Round(float64(src[i])))
	}
}

func reciprocal32Go(dst, a []float32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
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
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
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
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
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
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
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
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		dst[i] = float32(math.Tanh(float64(src[i])))
	}
}

// exp32Go computes e^x using math.Exp.
func exp32Go(dst, src []float32) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
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

// logGo computes the natural logarithm: dst[i] = ln(src[i]).
// Edge cases follow math.Log: ln(0) = -Inf, ln(x<0) = NaN, ln(+Inf) = +Inf,
// ln(NaN) = NaN. This is the scalar reference and the fallback when no SIMD log
// kernel is selected.
func logGo(dst, src []float32) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		dst[i] = float32(math.Log(float64(src[i])))
	}
}

// log2Go computes the base-2 logarithm: dst[i] = log2(src[i]).
func log2Go(dst, src []float32) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		dst[i] = float32(math.Log2(float64(src[i])))
	}
}

// log10Go computes the base-10 logarithm: dst[i] = log10(src[i]).
func log10Go(dst, src []float32) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		dst[i] = float32(math.Log10(float64(src[i])))
	}
}

// powGo raises each element to a scalar power: dst[i] = src[i]**exp.
// Edge cases follow math.Pow (for example pow(x, 0) = 1, pow(negative,
// non-integer) = NaN, pow(0, negative) = +Inf).
func powGo(dst, src []float32, exp float32) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	e := float64(exp)
	for i := range dst {
		dst[i] = float32(math.Pow(float64(src[i]), e))
	}
}

// powElemGo raises each base to its own exponent: dst[i] = base[i]**exp[i].
func powElemGo(dst, base, exp []float32) {
	if len(dst) == 0 {
		return
	}
	_ = base[len(dst)-1]
	_ = exp[len(dst)-1]
	for i := range dst {
		dst[i] = float32(math.Pow(float64(base[i]), float64(exp[i])))
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

// int16ToFloat32ScaleGo converts int16 samples to float32 and scales.
// dst[i] = float32(src[i]) * scale
// int16 values are exactly representable in float32, so this is a single
// rounding (the multiply); the SIMD paths produce bit-identical results.
func int16ToFloat32ScaleGo(dst []float32, src []int16, scale float32) {
	for i := range src {
		dst[i] = float32(src[i]) * scale
	}
}

// float32ToInt16ScaleGo scales float32 samples and converts to int16 PCM.
// dst[i] = clamp(roundTiesToEven(src[i]*scale), -32768, 32767), with
// NaN -> 0, +Inf -> 32767, -Inf -> -32768.
//
// These are exactly the results of ARM64 FCVTNS + SQXTN; the AVX2 path and this
// fallback are written to match that bit-for-bit so output is identical across
// architectures. Rounding is round-to-nearest, ties to even (one LSB tighter
// than a truncating int16(f*scale) cast).
func float32ToInt16ScaleGo(dst []int16, src []float32, scale float32) {
	for i := range src {
		v := src[i] * scale
		switch {
		case v != v: // NaN
			dst[i] = 0
		case v >= math.MaxInt16: // includes +Inf
			dst[i] = math.MaxInt16
		case v <= math.MinInt16: // includes -Inf
			dst[i] = math.MinInt16
		default:
			dst[i] = int16(math.RoundToEven(float64(v)))
		}
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

// realFFTUnpackHalf is the constant 0.5 used in real FFT unpack computation.
const realFFTUnpackHalf = 0.5

// realFFTUnpack32Go performs the unpacking step of real FFT.
// For k in [1, n-1]:
//
//	X[k] = 0.5*(Z[k] + conj(Z[n-k])) + W[k]*(-0.5i)*(Z[k] - conj(Z[n-k]))
func realFFTUnpack32Go(outRe, outIm, zRe, zIm, twRe, twIm []float32, n int) {
	// Process k from 1 to n-1
	// For each k, we need Z[k] and Z[n-k] (mirrored)
	for k := 1; k < n; k++ {
		nk := n - k // Mirror index

		// Load Z[k] and conj(Z[n-k])
		zkRe, zkIm := zRe[k], zIm[k]
		znkRe, znkIm := zRe[nk], -zIm[nk] // Conjugate

		// even = 0.5 * (Z[k] + conj(Z[n-k]))
		evenRe := realFFTUnpackHalf * (zkRe + znkRe)
		evenIm := realFFTUnpackHalf * (zkIm + znkIm)

		// diff = Z[k] - conj(Z[n-k])
		diffRe := zkRe - znkRe
		diffIm := zkIm - znkIm

		// odd = W[k] * (-0.5i) * diff
		// W[k] is at twRe[k-1], twIm[k-1]
		wr, wi := twRe[k-1], twIm[k-1]
		// (-0.5i) * diff = 0.5*diffIm - 0.5i*diffRe
		// W * (-0.5i * diff) = (wr + i*wi) * (0.5*diffIm - 0.5i*diffRe)
		//   = 0.5*(wr*diffIm + wi*diffRe) + 0.5i*(wi*diffIm - wr*diffRe)
		oddRe := realFFTUnpackHalf * (wr*diffIm + wi*diffRe)
		oddIm := realFFTUnpackHalf * (wi*diffIm - wr*diffRe)

		// X[k] = even + odd
		outRe[k] = evenRe + oddRe
		outIm[k] = evenIm + oddIm
	}
}

// reverse32Go reverses a slice in pure Go.
// dst[i] = src[n-1-i] for i in [0, n)
func reverse32Go(dst, src []float32) {
	n := len(src)
	// Check for in-place operation
	if &dst[0] == &src[0] {
		// In-place reversal: swap elements from both ends
		for i, j := 0, n-1; i < j; i, j = i+1, j-1 {
			dst[i], dst[j] = dst[j], dst[i]
		}
		return
	}
	// Out-of-place: copy in reverse order
	// Process 4 elements at a time for better performance
	i := 0
	for ; i+3 < n; i += 4 {
		j := n - 1 - i
		dst[i] = src[j]
		dst[i+1] = src[j-1]
		dst[i+2] = src[j-2]
		dst[i+3] = src[j-3]
	}
	// Handle remainder
	for ; i < n; i++ {
		dst[i] = src[n-1-i]
	}
}

// addSub32Go computes element-wise sum and difference in pure Go.
// sumDst[i] = a[i] + b[i], diffDst[i] = a[i] - b[i]
func addSub32Go(sumDst, diffDst, a, b []float32) {
	n := len(a)
	// Process 4 elements at a time for better performance
	i := 0
	for ; i+3 < n; i += 4 {
		a0, a1, a2, a3 := a[i], a[i+1], a[i+2], a[i+3]
		b0, b1, b2, b3 := b[i], b[i+1], b[i+2], b[i+3]
		sumDst[i] = a0 + b0
		sumDst[i+1] = a1 + b1
		sumDst[i+2] = a2 + b2
		sumDst[i+3] = a3 + b3
		diffDst[i] = a0 - b0
		diffDst[i+1] = a1 - b1
		diffDst[i+2] = a2 - b2
		diffDst[i+3] = a3 - b3
	}
	// Handle remainder
	for ; i < n; i++ {
		sumDst[i] = a[i] + b[i]
		diffDst[i] = a[i] - b[i]
	}
}
