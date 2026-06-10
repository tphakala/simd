package f64

import "math"

// Loop unroll factors for SIMD-width matching
const (
	unrollFactor = 4 // Match AVX 256-bit = 4 x float64
	unrollMask   = unrollFactor - 1
)

// Numerical stability thresholds
const (
	tanhClampThreshold   = 2.5   // fast approximation threshold: tanh(±2.5) saturates to ±1
	expOverflowThreshold = 709.0 // exp(709.78) = max float64; clamp to prevent overflow
)

// Pure Go implementations - used as fallback on all architectures

func dotProductGo(a, b []float64) float64 {
	var sum float64
	n := min(len(a), len(b))
	n4 := n &^ unrollMask // Round down to multiple of 4

	// Unrolled loop: 4 FMAs per iteration
	for i := 0; i < n4; i += 4 {
		sum = math.FMA(a[i], b[i], sum)
		sum = math.FMA(a[i+1], b[i+1], sum)
		sum = math.FMA(a[i+2], b[i+2], sum)
		sum = math.FMA(a[i+3], b[i+3], sum)
	}

	// Handle remainder
	for i := n4; i < n; i++ {
		sum = math.FMA(a[i], b[i], sum)
	}
	return sum
}

func addGo(dst, a, b []float64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func subGo(dst, a, b []float64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func mulGo(dst, a, b []float64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func divGo(dst, a, b []float64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func scaleGo(dst, a []float64, s float64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] * s
	}
}

func addScalarGo(dst, a []float64, s float64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] + s
	}
}

func sumGo(a []float64) float64 {
	var sum float64
	for _, v := range a {
		sum += v
	}
	return sum
}

func minGo(a []float64) float64 {
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

func maxGo(a []float64) float64 {
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

func absGo(dst, a []float64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = math.Abs(a[i])
	}
}

func negGo(dst, a []float64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = -a[i]
	}
}

func fmaGo(dst, a, b, c []float64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	_ = c[len(dst)-1]
	for i := range dst {
		dst[i] = math.FMA(a[i], b[i], c[i])
	}
}

func clampGo(dst, a []float64, minVal, maxVal float64) {
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

func sqrt64Go(dst, a []float64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = math.Sqrt(a[i])
	}
}

func round64Go(dst, src []float64) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		dst[i] = math.Round(src[i])
	}
}

func subFromScalarGo(dst, a []float64, s float64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = s - a[i]
	}
}

func reciprocal64Go(dst, a []float64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = 1.0 / a[i]
	}
}

func variance64Go(a []float64, mean float64) float64 {
	var sum float64
	for _, v := range a {
		diff := v - mean
		sum += diff * diff
	}
	return sum / float64(len(a))
}

// varianceFullGo computes variance including mean calculation (two-pass).
// Used for fair benchmarking against the public Variance function.
func varianceFullGo(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	// First pass: compute mean
	var sum float64
	for _, v := range a {
		sum += v
	}
	mean := sum / float64(len(a))
	// Second pass: compute variance
	return variance64Go(a, mean)
}

func euclideanDistance64Go(a, b []float64) float64 {
	var sum float64
	n := len(a)
	n4 := n &^ unrollMask // Round down to multiple of 4

	// Unrolled loop: 4 operations per iteration
	for i := 0; i < n4; i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		sum = math.FMA(d0, d0, sum)
		sum = math.FMA(d1, d1, sum)
		sum = math.FMA(d2, d2, sum)
		sum = math.FMA(d3, d3, sum)
	}

	// Handle remainder
	for i := n4; i < n; i++ {
		diff := a[i] - b[i]
		sum = math.FMA(diff, diff, sum)
	}
	return math.Sqrt(sum)
}

func cumulativeSum64Go(dst, a []float64) {
	if len(dst) == 0 {
		return
	}
	sum := 0.0
	for i := range dst {
		sum += a[i]
		dst[i] = sum
	}
}

func dotProductBatch64Go(results []float64, rows [][]float64, vec []float64) {
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

func convolveValid64Go(dst, signal, kernel []float64) {
	kLen := len(kernel)
	for i := range dst {
		dst[i] = dotProductGo(signal[i:i+kLen], kernel)
	}
}

// convolveDecimate64Go is the pure-Go decimating valid convolution. It is the
// correctness oracle for the SIMD paths and the fallback on non-asm platforms.
func convolveDecimate64Go(dst, signal, kernel []float64, factor, phase int) {
	kLen := len(kernel)
	pos := phase
	for k := range dst {
		dst[k] = dotProductGo(signal[pos:pos+kLen], kernel)
		pos += factor
	}
}

func accumulateAdd64Go(dst, src []float64) {
	for i := range dst {
		dst[i] += src[i]
	}
}

// autocorrelateGo is the pure-Go reference autocorrelation. For each lag in
// 0..maxLag it sums x[i]*x[i-lag] over i in lag..len(x)-1, left to right with
// separate multiply and add. It is the bit-exact contract the AVX2 and NEON
// kernels reproduce: each lag is an independent accumulator fed in increasing-i
// order, so vectorizing ACROSS lags (one lane per lag, separate mul+add, never
// FMA) yields identical bytes to this loop on every build.
func autocorrelateGo(autoc, x []float64, maxLag int) {
	n := len(x)
	for lag := 0; lag <= maxLag; lag++ {
		var sum float64
		for i := lag; i < n; i++ {
			sum += x[i] * x[i-lag]
		}
		autoc[lag] = sum
	}
}

// autocorrTriangularSeed fills autoc[lag] (lag in 0..maxLag) with the partial
// sum over the prologue samples i in lag..pmax-1, left to right. The SIMD
// kernels then continue each lag's accumulator over the steady region
// i in pmax..n-1, so the complete sum is accumulated in the same order as
// autocorrelateGo and stays bit-identical. pmax is the padded steady-region
// start (a multiple-of-lanes boundary) chosen so every vectorized load is in
// bounds. Shared by the amd64 and arm64 orchestrators.
func autocorrTriangularSeed(autoc, x []float64, maxLag, pmax int) {
	for lag := 0; lag <= maxLag; lag++ {
		var s float64
		for i := lag; i < pmax; i++ {
			s += x[i] * x[i-lag]
		}
		autoc[lag] = s
	}
}

func interleave2Go(dst, a, b []float64) {
	for i := range a {
		dst[i*2] = a[i]
		dst[i*2+1] = b[i]
	}
}

func deinterleave2Go(a, b, src []float64) {
	for i := range a {
		a[i] = src[i*2]
		b[i] = src[i*2+1]
	}
}

// interleaveNGo is the generic strided interleave: dst[i*nc+c] = srcs[c][i] for
// nc = len(srcs) streams and n frames. It is the portable fallback and the
// correctness reference for the SIMD specializations. The caller guarantees
// len(dst) >= n*nc and len(srcs[c]) >= n for every c.
func interleaveNGo(dst []float64, srcs [][]float64, n int) {
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
func deinterleaveNGo(dsts [][]float64, src []float64, n int) {
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

// cubicInterpDotGo computes: Σ hist[i] * (a[i] + x*(b[i] + x*(c[i] + x*d[i])))
// Uses Horner's method for numerical stability.
func cubicInterpDotGo(hist, a, b, c, d []float64, x float64) float64 {
	var sum float64
	n := len(hist)
	n4 := n &^ unrollMask // Round down to multiple of 4

	// Unrolled loop: 4 elements per iteration
	for i := 0; i < n4; i += 4 {
		// Horner's method: coef = a + x*(b + x*(c + x*d))
		coef0 := math.FMA(x, math.FMA(x, math.FMA(x, d[i], c[i]), b[i]), a[i])
		coef1 := math.FMA(x, math.FMA(x, math.FMA(x, d[i+1], c[i+1]), b[i+1]), a[i+1])
		coef2 := math.FMA(x, math.FMA(x, math.FMA(x, d[i+2], c[i+2]), b[i+2]), a[i+2])
		coef3 := math.FMA(x, math.FMA(x, math.FMA(x, d[i+3], c[i+3]), b[i+3]), a[i+3])

		sum = math.FMA(hist[i], coef0, sum)
		sum = math.FMA(hist[i+1], coef1, sum)
		sum = math.FMA(hist[i+2], coef2, sum)
		sum = math.FMA(hist[i+3], coef3, sum)
	}

	// Handle remainder
	for i := n4; i < n; i++ {
		coef := math.FMA(x, math.FMA(x, math.FMA(x, d[i], c[i]), b[i]), a[i])
		sum = math.FMA(hist[i], coef, sum)
	}

	return sum
}

func convolveValidMultiGo(dsts [][]float64, signal []float64, kernels [][]float64, n, _ int) {
	// Kernel-major loop order: each kernel stays hot in cache for entire signal pass
	for k, kernel := range kernels {
		convolveValid64Go(dsts[k][:n], signal, kernel)
	}
}

func minIdxGo64(a []float64) int {
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

func maxIdxGo64(a []float64) int {
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

func addScaledGo64(dst []float64, alpha float64, s []float64) {
	if len(dst) == 0 {
		return
	}
	_ = s[len(dst)-1]
	for i := range dst {
		dst[i] += alpha * s[i]
	}
}

// sigmoid64Go computes sigmoid(x) = 1 / (1 + e^(-x)) using math.Exp.
// This is accurate but slower than SIMD approximations.
func sigmoid64Go(dst, src []float64) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		// sigmoid(x) = 1/(1+e^-x). Clamp the exponent to ±expOverflowThreshold so
		// e^z cannot overflow to +Inf; the quotient then saturates smoothly to 0
		// (x -> -inf) or 1 (x -> +inf). This matches the accurate exp-based SIMD
		// kernels (issue #33) within tolerance. The previous hard flush to 0 for
		// x < -20 diverged from those kernels (sigmoid(-25) is ~1.4e-11, not 0),
		// so TestSigmoidAccuracy failed on the scalar fallback path that runs on
		// CPUs without AVX2.
		z := -src[i]
		switch {
		case z > expOverflowThreshold:
			z = expOverflowThreshold
		case z < -expOverflowThreshold:
			z = -expOverflowThreshold
		}
		dst[i] = 1.0 / (1.0 + math.Exp(z))
	}
}

// relu64Go computes ReLU activation: dst[i] = max(0, src[i]).
func relu64Go(dst, src []float64) {
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

// clampScale64Go performs fused clamp and scale operation.
// dst[i] = (clamp(src[i], minVal, maxVal) - minVal) * scale
func clampScale64Go(dst, src []float64, minVal, maxVal, scale float64) {
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

// tanh64Go computes hyperbolic tangent using math.Tanh for accuracy.
// This is the accurate implementation used as a fallback when SIMD is unavailable.
func tanh64Go(dst, src []float64) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		dst[i] = math.Tanh(src[i])
	}
}

// exp64Go computes exponential function: dst[i] = e^src[i].
// Uses math.Exp with overflow/underflow protection.
func exp64Go(dst, src []float64) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		x := src[i]
		// Clamp to prevent overflow/underflow
		switch {
		case x > expOverflowThreshold:
			dst[i] = math.Exp(expOverflowThreshold)
		case x < -expOverflowThreshold:
			dst[i] = 0.0
		default:
			dst[i] = math.Exp(x)
		}
	}
}

// logGo computes the natural logarithm: dst[i] = ln(src[i]).
// Edge cases follow math.Log: ln(0) = -Inf, ln(x<0) = NaN, ln(+Inf) = +Inf,
// ln(NaN) = NaN. This is the scalar reference and the fallback when no SIMD log
// kernel is selected.
func logGo(dst, src []float64) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		dst[i] = math.Log(src[i])
	}
}

// log2Go computes the base-2 logarithm: dst[i] = log2(src[i]).
func log2Go(dst, src []float64) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		dst[i] = math.Log2(src[i])
	}
}

// log10Go computes the base-10 logarithm: dst[i] = log10(src[i]).
func log10Go(dst, src []float64) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		dst[i] = math.Log10(src[i])
	}
}

// powGo raises each element to a scalar power: dst[i] = src[i]**exp.
// Edge cases follow math.Pow (for example pow(x, 0) = 1, pow(negative,
// non-integer) = NaN, pow(0, negative) = +Inf).
func powGo(dst, src []float64, exp float64) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		dst[i] = math.Pow(src[i], exp)
	}
}

// powElemGo raises each base to its own exponent: dst[i] = base[i]**exp[i].
func powElemGo(dst, base, exp []float64) {
	if len(dst) == 0 {
		return
	}
	_ = base[len(dst)-1]
	_ = exp[len(dst)-1]
	for i := range dst {
		dst[i] = math.Pow(base[i], exp[i])
	}
}
