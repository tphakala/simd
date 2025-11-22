package f64

import "math"

// Loop unroll factors for SIMD-width matching
const (
	unrollFactor = 4 // Match AVX 256-bit = 4 x float64
	unrollMask   = unrollFactor - 1
)

// Pure Go implementations - used as fallback on all architectures

func dotProductGo(a, b []float64) float64 {
	var sum float64
	n := len(a)
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
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func subGo(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func mulGo(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func divGo(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] / b[i]
	}
}

func scaleGo(dst, a []float64, s float64) {
	for i := range dst {
		dst[i] = a[i] * s
	}
}

func addScalarGo(dst, a []float64, s float64) {
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
	m := a[0]
	for _, v := range a[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func maxGo(a []float64) float64 {
	m := a[0]
	for _, v := range a[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func absGo(dst, a []float64) {
	for i := range dst {
		dst[i] = math.Abs(a[i])
	}
}

func negGo(dst, a []float64) {
	for i := range dst {
		dst[i] = -a[i]
	}
}

func fmaGo(dst, a, b, c []float64) {
	for i := range dst {
		dst[i] = math.FMA(a[i], b[i], c[i])
	}
}

func clampGo(dst, a []float64, minVal, maxVal float64) {
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
	for i := range dst {
		dst[i] = math.Sqrt(a[i])
	}
}

func reciprocal64Go(dst, a []float64) {
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
