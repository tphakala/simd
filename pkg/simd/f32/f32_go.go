package f32

import "math"

// Loop unroll factors for SIMD-width matching
const (
	unrollFactor = 8 // Match AVX 256-bit = 8 x float32
	unrollMask   = unrollFactor - 1
)

// Pure Go implementations

func dotProductGo(a, b []float32) float32 {
	var sum float32
	n := len(a)
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
