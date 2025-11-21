//go:build arm64

package f64

import "github.com/tphakala/simd/pkg/simd/cpu"

var (
	hasNEON = cpu.ARM64.NEON
)

func dotProduct(a, b []float64) float64 {
	if hasNEON && len(a) >= 2 {
		return dotProductNEON(a, b)
	}
	return dotProductGo(a, b)
}

func add(dst, a, b []float64) {
	if hasNEON && len(dst) >= 2 {
		addNEON(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func sub(dst, a, b []float64) {
	if hasNEON && len(dst) >= 2 {
		subNEON(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

func mul(dst, a, b []float64) {
	if hasNEON && len(dst) >= 2 {
		mulNEON(dst, a, b)
		return
	}
	mulGo(dst, a, b)
}

func div(dst, a, b []float64) {
	if hasNEON && len(dst) >= 2 {
		divNEON(dst, a, b)
		return
	}
	divGo(dst, a, b)
}

func scale(dst, a []float64, s float64) {
	if hasNEON && len(dst) >= 2 {
		scaleNEON(dst, a, s)
		return
	}
	scaleGo(dst, a, s)
}

func addScalar(dst, a []float64, s float64) {
	if hasNEON && len(dst) >= 2 {
		addScalarNEON(dst, a, s)
		return
	}
	addScalarGo(dst, a, s)
}

func sum(a []float64) float64 {
	if hasNEON && len(a) >= 2 {
		return sumNEON(a)
	}
	return sumGo(a)
}

func min64(a []float64) float64 {
	if hasNEON && len(a) >= 2 {
		return minNEON(a)
	}
	return minGo(a)
}

func max64(a []float64) float64 {
	if hasNEON && len(a) >= 2 {
		return maxNEON(a)
	}
	return maxGo(a)
}

func abs64(dst, a []float64) {
	if hasNEON && len(dst) >= 2 {
		absNEON(dst, a)
		return
	}
	absGo(dst, a)
}

func neg64(dst, a []float64) {
	if hasNEON && len(dst) >= 2 {
		negNEON(dst, a)
		return
	}
	negGo(dst, a)
}

func fma64(dst, a, b, c []float64) {
	if hasNEON && len(dst) >= 2 {
		fmaNEON(dst, a, b, c)
		return
	}
	fmaGo(dst, a, b, c)
}

func clamp64(dst, a []float64, minVal, maxVal float64) {
	if hasNEON && len(dst) >= 2 {
		clampNEON(dst, a, minVal, maxVal)
		return
	}
	clampGo(dst, a, minVal, maxVal)
}

func sqrt64(dst, a []float64) {
	if hasNEON && len(dst) >= 2 {
		sqrtNEON(dst, a)
		return
	}
	sqrt64Go(dst, a)
}

func reciprocal64(dst, a []float64) {
	if hasNEON && len(dst) >= 2 {
		reciprocalNEON(dst, a)
		return
	}
	reciprocal64Go(dst, a)
}

func variance64(a []float64, mean float64) float64 {
	if hasNEON && len(a) >= 2 {
		return varianceNEON(a, mean)
	}
	return variance64Go(a, mean)
}

func euclideanDistance64(a, b []float64) float64 {
	if hasNEON && len(a) >= 2 {
		return euclideanDistanceNEON(a, b)
	}
	return euclideanDistance64Go(a, b)
}

func cumulativeSum64(dst, a []float64) {
	// CumulativeSum is inherently sequential
	cumulativeSum64Go(dst, a)
}

func dotProductBatch64(results []float64, rows [][]float64, vec []float64) {
	// Batch dot product benefits from cache locality of vec
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

func convolveValid64(dst, signal, kernel []float64) {
	// Convolution as sliding dot products
	kLen := len(kernel)
	for i := range dst {
		dst[i] = dotProduct(signal[i:i+kLen], kernel)
	}
}

//go:noescape
func dotProductNEON(a, b []float64) float64

//go:noescape
func addNEON(dst, a, b []float64)

//go:noescape
func subNEON(dst, a, b []float64)

//go:noescape
func mulNEON(dst, a, b []float64)

//go:noescape
func divNEON(dst, a, b []float64)

//go:noescape
func scaleNEON(dst, a []float64, s float64)

//go:noescape
func addScalarNEON(dst, a []float64, s float64)

//go:noescape
func sumNEON(a []float64) float64

//go:noescape
func minNEON(a []float64) float64

//go:noescape
func maxNEON(a []float64) float64

//go:noescape
func absNEON(dst, a []float64)

//go:noescape
func negNEON(dst, a []float64)

//go:noescape
func fmaNEON(dst, a, b, c []float64)

//go:noescape
func clampNEON(dst, a []float64, minVal, maxVal float64)

//go:noescape
func sqrtNEON(dst, a []float64)

//go:noescape
func reciprocalNEON(dst, a []float64)

//go:noescape
func varianceNEON(a []float64, mean float64) float64

//go:noescape
func euclideanDistanceNEON(a, b []float64) float64
