//go:build amd64

package f64

import "github.com/tphakala/simd/pkg/simd/cpu"

var (
	hasAVX = cpu.X86.AVX && cpu.X86.FMA
)

// Dispatch functions - select optimal implementation at runtime

func dotProduct(a, b []float64) float64 {
	if hasAVX && len(a) >= 4 {
		return dotProductAVX(a, b)
	}
	return dotProductGo(a, b)
}

func add(dst, a, b []float64) {
	if hasAVX && len(dst) >= 4 {
		addAVX(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func sub(dst, a, b []float64) {
	if hasAVX && len(dst) >= 4 {
		subAVX(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

func mul(dst, a, b []float64) {
	if hasAVX && len(dst) >= 4 {
		mulAVX(dst, a, b)
		return
	}
	mulGo(dst, a, b)
}

func div(dst, a, b []float64) {
	if hasAVX && len(dst) >= 4 {
		divAVX(dst, a, b)
		return
	}
	divGo(dst, a, b)
}

func scale(dst, a []float64, s float64) {
	if hasAVX && len(dst) >= 4 {
		scaleAVX(dst, a, s)
		return
	}
	scaleGo(dst, a, s)
}

func addScalar(dst, a []float64, s float64) {
	if hasAVX && len(dst) >= 4 {
		addScalarAVX(dst, a, s)
		return
	}
	addScalarGo(dst, a, s)
}

func sum(a []float64) float64 {
	if hasAVX && len(a) >= 4 {
		return sumAVX(a)
	}
	return sumGo(a)
}

func min64(a []float64) float64 {
	if hasAVX && len(a) >= 4 {
		return minAVX(a)
	}
	return minGo(a)
}

func max64(a []float64) float64 {
	if hasAVX && len(a) >= 4 {
		return maxAVX(a)
	}
	return maxGo(a)
}

func abs64(dst, a []float64) {
	if hasAVX && len(dst) >= 4 {
		absAVX(dst, a)
		return
	}
	absGo(dst, a)
}

func neg64(dst, a []float64) {
	if hasAVX && len(dst) >= 4 {
		negAVX(dst, a)
		return
	}
	negGo(dst, a)
}

func fma64(dst, a, b, c []float64) {
	if hasAVX && len(dst) >= 4 {
		fmaAVX(dst, a, b, c)
		return
	}
	fmaGo(dst, a, b, c)
}

func clamp64(dst, a []float64, minVal, maxVal float64) {
	if hasAVX && len(dst) >= 4 {
		clampAVX(dst, a, minVal, maxVal)
		return
	}
	clampGo(dst, a, minVal, maxVal)
}

func sqrt64(dst, a []float64) {
	if hasAVX && len(dst) >= 4 {
		sqrtAVX(dst, a)
		return
	}
	sqrt64Go(dst, a)
}

func reciprocal64(dst, a []float64) {
	if hasAVX && len(dst) >= 4 {
		reciprocalAVX(dst, a)
		return
	}
	reciprocal64Go(dst, a)
}

func variance64(a []float64, mean float64) float64 {
	if hasAVX && len(a) >= 4 {
		return varianceAVX(a, mean)
	}
	return variance64Go(a, mean)
}

func euclideanDistance64(a, b []float64) float64 {
	if hasAVX && len(a) >= 4 {
		return euclideanDistanceAVX(a, b)
	}
	return euclideanDistance64Go(a, b)
}

func cumulativeSum64(dst, a []float64) {
	// CumulativeSum is inherently sequential, so Go implementation is fine
	cumulativeSum64Go(dst, a)
}

// Assembly function declarations
//
//go:noescape
func dotProductAVX(a, b []float64) float64

//go:noescape
func addAVX(dst, a, b []float64)

//go:noescape
func subAVX(dst, a, b []float64)

//go:noescape
func mulAVX(dst, a, b []float64)

//go:noescape
func divAVX(dst, a, b []float64)

//go:noescape
func scaleAVX(dst, a []float64, s float64)

//go:noescape
func addScalarAVX(dst, a []float64, s float64)

//go:noescape
func sumAVX(a []float64) float64

//go:noescape
func minAVX(a []float64) float64

//go:noescape
func maxAVX(a []float64) float64

//go:noescape
func absAVX(dst, a []float64)

//go:noescape
func negAVX(dst, a []float64)

//go:noescape
func fmaAVX(dst, a, b, c []float64)

//go:noescape
func clampAVX(dst, a []float64, minVal, maxVal float64)

//go:noescape
func sqrtAVX(dst, a []float64)

//go:noescape
func reciprocalAVX(dst, a []float64)

//go:noescape
func varianceAVX(a []float64, mean float64) float64

//go:noescape
func euclideanDistanceAVX(a, b []float64) float64
