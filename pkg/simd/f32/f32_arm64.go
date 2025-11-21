//go:build arm64

package f32

import "github.com/tphakala/simd/pkg/simd/cpu"

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
