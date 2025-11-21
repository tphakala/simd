//go:build amd64

package f32

import "github.com/tphakala/simd/pkg/simd/cpu"

var (
	hasAVX = cpu.X86.AVX && cpu.X86.FMA
)

func dotProduct(a, b []float32) float32 {
	if hasAVX && len(a) >= 8 {
		return dotProductAVX(a, b)
	}
	return dotProductGo(a, b)
}

func add(dst, a, b []float32) {
	if hasAVX && len(dst) >= 8 {
		addAVX(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func sub(dst, a, b []float32) {
	if hasAVX && len(dst) >= 8 {
		subAVX(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

func mul(dst, a, b []float32) {
	if hasAVX && len(dst) >= 8 {
		mulAVX(dst, a, b)
		return
	}
	mulGo(dst, a, b)
}

func div(dst, a, b []float32) {
	if hasAVX && len(dst) >= 8 {
		divAVX(dst, a, b)
		return
	}
	divGo(dst, a, b)
}

func scale(dst, a []float32, s float32) {
	if hasAVX && len(dst) >= 8 {
		scaleAVX(dst, a, s)
		return
	}
	scaleGo(dst, a, s)
}

func addScalar(dst, a []float32, s float32) {
	if hasAVX && len(dst) >= 8 {
		addScalarAVX(dst, a, s)
		return
	}
	addScalarGo(dst, a, s)
}

func sum(a []float32) float32 {
	if hasAVX && len(a) >= 8 {
		return sumAVX(a)
	}
	return sumGo(a)
}

func min32(a []float32) float32 {
	if hasAVX && len(a) >= 8 {
		return minAVX(a)
	}
	return minGo(a)
}

func max32(a []float32) float32 {
	if hasAVX && len(a) >= 8 {
		return maxAVX(a)
	}
	return maxGo(a)
}

func abs32(dst, a []float32) {
	if hasAVX && len(dst) >= 8 {
		absAVX(dst, a)
		return
	}
	absGo(dst, a)
}

func neg32(dst, a []float32) {
	if hasAVX && len(dst) >= 8 {
		negAVX(dst, a)
		return
	}
	negGo(dst, a)
}

func fma32(dst, a, b, c []float32) {
	if hasAVX && len(dst) >= 8 {
		fmaAVX(dst, a, b, c)
		return
	}
	fmaGo(dst, a, b, c)
}

func clamp32(dst, a []float32, minVal, maxVal float32) {
	if hasAVX && len(dst) >= 8 {
		clampAVX(dst, a, minVal, maxVal)
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

//go:noescape
func dotProductAVX(a, b []float32) float32

//go:noescape
func addAVX(dst, a, b []float32)

//go:noescape
func subAVX(dst, a, b []float32)

//go:noescape
func mulAVX(dst, a, b []float32)

//go:noescape
func divAVX(dst, a, b []float32)

//go:noescape
func scaleAVX(dst, a []float32, s float32)

//go:noescape
func addScalarAVX(dst, a []float32, s float32)

//go:noescape
func sumAVX(a []float32) float32

//go:noescape
func minAVX(a []float32) float32

//go:noescape
func maxAVX(a []float32) float32

//go:noescape
func absAVX(dst, a []float32)

//go:noescape
func negAVX(dst, a []float32)

//go:noescape
func fmaAVX(dst, a, b, c []float32)

//go:noescape
func clampAVX(dst, a []float32, minVal, maxVal float32)
