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

func convolveValidMulti32(dsts [][]float32, signal []float32, kernels [][]float32, n, kLen int) {
	for i := range n {
		sig := signal[i : i+kLen]
		for k, kernel := range kernels {
			dsts[k][i] = dotProduct(sig, kernel)
		}
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

//go:noescape
func varianceNEON32(a []float32, mean float32) float32

//go:noescape
func euclideanDistanceNEON32(a, b []float32) float32
