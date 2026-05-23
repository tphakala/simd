//go:build arm64

package f16

import "github.com/tphakala/simd/cpu"

// neonWidth is the number of FP16 elements per NEON vector (128-bit / 16-bit = 8).
const neonWidth = 8

var (
	hasFP16 = cpu.ARM64.FP16
	hasNEON = cpu.ARM64.NEON
)

func toFloat32(h Float16) float32 {
	return toFloat32Go(h)
}

func fromFloat32(f float32) Float16 {
	return fromFloat32Go(f)
}

func toFloat32Slice(dst []float32, src []Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		// Process vectorized portion (multiples of neonWidth)
		nVec := (n / neonWidth) * neonWidth
		toFloat32SliceNEON(dst[:nVec], src[:nVec])
		// Handle remainder with Go
		toFloat32SliceGo(dst[nVec:], src[nVec:])
		return
	}
	toFloat32SliceGo(dst, src)
}

func fromFloat32Slice(dst []Float16, src []float32) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		// Process vectorized portion (multiples of neonWidth)
		nVec := (n / neonWidth) * neonWidth
		fromFloat32SliceNEON(dst[:nVec], src[:nVec])
		// Handle remainder with Go
		fromFloat32SliceGo(dst[nVec:], src[nVec:])
		return
	}
	fromFloat32SliceGo(dst, src)
}

func dotProduct(a, b []Float16) float32 {
	n := min(len(a), len(b))
	if hasFP16 && n >= neonWidth {
		// Process vectorized portion
		nVec := (n / neonWidth) * neonWidth
		result := dotProductNEON(a[:nVec], b[:nVec])
		// Handle remainder with Go
		result += dotProductGo(a[nVec:n], b[nVec:n])
		return result
	}
	return dotProductGo(a, b)
}

func dotProductF32(a, b []Float16) float32 {
	// Task 3 will swap this to a NEON kernel that widens FP16 to FP32 first.
	// For now, the Go fallback already widens before multiplying, so it
	// gives the right answer on every input, just slower than the eventual
	// NEON path.
	return dotProductGo(a, b)
}

func add(dst, a, b []Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		addNEON(dst[:nVec], a[:nVec], b[:nVec])
		addGo(dst[nVec:], a[nVec:], b[nVec:])
		return
	}
	addGo(dst, a, b)
}

func sub(dst, a, b []Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		subNEON(dst[:nVec], a[:nVec], b[:nVec])
		subGo(dst[nVec:], a[nVec:], b[nVec:])
		return
	}
	subGo(dst, a, b)
}

func mul(dst, a, b []Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		mulNEON(dst[:nVec], a[:nVec], b[:nVec])
		mulGo(dst[nVec:], a[nVec:], b[nVec:])
		return
	}
	mulGo(dst, a, b)
}

func scale(dst, a []Float16, s Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		scaleNEON(dst[:nVec], a[:nVec], s)
		scaleGo(dst[nVec:], a[nVec:], s)
		return
	}
	scaleGo(dst, a, s)
}

func fma16(dst, a, b, c []Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		fmaNEON(dst[:nVec], a[:nVec], b[:nVec], c[:nVec])
		fmaGo(dst[nVec:], a[nVec:], b[nVec:], c[nVec:])
		return
	}
	fmaGo(dst, a, b, c)
}

func sum(a []Float16) float32 {
	n := len(a)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		result := sumNEON(a[:nVec])
		result += sumGo(a[nVec:])
		return result
	}
	return sumGo(a)
}

func abs16(dst, a []Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		absNEON(dst[:nVec], a[:nVec])
		absGo(dst[nVec:], a[nVec:])
		return
	}
	absGo(dst, a)
}

func neg16(dst, a []Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		negNEON(dst[:nVec], a[:nVec])
		negGo(dst[nVec:], a[nVec:])
		return
	}
	negGo(dst, a)
}

func relu16(dst, src []Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		reluNEON(dst[:nVec], src[:nVec])
		reluGo(dst[nVec:], src[nVec:])
		return
	}
	reluGo(dst, src)
}

func sigmoid16(dst, src []Float16) {
	// Sigmoid is complex - use Go fallback for now
	sigmoidGo(dst, src)
}

func min16(a []Float16) Float16 {
	n := len(a)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		minVec := minNEON(a[:nVec])
		if nVec < n {
			minRem := minGo(a[nVec:])
			if toFloat32Go(minRem) < toFloat32Go(minVec) {
				return minRem
			}
		}
		return minVec
	}
	return minGo(a)
}

func max16(a []Float16) Float16 {
	n := len(a)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		maxVec := maxNEON(a[:nVec])
		if nVec < n {
			maxRem := maxGo(a[nVec:])
			if toFloat32Go(maxRem) > toFloat32Go(maxVec) {
				return maxRem
			}
		}
		return maxVec
	}
	return maxGo(a)
}

func div16(dst, a, b []Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		divNEON(dst[:nVec], a[:nVec], b[:nVec])
		divGo(dst[nVec:], a[nVec:], b[nVec:])
		return
	}
	divGo(dst, a, b)
}

func addScalar16(dst, a []Float16, s Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		addScalarNEON(dst[:nVec], a[:nVec], s)
		addScalarGo(dst[nVec:], a[nVec:], s)
		return
	}
	addScalarGo(dst, a, s)
}

func clamp16(dst, a []Float16, minVal, maxVal Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		clampNEON(dst[:nVec], a[:nVec], minVal, maxVal)
		clampGo(dst[nVec:], a[nVec:], minVal, maxVal)
		return
	}
	clampGo(dst, a, minVal, maxVal)
}

func sqrt16(dst, a []Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		sqrtNEON(dst[:nVec], a[:nVec])
		sqrtGo(dst[nVec:], a[nVec:])
		return
	}
	sqrtGo(dst, a)
}

func reciprocal16(dst, a []Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		reciprocalNEON(dst[:nVec], a[:nVec])
		reciprocalGo(dst[nVec:], a[nVec:])
		return
	}
	reciprocalGo(dst, a)
}

func exp16(dst, src []Float16) {
	// Exp requires polynomial approximation - use Go for now
	expGo(dst, src)
}

func tanh16(dst, src []Float16) {
	// Tanh requires polynomial approximation - use Go for now
	tanhGo(dst, src)
}

func minIdx16(a []Float16) int {
	// MinIdx doesn't vectorize well - use Go
	return minIdxGo(a)
}

func maxIdx16(a []Float16) int {
	// MaxIdx doesn't vectorize well - use Go
	return maxIdxGo(a)
}

func addScaled16(dst []Float16, alpha Float16, s []Float16) {
	n := len(dst)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		addScaledNEON(dst[:nVec], alpha, s[:nVec])
		addScaledGo(dst[nVec:], alpha, s[nVec:])
		return
	}
	addScaledGo(dst, alpha, s)
}

func euclideanDistance16(a, b []Float16) float32 {
	// Use Go implementation - could optimize later
	return euclideanDistanceGo(a, b)
}

func variance16(a []Float16, mean float32) float32 {
	return varianceGo(a, mean)
}

func cumulativeSum16(dst, a []Float16) {
	// Cumulative sum is inherently sequential
	cumulativeSumGo(dst, a)
}

func dotProductBatch16(results []float32, rows [][]Float16, vec []Float16) {
	// Use per-row SIMD via dotProduct
	for i, row := range rows {
		results[i] = dotProduct(row, vec)
	}
}

func accumulateAdd16(dst, src []Float16) {
	n := len(src)
	if hasFP16 && n >= neonWidth {
		nVec := (n / neonWidth) * neonWidth
		accumulateAddNEON(dst[:nVec], src[:nVec])
		accumulateAddGo(dst[nVec:], src[nVec:])
		return
	}
	accumulateAddGo(dst, src)
}

func convolveValid16(dst, signal, kernel []Float16) {
	convolveValidGo(dst, signal, kernel)
}

func interleave2_16(dst, a, b []Float16) {
	interleave2Go(dst, a, b)
}

func deinterleave2_16(a, b, src []Float16) {
	deinterleave2Go(a, b, src)
}

func clampScale16(dst, src []Float16, minVal, maxVal, scale Float16) {
	clampScaleGo(dst, src, minVal, maxVal, scale)
}

//go:noescape
func toFloat32SliceNEON(dst []float32, src []Float16)

//go:noescape
func fromFloat32SliceNEON(dst []Float16, src []float32)

//go:noescape
func dotProductNEON(a, b []Float16) float32

//go:noescape
func addNEON(dst, a, b []Float16)

//go:noescape
func subNEON(dst, a, b []Float16)

//go:noescape
func mulNEON(dst, a, b []Float16)

//go:noescape
func scaleNEON(dst, a []Float16, s Float16)

//go:noescape
func fmaNEON(dst, a, b, c []Float16)

//go:noescape
func sumNEON(a []Float16) float32

//go:noescape
func absNEON(dst, a []Float16)

//go:noescape
func negNEON(dst, a []Float16)

//go:noescape
func reluNEON(dst, src []Float16)

//go:noescape
func minNEON(a []Float16) Float16

//go:noescape
func maxNEON(a []Float16) Float16

//go:noescape
func divNEON(dst, a, b []Float16)

//go:noescape
func addScalarNEON(dst, a []Float16, s Float16)

//go:noescape
func clampNEON(dst, a []Float16, minVal, maxVal Float16)

//go:noescape
func sqrtNEON(dst, a []Float16)

//go:noescape
func reciprocalNEON(dst, a []Float16)

//go:noescape
func addScaledNEON(dst []Float16, alpha Float16, s []Float16)

//go:noescape
func accumulateAddNEON(dst, src []Float16)
