//go:build amd64

package f64

import "github.com/tphakala/simd/cpu"

// Minimum number of float64 elements required for SIMD operations.
// AVX processes 4 float64 values per 256-bit register.
// AVX-512 processes 8 float64 values per 512-bit register.
const (
	minAVXElements    = 4
	minAVX512Elements = 8
)

// minSIMDElements is set at init time based on which SIMD implementation is selected.
// Used by min64/max64 to determine when to fall back to scalar code.
var minSIMDElements = minAVXElements

// Function pointer types for SIMD operations
type (
	dotProductFunc        func(a, b []float64) float64
	binaryOpFunc          func(dst, a, b []float64)
	scaleFunc             func(dst, a []float64, s float64)
	unaryOpFunc           func(dst, a []float64)
	reduceFunc            func(a []float64) float64
	fmaFunc               func(dst, a, b, c []float64)
	clampFunc             func(dst, a []float64, minVal, maxVal float64)
	varianceFunc          func(a []float64, mean float64) float64
	euclideanDistanceFunc func(a, b []float64) float64
	interleave2Func       func(dst, a, b []float64)
	deinterleave2Func     func(a, b, src []float64)
	addScaledFunc         func(dst []float64, alpha float64, s []float64)
)

// Function pointers - assigned at init time based on CPU features
var (
	dotProductImpl        dotProductFunc
	addImpl               binaryOpFunc
	subImpl               binaryOpFunc
	mulImpl               binaryOpFunc
	divImpl               binaryOpFunc
	scaleImpl             scaleFunc
	addScalarImpl         scaleFunc
	sumImpl               reduceFunc
	minImpl               reduceFunc
	maxImpl               reduceFunc
	absImpl               unaryOpFunc
	negImpl               unaryOpFunc
	sqrtImpl              unaryOpFunc
	reciprocalImpl        unaryOpFunc
	fmaImpl               fmaFunc
	clampImpl             clampFunc
	varianceImpl          varianceFunc
	euclideanDistanceImpl euclideanDistanceFunc
	interleave2Impl       interleave2Func
	deinterleave2Impl     deinterleave2Func
	addScaledImpl         addScaledFunc
)

func init() {
	// Select optimal implementation based on CPU features
	// Priority: AVX-512 > AVX+FMA > SSE2 > Go
	switch {
	case cpu.X86.AVX512F && cpu.X86.AVX512VL:
		initAVX512()
	case cpu.X86.AVX && cpu.X86.FMA:
		initAVX()
	case cpu.X86.SSE2:
		initSSE2()
	default:
		initGo()
	}
}

func initAVX512() {
	minSIMDElements = minAVX512Elements
	dotProductImpl = dotProductAVX512
	addImpl = addAVX512
	subImpl = subAVX512
	mulImpl = mulAVX512
	divImpl = divAVX512
	scaleImpl = scaleAVX512
	addScalarImpl = addScalarAVX512
	sumImpl = sumAVX512
	minImpl = minAVX512
	maxImpl = maxAVX512
	absImpl = absAVX512
	negImpl = negAVX512
	sqrtImpl = sqrtAVX512
	reciprocalImpl = reciprocalAVX512
	fmaImpl = fmaAVX512
	clampImpl = clampAVX512
	varianceImpl = varianceAVX512
	euclideanDistanceImpl = euclideanDistanceAVX512
	interleave2Impl = interleave2AVX
	deinterleave2Impl = deinterleave2AVX
	addScaledImpl = addScaledAVX512
}

func initAVX() {
	dotProductImpl = dotProductAVX
	addImpl = addAVX
	subImpl = subAVX
	mulImpl = mulAVX
	divImpl = divAVX
	scaleImpl = scaleAVX
	addScalarImpl = addScalarAVX
	sumImpl = sumAVX
	minImpl = minAVX
	maxImpl = maxAVX
	absImpl = absAVX
	negImpl = negAVX
	sqrtImpl = sqrtAVX
	reciprocalImpl = reciprocalAVX
	fmaImpl = fmaAVX
	clampImpl = clampAVX
	varianceImpl = varianceAVX
	euclideanDistanceImpl = euclideanDistanceAVX
	interleave2Impl = interleave2AVX
	deinterleave2Impl = deinterleave2AVX
	addScaledImpl = addScaledAVX
}

func initSSE2() {
	dotProductImpl = dotProductSSE2
	addImpl = addSSE2
	subImpl = subSSE2
	mulImpl = mulSSE2
	divImpl = divSSE2
	scaleImpl = scaleSSE2
	addScalarImpl = addScalarSSE2
	sumImpl = sumSSE2
	minImpl = minSSE2
	maxImpl = maxSSE2
	absImpl = absSSE2
	negImpl = negSSE2
	sqrtImpl = sqrtSSE2
	reciprocalImpl = reciprocalSSE2
	fmaImpl = fmaSSE2
	clampImpl = clampSSE2
	varianceImpl = varianceSSE2
	euclideanDistanceImpl = euclideanDistanceSSE2
	interleave2Impl = interleave2SSE2
	deinterleave2Impl = deinterleave2SSE2
	addScaledImpl = addScaledSSE2
}

func initGo() {
	dotProductImpl = dotProductGo
	addImpl = addGo
	subImpl = subGo
	mulImpl = mulGo
	divImpl = divGo
	scaleImpl = scaleGo
	addScalarImpl = addScalarGo
	sumImpl = sumGo
	minImpl = minGo
	maxImpl = maxGo
	absImpl = absGo
	negImpl = negGo
	sqrtImpl = sqrt64Go
	reciprocalImpl = reciprocal64Go
	fmaImpl = fmaGo
	clampImpl = clampGo
	varianceImpl = variance64Go
	euclideanDistanceImpl = euclideanDistance64Go
	interleave2Impl = interleave2Go
	deinterleave2Impl = deinterleave2Go
	addScaledImpl = addScaledGo64
}

// Dispatch functions - call function pointers (zero overhead after init)

func dotProduct(a, b []float64) float64 {
	return dotProductImpl(a, b)
}

func add(dst, a, b []float64) {
	addImpl(dst, a, b)
}

func sub(dst, a, b []float64) {
	subImpl(dst, a, b)
}

func mul(dst, a, b []float64) {
	mulImpl(dst, a, b)
}

func div(dst, a, b []float64) {
	divImpl(dst, a, b)
}

func scale(dst, a []float64, s float64) {
	scaleImpl(dst, a, s)
}

func addScalar(dst, a []float64, s float64) {
	addScalarImpl(dst, a, s)
}

func sum(a []float64) float64 {
	return sumImpl(a)
}

func min64(a []float64) float64 {
	// AVX/AVX-512 requires at least 4/8 elements for initial vector load
	// Fall back to Go for small slices to avoid reading beyond bounds
	if len(a) < minSIMDElements {
		return minGo(a)
	}
	return minImpl(a)
}

func max64(a []float64) float64 {
	// AVX/AVX-512 requires at least 4/8 elements for initial vector load
	// Fall back to Go for small slices to avoid reading beyond bounds
	if len(a) < minSIMDElements {
		return maxGo(a)
	}
	return maxImpl(a)
}

func abs64(dst, a []float64) {
	absImpl(dst, a)
}

func neg64(dst, a []float64) {
	negImpl(dst, a)
}

func fma64(dst, a, b, c []float64) {
	fmaImpl(dst, a, b, c)
}

func clamp64(dst, a []float64, minVal, maxVal float64) {
	clampImpl(dst, a, minVal, maxVal)
}

func sqrt64(dst, a []float64) {
	sqrtImpl(dst, a)
}

func reciprocal64(dst, a []float64) {
	reciprocalImpl(dst, a)
}

func variance64(a []float64, mean float64) float64 {
	return varianceImpl(a, mean)
}

func euclideanDistance64(a, b []float64) float64 {
	return euclideanDistanceImpl(a, b)
}

func cumulativeSum64(dst, a []float64) {
	// CumulativeSum is inherently sequential
	cumulativeSum64Go(dst, a)
}

func dotProductBatch64(results []float64, rows [][]float64, vec []float64) {
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
	kLen := len(kernel)
	for i := range dst {
		dst[i] = dotProduct(signal[i:i+kLen], kernel)
	}
}

func accumulateAdd64(dst, src []float64) {
	// AccumulateAdd is dst += src, which is the same as add(dst, dst, src)
	addImpl(dst, dst, src)
}

func interleave2_64(dst, a, b []float64) {
	// Need at least 2 pairs for SIMD to be worthwhile (AVX processes 4 at a time)
	if len(a) >= minAVXElements {
		interleave2Impl(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2_64(a, b, src []float64) {
	// Need at least 2 pairs for SIMD to be worthwhile
	if len(a) >= minAVXElements {
		deinterleave2Impl(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

func convolveValidMulti64(dsts [][]float64, signal []float64, kernels [][]float64, n, kLen int) {
	// Use the SIMD dotProduct for each kernel application
	for i := range n {
		sig := signal[i : i+kLen]
		for k, kernel := range kernels {
			dsts[k][i] = dotProduct(sig, kernel)
		}
	}
}

func minIdx64(a []float64) int {
	return minIdxGo64(a)
}

func maxIdx64(a []float64) int {
	return maxIdxGo64(a)
}

func addScaled64(dst []float64, alpha float64, s []float64) {
	addScaledImpl(dst, alpha, s)
}

// AVX+FMA assembly function declarations (4x float64 per iteration)
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

//go:noescape
func addScaledAVX(dst []float64, alpha float64, s []float64)

// AVX-512 assembly function declarations (8x float64 per iteration)
//
//go:noescape
func dotProductAVX512(a, b []float64) float64

//go:noescape
func addAVX512(dst, a, b []float64)

//go:noescape
func subAVX512(dst, a, b []float64)

//go:noescape
func mulAVX512(dst, a, b []float64)

//go:noescape
func divAVX512(dst, a, b []float64)

//go:noescape
func scaleAVX512(dst, a []float64, s float64)

//go:noescape
func addScalarAVX512(dst, a []float64, s float64)

//go:noescape
func sumAVX512(a []float64) float64

//go:noescape
func minAVX512(a []float64) float64

//go:noescape
func maxAVX512(a []float64) float64

//go:noescape
func absAVX512(dst, a []float64)

//go:noescape
func negAVX512(dst, a []float64)

//go:noescape
func fmaAVX512(dst, a, b, c []float64)

//go:noescape
func clampAVX512(dst, a []float64, minVal, maxVal float64)

//go:noescape
func sqrtAVX512(dst, a []float64)

//go:noescape
func reciprocalAVX512(dst, a []float64)

//go:noescape
func varianceAVX512(a []float64, mean float64) float64

//go:noescape
func euclideanDistanceAVX512(a, b []float64) float64

//go:noescape
func addScaledAVX512(dst []float64, alpha float64, s []float64)

// SSE2 assembly function declarations (2x float64 per iteration)
//
//go:noescape
func dotProductSSE2(a, b []float64) float64

//go:noescape
func addSSE2(dst, a, b []float64)

//go:noescape
func subSSE2(dst, a, b []float64)

//go:noescape
func mulSSE2(dst, a, b []float64)

//go:noescape
func divSSE2(dst, a, b []float64)

//go:noescape
func scaleSSE2(dst, a []float64, s float64)

//go:noescape
func addScalarSSE2(dst, a []float64, s float64)

//go:noescape
func sumSSE2(a []float64) float64

//go:noescape
func minSSE2(a []float64) float64

//go:noescape
func maxSSE2(a []float64) float64

//go:noescape
func absSSE2(dst, a []float64)

//go:noescape
func negSSE2(dst, a []float64)

//go:noescape
func fmaSSE2(dst, a, b, c []float64)

//go:noescape
func clampSSE2(dst, a []float64, minVal, maxVal float64)

//go:noescape
func sqrtSSE2(dst, a []float64)

//go:noescape
func reciprocalSSE2(dst, a []float64)

//go:noescape
func varianceSSE2(a []float64, mean float64) float64

//go:noescape
func euclideanDistanceSSE2(a, b []float64) float64

//go:noescape
func addScaledSSE2(dst []float64, alpha float64, s []float64)

// Interleave/Deinterleave assembly function declarations
//
//go:noescape
func interleave2AVX(dst, a, b []float64)

//go:noescape
func deinterleave2AVX(a, b, src []float64)

//go:noescape
func interleave2SSE2(dst, a, b []float64)

//go:noescape
func deinterleave2SSE2(a, b, src []float64)
