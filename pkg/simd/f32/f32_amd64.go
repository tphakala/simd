//go:build amd64

package f32

import "github.com/tphakala/simd/pkg/simd/cpu"

// Minimum number of float32 elements required for SIMD operations.
// AVX processes 8 float32 values per 256-bit register.
// AVX-512 processes 16 float32 values per 512-bit register.
const (
	minAVXElements    = 8
	minAVX512Elements = 16
)

// minSIMDElements is set at init time based on which SIMD implementation is selected.
// Used by min32/max32 to determine when to fall back to scalar code.
var minSIMDElements = minAVXElements

// Function pointer types for SIMD operations
type (
	dotProductFunc func(a, b []float32) float32
	binaryOpFunc   func(dst, a, b []float32)
	scaleFunc      func(dst, a []float32, s float32)
	unaryOpFunc    func(dst, a []float32)
	reduceFunc     func(a []float32) float32
	fmaFunc        func(dst, a, b, c []float32)
	clampFunc      func(dst, a []float32, minVal, maxVal float32)
)

// Function pointers - assigned at init time based on CPU features
var (
	dotProductImpl dotProductFunc
	addImpl        binaryOpFunc
	subImpl        binaryOpFunc
	mulImpl        binaryOpFunc
	divImpl        binaryOpFunc
	scaleImpl      scaleFunc
	addScalarImpl  scaleFunc
	sumImpl        reduceFunc
	minImpl        reduceFunc
	maxImpl        reduceFunc
	absImpl        unaryOpFunc
	negImpl        unaryOpFunc
	fmaImpl        fmaFunc
	clampImpl      clampFunc
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
		initSSE()
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
	fmaImpl = fmaAVX512
	clampImpl = clampAVX512
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
	fmaImpl = fmaAVX
	clampImpl = clampAVX
}

func initSSE() {
	dotProductImpl = dotProductSSE
	addImpl = addSSE
	subImpl = subSSE
	mulImpl = mulSSE
	divImpl = divSSE
	scaleImpl = scaleSSE
	addScalarImpl = addScalarSSE
	sumImpl = sumSSE
	minImpl = minSSE
	maxImpl = maxSSE
	absImpl = absSSE
	negImpl = negSSE
	fmaImpl = fmaSSE
	clampImpl = clampSSE
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
	fmaImpl = fmaGo
	clampImpl = clampGo
}

// Dispatch functions - call function pointers (zero overhead after init)

func dotProduct(a, b []float32) float32 {
	return dotProductImpl(a, b)
}

func add(dst, a, b []float32) {
	addImpl(dst, a, b)
}

func sub(dst, a, b []float32) {
	subImpl(dst, a, b)
}

func mul(dst, a, b []float32) {
	mulImpl(dst, a, b)
}

func div(dst, a, b []float32) {
	divImpl(dst, a, b)
}

func scale(dst, a []float32, s float32) {
	scaleImpl(dst, a, s)
}

func addScalar(dst, a []float32, s float32) {
	addScalarImpl(dst, a, s)
}

func sum(a []float32) float32 {
	return sumImpl(a)
}

func min32(a []float32) float32 {
	// AVX/AVX-512 requires at least 8/16 elements for initial vector load
	// Fall back to Go for small slices to avoid reading beyond bounds
	if len(a) < minSIMDElements {
		return minGo(a)
	}
	return minImpl(a)
}

func max32(a []float32) float32 {
	// AVX/AVX-512 requires at least 8/16 elements for initial vector load
	// Fall back to Go for small slices to avoid reading beyond bounds
	if len(a) < minSIMDElements {
		return maxGo(a)
	}
	return maxImpl(a)
}

func abs32(dst, a []float32) {
	absImpl(dst, a)
}

func neg32(dst, a []float32) {
	negImpl(dst, a)
}

func fma32(dst, a, b, c []float32) {
	fmaImpl(dst, a, b, c)
}

func clamp32(dst, a []float32, minVal, maxVal float32) {
	clampImpl(dst, a, minVal, maxVal)
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

// AVX+FMA assembly function declarations (8x float32 per iteration)
//
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

// AVX-512 assembly function declarations (16x float32 per iteration)
//
//go:noescape
func dotProductAVX512(a, b []float32) float32

//go:noescape
func addAVX512(dst, a, b []float32)

//go:noescape
func subAVX512(dst, a, b []float32)

//go:noescape
func mulAVX512(dst, a, b []float32)

//go:noescape
func divAVX512(dst, a, b []float32)

//go:noescape
func scaleAVX512(dst, a []float32, s float32)

//go:noescape
func addScalarAVX512(dst, a []float32, s float32)

//go:noescape
func sumAVX512(a []float32) float32

//go:noescape
func minAVX512(a []float32) float32

//go:noescape
func maxAVX512(a []float32) float32

//go:noescape
func absAVX512(dst, a []float32)

//go:noescape
func negAVX512(dst, a []float32)

//go:noescape
func fmaAVX512(dst, a, b, c []float32)

//go:noescape
func clampAVX512(dst, a []float32, minVal, maxVal float32)

// SSE assembly function declarations (4x float32 per iteration)
//
//go:noescape
func dotProductSSE(a, b []float32) float32

//go:noescape
func addSSE(dst, a, b []float32)

//go:noescape
func subSSE(dst, a, b []float32)

//go:noescape
func mulSSE(dst, a, b []float32)

//go:noescape
func divSSE(dst, a, b []float32)

//go:noescape
func scaleSSE(dst, a []float32, s float32)

//go:noescape
func addScalarSSE(dst, a []float32, s float32)

//go:noescape
func sumSSE(a []float32) float32

//go:noescape
func minSSE(a []float32) float32

//go:noescape
func maxSSE(a []float32) float32

//go:noescape
func absSSE(dst, a []float32)

//go:noescape
func negSSE(dst, a []float32)

//go:noescape
func fmaSSE(dst, a, b, c []float32)

//go:noescape
func clampSSE(dst, a []float32, minVal, maxVal float32)
