//go:build amd64

package f32

import "github.com/tphakala/simd/cpu"

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
	reduceIdxFunc  func(a []float32) int
	fmaFunc        func(dst, a, b, c []float32)
	clampFunc      func(dst, a []float32, minVal, maxVal float32)
	addScaledFunc  func(dst []float32, alpha float32, s []float32)
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
	sqrtImpl       unaryOpFunc
	reciprocalImpl unaryOpFunc
	fmaImpl        fmaFunc
	clampImpl      clampFunc
	minIdxImpl     reduceIdxFunc
	maxIdxImpl     reduceIdxFunc
	addScaledImpl  addScaledFunc
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
	sqrtImpl = sqrtAVX512
	reciprocalImpl = reciprocalAVX512
	fmaImpl = fmaAVX512
	clampImpl = clampAVX512
	minIdxImpl = minIdxGo
	maxIdxImpl = maxIdxGo
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
	minIdxImpl = minIdxGo
	maxIdxImpl = maxIdxGo
	addScaledImpl = addScaledAVX
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
	sqrtImpl = sqrtSSE
	reciprocalImpl = reciprocalSSE
	fmaImpl = fmaSSE
	clampImpl = clampSSE
	minIdxImpl = minIdxGo
	maxIdxImpl = maxIdxGo
	addScaledImpl = addScaledSSE
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
	sqrtImpl = sqrt32Go
	reciprocalImpl = reciprocal32Go
	fmaImpl = fmaGo
	clampImpl = clampGo
	minIdxImpl = minIdxGo
	maxIdxImpl = maxIdxGo
	addScaledImpl = addScaledGo
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

func sqrt32(dst, a []float32) {
	sqrtImpl(dst, a)
}

func reciprocal32(dst, a []float32) {
	reciprocalImpl(dst, a)
}

func minIdx32(a []float32) int {
	return minIdxImpl(a)
}

func maxIdx32(a []float32) int {
	return maxIdxImpl(a)
}

func addScaled32(dst []float32, alpha float32, s []float32) {
	addScaledImpl(dst, alpha, s)
}

func cumulativeSum32(dst, a []float32) {
	// CumulativeSum is inherently sequential
	cumulativeSum32Go(dst, a)
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
	// AccumulateAdd is dst += src, which is the same as add(dst, dst, src)
	addImpl(dst, dst, src)
}

func interleave2_32(dst, a, b []float32) {
	// Need at least 8 pairs for SIMD to be worthwhile (AVX processes 8 at a time)
	if len(a) >= minAVXElements {
		interleave2AVX(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2_32(a, b, src []float32) {
	if len(a) >= minAVXElements {
		deinterleave2AVX(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

func convolveValidMulti32(dsts [][]float32, signal []float32, kernels [][]float32, n, _ int) {
	// Kernel-major loop order: each kernel stays hot in cache for entire signal pass
	for k, kernel := range kernels {
		convolveValid32(dsts[k][:n], signal, kernel)
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

//go:noescape
func clampScaleAVX(dst, src []float32, minVal, maxVal, scale float32)

//go:noescape
func sqrtAVX(dst, a []float32)

//go:noescape
func reciprocalAVX(dst, a []float32)

//go:noescape
func addScaledAVX(dst []float32, alpha float32, s []float32)

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

//go:noescape
func sqrtAVX512(dst, a []float32)

//go:noescape
func reciprocalAVX512(dst, a []float32)

//go:noescape
func addScaledAVX512(dst []float32, alpha float32, s []float32)

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

//go:noescape
func sqrtSSE(dst, a []float32)

//go:noescape
func reciprocalSSE(dst, a []float32)

//go:noescape
func addScaledSSE(dst []float32, alpha float32, s []float32)

// Interleave/Deinterleave assembly function declarations
//
//go:noescape
func interleave2AVX(dst, a, b []float32)

//go:noescape
func deinterleave2AVX(a, b, src []float32)

func variance32(a []float32, mean float32) float32 {
	return variance32Go(a, mean)
}

func euclideanDistance32(a, b []float32) float32 {
	return euclideanDistance32Go(a, b)
}

func cubicInterpDot32(hist, a, b, c, d []float32, x float32) float32 {
	// Use AVX+FMA if available and have enough elements
	if cpu.X86.AVX && cpu.X86.FMA && len(hist) >= minAVXElements {
		return cubicInterpDotAVX(hist, a, b, c, d, x)
	}
	return cubicInterpDotGo(hist, a, b, c, d, x)
}

// CubicInterpDot assembly function declaration
//
//go:noescape
func cubicInterpDotAVX(hist, a, b, c, d []float32, x float32) float32

func sigmoid32(dst, src []float32) {
	// Use AVX+FMA if available and have enough elements on both slices
	if cpu.X86.AVX && cpu.X86.FMA && len(dst) >= minAVXElements && len(src) >= minAVXElements {
		sigmoidAVX(dst, src)
		return
	}
	sigmoid32Go(dst, src)
}

// Sigmoid assembly function declaration
//
//go:noescape
func sigmoidAVX(dst, src []float32)

func relu32(dst, src []float32) {
	if cpu.X86.AVX && len(dst) >= minAVXElements && len(src) >= minAVXElements {
		reluAVX(dst, src)
		return
	}
	relu32Go(dst, src)
}

//go:noescape
func reluAVX(dst, src []float32)

func clampScale32(dst, src []float32, minVal, maxVal, scale float32) {
	if cpu.X86.AVX && len(dst) >= minAVXElements && len(src) >= minAVXElements {
		clampScaleAVX(dst, src, minVal, maxVal, scale)
		return
	}
	clampScale32Go(dst, src, minVal, maxVal, scale)
}

func tanh32(dst, src []float32) {
	if cpu.X86.AVX && len(dst) >= minAVXElements && len(src) >= minAVXElements {
		tanhAVX(dst, src)
		return
	}
	tanh32Go(dst, src)
}

//go:noescape
func tanhAVX(dst, src []float32)

func exp32(dst, src []float32) {
	// Exp is complex, use Go implementation with math.Exp for now
	// Can be optimized with AVX polynomial approximation later
	exp32Go(dst, src)
}

func int32ToFloat32Scale(dst []float32, src []int32, scale float32) {
	// Use AVX if available and have enough elements
	if cpu.X86.AVX && len(dst) >= minAVXElements {
		int32ToFloat32ScaleAVX(dst, src, scale)
		return
	}
	int32ToFloat32ScaleGo(dst, src, scale)
}

//go:noescape
func int32ToFloat32ScaleAVX(dst []float32, src []int32, scale float32)

// ============================================================================
// SPLIT-FORMAT COMPLEX OPERATIONS
// ============================================================================

func mulComplex32(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	// Use AVX+FMA if available and have enough elements
	if cpu.X86.AVX && cpu.X86.FMA && len(dstRe) >= minAVXElements {
		mulComplexAVX(dstRe, dstIm, aRe, aIm, bRe, bIm)
		return
	}
	mulComplex32Go(dstRe, dstIm, aRe, aIm, bRe, bIm)
}

func mulConjComplex32(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	// Use AVX+FMA if available and have enough elements
	if cpu.X86.AVX && cpu.X86.FMA && len(dstRe) >= minAVXElements {
		mulConjComplexAVX(dstRe, dstIm, aRe, aIm, bRe, bIm)
		return
	}
	mulConjComplex32Go(dstRe, dstIm, aRe, aIm, bRe, bIm)
}

func absSqComplex32(dst, aRe, aIm []float32) {
	// Use AVX+FMA if available and have enough elements
	if cpu.X86.AVX && cpu.X86.FMA && len(dst) >= minAVXElements {
		absSqComplexAVX(dst, aRe, aIm)
		return
	}
	absSqComplex32Go(dst, aRe, aIm)
}

// Split-format complex assembly function declarations
//
//go:noescape
func mulComplexAVX(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)

//go:noescape
func mulConjComplexAVX(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)

//go:noescape
func absSqComplexAVX(dst, aRe, aIm []float32)
