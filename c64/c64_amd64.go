//go:build amd64

package c64

import "github.com/tphakala/simd/cpu"

// Function pointer types for SIMD operations
type (
	binaryOpFunc  func(dst, a, b []complex64)
	scaleFunc     func(dst, a []complex64, s complex64)
	unaryAbsFunc  func(dst []float32, a []complex64)
	unaryConjFunc func(dst, a []complex64)
	fromRealFunc  func(dst []complex64, src []float32)
)

// Function pointers - assigned at init time based on CPU features
var (
	mulImpl      binaryOpFunc
	mulConjImpl  binaryOpFunc
	scaleImpl    scaleFunc
	addImpl      binaryOpFunc
	subImpl      binaryOpFunc
	absImpl      unaryAbsFunc
	absSqImpl    unaryAbsFunc
	conjImpl     unaryConjFunc
	fromRealImpl fromRealFunc
)

func init() {
	// Select optimal implementation based on CPU features
	// Priority: AVX-512 > AVX+FMA > SSE4.1 > Go
	// Note: "SSE2" routines use BLENDPS which requires SSE4.1
	switch {
	case cpu.X86.AVX512F && cpu.X86.AVX512VL:
		initAVX512()
	case cpu.X86.AVX && cpu.X86.FMA:
		initAVX()
	case cpu.X86.SSE41:
		initSSE2() // Misnamed - actually uses SSE4.1 (BLENDPS)
	default:
		initGo()
	}
}

func initAVX512() {
	mulImpl = mulAVX512
	mulConjImpl = mulConjAVX512
	scaleImpl = scaleAVX512
	addImpl = addAVX512
	subImpl = subAVX512
	absImpl = absAVX512
	absSqImpl = absSqAVX512
	conjImpl = conjAVX512
	fromRealImpl = fromRealAVX512
}

func initAVX() {
	mulImpl = mulAVX
	mulConjImpl = mulConjAVX
	scaleImpl = scaleAVX
	addImpl = addAVX
	subImpl = subAVX
	absImpl = absAVX
	absSqImpl = absSqAVX
	conjImpl = conjAVX
	fromRealImpl = fromRealAVX
}

func initSSE2() {
	mulImpl = mulSSE2
	mulConjImpl = mulConjSSE2
	scaleImpl = scaleSSE2
	addImpl = addSSE2
	subImpl = subSSE2
	absImpl = absSSE2
	absSqImpl = absSqSSE2
	conjImpl = conjSSE2
	fromRealImpl = fromRealSSE2
}

func initGo() {
	mulImpl = mulGo
	mulConjImpl = mulConjGo
	scaleImpl = scaleGo
	addImpl = addGo
	subImpl = subGo
	absImpl = absGo
	absSqImpl = absSqGo
	conjImpl = conjGo
	fromRealImpl = fromRealGo
}

// Dispatch functions - call function pointers (zero overhead after init)

func mul64(dst, a, b []complex64) {
	mulImpl(dst, a, b)
}

func mulConj64(dst, a, b []complex64) {
	mulConjImpl(dst, a, b)
}

func scale64(dst, a []complex64, s complex64) {
	scaleImpl(dst, a, s)
}

func add64(dst, a, b []complex64) {
	addImpl(dst, a, b)
}

func sub64(dst, a, b []complex64) {
	subImpl(dst, a, b)
}

func abs64(dst []float32, a []complex64) {
	absImpl(dst, a)
}

func absSq64(dst []float32, a []complex64) {
	absSqImpl(dst, a)
}

func conj64(dst, a []complex64) {
	conjImpl(dst, a)
}

func fromReal64(dst []complex64, src []float32) {
	fromRealImpl(dst, src)
}

// AVX+FMA assembly function declarations (4x complex64 per iteration)
//
//go:noescape
func mulAVX(dst, a, b []complex64)

//go:noescape
func mulConjAVX(dst, a, b []complex64)

//go:noescape
func scaleAVX(dst, a []complex64, s complex64)

//go:noescape
func addAVX(dst, a, b []complex64)

//go:noescape
func subAVX(dst, a, b []complex64)

// AVX-512 assembly function declarations (8x complex64 per iteration)
//
//go:noescape
func mulAVX512(dst, a, b []complex64)

//go:noescape
func mulConjAVX512(dst, a, b []complex64)

//go:noescape
func scaleAVX512(dst, a []complex64, s complex64)

//go:noescape
func addAVX512(dst, a, b []complex64)

//go:noescape
func subAVX512(dst, a, b []complex64)

// SSE2 assembly function declarations (2x complex64 per iteration)
//
//go:noescape
func mulSSE2(dst, a, b []complex64)

//go:noescape
func mulConjSSE2(dst, a, b []complex64)

//go:noescape
func scaleSSE2(dst, a []complex64, s complex64)

//go:noescape
func addSSE2(dst, a, b []complex64)

//go:noescape
func subSSE2(dst, a, b []complex64)

// Abs assembly function declarations

//go:noescape
func absAVX512(dst []float32, a []complex64)

//go:noescape
func absAVX(dst []float32, a []complex64)

//go:noescape
func absSSE2(dst []float32, a []complex64)

// AbsSq assembly function declarations

//go:noescape
func absSqAVX512(dst []float32, a []complex64)

//go:noescape
func absSqAVX(dst []float32, a []complex64)

//go:noescape
func absSqSSE2(dst []float32, a []complex64)

// Conj assembly function declarations

//go:noescape
func conjAVX512(dst, a []complex64)

//go:noescape
func conjAVX(dst, a []complex64)

//go:noescape
func conjSSE2(dst, a []complex64)

// FromReal assembly function declarations

//go:noescape
func fromRealAVX512(dst []complex64, src []float32)

//go:noescape
func fromRealAVX(dst []complex64, src []float32)

//go:noescape
func fromRealSSE2(dst []complex64, src []float32)
