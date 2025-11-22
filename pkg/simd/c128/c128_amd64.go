//go:build amd64

package c128

import "github.com/tphakala/simd/pkg/simd/cpu"

// Minimum number of complex128 elements required for SIMD operations.
// AVX processes 2 complex128 values per 256-bit register (4 float64).
// AVX-512 processes 4 complex128 values per 512-bit register (8 float64).
const (
	minAVXElements    = 2
	minAVX512Elements = 4
)

// Function pointer types for SIMD operations
type (
	binaryOpFunc func(dst, a, b []complex128)
	scaleFunc    func(dst, a []complex128, s complex128)
)

// Function pointers - assigned at init time based on CPU features
var (
	mulImpl     binaryOpFunc
	mulConjImpl binaryOpFunc
	scaleImpl   scaleFunc
	addImpl     binaryOpFunc
	subImpl     binaryOpFunc
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
	mulImpl = mulAVX512
	mulConjImpl = mulConjAVX512
	scaleImpl = scaleAVX512
	addImpl = addAVX512
	subImpl = subAVX512
}

func initAVX() {
	mulImpl = mulAVX
	mulConjImpl = mulConjAVX
	scaleImpl = scaleAVX
	addImpl = addAVX
	subImpl = subAVX
}

func initSSE2() {
	mulImpl = mulSSE2
	mulConjImpl = mulConjSSE2
	scaleImpl = scaleSSE2
	addImpl = addSSE2
	subImpl = subSSE2
}

func initGo() {
	mulImpl = mulGo
	mulConjImpl = mulConjGo
	scaleImpl = scaleGo
	addImpl = addGo
	subImpl = subGo
}

// Dispatch functions - call function pointers (zero overhead after init)

func mul128(dst, a, b []complex128) {
	mulImpl(dst, a, b)
}

func mulConj128(dst, a, b []complex128) {
	mulConjImpl(dst, a, b)
}

func scale128(dst, a []complex128, s complex128) {
	scaleImpl(dst, a, s)
}

func add128(dst, a, b []complex128) {
	addImpl(dst, a, b)
}

func sub128(dst, a, b []complex128) {
	subImpl(dst, a, b)
}

// AVX+FMA assembly function declarations (2x complex128 per iteration)
//
//go:noescape
func mulAVX(dst, a, b []complex128)

//go:noescape
func mulConjAVX(dst, a, b []complex128)

//go:noescape
func scaleAVX(dst, a []complex128, s complex128)

//go:noescape
func addAVX(dst, a, b []complex128)

//go:noescape
func subAVX(dst, a, b []complex128)

// AVX-512 assembly function declarations (4x complex128 per iteration)
//
//go:noescape
func mulAVX512(dst, a, b []complex128)

//go:noescape
func mulConjAVX512(dst, a, b []complex128)

//go:noescape
func scaleAVX512(dst, a []complex128, s complex128)

//go:noescape
func addAVX512(dst, a, b []complex128)

//go:noescape
func subAVX512(dst, a, b []complex128)

// SSE2 assembly function declarations (1x complex128 per iteration)
//
//go:noescape
func mulSSE2(dst, a, b []complex128)

//go:noescape
func mulConjSSE2(dst, a, b []complex128)

//go:noescape
func scaleSSE2(dst, a []complex128, s complex128)

//go:noescape
func addSSE2(dst, a, b []complex128)

//go:noescape
func subSSE2(dst, a, b []complex128)
