//go:build amd64

package c128

import "github.com/tphakala/simd/cpu"

// Minimum number of complex128 elements required for SIMD operations.
// AVX processes 2 complex128 values per 256-bit register (4 float64).
// AVX-512 processes 4 complex128 values per 512-bit register (8 float64).
const (
	minAVXElements    = 2
	minAVX512Elements = 4
)

// Function pointer types for SIMD operations
type (
	binaryOpFunc  func(dst, a, b []complex128)
	reduceFunc    func(a, b []complex128) complex128
	scaleFunc     func(dst, a []complex128, s complex128)
	unaryAbsFunc  func(dst []float64, a []complex128)
	unaryConjFunc func(dst, a []complex128)
	fromRealFunc  func(dst []complex128, src []float64)
)

// Function pointers - assigned at init time based on CPU features
var (
	mulImpl            binaryOpFunc
	mulConjImpl        binaryOpFunc
	dotProductImpl     reduceFunc
	dotProductConjImpl reduceFunc
	scaleImpl          scaleFunc
	addImpl            binaryOpFunc
	subImpl            binaryOpFunc
	absImpl            unaryAbsFunc
	absSqImpl          unaryAbsFunc
	conjImpl           unaryConjFunc
	fromRealImpl       fromRealFunc
)

func init() {
	selectImpl(
		cpu.X86.AVX512F && cpu.X86.AVX512VL,
		cpu.X86.AVX && cpu.X86.FMA,
		cpu.X86.AVX,
		cpu.X86.SSE2,
	)
}

// selectImpl assigns the operation implementations from CPU feature predicates.
// Priority: AVX-512 > AVX+FMA > AVX (no FMA) > SSE2 > Go.
// It is split out from init so the dispatch priority can be unit-tested on any
// host, including the AVX-without-FMA path that never runs on FMA-capable CI.
func selectImpl(avx512, avxFMA, avx, sse2 bool) {
	switch {
	case avx512:
		initAVX512()
	case avxFMA:
		initAVX()
	case avx:
		initAVXNoFMA()
	case sse2:
		initSSE2()
	default:
		initGo()
	}
}

func initAVX512() {
	mulImpl = mulAVX512
	mulConjImpl = mulConjAVX512
	// No AVX-512 dot kernels yet (no AVX-512 hardware to verify; see #75/#96):
	// reuse the AVX kernels so the tier still gets the speedup.
	dotProductImpl = dotProductAVX
	dotProductConjImpl = dotProductConjAVX
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
	dotProductImpl = dotProductAVX
	dotProductConjImpl = dotProductConjAVX
	scaleImpl = scaleAVX
	addImpl = addAVX
	subImpl = subAVX
	absImpl = absAVX
	absSqImpl = absSqAVX
	conjImpl = conjAVX
	fromRealImpl = fromRealAVX
}

// initAVXNoFMA runs on AVX-capable CPUs that lack FMA (rare but possible:
// some early Sandy/Ivy Bridge generations, certain Atom/Pentium SKUs).
// The mul/mulConj/scale AVX kernels depend on VFMADDSUB213PD, so they fall
// back to SSE2; the FMA-free AVX kernels (add/sub/abs/absSq/conj) stay on AVX.
func initAVXNoFMA() {
	mulImpl = mulSSE2
	mulConjImpl = mulConjSSE2
	scaleImpl = scaleSSE2
	// The dot kernels use VFMADDSUB213PD as well, so they follow mul/mulConj
	// down to SSE2 on AVX-without-FMA parts.
	dotProductImpl = dotProductSSE2
	dotProductConjImpl = dotProductConjSSE2
	addImpl = addAVX
	subImpl = subAVX
	absImpl = absAVX
	absSqImpl = absSqAVX
	conjImpl = conjAVX
	// FromReal is pure interleaving (no FMA), so the AVX kernel is safe here.
	fromRealImpl = fromRealAVX
}

func initSSE2() {
	mulImpl = mulSSE2
	mulConjImpl = mulConjSSE2
	scaleImpl = scaleSSE2
	dotProductImpl = dotProductSSE2
	dotProductConjImpl = dotProductConjSSE2
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
	dotProductImpl = dotProductGo
	dotProductConjImpl = dotProductConjGo
	scaleImpl = scaleGo
	addImpl = addGo
	subImpl = subGo
	absImpl = absGo
	absSqImpl = absSqGo
	conjImpl = conjGo
	fromRealImpl = fromRealGo
}

// Dispatch functions - call function pointers (zero overhead after init)

func mul128(dst, a, b []complex128) {
	mulImpl(dst, a, b)
}

func mulConj128(dst, a, b []complex128) {
	mulConjImpl(dst, a, b)
}

func dotProduct128(a, b []complex128) complex128 {
	return dotProductImpl(a, b)
}

func dotProductConj128(a, b []complex128) complex128 {
	return dotProductConjImpl(a, b)
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

func abs128(dst []float64, a []complex128) {
	absImpl(dst, a)
}

func absSq128(dst []float64, a []complex128) {
	absSqImpl(dst, a)
}

func conj128(dst, a []complex128) {
	conjImpl(dst, a)
}

func fromReal128(dst []complex128, src []float64) {
	fromRealImpl(dst, src)
}

// AVX+FMA assembly function declarations (2x complex128 per iteration)
//
//go:noescape
func mulAVX(dst, a, b []complex128)

//go:noescape
func mulConjAVX(dst, a, b []complex128)

//go:noescape
func dotProductAVX(a, b []complex128) complex128

//go:noescape
func dotProductConjAVX(a, b []complex128) complex128

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
func dotProductSSE2(a, b []complex128) complex128

//go:noescape
func dotProductConjSSE2(a, b []complex128) complex128

//go:noescape
func scaleSSE2(dst, a []complex128, s complex128)

//go:noescape
func addSSE2(dst, a, b []complex128)

//go:noescape
func subSSE2(dst, a, b []complex128)

// Abs assembly function declarations

//go:noescape
func absAVX512(dst []float64, a []complex128)

//go:noescape
func absAVX(dst []float64, a []complex128)

//go:noescape
func absSSE2(dst []float64, a []complex128)

// AbsSq assembly function declarations

//go:noescape
func absSqAVX512(dst []float64, a []complex128)

//go:noescape
func absSqAVX(dst []float64, a []complex128)

//go:noescape
func absSqSSE2(dst []float64, a []complex128)

// Conj assembly function declarations

//go:noescape
func conjAVX512(dst, a []complex128)

//go:noescape
func conjAVX(dst, a []complex128)

//go:noescape
func conjSSE2(dst, a []complex128)

// FromReal assembly function declarations

//go:noescape
func fromRealAVX512(dst []complex128, src []float64)

//go:noescape
func fromRealAVX(dst []complex128, src []float64)

//go:noescape
func fromRealSSE2(dst []complex128, src []float64)
