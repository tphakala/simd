//go:build arm64

package cint

import "github.com/tphakala/simd/cpu"

// hasNEON gates the SIMD kernels. Every kernel does integer ALU work on NEON
// vectors, so it checks the CPU feature explicitly and falls back to the pure-Go
// reference otherwise.
//
// Dispatch is a direct CPU-flag switch, not an init-time function-pointer table:
// a pointer table defeats //go:noescape and forces the slice arguments to the
// heap, so every call would allocate. The explicit branch keeps the operations
// zero-allocation.
var hasNEON = cpu.ARM64.NEON

// Tier thresholds: one vector block each. Add, Sub and MulByScalar step 4 int32
// (one .4S register) per block, so they gate at 4. Mul and MulConj deinterleave 4
// complex (8 int32) per block via LD2/ST2, so they gate at 8. Every kernel is
// correct at any length (each falls through to a scalar tail), so these are
// performance cuts, never a safety requirement, and are independent literals so
// retuning one cannot move another.
const (
	minNEONAdd         = 4
	minNEONSub         = 4
	minNEONMulByScalar = 4
	minNEONMul         = 8
	minNEONMulConj     = 8
)

func addCint(dst, a, b []int32) {
	if hasNEON && len(dst) >= minNEONAdd {
		addNEON(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func subCint(dst, a, b []int32) {
	if hasNEON && len(dst) >= minNEONSub {
		subNEON(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

func mulByScalarCint(a []int32, s int16) {
	if hasNEON && len(a) >= minNEONMulByScalar {
		mulByScalarNEON(a, s)
		return
	}
	mulByScalarGo(a, s)
}

func mulCint(dst, a []int32, tw []int16) {
	if hasNEON && len(dst) >= minNEONMul {
		mulNEON(dst, a, tw)
		return
	}
	mulGo(dst, a, tw)
}

func mulConjCint(dst, a []int32, tw []int16) {
	if hasNEON && len(dst) >= minNEONMulConj {
		mulConjNEON(dst, a, tw)
		return
	}
	mulConjGo(dst, a, tw)
}

//go:noescape
func addNEON(dst, a, b []int32)

//go:noescape
func subNEON(dst, a, b []int32)

//go:noescape
func mulByScalarNEON(a []int32, s int16)

//go:noescape
func mulNEON(dst, a []int32, tw []int16)

//go:noescape
func mulConjNEON(dst, a []int32, tw []int16)
