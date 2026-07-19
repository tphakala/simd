//go:build amd64

package cint

import "github.com/tphakala/simd/cpu"

// hasAVX2 gates the SIMD kernels. Every kernel does integer ALU work on 256-bit
// lanes (VPADDD / VPSUBD / VPMULDQ / VPSRLQ / VPSLLQ / VPBLENDD / VPMOVSXWD),
// which requires AVX2, so this checks the CPU feature explicitly rather than
// relying on length alone and falls back to the pure-Go reference otherwise.
//
// Dispatch is a direct CPU-flag switch, not an init-time function-pointer table:
// a pointer table defeats //go:noescape and forces the slice arguments to the
// heap, so every call would allocate. The explicit branch keeps the operations
// zero-allocation.
var hasAVX2 = cpu.X86.AVX2

// Tier thresholds: the minimum lane count at which the AVX2 kernel is worth its
// setup over the scalar loop, one 8-lane (256-bit) block each. Add, Sub and
// MulByScalar step 8 int32 per block; Mul and MulConj step 8 int32 (4 complex)
// per block, so all five gate at 8. Every kernel is correct at any length (each
// falls through to a scalar tail), so these are performance cuts, never a safety
// requirement. They are independent literals so retuning one cannot move another.
const (
	minAVX2Add         = 8
	minAVX2Sub         = 8
	minAVX2MulByScalar = 8
	minAVX2Mul         = 8
	minAVX2MulConj     = 8
)

func addCint(dst, a, b []int32) {
	if hasAVX2 && len(dst) >= minAVX2Add {
		addAVX2(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func subCint(dst, a, b []int32) {
	if hasAVX2 && len(dst) >= minAVX2Sub {
		subAVX2(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

func mulByScalarCint(a []int32, s int16) {
	if hasAVX2 && len(a) >= minAVX2MulByScalar {
		mulByScalarAVX2(a, s)
		return
	}
	mulByScalarGo(a, s)
}

func mulCint(dst, a []int32, tw []int16) {
	if hasAVX2 && len(dst) >= minAVX2Mul {
		mulAVX2(dst, a, tw)
		return
	}
	mulGo(dst, a, tw)
}

func mulConjCint(dst, a []int32, tw []int16) {
	if hasAVX2 && len(dst) >= minAVX2MulConj {
		mulConjAVX2(dst, a, tw)
		return
	}
	mulConjGo(dst, a, tw)
}

//go:noescape
func addAVX2(dst, a, b []int32)

//go:noescape
func subAVX2(dst, a, b []int32)

//go:noescape
func mulByScalarAVX2(a []int32, s int16)

//go:noescape
func mulAVX2(dst, a []int32, tw []int16)

//go:noescape
func mulConjAVX2(dst, a []int32, tw []int16)
