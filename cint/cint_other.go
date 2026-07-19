//go:build !amd64 && !arm64

package cint

// Fallback dispatch for architectures without a SIMD kernel: every operation
// routes to the pure-Go reference.

func addCint(dst, a, b []int32)          { addGo(dst, a, b) }
func subCint(dst, a, b []int32)          { subGo(dst, a, b) }
func mulByScalarCint(a []int32, s int16) { mulByScalarGo(a, s) }
func mulCint(dst, a []int32, tw []int16) { mulGo(dst, a, tw) }
func mulConjCint(dst, a []int32, tw []int16) {
	mulConjGo(dst, a, tw)
}
