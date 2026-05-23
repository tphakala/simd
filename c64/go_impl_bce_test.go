package c64

import "testing"

// TestGoFallbacks_EmptyDst exercises the `if len(dst) == 0 { return }` BCE
// guard branches in the Go fallbacks. On AVX-capable hosts those branches
// are not normally hit because the public API short-circuits before dispatch.
func TestGoFallbacks_EmptyDst(_ *testing.T) {
	src := []complex64{1 + 2i, 3 + 4i, 5 + 6i}
	srcReal := []float32{1, 2, 3}
	dst := []complex64{}
	dstReal := []float32{}

	mulGo(dst, src, src)
	mulConjGo(dst, src, src)
	scaleGo(dst, src, 1+1i)
	addGo(dst, src, src)
	subGo(dst, src, src)
	absGo(dstReal, src)
	absSqGo(dstReal, src)
	conjGo(dst, src)
	fromRealGo(dst, srcReal)
}
