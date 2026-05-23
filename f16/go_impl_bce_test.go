package f16

import "testing"

// TestGoFallbacks_EmptyDst exercises the `if len(dst) == 0 { return }` BCE
// guard branches in the Go fallbacks. On NEON/AVX hosts those branches are
// not normally hit because the public API short-circuits before dispatch.
func TestGoFallbacks_EmptyDst(_ *testing.T) {
	src := []Float16{FromFloat32(1), FromFloat32(2), FromFloat32(3)}
	src32 := []float32{1, 2, 3}
	dst := []Float16{}
	dst32 := []float32{}

	addGo(dst, src, src)
	subGo(dst, src, src)
	mulGo(dst, src, src)
	divGo(dst, src, src)
	scaleGo(dst, src, FromFloat32(2))
	addScalarGo(dst, src, FromFloat32(1))
	absGo(dst, src)
	negGo(dst, src)
	fmaGo(dst, src, src, src)
	clampGo(dst, src, FromFloat32(0), FromFloat32(1))
	sqrtGo(dst, src)
	reciprocalGo(dst, src)
	reluGo(dst, src)
	sigmoidGo(dst, src)
	expGo(dst, src)
	tanhGo(dst, src)
	toFloat32SliceGo(dst32, src)
	fromFloat32SliceGo(dst, src32)
	dotProductGo(nil, src)
	dotProductGo(src, nil)
}
