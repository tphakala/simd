package f32

import "testing"

// TestGoFallbacks_EmptyDst exercises the `if len(dst) == 0 { return }` BCE
// guard branches in the Go fallbacks. On AVX/NEON hosts those branches are
// not normally hit because the public API short-circuits before dispatch.
func TestGoFallbacks_EmptyDst(_ *testing.T) {
	src := []float32{1, 2, 3}
	dst := []float32{}

	addGo(dst, src, src)
	subGo(dst, src, src)
	mulGo(dst, src, src)
	divGo(dst, src, src)
	scaleGo(dst, src, 2)
	addScalarGo(dst, src, 1)
	absGo(dst, src)
	negGo(dst, src)
	fmaGo(dst, src, src, src)
	clampGo(dst, src, 0, 1)
	sqrt32Go(dst, src)
	reciprocal32Go(dst, src)
	sigmoid32Go(dst, src)
	relu32Go(dst, src)
	clampScale32Go(dst, src, 0, 1, 1)
	tanh32Go(dst, src)
	exp32Go(dst, src)
}
