//go:build arm64

package f32

import "testing"

// expE1NEONFunc adapts expE1NEON to expE1Func. The kernel needs len(dst) >= 4 and
// len(src) >= len(dst), so shorter inputs go to the Go fallback, as the
// dispatcher does.
func expE1NEONFunc(dst, src []float32) {
	n := min(len(dst), len(src))
	if n < 4 {
		expE1Go(dst[:n], src[:n])
		return
	}
	expE1NEON(dst[:n], src[:n])
}

// TestExpE1NEONDirect runs the suite against the NEON kernel itself, so the
// kernel stays covered when SIMD_DISABLE sends the public path to Go.
func TestExpE1NEONDirect(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON not available")
	}
	runExpE1Suite(t, expE1NEONFunc)
}
