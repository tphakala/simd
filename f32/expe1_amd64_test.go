//go:build amd64

package f32

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// expE1AVX2Func adapts expE1AVX2 to expE1Func. The kernel needs len(dst) >= 8 and
// len(src) >= len(dst), so shorter inputs go to the Go fallback, as the
// dispatcher does.
func expE1AVX2Func(dst, src []float32) {
	n := min(len(dst), len(src))
	if n < minAVXElements {
		expE1Go(dst[:n], src[:n])
		return
	}
	expE1AVX2(dst[:n], src[:n])
}

// TestExpE1AVX2Direct runs the suite against the AVX2+FMA kernel itself. The
// public path picks its kernel from cpu.X86 flags directly, so the tier-forcing
// helper does not steer it; this test is what keeps the kernel covered when
// SIMD_DISABLE sends the public path to Go. It skips without AVX2 and FMA so the
// legs below that level never execute a VEX.256 integer op or a VFMADD.
func TestExpE1AVX2Direct(t *testing.T) {
	if !cpu.X86.AVX2 || !cpu.X86.FMA {
		t.Skip("AVX2 and FMA not both available")
	}
	runExpE1Suite(t, expE1AVX2Func)
}
