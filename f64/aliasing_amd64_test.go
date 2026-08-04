//go:build amd64

package f64

import (
	"testing"

	"github.com/tphakala/simd/cpu"
	"github.com/tphakala/simd/internal/aliastest"
)

// forTiers runs the aliasing sweep under every forceable amd64 tier, including
// the AVX-without-FMA tier (#201) that never runs on FMA-capable CI. The list is
// in descending priority so aliastest.ForTiers re-binds the host default on
// cleanup. Forcing a tier only redirects the function-pointer-dispatched ops; the
// transcendentals select inline from cpu.X86 and length, so they are not forced
// per tier (they have no distinct SSE kernel, so the size sweep covers their Go
// and native-SIMD kernels; see the note in aliasing_test.go).
func forTiers(t *testing.T, run func(t *testing.T)) {
	t.Helper()
	aliastest.ForTiers(t, []aliastest.Tier{
		{Name: "AVX512", Bind: initAVX512, Supported: cpu.X86.AVX512F && cpu.X86.AVX512VL},
		{Name: "AVX", Bind: initAVX, Supported: cpu.X86.AVX && cpu.X86.FMA},
		{Name: "AVXNoFMA", Bind: initAVXNoFMA, Supported: cpu.X86.AVX},
		{Name: "SSE2", Bind: initSSE2, Supported: cpu.X86.SSE2},
		{Name: "Go", Bind: initGo, Supported: true},
	}, run)
}
