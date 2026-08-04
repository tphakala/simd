//go:build amd64

package c64

import (
	"testing"

	"github.com/tphakala/simd/cpu"
	"github.com/tphakala/simd/internal/aliastest"
)

// forTiers runs the aliasing sweep under every forceable amd64 tier. The
// SSE-named tier actually uses SSE4.1 (BLENDPS); it is gated on SSE41
// accordingly. The list is in descending priority so aliastest.ForTiers re-binds
// the host default on cleanup.
func forTiers(t *testing.T, run func(t *testing.T)) {
	t.Helper()
	aliastest.ForTiers(t, []aliastest.Tier{
		{Name: "AVX512", Bind: initAVX512, Supported: cpu.X86.AVX512F && cpu.X86.AVX512VL},
		{Name: "AVX", Bind: initAVX, Supported: cpu.X86.AVX && cpu.X86.FMA},
		{Name: "SSE41", Bind: initSSE2, Supported: cpu.X86.SSE41},
		{Name: "Go", Bind: initGo, Supported: true},
	}, run)
}
