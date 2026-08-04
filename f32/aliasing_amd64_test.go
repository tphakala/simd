//go:build amd64

package f32

import (
	"testing"

	"github.com/tphakala/simd/cpu"
	"github.com/tphakala/simd/internal/aliastest"
)

// forTiers runs the aliasing sweep under every forceable amd64 tier. The list is
// in descending priority so aliastest.ForTiers re-binds the host default on
// cleanup. AVX-512 is exercised only where the host has it.
func forTiers(t *testing.T, run func(t *testing.T)) {
	t.Helper()
	aliastest.ForTiers(t, []aliastest.Tier{
		{Name: "AVX512", Bind: initAVX512, Supported: cpu.X86.AVX512F && cpu.X86.AVX512VL},
		{Name: "AVX", Bind: initAVX, Supported: cpu.X86.AVX && cpu.X86.FMA},
		{Name: "SSE", Bind: initSSE, Supported: cpu.X86.SSE2},
		{Name: "Go", Bind: initGo, Supported: true},
	}, run)
}

// TestAliasingDirectKernels exercises the exact-overlay property of the SSE and
// Go kernels of CopySign and AbsPow34 directly. Both ops dispatch inline on
// cpu.X86.AVX with no length guard (f32_amd64.go copySign32, absPow34_32), so on
// an AVX host the tier-forcing sweep only ever reaches their AVX kernel; their SSE
// and Go kernels are never covered by TestAliasingSweep. The package doc promises
// the overlay on the amd64 SSE path, so it is checked here directly. The AVX
// kernels are included for symmetry (also covered by the main sweep). CopySign is
// binary (dst, mag, sign); AbsPow34 is unary (dst, src).
func TestAliasingDirectKernels(t *testing.T) {
	for _, n := range aliastest.Sizes {
		aliastest.Binary(t, n, aliasEqF32, genF32, copySign32Go)
		aliastest.Unary(t, n, aliasEqF32, genF32, absPow34Go)
		if cpu.X86.SSE2 {
			aliastest.Binary(t, n, aliasEqF32, genF32, copySignSSE)
			aliastest.Unary(t, n, aliasEqF32, genF32, absPow34SSE)
		}
		if cpu.X86.AVX {
			aliastest.Binary(t, n, aliasEqF32, genF32, copySignAVX)
			aliastest.Unary(t, n, aliasEqF32, genF32, absPow34AVX)
		}
	}
}
