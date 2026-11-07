//go:build amd64

package f64

import (
	"math/rand"
	"slices"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// quantPow34Kernels lists the kernels, each called directly so the portable
// loop is tested next to the one the public API dispatches to.
func quantPow34Kernels() []quantPow34Kernel {
	return []quantPow34Kernel{{"go", quantizePow34BandsGo}, {"sse2", quantizePow34BandsSSE2}}
}

// TestQuantizePow34Bands_NoSSE2 checks that the public API still gives the
// reference result with cpu.X86.SSE2 cleared, the state SIMD_DISABLE=all
// leaves, which routes the wrapper to the Go path.
func TestQuantizePow34Bands_NoSSE2(t *testing.T) {
	saved := cpu.X86.SSE2
	cpu.X86.SSE2 = false
	t.Cleanup(func() { cpu.X86.SSE2 = saved })

	rng := rand.New(rand.NewSource(3402))
	for _, widths := range quantPow34Layouts() {
		src, scales := quantPow34RandomBands(rng, widths, 2, false)
		want := quantPow34Want(src, widths, scales, 0.4054, 8206)
		got := make([]int32, len(want))
		QuantizePow34Bands(got, src, widths, scales, 0.4054, 8206)
		if !slices.Equal(got, want) {
			t.Fatalf("widths %v: got %v, want %v", widths, got, want)
		}
	}
}
