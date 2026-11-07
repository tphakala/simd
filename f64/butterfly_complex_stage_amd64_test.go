//go:build amd64

package f64

import (
	"math/rand/v2"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestButterflyComplexStageKernels calls the amd64 stage kernels directly, so the
// SSE2 and AVX kernels are held to the oracle whatever the dispatcher picks. The
// dispatched sweep cannot do this: every path computes the same bits, so a
// dispatcher that routed everything to the Go fallback would leave it green. The
// shapes include ones below the dispatcher's blocks*span threshold, which only
// ever reach a kernel through this test.
func TestButterflyComplexStageKernels(t *testing.T) {
	kernels := []struct {
		name string
		ok   bool
		fn   func(re, im []float64, span, blocks int, twRe, twIm []float64)
	}{
		{"SSE2", cpu.X86.SSE2, butterflyComplexStageSSE2},
		{"AVX", cpu.X86.AVX, butterflyComplexStageAVX},
	}
	rng := rand.New(rand.NewPCG(0x6b3d, 0x21f4))
	for _, k := range kernels {
		t.Run(k.name, func(t *testing.T) {
			if !k.ok {
				t.Skip("CPU feature not available")
			}
			for _, span := range stageSpans {
				for _, blocks := range stageBlockCounts {
					re, im, twRe, twIm := stageRandData(rng, 2*span*blocks, span)
					reWant := append([]float64(nil), re...)
					imWant := append([]float64(nil), im...)
					butterflyComplexStageRef(reWant, imWant, span, twRe, twIm)

					k.fn(re, im, span, blocks, twRe, twIm)

					for i := range re {
						if !bitsEqF64(re[i], reWant[i]) || !bitsEqF64(im[i], imWant[i]) {
							t.Fatalf("span=%d blocks=%d: element %d = (%v, %v), want (%v, %v)",
								span, blocks, i, re[i], im[i], reWant[i], imWant[i])
						}
					}
				}
			}
		})
	}
}
