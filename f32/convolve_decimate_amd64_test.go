//go:build amd64

package f32

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestConvolveDecimate_Backends validates every amd64 fused kernel this CPU can
// execute by swapping the dispatch pointers, the same technique the existing
// init tests use. SSE2 and the Go fallback always run here; AVX is the default
// on this host; AVX-512 runs only on capable hardware/CI. Each backend is held
// to the bit-exact oracle (fused kernel == per-window DotProductUnsafe loop),
// which on amd64 means dotProduct uses the matching backend too.
func TestConvolveDecimate_Backends(t *testing.T) {
	savedConv := convolveDecimateImpl
	savedDot := dotProductImpl
	savedMin := minSIMDElements
	t.Cleanup(func() {
		convolveDecimateImpl = savedConv
		dotProductImpl = savedDot
		minSIMDElements = savedMin
	})

	backends := []struct {
		name string
		init func()
		ok   bool
	}{
		{"Go", initGo, true},
		{"SSE", initSSE, cpu.X86.SSE2},
		{"AVX", initAVX, cpu.X86.AVX && cpu.X86.FMA},
		{"AVX512", initAVX512, cpu.X86.AVX512F && cpu.X86.AVX512VL},
	}

	for _, be := range backends {
		t.Run(be.name, func(t *testing.T) {
			if !be.ok {
				t.Skipf("%s not supported on this CPU", be.name)
			}
			be.init()
			checkConvolveDecimateExact(t)
		})
	}
}
