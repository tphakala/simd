//go:build amd64

package f32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestMaxAbsKernelsAMD64 exercises each amd64 MaxAbs kernel directly (bypassing
// dispatch), so SSE is covered even on an AVX2 host. Each kernel does a
// full-width initial load, so only lengths >= the kernel stride are passed
// (SSE: 4 lanes, AVX: 8 lanes).
func TestMaxAbsKernelsAMD64(t *testing.T) {
	mk := func(n int) []float32 {
		a := make([]float32, n)
		for i := range a {
			a[i] = float32(math.Cos(float64(i)*0.37)) * float32((i%7)-3) * 9.5
		}
		return a
	}

	t.Run("SSE", func(t *testing.T) {
		for _, n := range []int{4, 5, 6, 7, 8, 9, 16, 17, 31, 64, 127} {
			a := mk(n)
			if got, want := maxAbsSSE(a), maxAbsGo(a); got != want {
				t.Errorf("n=%d: maxAbsSSE=%v want %v", n, got, want)
			}
		}
	})

	t.Run("AVX", func(t *testing.T) {
		if !cpu.X86.AVX {
			t.Skip("AVX not supported on this CPU")
		}
		for _, n := range []int{8, 9, 10, 15, 16, 17, 31, 33, 64, 127, 256} {
			a := mk(n)
			if got, want := maxAbsAVX(a), maxAbsGo(a); got != want {
				t.Errorf("n=%d: maxAbsAVX=%v want %v", n, got, want)
			}
		}
	})
}
