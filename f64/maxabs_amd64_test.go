//go:build amd64

package f64

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestMaxAbsKernelsAMD64 exercises each amd64 MaxAbs kernel directly (bypassing
// dispatch), so SSE2 is covered even on an AVX2 host. Each kernel does a
// full-width initial load, so only lengths >= the kernel stride are passed.
func TestMaxAbsKernelsAMD64(t *testing.T) {
	mk := func(n int) []float64 {
		a := make([]float64, n)
		for i := range a {
			a[i] = math.Cos(float64(i)*0.37) * float64((i%7)-3) * 9.5
		}
		return a
	}

	t.Run("SSE2", func(t *testing.T) {
		for _, n := range []int{2, 3, 4, 5, 8, 9, 16, 17, 31, 64, 127} {
			a := mk(n)
			if got, want := maxAbsSSE2(a), maxAbsGo(a); got != want {
				t.Errorf("n=%d: maxAbsSSE2=%v want %v", n, got, want)
			}
		}
	})

	t.Run("AVX", func(t *testing.T) {
		if !cpu.X86.AVX {
			t.Skip("AVX not supported on this CPU")
		}
		for _, n := range []int{4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 33, 64, 127} {
			a := mk(n)
			if got, want := maxAbsAVX(a), maxAbsGo(a); got != want {
				t.Errorf("n=%d: maxAbsAVX=%v want %v", n, got, want)
			}
		}
	})
}
