//go:build amd64

package f32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// refConvMaxAbsDot computes max|valid-conv output| using the given per-window dot
// product, so each amd64 fused kernel can be checked for bit-exactness against the
// exact dotProduct kernel whose accumulation it replicates.
func refConvMaxAbsDot(signal, kernel []float32, dot func(a, b []float32) float32) float32 {
	kLen := len(kernel)
	validLen := len(signal) - kLen + 1
	var m float32
	for i := range validLen {
		v := dot(signal[i:i+kLen], kernel)
		if v < 0 {
			v = -v
		}
		if v > m {
			m = v
		}
	}
	return m
}

// TestConvolveValidMaxAbsKernelsAMD64 exercises each fused kernel directly
// (bypassing dispatch) so SSE is covered even on an AVX2 host, and asserts each
// is bit-identical to max|its matching per-window dotProduct|.
func TestConvolveValidMaxAbsKernelsAMD64(t *testing.T) {
	mkPair := func(sl, kl int) ([]float32, []float32) {
		s := make([]float32, sl)
		k := make([]float32, kl)
		for i := range s {
			s[i] = float32(math.Sin(float64(i)*0.31)) - 0.3*float32(i%4)
		}
		for i := range k {
			k[i] = 0.4 - float32(i)*0.07
		}
		return s, k
	}
	type cfg struct{ sl, kl int }
	cfgs := []cfg{{4, 1}, {8, 2}, {16, 3}, {17, 4}, {33, 5}, {64, 8}, {128, 16}, {200, 17}, {256, 32}}

	t.Run("SSE", func(t *testing.T) {
		for _, c := range cfgs {
			s, k := mkPair(c.sl, c.kl)
			got := convolveValidMaxAbsSSE(s, k)
			want := refConvMaxAbsDot(s, k, dotProductSSE)
			if got != want {
				t.Errorf("sl=%d kl=%d: SSE=%v want %v", c.sl, c.kl, got, want)
			}
		}
	})

	t.Run("AVX", func(t *testing.T) {
		if !cpu.X86.AVX || !cpu.X86.FMA {
			t.Skip("AVX+FMA not supported on this CPU")
		}
		for _, c := range cfgs {
			s, k := mkPair(c.sl, c.kl)
			got := convolveValidMaxAbsAVX(s, k)
			want := refConvMaxAbsDot(s, k, dotProductAVX)
			if got != want {
				t.Errorf("sl=%d kl=%d: AVX=%v want %v", c.sl, c.kl, got, want)
			}
		}
	})

	t.Run("NoAlloc", func(t *testing.T) {
		s, k := mkPair(256, 32)
		if n := testing.AllocsPerRun(50, func() { _ = convolveValidMaxAbsSSE(s, k) }); n != 0 {
			t.Errorf("convolveValidMaxAbsSSE allocated %v, want 0", n)
		}
		if cpu.X86.AVX && cpu.X86.FMA {
			if n := testing.AllocsPerRun(50, func() { _ = convolveValidMaxAbsAVX(s, k) }); n != 0 {
				t.Errorf("convolveValidMaxAbsAVX allocated %v, want 0", n)
			}
		}
	})
}
