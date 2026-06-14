//go:build amd64

package f64

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// refConvMaxAbsDot computes max|valid-conv output| using the given per-window dot
// product, so each amd64 fused kernel can be checked for bit-exactness against the
// exact dotProduct kernel whose accumulation it replicates.
func refConvMaxAbsDot(signal, kernel []float64, dot func(a, b []float64) float64) float64 {
	kLen := len(kernel)
	validLen := len(signal) - kLen + 1
	var m float64
	for i := range validLen {
		if v := math.Abs(dot(signal[i:i+kLen], kernel)); v > m {
			m = v
		}
	}
	return m
}

// TestConvolveValidMaxAbsKernelsAMD64 exercises each fused kernel directly
// (bypassing dispatch) so SSE2 is covered even on an AVX2 host, and asserts each
// is bit-identical to max|its matching per-window dotProduct|.
func TestConvolveValidMaxAbsKernelsAMD64(t *testing.T) {
	mkPair := func(sl, kl int) ([]float64, []float64) {
		s := make([]float64, sl)
		k := make([]float64, kl)
		for i := range s {
			s[i] = math.Sin(float64(i)*0.31) - 0.3*float64(i%4)
		}
		for i := range k {
			k[i] = 0.4 - float64(i)*0.07
		}
		return s, k
	}
	type cfg struct{ sl, kl int }
	cfgs := []cfg{{4, 1}, {8, 2}, {16, 3}, {17, 4}, {33, 5}, {64, 8}, {128, 16}, {200, 17}, {256, 32}}

	t.Run("SSE2", func(t *testing.T) {
		for _, c := range cfgs {
			s, k := mkPair(c.sl, c.kl)
			got := convolveValidMaxAbsSSE2(s, k)
			want := refConvMaxAbsDot(s, k, dotProductSSE2)
			if got != want {
				t.Errorf("sl=%d kl=%d: SSE2=%v want %v", c.sl, c.kl, got, want)
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
		if n := testing.AllocsPerRun(50, func() { _ = convolveValidMaxAbsSSE2(s, k) }); n != 0 {
			t.Errorf("convolveValidMaxAbsSSE2 allocated %v, want 0", n)
		}
		if cpu.X86.AVX && cpu.X86.FMA {
			if n := testing.AllocsPerRun(50, func() { _ = convolveValidMaxAbsAVX(s, k) }); n != 0 {
				t.Errorf("convolveValidMaxAbsAVX allocated %v, want 0", n)
			}
		}
	})
}
