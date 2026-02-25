//go:build amd64

package f64

import (
	"fmt"
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

func TestCubicInterpDotAVX_Direct(t *testing.T) {
	if !cpu.X86.AVX || !cpu.X86.FMA {
		t.Skip("AVX+FMA not supported on this CPU")
	}

	sizes := []int{4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 64, 241, 1000}
	for _, n := range sizes {
		hist := make([]float64, n)
		a := make([]float64, n)
		b := make([]float64, n)
		c := make([]float64, n)
		d := make([]float64, n)

		for i := range n {
			fi := float64(i)
			hist[i] = math.Sin(fi*0.17) + 0.01*fi
			a[i] = math.Cos(fi*0.11) * 0.7
			b[i] = math.Sin(fi*0.07) * 0.5
			c[i] = math.Cos(fi*0.03) * 0.25
			d[i] = math.Sin(fi*0.19) * 0.125
		}

		x := 0.73125
		got := cubicInterpDotAVX(hist, a, b, c, d, x)
		want := cubicInterpDotRef64(hist, a, b, c, d, x)

		tol := math.Abs(want)*1e-10 + 1e-12
		if math.Abs(got-want) > tol {
			t.Fatalf("n=%d: cubicInterpDotAVX=%v, want=%v, diff=%v", n, got, want, got-want)
		}
	}
}

func BenchmarkCubicInterpDotAVX_Direct(b *testing.B) {
	if !cpu.X86.AVX || !cpu.X86.FMA {
		b.Skip("AVX+FMA not supported on this CPU")
	}

	for _, n := range []int{64, 241, 1000} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			hist := make([]float64, n)
			a := make([]float64, n)
			bb := make([]float64, n)
			c := make([]float64, n)
			d := make([]float64, n)

			for i := range n {
				fi := float64(i)
				hist[i] = math.Sin(fi*0.17) + 0.01*fi
				a[i] = math.Cos(fi*0.11) * 0.7
				bb[i] = math.Sin(fi*0.07) * 0.5
				c[i] = math.Cos(fi*0.03) * 0.25
				d[i] = math.Sin(fi*0.19) * 0.125
			}

			x := 0.73125
			b.SetBytes(int64(n * 8 * 5))
			b.ResetTimer()

			var result float64
			for i := 0; i < b.N; i++ {
				result = cubicInterpDotAVX(hist, a, bb, c, d, x)
			}
			sink64 = result
		})
	}
}
