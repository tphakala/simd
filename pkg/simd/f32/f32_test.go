package f32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/pkg/simd/cpu"
)

func TestDotProduct(t *testing.T) {
	t.Logf("CPU: %s", cpu.Info())

	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"empty", nil, nil, 0},
		{"single", []float32{2}, []float32{3}, 6},
		{"two", []float32{1, 2}, []float32{3, 4}, 11},
		{"four", []float32{1, 2, 3, 4}, []float32{5, 6, 7, 8}, 70},
		{"eight", []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{8, 7, 6, 5, 4, 3, 2, 1}, 120},
		{"nine", []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, []float32{9, 8, 7, 6, 5, 4, 3, 2, 1}, 165},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DotProduct(tt.a, tt.b)
			if got != tt.want {
				t.Errorf("DotProduct() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAdd(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	b := []float32{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
	dst := make([]float32, len(a))

	Add(dst, a, b)

	for i := range dst {
		if dst[i] != 11 {
			t.Errorf("Add()[%d] = %v, want 11", i, dst[i])
		}
	}
}

func TestMul(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	b := []float32{2, 2, 2, 2, 2, 2, 2, 2, 2, 2}
	dst := make([]float32, len(a))

	Mul(dst, a, b)

	for i := range dst {
		want := float32(i+1) * 2
		if dst[i] != want {
			t.Errorf("Mul()[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestSum(t *testing.T) {
	a := make([]float32, 100)
	for i := range a {
		a[i] = float32(i + 1)
	}

	got := Sum(a)
	want := float32(5050)
	if got != want {
		t.Errorf("Sum() = %v, want %v", got, want)
	}
}

func TestMinMax(t *testing.T) {
	a := []float32{5, 2, 8, 1, 9, 3, 7, 4, 6, 10}

	if got := Min(a); got != 1 {
		t.Errorf("Min() = %v, want 1", got)
	}
	if got := Max(a); got != 10 {
		t.Errorf("Max() = %v, want 10", got)
	}
}

func TestAbs(t *testing.T) {
	a := []float32{-1, 2, -3, 4, -5, 6, -7, 8, -9, 10}
	dst := make([]float32, len(a))

	Abs(dst, a)

	for i := range dst {
		want := float32(math.Abs(float64(a[i])))
		if dst[i] != want {
			t.Errorf("Abs()[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestFMA(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float32{2, 2, 2, 2, 2, 2, 2, 2}
	c := []float32{1, 1, 1, 1, 1, 1, 1, 1}
	dst := make([]float32, len(a))

	FMA(dst, a, b, c)

	for i := range dst {
		want := a[i]*b[i] + c[i]
		if dst[i] != want {
			t.Errorf("FMA()[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestClamp(t *testing.T) {
	a := []float32{-5, -2, 0, 5, 10, 15, 20}
	dst := make([]float32, len(a))

	Clamp(dst, a, 0, 10)

	want := []float32{0, 0, 0, 5, 10, 10, 10}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("Clamp()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

// Benchmarks

func BenchmarkDotProduct_100(b *testing.B) {
	benchmarkDotProduct32(b, 100)
}

func BenchmarkDotProduct_1000(b *testing.B) {
	benchmarkDotProduct32(b, 1000)
}

func BenchmarkDotProduct_10000(b *testing.B) {
	benchmarkDotProduct32(b, 10000)
}

func benchmarkDotProduct32(b *testing.B, n int) {
	b.Helper()
	a := make([]float32, n)
	x := make([]float32, n)
	for i := range a {
		a[i] = float32(i)
		x[i] = float32(n - i)
	}

	b.ResetTimer()
	b.SetBytes(int64(n * 4 * 2))

	var result float32
	for i := 0; i < b.N; i++ {
		result = DotProduct(a, x)
	}
	_ = result
}

func BenchmarkAdd_1000(b *testing.B) {
	a := make([]float32, 1000)
	c := make([]float32, 1000)
	dst := make([]float32, 1000)

	b.SetBytes(1000 * 4 * 3)

	for b.Loop() {
		Add(dst, a, c)
	}
}

func BenchmarkMul_1000(b *testing.B) {
	a := make([]float32, 1000)
	c := make([]float32, 1000)
	dst := make([]float32, 1000)

	b.SetBytes(1000 * 4 * 3)

	for b.Loop() {
		Mul(dst, a, c)
	}
}

func BenchmarkFMA_1000(b *testing.B) {
	a := make([]float32, 1000)
	c := make([]float32, 1000)
	d := make([]float32, 1000)
	dst := make([]float32, 1000)

	b.SetBytes(1000 * 4 * 4)

	for b.Loop() {
		FMA(dst, a, c, d)
	}
}
