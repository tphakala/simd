package f64

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

func TestDotProduct(t *testing.T) {
	t.Logf("CPU: %s", cpu.Info())

	tests := []struct {
		name string
		a, b []float64
		want float64
	}{
		{"empty", nil, nil, 0},
		{"single", []float64{2}, []float64{3}, 6},
		{"two", []float64{1, 2}, []float64{3, 4}, 11},
		{"three", []float64{1, 2, 3}, []float64{4, 5, 6}, 32},
		{"four", []float64{1, 2, 3, 4}, []float64{5, 6, 7, 8}, 70},
		{"five", []float64{1, 2, 3, 4, 5}, []float64{6, 7, 8, 9, 10}, 130},
		{"eight", []float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{8, 7, 6, 5, 4, 3, 2, 1}, 120},
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

func TestDotProduct_Large(t *testing.T) {
	// Test with 277 elements (typical FIR filter size)
	n := 277
	a := make([]float64, n)
	b := make([]float64, n)
	for i := range n {
		a[i] = float64(i + 1)
		b[i] = 1.0
	}

	got := DotProduct(a, b)
	want := float64(n*(n+1)) / 2 // Sum 1..n = n*(n+1)/2
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("DotProduct(large) = %v, want %v", got, want)
	}
}

func TestAdd(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	b := []float64{8, 7, 6, 5, 4, 3, 2, 1}
	dst := make([]float64, len(a))

	Add(dst, a, b)

	for i := range dst {
		if dst[i] != 9 {
			t.Errorf("Add()[%d] = %v, want 9", i, dst[i])
		}
	}
}

func TestSub(t *testing.T) {
	a := []float64{10, 20, 30, 40, 50}
	b := []float64{1, 2, 3, 4, 5}
	dst := make([]float64, len(a))

	Sub(dst, a, b)

	want := []float64{9, 18, 27, 36, 45}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("Sub()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestMul(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	b := []float64{2, 3, 4, 5, 6}
	dst := make([]float64, len(a))

	Mul(dst, a, b)

	want := []float64{2, 6, 12, 20, 30}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("Mul()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestDiv(t *testing.T) {
	a := []float64{10, 20, 30, 40, 50}
	b := []float64{2, 4, 5, 8, 10}
	dst := make([]float64, len(a))

	Div(dst, a, b)

	want := []float64{5, 5, 6, 5, 5}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("Div()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestScale(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	dst := make([]float64, len(a))

	Scale(dst, a, 3.0)

	want := []float64{3, 6, 9, 12, 15}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("Scale()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestSum(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want float64
	}{
		{"empty", nil, 0},
		{"single", []float64{5}, 5},
		{"multi", []float64{1, 2, 3, 4, 5}, 15},
		{"large", make100(), 5050},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Sum(tt.a)
			if got != tt.want {
				t.Errorf("Sum() = %v, want %v", got, tt.want)
			}
		})
	}
}

func make100() []float64 {
	a := make([]float64, 100)
	for i := range a {
		a[i] = float64(i + 1)
	}
	return a
}

func TestMinMax(t *testing.T) {
	a := []float64{5, 2, 8, 1, 9, 3, 7, 4, 6}

	if got := Min(a); got != 1 {
		t.Errorf("Min() = %v, want 1", got)
	}
	if got := Max(a); got != 9 {
		t.Errorf("Max() = %v, want 9", got)
	}
}

func TestAbs(t *testing.T) {
	a := []float64{-1, 2, -3, 4, -5}
	dst := make([]float64, len(a))

	Abs(dst, a)

	want := []float64{1, 2, 3, 4, 5}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("Abs()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestNeg(t *testing.T) {
	a := []float64{1, -2, 3, -4, 5}
	dst := make([]float64, len(a))

	Neg(dst, a)

	want := []float64{-1, 2, -3, 4, -5}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("Neg()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestFMA(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	b := []float64{2, 3, 4, 5, 6}
	c := []float64{1, 1, 1, 1, 1}
	dst := make([]float64, len(a))

	FMA(dst, a, b, c)

	// dst[i] = a[i]*b[i] + c[i]
	want := []float64{3, 7, 13, 21, 31}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("FMA()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestClamp(t *testing.T) {
	a := []float64{-5, 0, 5, 10, 15}
	dst := make([]float64, len(a))

	Clamp(dst, a, 0, 10)

	want := []float64{0, 0, 5, 10, 10}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("Clamp()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestAccumulateAdd(t *testing.T) {
	tests := []struct {
		name   string
		dst    []float64
		src    []float64
		offset int
		want   []float64
	}{
		{
			"basic",
			[]float64{1, 2, 3, 4, 5},
			[]float64{10, 20},
			1,
			[]float64{1, 12, 23, 4, 5},
		},
		{
			"at start",
			[]float64{1, 2, 3, 4},
			[]float64{10, 20, 30},
			0,
			[]float64{11, 22, 33, 4},
		},
		{
			"at end",
			[]float64{1, 2, 3, 4},
			[]float64{10, 20},
			2,
			[]float64{1, 2, 13, 24},
		},
		{
			"empty src",
			[]float64{1, 2, 3},
			[]float64{},
			0,
			[]float64{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.dst))
			copy(dst, tt.dst)
			AccumulateAdd(dst, tt.src, tt.offset)
			for i := range dst {
				if dst[i] != tt.want[i] {
					t.Errorf("AccumulateAdd()[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestAccumulateAdd_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for negative offset")
		}
	}()
	dst := make([]float64, 5)
	src := []float64{1, 2}
	AccumulateAdd(dst, src, -1)
}

func TestAccumulateAdd_PanicsOverflow(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for overflow")
		}
	}()
	dst := make([]float64, 5)
	src := []float64{1, 2, 3}
	AccumulateAdd(dst, src, 4) // 4+3 > 5
}

// Benchmarks

func BenchmarkDotProduct_20(b *testing.B) {
	benchmarkDotProduct(b, 20)
}

func BenchmarkDotProduct_80(b *testing.B) {
	benchmarkDotProduct(b, 80)
}

func BenchmarkDotProduct_277(b *testing.B) {
	benchmarkDotProduct(b, 277)
}

func BenchmarkDotProduct_1000(b *testing.B) {
	benchmarkDotProduct(b, 1000)
}

func benchmarkDotProduct(b *testing.B, n int) {
	b.Helper()
	a := make([]float64, n)
	x := make([]float64, n)
	for i := range a {
		a[i] = float64(i)
		x[i] = float64(n - i)
	}

	b.ResetTimer()
	b.SetBytes(int64(n * 8 * 2)) // 2 slices of float64

	var result float64
	for i := 0; i < b.N; i++ {
		result = DotProduct(a, x)
	}
	_ = result
}

func BenchmarkAdd_1000(b *testing.B) {
	a := make([]float64, 1000)
	c := make([]float64, 1000)
	dst := make([]float64, 1000)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i)
	}

	b.SetBytes(1000 * 8 * 3)

	for b.Loop() {
		Add(dst, a, c)
	}
}

func BenchmarkMul_1000(b *testing.B) {
	a := make([]float64, 1000)
	c := make([]float64, 1000)
	dst := make([]float64, 1000)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i)
	}

	b.SetBytes(1000 * 8 * 3)

	for b.Loop() {
		Mul(dst, a, c)
	}
}

func BenchmarkSum_1000(b *testing.B) {
	a := make([]float64, 1000)
	for i := range a {
		a[i] = float64(i)
	}

	b.SetBytes(1000 * 8)

	var result float64
	for b.Loop() {
		result = Sum(a)
	}
	_ = result
}

func BenchmarkFMA_1000(b *testing.B) {
	a := make([]float64, 1000)
	c := make([]float64, 1000)
	d := make([]float64, 1000)
	dst := make([]float64, 1000)
	for i := range a {
		a[i] = float64(i)
		c[i] = float64(i)
		d[i] = float64(i)
	}

	b.SetBytes(1000 * 8 * 4)

	for b.Loop() {
		FMA(dst, a, c, d)
	}
}

func BenchmarkAccumulateAdd_1000(b *testing.B) {
	dst := make([]float64, 1000)
	src := make([]float64, 500)
	for i := range src {
		src[i] = float64(i)
	}

	b.SetBytes(500 * 8 * 2) // read src, read+write dst

	for b.Loop() {
		// Reset dst for consistent benchmark
		for i := range dst {
			dst[i] = float64(i)
		}
		AccumulateAdd(dst, src, 250)
	}
}

// Tests for MinIdx and MaxIdx

func TestMinIdx(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want int
	}{
		{"empty", nil, -1},
		{"single", []float64{5}, 0},
		{"min at start", []float64{1, 2, 3, 4, 5}, 0},
		{"min at end", []float64{5, 4, 3, 2, 1}, 4},
		{"min in middle", []float64{5, 2, 8, 1, 9, 3}, 3},
		{"duplicates", []float64{3, 1, 4, 1, 5}, 1}, // First occurrence
		{"negative", []float64{-1, -5, -3, -2}, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MinIdx(tt.a)
			if got != tt.want {
				t.Errorf("MinIdx() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMinIdx_Large(t *testing.T) {
	// Test with large slice to exercise SIMD paths
	n := 1000
	a := make([]float64, n)
	for i := range a {
		a[i] = float64(i + 100)
	}
	// Put minimum at position 777
	a[777] = -1.0

	got := MinIdx(a)
	if got != 777 {
		t.Errorf("MinIdx(large) = %v, want 777", got)
	}
}

func TestMaxIdx(t *testing.T) {
	tests := []struct {
		name string
		a    []float64
		want int
	}{
		{"empty", nil, -1},
		{"single", []float64{5}, 0},
		{"max at start", []float64{9, 2, 3, 4, 5}, 0},
		{"max at end", []float64{1, 2, 3, 4, 9}, 4},
		{"max in middle", []float64{5, 2, 8, 1, 3}, 2},
		{"duplicates", []float64{3, 5, 4, 5, 2}, 1}, // First occurrence
		{"negative", []float64{-5, -1, -3, -2}, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MaxIdx(tt.a)
			if got != tt.want {
				t.Errorf("MaxIdx() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMaxIdx_Large(t *testing.T) {
	// Test with large slice to exercise SIMD paths
	n := 1000
	a := make([]float64, n)
	for i := range a {
		a[i] = float64(i)
	}
	// Put maximum at position 333
	a[333] = 99999.0

	got := MaxIdx(a)
	if got != 333 {
		t.Errorf("MaxIdx(large) = %v, want 333", got)
	}
}

// Tests for AddScaled

func TestAddScaled(t *testing.T) {
	dst := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	s := []float64{1, 1, 1, 1, 1, 1, 1, 1}
	alpha := 2.0

	// dst[i] += alpha * s[i] = dst[i] + 2*1 = dst[i] + 2
	want := []float64{3, 4, 5, 6, 7, 8, 9, 10}

	AddScaled(dst, alpha, s)

	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("AddScaled()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestAddScaled_Large(t *testing.T) {
	// Test with large slice to exercise SIMD paths
	n := 1000
	dst := make([]float64, n)
	s := make([]float64, n)
	want := make([]float64, n)

	for i := range n {
		dst[i] = float64(i)
		s[i] = float64(i * 2)
		want[i] = float64(i) + 0.5*float64(i*2) // dst[i] + 0.5 * s[i]
	}

	AddScaled(dst, 0.5, s)

	for i := range dst {
		if math.Abs(dst[i]-want[i]) > 1e-10 {
			t.Errorf("AddScaled_Large()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestAddScaled_Empty(_ *testing.T) {
	// Should not panic on empty slices
	var dst, s []float64
	AddScaled(dst, 2.0, s)
}

func TestAddScaled_DifferentLengths(t *testing.T) {
	dst := []float64{1, 2, 3, 4, 5}
	s := []float64{10, 20, 30} // Shorter
	alpha := 1.0

	AddScaled(dst, alpha, s)

	// Only first 3 elements should be modified
	want := []float64{11, 22, 33, 4, 5}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("AddScaled_DifferentLengths()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestAddScaled_ZeroAlpha(t *testing.T) {
	dst := []float64{1, 2, 3, 4}
	s := []float64{10, 20, 30, 40}

	AddScaled(dst, 0.0, s)

	// dst should be unchanged
	want := []float64{1, 2, 3, 4}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("AddScaled_ZeroAlpha()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestAddScaled_NegativeAlpha(t *testing.T) {
	dst := []float64{10, 20, 30, 40}
	s := []float64{1, 2, 3, 4}

	AddScaled(dst, -2.0, s)

	// dst[i] += -2 * s[i]
	want := []float64{8, 16, 24, 32}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("AddScaled_NegativeAlpha()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

// Benchmarks for new functions

func BenchmarkMinIdx_1000(b *testing.B) {
	a := make([]float64, 1000)
	for i := range a {
		a[i] = float64(i)
	}
	a[500] = -1.0 // Put min in middle

	b.SetBytes(1000 * 8)

	var result int
	for b.Loop() {
		result = MinIdx(a)
	}
	_ = result
}

func BenchmarkMaxIdx_1000(b *testing.B) {
	a := make([]float64, 1000)
	for i := range a {
		a[i] = float64(i)
	}
	a[500] = 99999.0 // Put max in middle

	b.SetBytes(1000 * 8)

	var result int
	for b.Loop() {
		result = MaxIdx(a)
	}
	_ = result
}

func BenchmarkAddScaled_1000(b *testing.B) {
	dst := make([]float64, 1000)
	s := make([]float64, 1000)
	for i := range dst {
		dst[i] = float64(i)
		s[i] = float64(i)
	}

	b.SetBytes(1000 * 8 * 2) // read s, read+write dst

	for b.Loop() {
		AddScaled(dst, 0.5, s)
	}
}

// Tests for CubicInterpDot

// cubicInterpDotRef is a reference implementation for testing
func cubicInterpDotRef64(hist, a, b, c, d []float64, x float64) float64 {
	var sum float64
	for i := range hist {
		coef := a[i] + x*(b[i]+x*(c[i]+x*d[i]))
		sum += hist[i] * coef
	}
	return sum
}

func TestCubicInterpDot(t *testing.T) {
	t.Logf("CPU: %s", cpu.Info())

	tests := []struct {
		name string
		n    int
		x    float64
	}{
		{"empty", 0, 0.5},
		{"single", 1, 0.5},
		{"two", 2, 0.5},
		{"three", 3, 0.5},
		{"four", 4, 0.5},       // NEON vector width
		{"five", 5, 0.5},       // NEON + scalar remainder
		{"eight", 8, 0.5},      // AVX vector width
		{"nine", 9, 0.5},       // AVX + scalar remainder
		{"sixteen", 16, 0.5},   // 2x AVX vectors
		{"seventeen", 17, 0.5}, // 2x AVX + scalar remainder
		{"x=0", 16, 0.0},       // Edge case: x=0 means only a[] matters
		{"x=0.999", 16, 0.999}, // Near boundary
		{"x=1", 16, 1.0},       // Edge case
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.n == 0 {
				got := CubicInterpDot(nil, nil, nil, nil, nil, tt.x)
				if got != 0 {
					t.Errorf("CubicInterpDot(empty) = %v, want 0", got)
				}
				return
			}

			hist := make([]float64, tt.n)
			a := make([]float64, tt.n)
			b := make([]float64, tt.n)
			c := make([]float64, tt.n)
			d := make([]float64, tt.n)

			for i := range tt.n {
				hist[i] = float64(i+1) * 0.1
				a[i] = float64(i) * 0.5
				b[i] = float64(i) * 0.3
				c[i] = float64(i) * 0.2
				d[i] = float64(i) * 0.1
			}

			got := CubicInterpDot(hist, a, b, c, d, tt.x)
			want := cubicInterpDotRef64(hist, a, b, c, d, tt.x)

			// Use relative tolerance for floating point comparison
			tol := 1e-10
			if want != 0 {
				tol = math.Abs(want) * 1e-10
			}
			if math.Abs(got-want) > tol {
				t.Errorf("CubicInterpDot() = %v, want %v, diff = %v", got, want, got-want)
			}
		})
	}
}

func TestCubicInterpDot_Large(t *testing.T) {
	// Test with 64 taps (typical for high-quality audio resampling)
	sizes := []int{32, 64, 100, 241, 1000}

	for _, n := range sizes {
		hist := make([]float64, n)
		a := make([]float64, n)
		b := make([]float64, n)
		c := make([]float64, n)
		d := make([]float64, n)

		for i := range n {
			hist[i] = float64(i+1) * 0.01
			a[i] = float64(i) * 0.1
			b[i] = float64(i) * 0.05
			c[i] = float64(i) * 0.02
			d[i] = float64(i) * 0.01
		}

		x := 0.75
		got := CubicInterpDot(hist, a, b, c, d, x)
		want := cubicInterpDotRef64(hist, a, b, c, d, x)

		// Use relative tolerance
		tol := math.Abs(want) * 1e-9
		if math.Abs(got-want) > tol {
			t.Errorf("CubicInterpDot(n=%d) = %v, want %v, diff = %v", n, got, want, got-want)
		}
	}
}

func TestCubicInterpDot_DifferentLengths(t *testing.T) {
	// Test with slices of different lengths - should use minimum
	hist := []float64{1, 2, 3, 4, 5}
	a := []float64{1, 1, 1, 1, 1, 1, 1}
	b := []float64{0.5, 0.5, 0.5}
	c := []float64{0.1, 0.1, 0.1, 0.1}
	d := []float64{0.01, 0.01, 0.01, 0.01, 0.01, 0.01}

	// Minimum length is 3 (from b)
	n := 3
	x := 0.5
	got := CubicInterpDot(hist, a, b, c, d, x)
	want := cubicInterpDotRef64(hist[:n], a[:n], b[:n], c[:n], d[:n], x)

	if math.Abs(got-want) > 1e-10 {
		t.Errorf("CubicInterpDot(different lengths) = %v, want %v", got, want)
	}
}

func TestCubicInterpDotUnsafe(t *testing.T) {
	n := 16
	hist := make([]float64, n)
	a := make([]float64, n)
	b := make([]float64, n)
	c := make([]float64, n)
	d := make([]float64, n)

	for i := range n {
		hist[i] = float64(i+1) * 0.1
		a[i] = float64(i) * 0.5
		b[i] = float64(i) * 0.3
		c[i] = float64(i) * 0.2
		d[i] = float64(i) * 0.1
	}

	x := 0.5
	got := CubicInterpDotUnsafe(hist, a, b, c, d, x)
	want := cubicInterpDotRef64(hist, a, b, c, d, x)

	if math.Abs(got-want) > 1e-10 {
		t.Errorf("CubicInterpDotUnsafe() = %v, want %v", got, want)
	}
}

func BenchmarkCubicInterpDot_64(b *testing.B) {
	benchmarkCubicInterpDot64(b, 64)
}

func BenchmarkCubicInterpDot_241(b *testing.B) {
	benchmarkCubicInterpDot64(b, 241)
}

func BenchmarkCubicInterpDot_1000(b *testing.B) {
	benchmarkCubicInterpDot64(b, 1000)
}

func benchmarkCubicInterpDot64(b *testing.B, n int) {
	b.Helper()
	hist := make([]float64, n)
	a := make([]float64, n)
	bb := make([]float64, n)
	c := make([]float64, n)
	d := make([]float64, n)

	for i := range n {
		hist[i] = float64(i+1) * 0.01
		a[i] = float64(i) * 0.1
		bb[i] = float64(i) * 0.05
		c[i] = float64(i) * 0.02
		d[i] = float64(i) * 0.01
	}

	x := 0.75
	b.ResetTimer()
	b.SetBytes(int64(n * 8 * 5)) // 5 slices of float64

	var result float64
	for i := 0; i < b.N; i++ {
		result = CubicInterpDot(hist, a, bb, c, d, x)
	}
	_ = result
}

func TestSigmoid(t *testing.T) {
	testCases := []struct {
		name string
		src  []float64
	}{
		{"zeros", []float64{0, 0, 0, 0, 0, 0, 0, 0}},
		{"ones", []float64{1, 1, 1, 1, 1, 1, 1, 1}},
		{"negative", []float64{-1, -2, -3, -4, -5, -6, -7, -8}},
		{"positive", []float64{1, 2, 3, 4, 5, 6, 7, 8}},
		{"mixed", []float64{-3, -1, 0, 1, 3, -5, 5, -8}},
		{"large_positive", []float64{10, 20, 30}},
		{"large_negative", []float64{-10, -20, -30}},
		{"nine_elements", []float64{-2, -1, 0, 1, 2, 3, 4, 5, 6}},
		{"seventeen", make([]float64, 17)},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dst := make([]float64, len(tc.src))
			Sigmoid(dst, tc.src)

			// Verify all values are in [0,1]
			for i, v := range dst {
				if v < 0 || v > 1 {
					t.Errorf("sigmoid(%v) = %v, want in range [0,1]", tc.src[i], v)
				}
			}

			// Verify special values
			if len(tc.src) > 0 && tc.src[0] == 0 {
				want := 0.5
				if math.Abs(dst[0]-want) > 1e-6 {
					t.Errorf("sigmoid(0) = %v, want ~%v", dst[0], want)
				}
			}

			// Test in-place operation consistency
			inPlace := make([]float64, len(tc.src))
			copy(inPlace, tc.src)
			SigmoidInPlace(inPlace)
			for i := range dst {
				if math.Abs(dst[i]-inPlace[i]) > 1e-10 {
					t.Errorf("SigmoidInPlace mismatch at %d: got %v, want %v", i, inPlace[i], dst[i])
				}
			}
		})
	}
}

func TestReLU(t *testing.T) {
	src := []float64{-5, -2, -0.5, 0, 0.5, 2, 5}
	want := []float64{0, 0, 0, 0, 0.5, 2, 5}
	dst := make([]float64, len(src))

	ReLU(dst, src)
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("ReLU at %d: got %v, want %v", i, dst[i], want[i])
		}
	}

	// Test in-place
	inPlace := make([]float64, len(src))
	copy(inPlace, src)
	ReLUInPlace(inPlace)
	for i := range inPlace {
		if inPlace[i] != want[i] {
			t.Errorf("ReLUInPlace at %d: got %v, want %v", i, inPlace[i], want[i])
		}
	}
}

func TestClampScale(t *testing.T) {
	src := []float64{-5, 0, 5, 10, 15}
	dst := make([]float64, len(src))
	minVal := 0.0
	maxVal := 10.0
	scale := 0.1

	ClampScale(dst, src, minVal, maxVal, scale)

	// After clamping to [0,10] and scaling by 0.1:
	// -5 -> 0 -> 0*0.1 = 0
	// 0 -> 0 -> 0*0.1 = 0
	// 5 -> 5 -> 5*0.1 = 0.5
	// 10 -> 10 -> 10*0.1 = 1.0
	// 15 -> 10 -> 10*0.1 = 1.0
	want := []float64{0, 0, 0.5, 1.0, 1.0}

	for i := range dst {
		if math.Abs(dst[i]-want[i]) > 1e-10 {
			t.Errorf("ClampScale at %d: got %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestTanh(t *testing.T) {
	src := []float64{-3, -1, 0, 1, 3}
	dst := make([]float64, len(src))

	Tanh(dst, src)

	// Verify all values are in [-1,1]
	for i, v := range dst {
		if v < -1 || v > 1 {
			t.Errorf("tanh(%v) = %v, want in range [-1,1]", src[i], v)
		}
	}

	// Verify tanh(0) = 0
	if len(dst) > 2 && math.Abs(dst[2]) > 1e-10 {
		t.Errorf("tanh(0) = %v, want ~0", dst[2])
	}

	// Test in-place
	inPlace := make([]float64, len(src))
	copy(inPlace, src)
	TanhInPlace(inPlace)
	for i := range dst {
		if math.Abs(dst[i]-inPlace[i]) > 1e-10 {
			t.Errorf("TanhInPlace mismatch at %d: got %v, want %v", i, inPlace[i], dst[i])
		}
	}
}

func TestExp(t *testing.T) {
	src := []float64{-2, -1, 0, 1, 2}
	dst := make([]float64, len(src))

	Exp(dst, src)

	// Verify exp(0) = 1
	if len(dst) > 2 && math.Abs(dst[2]-1.0) > 1e-10 {
		t.Errorf("exp(0) = %v, want 1.0", dst[2])
	}

	// Verify all values are positive
	for i, v := range dst {
		if v <= 0 {
			t.Errorf("exp(%v) = %v, want positive", src[i], v)
		}
	}

	// Test in-place
	inPlace := make([]float64, len(src))
	copy(inPlace, src)
	ExpInPlace(inPlace)
	for i := range dst {
		if math.Abs(dst[i]-inPlace[i]) > 1e-10 {
			t.Errorf("ExpInPlace mismatch at %d: got %v, want %v", i, inPlace[i], dst[i])
		}
	}
}
