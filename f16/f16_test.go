package f16

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// Helper function to compare float32 values with tolerance
func almostEqual32(a, b, tol float32) bool {
	return math.Abs(float64(a-b)) <= float64(tol)
}

func TestCPUInfo(t *testing.T) {
	t.Logf("CPU: %s", cpu.Info())
	t.Logf("HasFP16: %v", cpu.HasFP16())
}

// =============================================================================
// Conversion Tests
// =============================================================================

func TestToFloat32(t *testing.T) {
	tests := []struct {
		name string
		h    Float16
		want float32
	}{
		{"zero", 0x0000, 0.0},
		{"neg_zero", 0x8000, 0.0}, // -0.0 equals 0.0
		{"one", 0x3C00, 1.0},
		{"neg_one", 0xBC00, -1.0},
		{"two", 0x4000, 2.0},
		{"half", 0x3800, 0.5},
		{"max_normal", 0x7BFF, 65504.0},  // Largest FP16 normal
		{"min_normal", 0x0400, 0.00006103515625}, // Smallest positive normal
		{"inf", 0x7C00, float32(math.Inf(1))},
		{"neg_inf", 0xFC00, float32(math.Inf(-1))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ToFloat32(tt.h)
			if math.IsInf(float64(tt.want), 0) {
				// Determine expected sign: +1 for positive infinity, -1 for negative
				sign := 1
				if tt.want < 0 {
					sign = -1
				}
				if !math.IsInf(float64(got), sign) {
					t.Errorf("ToFloat32(0x%04X) = %v, want %v", tt.h, got, tt.want)
				}
			} else if !almostEqual32(got, tt.want, 1e-6) {
				t.Errorf("ToFloat32(0x%04X) = %v, want %v", tt.h, got, tt.want)
			}
		})
	}
}

func TestFromFloat32(t *testing.T) {
	tests := []struct {
		name string
		f    float32
		want Float16
	}{
		{"zero", 0.0, 0x0000},
		{"one", 1.0, 0x3C00},
		{"neg_one", -1.0, 0xBC00},
		{"two", 2.0, 0x4000},
		{"half", 0.5, 0x3800},
		{"inf", float32(math.Inf(1)), 0x7C00},
		{"neg_inf", float32(math.Inf(-1)), 0xFC00},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FromFloat32(tt.f)
			if got != tt.want {
				t.Errorf("FromFloat32(%v) = 0x%04X, want 0x%04X", tt.f, got, tt.want)
			}
		})
	}
}

func TestRoundTrip(t *testing.T) {
	// Values that should round-trip exactly
	values := []float32{0, 1, -1, 2, 0.5, 0.25, 100, -100, 1024, 2048}

	for _, v := range values {
		h := FromFloat32(v)
		back := ToFloat32(h)
		if back != v {
			t.Errorf("RoundTrip(%v): FromFloat32 -> 0x%04X -> ToFloat32 = %v", v, h, back)
		}
	}
}

func TestToFloat32Slice(t *testing.T) {
	sizes := []int{1, 4, 7, 8, 9, 15, 16, 17, 100}

	for _, n := range sizes {
		t.Run(string(rune('0'+n%10)), func(t *testing.T) {
			src := make([]Float16, n)
			dst := make([]float32, n)

			for i := range src {
				src[i] = FromFloat32(float32(i))
			}

			ToFloat32Slice(dst, src)

			for i := range dst {
				want := float32(i)
				if !almostEqual32(dst[i], want, 1e-3) {
					t.Errorf("ToFloat32Slice()[%d] = %v, want %v", i, dst[i], want)
				}
			}
		})
	}
}

func TestFromFloat32Slice(t *testing.T) {
	sizes := []int{1, 4, 7, 8, 9, 15, 16, 17, 100}

	for _, n := range sizes {
		t.Run(string(rune('0'+n%10)), func(t *testing.T) {
			src := make([]float32, n)
			dst := make([]Float16, n)

			for i := range src {
				src[i] = float32(i)
			}

			FromFloat32Slice(dst, src)

			for i := range dst {
				back := ToFloat32(dst[i])
				want := float32(i)
				if !almostEqual32(back, want, 1e-3) {
					t.Errorf("FromFloat32Slice()[%d] converts back to %v, want %v", i, back, want)
				}
			}
		})
	}
}

// =============================================================================
// DotProduct Tests
// =============================================================================

func TestDotProduct(t *testing.T) {
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
			// Convert to Float16
			a := make([]Float16, len(tt.a))
			b := make([]Float16, len(tt.b))
			for i := range tt.a {
				a[i] = FromFloat32(tt.a[i])
			}
			for i := range tt.b {
				b[i] = FromFloat32(tt.b[i])
			}

			got := DotProduct(a, b)
			// FP16 has limited precision, use wider tolerance
			if !almostEqual32(got, tt.want, tt.want*0.01+0.1) {
				t.Errorf("DotProduct() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDotProduct_Large(t *testing.T) {
	sizes := []int{100, 1000}

	for _, n := range sizes {
		a := make([]Float16, n)
		b := make([]Float16, n)

		// a[i] = i+1, b[i] = 1 => sum = n*(n+1)/2
		for i := range a {
			a[i] = FromFloat32(float32(i + 1))
			b[i] = FromFloat32(1.0)
		}

		got := DotProduct(a, b)
		want := float32(n * (n + 1) / 2)

		// Allow 1% relative error for large sums
		tol := want * 0.01
		if !almostEqual32(got, want, tol) {
			t.Errorf("DotProduct(n=%d) = %v, want %v", n, got, want)
		}
	}
}

// =============================================================================
// Arithmetic Tests
// =============================================================================

func TestAdd(t *testing.T) {
	sizes := []int{1, 4, 7, 8, 9, 16, 17}

	for _, n := range sizes {
		a := make([]Float16, n)
		b := make([]Float16, n)
		dst := make([]Float16, n)

		for i := range a {
			a[i] = FromFloat32(float32(i))
			b[i] = FromFloat32(float32(n - i))
		}

		Add(dst, a, b)

		for i := range dst {
			got := ToFloat32(dst[i])
			want := float32(n)
			if !almostEqual32(got, want, 0.1) {
				t.Errorf("Add(n=%d)[%d] = %v, want %v", n, i, got, want)
			}
		}
	}
}

func TestSub(t *testing.T) {
	a := make([]Float16, 10)
	b := make([]Float16, 10)
	dst := make([]Float16, 10)

	for i := range a {
		a[i] = FromFloat32(float32(i * 2))
		b[i] = FromFloat32(float32(i))
	}

	Sub(dst, a, b)

	for i := range dst {
		got := ToFloat32(dst[i])
		want := float32(i)
		if !almostEqual32(got, want, 0.1) {
			t.Errorf("Sub()[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestMul(t *testing.T) {
	sizes := []int{1, 4, 7, 8, 9, 16, 17}

	for _, n := range sizes {
		a := make([]Float16, n)
		b := make([]Float16, n)
		dst := make([]Float16, n)

		for i := range a {
			a[i] = FromFloat32(float32(i + 1))
			b[i] = FromFloat32(2.0)
		}

		Mul(dst, a, b)

		for i := range dst {
			got := ToFloat32(dst[i])
			want := float32((i + 1) * 2)
			if !almostEqual32(got, want, 0.1) {
				t.Errorf("Mul(n=%d)[%d] = %v, want %v", n, i, got, want)
			}
		}
	}
}

func TestScale(t *testing.T) {
	a := make([]Float16, 10)
	dst := make([]Float16, 10)
	s := FromFloat32(3.0)

	for i := range a {
		a[i] = FromFloat32(float32(i))
	}

	Scale(dst, a, s)

	for i := range dst {
		got := ToFloat32(dst[i])
		want := float32(i * 3)
		if !almostEqual32(got, want, 0.1) {
			t.Errorf("Scale()[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestFMA(t *testing.T) {
	sizes := []int{1, 4, 7, 8, 9, 16, 17}

	for _, n := range sizes {
		a := make([]Float16, n)
		b := make([]Float16, n)
		c := make([]Float16, n)
		dst := make([]Float16, n)

		for i := range a {
			a[i] = FromFloat32(float32(i))
			b[i] = FromFloat32(2.0)
			c[i] = FromFloat32(1.0)
		}

		FMA(dst, a, b, c)

		for i := range dst {
			got := ToFloat32(dst[i])
			want := float32(i*2 + 1)
			if !almostEqual32(got, want, 0.1) {
				t.Errorf("FMA(n=%d)[%d] = %v, want %v", n, i, got, want)
			}
		}
	}
}

// =============================================================================
// Reduction Tests
// =============================================================================

func TestSum(t *testing.T) {
	sizes := []int{1, 4, 7, 8, 9, 16, 17, 100}

	for _, n := range sizes {
		a := make([]Float16, n)
		for i := range a {
			a[i] = FromFloat32(1.0)
		}

		got := Sum(a)
		want := float32(n)

		if !almostEqual32(got, want, want*0.01+0.1) {
			t.Errorf("Sum(n=%d) = %v, want %v", n, got, want)
		}
	}
}

func TestMean(t *testing.T) {
	a := make([]Float16, 10)
	for i := range a {
		a[i] = FromFloat32(float32(i + 1))
	}

	got := Mean(a)
	want := float32(5.5) // (1+2+...+10)/10

	if !almostEqual32(got, want, 0.1) {
		t.Errorf("Mean() = %v, want %v", got, want)
	}
}

func TestMin(t *testing.T) {
	a := make([]Float16, 10)
	values := []float32{5, 2, 8, 1, 9, 3, 7, 4, 6, 10}
	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	got := ToFloat32(Min(a))
	want := float32(1)

	if !almostEqual32(got, want, 0.1) {
		t.Errorf("Min() = %v, want %v", got, want)
	}
}

func TestMax(t *testing.T) {
	a := make([]Float16, 10)
	values := []float32{5, 2, 8, 1, 9, 3, 7, 4, 6, 10}
	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	got := ToFloat32(Max(a))
	want := float32(10)

	if !almostEqual32(got, want, 0.1) {
		t.Errorf("Max() = %v, want %v", got, want)
	}
}

// =============================================================================
// Unary Operations Tests
// =============================================================================

func TestAbs(t *testing.T) {
	a := make([]Float16, 10)
	dst := make([]Float16, 10)
	values := []float32{-5, 2, -8, 1, -9, 3, -7, 4, -6, 10}

	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	Abs(dst, a)

	for i := range dst {
		got := ToFloat32(dst[i])
		want := float32(math.Abs(float64(values[i])))
		if !almostEqual32(got, want, 0.1) {
			t.Errorf("Abs()[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestNeg(t *testing.T) {
	a := make([]Float16, 10)
	dst := make([]Float16, 10)

	for i := range a {
		a[i] = FromFloat32(float32(i - 5))
	}

	Neg(dst, a)

	for i := range dst {
		got := ToFloat32(dst[i])
		want := float32(5 - i)
		if !almostEqual32(got, want, 0.1) {
			t.Errorf("Neg()[%d] = %v, want %v", i, got, want)
		}
	}
}

// =============================================================================
// Activation Function Tests
// =============================================================================

func TestReLU(t *testing.T) {
	src := []float32{-5, -2, -0.5, 0, 0.5, 2, 5}
	want := []float32{0, 0, 0, 0, 0.5, 2, 5}

	a := make([]Float16, len(src))
	dst := make([]Float16, len(src))

	for i, v := range src {
		a[i] = FromFloat32(v)
	}

	ReLU(dst, a)

	for i := range dst {
		got := ToFloat32(dst[i])
		if !almostEqual32(got, want[i], 0.1) {
			t.Errorf("ReLU()[%d] = %v, want %v", i, got, want[i])
		}
	}
}

func TestSigmoid(t *testing.T) {
	values := []float32{-10, -5, -1, 0, 1, 5, 10}
	a := make([]Float16, len(values))
	dst := make([]Float16, len(values))

	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	Sigmoid(dst, a)

	for i := range dst {
		got := ToFloat32(dst[i])
		// Verify in valid range
		if got < 0 || got > 1 {
			t.Errorf("Sigmoid()[%d] = %v, out of range [0,1]", i, got)
		}
		// Verify sigmoid(0) ≈ 0.5
		if values[i] == 0 && !almostEqual32(got, 0.5, 0.1) {
			t.Errorf("Sigmoid(0) = %v, want ~0.5", got)
		}
	}
}

// =============================================================================
// Edge Case Tests
// =============================================================================

func TestEmptySlices(t *testing.T) {
	// These should not panic
	var empty []Float16

	DotProduct(empty, empty)
	DotProductF32(empty, empty)
	Add(empty, empty, empty)
	Sub(empty, empty, empty)
	Mul(empty, empty, empty)
	Scale(empty, empty, 0)
	FMA(empty, empty, empty, empty)
	Sum(empty)
	Abs(empty, empty)
	Neg(empty, empty)
	ReLU(empty, empty)
	Sigmoid(empty, empty)
	Mean(empty)

	// Reductions with sentinel returns on empty input.
	if got := Min(empty); got != 0x7C00 {
		t.Errorf("Min(empty): got 0x%04X, want 0x7C00 (+Inf)", got)
	}
	if got := Max(empty); got != 0xFC00 {
		t.Errorf("Max(empty): got 0x%04X, want 0xFC00 (-Inf)", got)
	}
	if got := MinIdx(empty); got != -1 {
		t.Errorf("MinIdx(empty): got %d, want -1", got)
	}
	if got := MaxIdx(empty); got != -1 {
		t.Errorf("MaxIdx(empty): got %d, want -1", got)
	}

	var emptyF32 []float32
	var emptyF16 []Float16
	ToFloat32Slice(emptyF32, emptyF16)
	FromFloat32Slice(emptyF16, emptyF32)
}

// =============================================================================
// Benchmarks
// =============================================================================

func BenchmarkToFloat32Slice_1000(b *testing.B) {
	src := make([]Float16, 1000)
	dst := make([]float32, 1000)
	for i := range src {
		src[i] = FromFloat32(float32(i))
	}

	b.SetBytes(1000 * 2) // 2 bytes per FP16

	for b.Loop() {
		ToFloat32Slice(dst, src)
	}
}

func BenchmarkFromFloat32Slice_1000(b *testing.B) {
	src := make([]float32, 1000)
	dst := make([]Float16, 1000)
	for i := range src {
		src[i] = float32(i)
	}

	b.SetBytes(1000 * 4) // 4 bytes per FP32

	for b.Loop() {
		FromFloat32Slice(dst, src)
	}
}

func BenchmarkDotProduct_1000(b *testing.B) {
	a := make([]Float16, 1000)
	c := make([]Float16, 1000)
	for i := range a {
		a[i] = FromFloat32(float32(i))
		c[i] = FromFloat32(float32(1000 - i))
	}

	b.SetBytes(1000 * 2 * 2) // 2 bytes per element, 2 arrays

	var result float32
	for b.Loop() {
		result = DotProduct(a, c)
	}
	_ = result
}

func BenchmarkAdd_1000(b *testing.B) {
	a := make([]Float16, 1000)
	c := make([]Float16, 1000)
	dst := make([]Float16, 1000)

	for i := range a {
		a[i] = FromFloat32(float32(i))
		c[i] = FromFloat32(float32(1000 - i))
	}

	b.SetBytes(1000 * 2 * 3) // 3 arrays

	for b.Loop() {
		Add(dst, a, c)
	}
}

func BenchmarkMul_1000(b *testing.B) {
	a := make([]Float16, 1000)
	c := make([]Float16, 1000)
	dst := make([]Float16, 1000)

	for i := range a {
		a[i] = FromFloat32(float32(i))
		c[i] = FromFloat32(float32(2))
	}

	b.SetBytes(1000 * 2 * 3)

	for b.Loop() {
		Mul(dst, a, c)
	}
}

func BenchmarkFMA_1000(b *testing.B) {
	a := make([]Float16, 1000)
	c := make([]Float16, 1000)
	d := make([]Float16, 1000)
	dst := make([]Float16, 1000)

	for i := range a {
		a[i] = FromFloat32(float32(i))
		c[i] = FromFloat32(float32(2))
		d[i] = FromFloat32(float32(1))
	}

	b.SetBytes(1000 * 2 * 4)

	for b.Loop() {
		FMA(dst, a, c, d)
	}
}

func BenchmarkSum_1000(b *testing.B) {
	a := make([]Float16, 1000)
	for i := range a {
		a[i] = FromFloat32(float32(i))
	}

	b.SetBytes(1000 * 2)

	var result float32
	for b.Loop() {
		result = Sum(a)
	}
	_ = result
}

func BenchmarkReLU_1000(b *testing.B) {
	src := make([]Float16, 1000)
	dst := make([]Float16, 1000)

	for i := range src {
		src[i] = FromFloat32(float32(i - 500))
	}

	b.SetBytes(1000 * 2 * 2)

	for b.Loop() {
		ReLU(dst, src)
	}
}

// =============================================================================
// New Operation Tests
// =============================================================================

func TestDiv(t *testing.T) {
	sizes := []int{1, 4, 7, 8, 9, 16, 17}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			a := make([]Float16, size)
			b := make([]Float16, size)
			dst := make([]Float16, size)

			for i := range a {
				a[i] = FromFloat32(float32(i + 2))
				b[i] = FromFloat32(float32(2))
			}

			Div(dst, a, b)

			for i := range dst {
				got := ToFloat32(dst[i])
				want := float32(i+2) / 2
				if !almostEqual32(got, want, 0.1) {
					t.Errorf("Div()[%d] = %v, want %v", i, got, want)
				}
			}
		})
	}
}

func TestAddScalar(t *testing.T) {
	sizes := []int{1, 4, 7, 8, 9, 16}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			a := make([]Float16, size)
			dst := make([]Float16, size)
			scalar := FromFloat32(10.0)

			for i := range a {
				a[i] = FromFloat32(float32(i))
			}

			AddScalar(dst, a, scalar)

			for i := range dst {
				got := ToFloat32(dst[i])
				want := float32(i) + 10.0
				if !almostEqual32(got, want, 0.1) {
					t.Errorf("AddScalar()[%d] = %v, want %v", i, got, want)
				}
			}
		})
	}
}

func TestClamp(t *testing.T) {
	values := []float32{-10, -5, 0, 5, 10, 15, 20}
	minVal := FromFloat32(0)
	maxVal := FromFloat32(10)

	a := make([]Float16, len(values))
	dst := make([]Float16, len(values))
	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	Clamp(dst, a, minVal, maxVal)

	want := []float32{0, 0, 0, 5, 10, 10, 10}
	for i := range dst {
		got := ToFloat32(dst[i])
		if !almostEqual32(got, want[i], 0.1) {
			t.Errorf("Clamp()[%d] = %v, want %v", i, got, want[i])
		}
	}
}

func TestSqrt(t *testing.T) {
	values := []float32{0, 1, 4, 9, 16, 25, 100}
	a := make([]Float16, len(values))
	dst := make([]Float16, len(values))

	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	Sqrt(dst, a)

	for i := range dst {
		got := ToFloat32(dst[i])
		want := float32(math.Sqrt(float64(values[i])))
		if !almostEqual32(got, want, 0.1) {
			t.Errorf("Sqrt()[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestReciprocal(t *testing.T) {
	values := []float32{1, 2, 4, 5, 10, 0.5}
	a := make([]Float16, len(values))
	dst := make([]Float16, len(values))

	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	Reciprocal(dst, a)

	for i := range dst {
		got := ToFloat32(dst[i])
		want := 1.0 / values[i]
		if !almostEqual32(got, want, 0.1) {
			t.Errorf("Reciprocal()[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestExp(t *testing.T) {
	values := []float32{0, 1, -1, 2, -2}
	a := make([]Float16, len(values))
	dst := make([]Float16, len(values))

	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	Exp(dst, a)

	for i := range dst {
		got := ToFloat32(dst[i])
		want := float32(math.Exp(float64(values[i])))
		// Use relative tolerance for exp
		tol := want * 0.05
		if tol < 0.1 {
			tol = 0.1
		}
		if !almostEqual32(got, want, tol) {
			t.Errorf("Exp()[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestTanh(t *testing.T) {
	values := []float32{0, 1, -1, 2, -2, 5, -5}
	a := make([]Float16, len(values))
	dst := make([]Float16, len(values))

	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	Tanh(dst, a)

	for i := range dst {
		got := ToFloat32(dst[i])
		want := float32(math.Tanh(float64(values[i])))
		if !almostEqual32(got, want, 0.1) {
			t.Errorf("Tanh()[%d] = %v, want %v", i, got, want)
		}
	}
}

func TestMinIdx(t *testing.T) {
	values := []float32{5, 2, 8, 1, 9, 3, 7}
	a := make([]Float16, len(values))
	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	got := MinIdx(a)
	want := 3 // index of 1

	if got != want {
		t.Errorf("MinIdx() = %v, want %v", got, want)
	}

	// Test empty
	if MinIdx(nil) != -1 {
		t.Error("MinIdx(nil) should return -1")
	}
}

func TestMaxIdx(t *testing.T) {
	values := []float32{5, 2, 8, 1, 9, 3, 7}
	a := make([]Float16, len(values))
	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	got := MaxIdx(a)
	want := 4 // index of 9

	if got != want {
		t.Errorf("MaxIdx() = %v, want %v", got, want)
	}

	// Test empty
	if MaxIdx(nil) != -1 {
		t.Error("MaxIdx(nil) should return -1")
	}
}

func TestAddScaled(t *testing.T) {
	sizes := []int{1, 4, 7, 8, 9, 16}
	for _, size := range sizes {
		t.Run(fmt.Sprintf("size_%d", size), func(t *testing.T) {
			dst := make([]Float16, size)
			s := make([]Float16, size)
			alpha := FromFloat32(2.0)

			for i := range dst {
				dst[i] = FromFloat32(float32(i))
				s[i] = FromFloat32(float32(i + 1))
			}

			AddScaled(dst, alpha, s)

			for i := range dst {
				got := ToFloat32(dst[i])
				want := float32(i) + 2.0*float32(i+1)
				if !almostEqual32(got, want, 0.2) {
					t.Errorf("AddScaled()[%d] = %v, want %v", i, got, want)
				}
			}
		})
	}
}

func TestNormalize(t *testing.T) {
	// Vector (3, 4) should normalize to (0.6, 0.8)
	a := []Float16{FromFloat32(3), FromFloat32(4)}
	dst := make([]Float16, 2)

	Normalize(dst, a)

	got0 := ToFloat32(dst[0])
	got1 := ToFloat32(dst[1])

	if !almostEqual32(got0, 0.6, 0.05) {
		t.Errorf("Normalize()[0] = %v, want 0.6", got0)
	}
	if !almostEqual32(got1, 0.8, 0.05) {
		t.Errorf("Normalize()[1] = %v, want 0.8", got1)
	}
}

func TestEuclideanDistance(t *testing.T) {
	// Distance between (0,0) and (3,4) = 5
	a := []Float16{FromFloat32(0), FromFloat32(0)}
	b := []Float16{FromFloat32(3), FromFloat32(4)}

	got := EuclideanDistance(a, b)
	want := float32(5)

	if !almostEqual32(got, want, 0.1) {
		t.Errorf("EuclideanDistance() = %v, want %v", got, want)
	}
}

func TestVariance(t *testing.T) {
	// Variance of [1,2,3,4,5] = 2
	values := []float32{1, 2, 3, 4, 5}
	a := make([]Float16, len(values))
	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	got := Variance(a)
	want := float32(2.0) // ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5 = 10/5 = 2

	if !almostEqual32(got, want, 0.2) {
		t.Errorf("Variance() = %v, want %v", got, want)
	}
}

func TestStdDev(t *testing.T) {
	values := []float32{1, 2, 3, 4, 5}
	a := make([]Float16, len(values))
	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	got := StdDev(a)
	want := float32(math.Sqrt(2.0)) // sqrt(variance)

	if !almostEqual32(got, want, 0.2) {
		t.Errorf("StdDev() = %v, want %v", got, want)
	}
}

func TestCumulativeSum(t *testing.T) {
	values := []float32{1, 2, 3, 4, 5}
	a := make([]Float16, len(values))
	dst := make([]Float16, len(values))
	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	CumulativeSum(dst, a)

	want := []float32{1, 3, 6, 10, 15}
	for i := range dst {
		got := ToFloat32(dst[i])
		if !almostEqual32(got, want[i], 0.1) {
			t.Errorf("CumulativeSum()[%d] = %v, want %v", i, got, want[i])
		}
	}
}

func TestDotProductBatch(t *testing.T) {
	vec := []Float16{FromFloat32(1), FromFloat32(2), FromFloat32(3)}
	rows := [][]Float16{
		{FromFloat32(1), FromFloat32(0), FromFloat32(0)},
		{FromFloat32(0), FromFloat32(1), FromFloat32(0)},
		{FromFloat32(1), FromFloat32(1), FromFloat32(1)},
	}
	results := make([]float32, 3)

	DotProductBatch(results, rows, vec)

	want := []float32{1, 2, 6}
	for i := range results {
		if !almostEqual32(results[i], want[i], 0.1) {
			t.Errorf("DotProductBatch()[%d] = %v, want %v", i, results[i], want[i])
		}
	}
}

func TestAccumulateAdd(t *testing.T) {
	dst := make([]Float16, 10)
	for i := range dst {
		dst[i] = FromFloat32(float32(i))
	}
	src := []Float16{FromFloat32(10), FromFloat32(20), FromFloat32(30)}

	AccumulateAdd(dst, src, 2)

	// dst[2:5] should be increased by src values
	want := []float32{0, 1, 12, 23, 34, 5, 6, 7, 8, 9}
	for i := range dst {
		got := ToFloat32(dst[i])
		if !almostEqual32(got, want[i], 0.1) {
			t.Errorf("AccumulateAdd()[%d] = %v, want %v", i, got, want[i])
		}
	}
}

func TestConvolveValid(t *testing.T) {
	signal := []Float16{
		FromFloat32(1), FromFloat32(2), FromFloat32(3),
		FromFloat32(4), FromFloat32(5),
	}
	kernel := []Float16{FromFloat32(1), FromFloat32(2)}
	dst := make([]Float16, 4) // len(signal) - len(kernel) + 1

	ConvolveValid(dst, signal, kernel)

	// [1,2,3,4,5] conv [1,2] = [1*1+2*2, 2*1+3*2, 3*1+4*2, 4*1+5*2] = [5, 8, 11, 14]
	want := []float32{5, 8, 11, 14}
	for i := range dst {
		got := ToFloat32(dst[i])
		if !almostEqual32(got, want[i], 0.2) {
			t.Errorf("ConvolveValid()[%d] = %v, want %v", i, got, want[i])
		}
	}
}

func TestInterleave2(t *testing.T) {
	a := []Float16{FromFloat32(1), FromFloat32(2), FromFloat32(3)}
	b := []Float16{FromFloat32(10), FromFloat32(20), FromFloat32(30)}
	dst := make([]Float16, 6)

	Interleave2(dst, a, b)

	want := []float32{1, 10, 2, 20, 3, 30}
	for i := range dst {
		got := ToFloat32(dst[i])
		if !almostEqual32(got, want[i], 0.1) {
			t.Errorf("Interleave2()[%d] = %v, want %v", i, got, want[i])
		}
	}
}

func TestDeinterleave2(t *testing.T) {
	src := []Float16{
		FromFloat32(1), FromFloat32(10),
		FromFloat32(2), FromFloat32(20),
		FromFloat32(3), FromFloat32(30),
	}
	a := make([]Float16, 3)
	b := make([]Float16, 3)

	Deinterleave2(a, b, src)

	wantA := []float32{1, 2, 3}
	wantB := []float32{10, 20, 30}

	for i := range a {
		gotA := ToFloat32(a[i])
		gotB := ToFloat32(b[i])
		if !almostEqual32(gotA, wantA[i], 0.1) {
			t.Errorf("Deinterleave2() a[%d] = %v, want %v", i, gotA, wantA[i])
		}
		if !almostEqual32(gotB, wantB[i], 0.1) {
			t.Errorf("Deinterleave2() b[%d] = %v, want %v", i, gotB, wantB[i])
		}
	}
}

func TestClampScale(t *testing.T) {
	values := []float32{-10, 0, 5, 10, 20}
	a := make([]Float16, len(values))
	dst := make([]Float16, len(values))
	for i, v := range values {
		a[i] = FromFloat32(v)
	}

	minVal := FromFloat32(0)
	maxVal := FromFloat32(10)
	scale := FromFloat32(0.1) // Normalize to [0, 1]

	ClampScale(dst, a, minVal, maxVal, scale)

	// (clamp(x, 0, 10) - 0) * 0.1
	want := []float32{0, 0, 0.5, 1.0, 1.0}
	for i := range dst {
		got := ToFloat32(dst[i])
		if !almostEqual32(got, want[i], 0.1) {
			t.Errorf("ClampScale()[%d] = %v, want %v", i, got, want[i])
		}
	}
}

func TestReLUInPlace(t *testing.T) {
	a := []Float16{FromFloat32(-5), FromFloat32(-2), FromFloat32(0), FromFloat32(2), FromFloat32(5)}
	ReLUInPlace(a)

	want := []float32{0, 0, 0, 2, 5}
	for i := range a {
		got := ToFloat32(a[i])
		if !almostEqual32(got, want[i], 0.1) {
			t.Errorf("ReLUInPlace()[%d] = %v, want %v", i, got, want[i])
		}
	}
}

func TestNewEmptySlices(_ *testing.T) {
	// These should not panic
	var empty []Float16

	Div(empty, empty, empty)
	AddScalar(empty, empty, 0)
	Clamp(empty, empty, 0, 0)
	Sqrt(empty, empty)
	Reciprocal(empty, empty)
	Exp(empty, empty)
	Tanh(empty, empty)
	AddScaled(empty, 0, empty)
	Normalize(empty, empty)
	EuclideanDistance(empty, empty)
	Variance(empty)
	StdDev(empty)
	CumulativeSum(empty, empty)
	DotProductBatch(nil, nil, empty)
	ConvolveValid(empty, empty, empty)
	Interleave2(empty, empty, empty)
	Deinterleave2(empty, empty, empty)
	ClampScale(empty, empty, 0, 0, 0)
	ReLUInPlace(empty)
	SigmoidInPlace(empty)
	ExpInPlace(empty)
	TanhInPlace(empty)
}

func TestMinGo_EmptySlice(t *testing.T) {
	got := minGo(nil)
	if got != 0x7C00 { // +Inf in FP16
		t.Errorf("minGo(nil): got 0x%04X, want 0x7C00 (+Inf)", got)
	}
}

func TestMaxGo_EmptySlice(t *testing.T) {
	got := maxGo(nil)
	if got != 0xFC00 { // -Inf in FP16
		t.Errorf("maxGo(nil): got 0x%04X, want 0xFC00 (-Inf)", got)
	}
}

func TestMinIdxGo_EmptySlice(t *testing.T) {
	if got := minIdxGo(nil); got != -1 {
		t.Errorf("minIdxGo(nil): got %d, want -1", got)
	}
}

func TestMaxIdxGo_EmptySlice(t *testing.T) {
	if got := maxIdxGo(nil); got != -1 {
		t.Errorf("maxIdxGo(nil): got %d, want -1", got)
	}
}

func TestDotProductF32_MatchesGoSum(t *testing.T) {
	sizes := []int{8, 16, 64, 256, 1024}
	rng := rand.New(rand.NewSource(42))
	for _, n := range sizes {
		a := make([]Float16, n)
		b := make([]Float16, n)
		for i := range a {
			a[i] = FromFloat32(rng.Float32()*2 - 1) // [-1, 1]
			b[i] = FromFloat32(rng.Float32()*2 - 1)
		}

		got := DotProductF32(a, b)
		want := dotProductGo(a, b)

		// Reference and SIMD differ in summation order; bound generously to
		// absorb up to n*ULP per accumulator chain.
		tol := math.Abs(float64(want))*float64(n)*1e-6 + 1e-5
		if math.Abs(float64(got-want)) > tol {
			t.Errorf("n=%d: got %v, want %v (tol %v)", n, got, want, tol)
		}
	}
}

func TestDotProductF32_Empty(t *testing.T) {
	if got := DotProductF32(nil, nil); got != 0 {
		t.Errorf("nil,nil: got %v, want 0", got)
	}
	if got := DotProductF32(nil, []Float16{FromFloat32(1), FromFloat32(2)}); got != 0 {
		t.Errorf("nil,non-nil: got %v, want 0", got)
	}
}

func TestDotProductF32_LengthMismatch(t *testing.T) {
	a := []Float16{FromFloat32(1), FromFloat32(2), FromFloat32(3)}
	b := []Float16{FromFloat32(4), FromFloat32(5)}
	// min(len) = 2; expected = 1*4 + 2*5 = 14
	if got := DotProductF32(a, b); got != 14 {
		t.Errorf("length-mismatch: got %v, want 14", got)
	}
}

func TestDotProductF32_TailLessThanWidth(t *testing.T) {
	// n = 11 exercises 1 NEON iteration (8) + 3 Go tail on ARM64,
	// and the all-Go path on AMD64.
	const n = 11
	a := make([]Float16, n)
	b := make([]Float16, n)
	for i := range a {
		a[i] = FromFloat32(float32(i + 1))
		b[i] = FromFloat32(float32(i + 1))
	}
	want := dotProductGo(a, b)
	got := DotProductF32(a, b)
	if math.Abs(float64(got-want)) > 1e-4 {
		t.Errorf("tail handling: got %v, want %v", got, want)
	}
}

// TestDotProductF32_Repro22 reproduces the scenario from issue #22 and
// asserts that DotProductF32 does not saturate. 300 is exact in FP16
// (1.171875 * 2^8), each product 300*300 = 90000 overflows FP16 max
// (65504) but fits FP32 easily; sum of 8 products is 720000.
//
// Logs both APIs for cross-platform comparison: DotProduct is +Inf on
// ARM64 with native FP16 SIMD and 720000 on AMD64 (where the Go fallback
// already widens before multiplying). The log line is captured in PR
// descriptions to document the platform-specific saturation.
func TestDotProductF32_Repro22(t *testing.T) {
	const n = 8
	a := make([]Float16, n)
	b := make([]Float16, n)
	for i := range a {
		a[i] = FromFloat32(300)
		b[i] = FromFloat32(300)
	}
	native := DotProduct(a, b)
	wide := DotProductF32(a, b)
	t.Logf("issue #22 reproducer: DotProduct=%v  DotProductF32=%v", native, wide)
	if wide != 720000 {
		t.Fatalf("DotProductF32: got %v, want 720000", wide)
	}
}
