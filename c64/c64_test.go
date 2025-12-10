package c64

import (
	"fmt"
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

const epsilon = 1e-5 // float32 has ~7 decimal digits precision

func complexClose(a, b complex64) bool {
	return math.Abs(float64(real(a)-real(b))) < epsilon && math.Abs(float64(imag(a)-imag(b))) < epsilon
}

func floatClose(a, b float32) bool {
	return math.Abs(float64(a-b)) < epsilon
}

func TestMul(t *testing.T) {
	t.Logf("CPU: %s", cpu.Info())

	tests := []struct {
		name string
		a, b []complex64
		want []complex64
	}{
		{"empty", nil, nil, nil},
		{"single", []complex64{1 + 2i}, []complex64{3 + 4i}, []complex64{-5 + 10i}},
		{"two", []complex64{1 + 2i, 3 + 4i}, []complex64{5 + 6i, 7 + 8i}, []complex64{-7 + 16i, -11 + 52i}},
		{"pure_real", []complex64{2 + 0i, 3 + 0i}, []complex64{4 + 0i, 5 + 0i}, []complex64{8 + 0i, 15 + 0i}},
		{"pure_imag", []complex64{0 + 2i, 0 + 3i}, []complex64{0 + 4i, 0 + 5i}, []complex64{-8 + 0i, -15 + 0i}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.a) == 0 {
				dst := make([]complex64, 0)
				Mul(dst, tt.a, tt.b)
				return
			}
			dst := make([]complex64, len(tt.a))
			Mul(dst, tt.a, tt.b)
			for i := range dst {
				if !complexClose(dst[i], tt.want[i]) {
					t.Errorf("Mul()[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestMul_Large(t *testing.T) {
	n := 100
	a := make([]complex64, n)
	b := make([]complex64, n)
	for i := range n {
		a[i] = complex(float32(i+1), float32(i+2))
		b[i] = complex(float32(i+3), float32(i+4))
	}

	dst := make([]complex64, n)
	Mul(dst, a, b)

	// Verify against Go's complex multiplication
	for i := range n {
		expected := a[i] * b[i]
		if !complexClose(dst[i], expected) {
			t.Errorf("Mul_Large()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestMulConj(t *testing.T) {
	tests := []struct {
		name string
		a, b []complex64
	}{
		{"single", []complex64{1 + 2i}, []complex64{3 + 4i}},
		{"two", []complex64{1 + 2i, 3 + 4i}, []complex64{5 + 6i, 7 + 8i}},
		{"mixed", []complex64{-1 + 2i, 3 - 4i, 5 + 0i}, []complex64{1 + 1i, 2 - 2i, 0 + 3i}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]complex64, len(tt.a))
			MulConj(dst, tt.a, tt.b)
			for i := range dst {
				// conj(b) = real(b) - imag(b)i
				conjB := complex(real(tt.b[i]), -imag(tt.b[i]))
				expected := tt.a[i] * conjB
				if !complexClose(dst[i], expected) {
					t.Errorf("MulConj()[%d] = %v, want %v", i, dst[i], expected)
				}
			}
		})
	}
}

func TestMulConj_Large(t *testing.T) {
	n := 100
	a := make([]complex64, n)
	b := make([]complex64, n)
	for i := range n {
		a[i] = complex(float32(i+1), float32(i+2))
		b[i] = complex(float32(i+3), float32(i+4))
	}

	dst := make([]complex64, n)
	MulConj(dst, a, b)

	for i := range n {
		conjB := complex(real(b[i]), -imag(b[i]))
		expected := a[i] * conjB
		if !complexClose(dst[i], expected) {
			t.Errorf("MulConj_Large()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestScale(t *testing.T) {
	tests := []struct {
		name string
		a    []complex64
		s    complex64
	}{
		{"single", []complex64{1 + 2i}, 3 + 4i},
		{"two", []complex64{1 + 2i, 3 + 4i}, 2 + 0i},
		{"by_i", []complex64{1 + 0i, 0 + 1i, 1 + 1i}, 0 + 1i},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]complex64, len(tt.a))
			Scale(dst, tt.a, tt.s)
			for i := range dst {
				expected := tt.a[i] * tt.s
				if !complexClose(dst[i], expected) {
					t.Errorf("Scale()[%d] = %v, want %v", i, dst[i], expected)
				}
			}
		})
	}
}

func TestAdd(t *testing.T) {
	a := []complex64{1 + 2i, 3 + 4i, 5 + 6i}
	b := []complex64{7 + 8i, 9 + 10i, 11 + 12i}
	dst := make([]complex64, len(a))

	Add(dst, a, b)

	for i := range dst {
		expected := a[i] + b[i]
		if !complexClose(dst[i], expected) {
			t.Errorf("Add()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestSub(t *testing.T) {
	a := []complex64{10 + 20i, 30 + 40i, 50 + 60i}
	b := []complex64{1 + 2i, 3 + 4i, 5 + 6i}
	dst := make([]complex64, len(a))

	Sub(dst, a, b)

	for i := range dst {
		expected := a[i] - b[i]
		if !complexClose(dst[i], expected) {
			t.Errorf("Sub()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

// Test various sizes to exercise SIMD remainder handling
func TestMul_Sizes(t *testing.T) {
	for size := 1; size <= 17; size++ {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			a := make([]complex64, size)
			b := make([]complex64, size)
			for i := range size {
				a[i] = complex(float32(i+1), float32(i+2))
				b[i] = complex(float32(i+3), float32(i+4))
			}

			dst := make([]complex64, size)
			Mul(dst, a, b)

			for i := range size {
				expected := a[i] * b[i]
				if !complexClose(dst[i], expected) {
					t.Errorf("size=%d, Mul()[%d] = %v, want %v", size, i, dst[i], expected)
				}
			}
		})
	}
}

// ============================================================================
// Tests for Abs, AbsSq, Conj, FromReal
// ============================================================================

func TestAbs(t *testing.T) {
	t.Logf("CPU: %s", cpu.Info())

	tests := []struct {
		name string
		a    []complex64
	}{
		{"single_1_0", []complex64{1 + 0i}},
		{"single_3_4", []complex64{3 + 4i}},    // Should give 5
		{"pair", []complex64{3 + 4i, 5 + 12i}}, // 5, 13
		{"pure_real", []complex64{5 + 0i, 10 + 0i}},
		{"pure_imag", []complex64{0 + 5i, 0 + 10i}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			Abs(dst, tt.a)
			for i := range dst {
				r := float64(real(tt.a[i]))
				im := float64(imag(tt.a[i]))
				expected := float32(math.Sqrt(r*r + im*im))
				if !floatClose(dst[i], expected) {
					t.Errorf("Abs()[%d] = %v, want %v", i, dst[i], expected)
				}
			}
		})
	}
}

func TestAbs_Large(t *testing.T) {
	n := 100
	a := make([]complex64, n)
	for i := range n {
		a[i] = complex(float32(i+1), float32(i+2))
	}

	dst := make([]float32, n)
	Abs(dst, a)

	for i := range n {
		r := float64(real(a[i]))
		im := float64(imag(a[i]))
		expected := float32(math.Sqrt(r*r + im*im))
		if !floatClose(dst[i], expected) {
			t.Errorf("Abs_Large()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestAbsSq(t *testing.T) {
	tests := []struct {
		name string
		a    []complex64
	}{
		{"single_3_4", []complex64{3 + 4i}},    // 25
		{"pair", []complex64{3 + 4i, 5 + 12i}}, // 25, 169
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.a))
			AbsSq(dst, tt.a)
			for i := range dst {
				r := real(tt.a[i])
				im := imag(tt.a[i])
				expected := r*r + im*im
				if !floatClose(dst[i], expected) {
					t.Errorf("AbsSq()[%d] = %v, want %v", i, dst[i], expected)
				}
			}
		})
	}
}

func TestAbsSq_Large(t *testing.T) {
	n := 100
	a := make([]complex64, n)
	for i := range n {
		a[i] = complex(float32(i+1), float32(i+2))
	}

	dst := make([]float32, n)
	AbsSq(dst, a)

	for i := range n {
		r := real(a[i])
		im := imag(a[i])
		expected := r*r + im*im
		if !floatClose(dst[i], expected) {
			t.Errorf("AbsSq_Large()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestConj(t *testing.T) {
	tests := []struct {
		name string
		a    []complex64
		want []complex64
	}{
		{"empty", nil, nil},
		{"single", []complex64{1 + 2i}, []complex64{1 - 2i}},
		{"pair", []complex64{1 + 2i, 3 + 4i}, []complex64{1 - 2i, 3 - 4i}},
		{"pure_real", []complex64{5 + 0i}, []complex64{5 + 0i}},
		{"pure_imag", []complex64{0 + 5i}, []complex64{0 - 5i}},
		{"zero", []complex64{0 + 0i}, []complex64{0 + 0i}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.a) == 0 {
				dst := make([]complex64, 0)
				Conj(dst, tt.a)
				return
			}
			dst := make([]complex64, len(tt.a))
			Conj(dst, tt.a)
			for i := range dst {
				if !complexClose(dst[i], tt.want[i]) {
					t.Errorf("Conj()[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestConj_Large(t *testing.T) {
	n := 100
	a := make([]complex64, n)
	for i := range n {
		a[i] = complex(float32(i+1), float32(i+2))
	}

	dst := make([]complex64, n)
	Conj(dst, a)

	for i := range n {
		expected := complex(real(a[i]), -imag(a[i]))
		if !complexClose(dst[i], expected) {
			t.Errorf("Conj_Large()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestFromReal(t *testing.T) {
	tests := []struct {
		name string
		src  []float32
		want []complex64
	}{
		{"empty", nil, nil},
		{"single", []float32{1}, []complex64{1 + 0i}},
		{"pair", []float32{1, 2}, []complex64{1 + 0i, 2 + 0i}},
		{"negative", []float32{-1, -2}, []complex64{-1 + 0i, -2 + 0i}},
		{"zero", []float32{0, 0}, []complex64{0 + 0i, 0 + 0i}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.src) == 0 {
				dst := make([]complex64, 0)
				FromReal(dst, tt.src)
				return
			}
			dst := make([]complex64, len(tt.src))
			FromReal(dst, tt.src)
			for i := range dst {
				if !complexClose(dst[i], tt.want[i]) {
					t.Errorf("FromReal()[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestFromReal_Large(t *testing.T) {
	n := 100
	src := make([]float32, n)
	for i := range n {
		src[i] = float32(i + 1)
	}

	dst := make([]complex64, n)
	FromReal(dst, src)

	for i := range n {
		expected := complex(src[i], 0)
		if !complexClose(dst[i], expected) {
			t.Errorf("FromReal_Large()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

// ============================================================================
// Benchmarks
// ============================================================================

func benchmarkBinaryOp(b *testing.B, size int, simdFn, goFn func(dst, a, bb []complex64)) {
	b.Helper()
	a := make([]complex64, size)
	bb := make([]complex64, size)
	dst := make([]complex64, size)
	for i := range size {
		a[i] = complex(float32(i+1), float32(i+2))
		bb[i] = complex(float32(i+3), float32(i+4))
	}

	b.Run("SIMD", func(b *testing.B) {
		b.SetBytes(int64(size * 8 * 3)) // 3 slices, 8 bytes per complex64
		for i := 0; i < b.N; i++ {
			simdFn(dst, a, bb)
		}
	})

	b.Run("Go", func(b *testing.B) {
		b.SetBytes(int64(size * 8 * 3))
		for i := 0; i < b.N; i++ {
			goFn(dst, a, bb)
		}
	})
}

func BenchmarkMul(b *testing.B) {
	benchmarkBinaryOp(b, 1024, Mul, mulGo)
}

func BenchmarkMulConj(b *testing.B) {
	benchmarkBinaryOp(b, 1024, MulConj, mulConjGo)
}

func BenchmarkAdd(b *testing.B) {
	benchmarkBinaryOp(b, 1024, Add, addGo)
}

func BenchmarkScale(b *testing.B) {
	size := 1024
	a := make([]complex64, size)
	dst := make([]complex64, size)
	s := complex64(1.5 + 2.5i)
	for i := range size {
		a[i] = complex(float32(i+1), float32(i+2))
	}

	b.Run("SIMD", func(b *testing.B) {
		b.SetBytes(int64(size * 8 * 2))
		for i := 0; i < b.N; i++ {
			Scale(dst, a, s)
		}
	})

	b.Run("Go", func(b *testing.B) {
		b.SetBytes(int64(size * 8 * 2))
		for i := 0; i < b.N; i++ {
			scaleGo(dst, a, s)
		}
	})
}

func benchmarkUnaryAbsOp(b *testing.B, size int, simdFn, goFn func([]float32, []complex64)) {
	b.Helper()
	a := make([]complex64, size)
	dst := make([]float32, size)
	for i := range size {
		a[i] = complex(float32(i+1), float32(i+2))
	}

	b.Run("SIMD", func(b *testing.B) {
		b.SetBytes(int64(size * 8)) // Input: complex64 (8 bytes)
		for i := 0; i < b.N; i++ {
			simdFn(dst, a)
		}
	})

	b.Run("Go", func(b *testing.B) {
		b.SetBytes(int64(size * 8))
		for i := 0; i < b.N; i++ {
			goFn(dst, a)
		}
	})
}

func benchmarkUnaryConjOp(b *testing.B, size int, simdFn, goFn func([]complex64, []complex64)) {
	b.Helper()
	a := make([]complex64, size)
	dst := make([]complex64, size)
	for i := range size {
		a[i] = complex(float32(i+1), float32(i+2))
	}

	b.Run("SIMD", func(b *testing.B) {
		b.SetBytes(int64(size * 8 * 2))
		for i := 0; i < b.N; i++ {
			simdFn(dst, a)
		}
	})

	b.Run("Go", func(b *testing.B) {
		b.SetBytes(int64(size * 8 * 2))
		for i := 0; i < b.N; i++ {
			goFn(dst, a)
		}
	})
}

func BenchmarkAbs(b *testing.B) {
	benchmarkUnaryAbsOp(b, 1024, Abs, absGo)
}

func BenchmarkAbsSq(b *testing.B) {
	benchmarkUnaryAbsOp(b, 1024, AbsSq, absSqGo)
}

func BenchmarkConj(b *testing.B) {
	benchmarkUnaryConjOp(b, 1024, Conj, conjGo)
}

func BenchmarkFromReal(b *testing.B) {
	size := 1024
	src := make([]float32, size)
	dst := make([]complex64, size)
	for i := range size {
		src[i] = float32(i + 1)
	}

	b.Run("SIMD", func(b *testing.B) {
		b.SetBytes(int64(size * 4)) // Input: float32 (4 bytes)
		for i := 0; i < b.N; i++ {
			FromReal(dst, src)
		}
	})

	b.Run("Go", func(b *testing.B) {
		b.SetBytes(int64(size * 4))
		for i := 0; i < b.N; i++ {
			fromRealGo(dst, src)
		}
	})
}

// ============================================================================
// Tests for Go fallback implementations
// ============================================================================

func TestMulGo(t *testing.T) {
	a := []complex64{1 + 2i, 3 + 4i}
	b := []complex64{5 + 6i, 7 + 8i}
	dst := make([]complex64, 2)
	mulGo(dst, a, b)
	for i := range dst {
		expected := a[i] * b[i]
		if !complexClose(dst[i], expected) {
			t.Errorf("mulGo()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestMulConjGo(t *testing.T) {
	a := []complex64{1 + 2i, 3 + 4i}
	b := []complex64{5 + 6i, 7 + 8i}
	dst := make([]complex64, 2)
	mulConjGo(dst, a, b)
	for i := range dst {
		conjB := complex(real(b[i]), -imag(b[i]))
		expected := a[i] * conjB
		if !complexClose(dst[i], expected) {
			t.Errorf("mulConjGo()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestScaleGo(t *testing.T) {
	a := []complex64{1 + 2i, 3 + 4i}
	s := complex64(2 + 1i)
	dst := make([]complex64, 2)
	scaleGo(dst, a, s)
	for i := range dst {
		expected := a[i] * s
		if !complexClose(dst[i], expected) {
			t.Errorf("scaleGo()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestAddGo(t *testing.T) {
	a := []complex64{1 + 2i, 3 + 4i}
	b := []complex64{5 + 6i, 7 + 8i}
	dst := make([]complex64, 2)
	addGo(dst, a, b)
	for i := range dst {
		expected := a[i] + b[i]
		if !complexClose(dst[i], expected) {
			t.Errorf("addGo()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestSubGo(t *testing.T) {
	a := []complex64{5 + 6i, 7 + 8i}
	b := []complex64{1 + 2i, 3 + 4i}
	dst := make([]complex64, 2)
	subGo(dst, a, b)
	for i := range dst {
		expected := a[i] - b[i]
		if !complexClose(dst[i], expected) {
			t.Errorf("subGo()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestAbsGo(t *testing.T) {
	a := []complex64{3 + 4i, 5 + 12i}
	dst := make([]float32, 2)
	absGo(dst, a)
	want := []float32{5, 13}
	for i := range dst {
		if !floatClose(dst[i], want[i]) {
			t.Errorf("absGo()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestAbsSqGo(t *testing.T) {
	a := []complex64{3 + 4i, 5 + 12i}
	dst := make([]float32, 2)
	absSqGo(dst, a)
	want := []float32{25, 169}
	for i := range dst {
		if !floatClose(dst[i], want[i]) {
			t.Errorf("absSqGo()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestConjGo(t *testing.T) {
	a := []complex64{1 + 2i, 3 + 4i}
	dst := make([]complex64, 2)
	conjGo(dst, a)
	want := []complex64{1 - 2i, 3 - 4i}
	for i := range dst {
		if !complexClose(dst[i], want[i]) {
			t.Errorf("conjGo()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestFromRealGo(t *testing.T) {
	src := []float32{1, 2, 3}
	dst := make([]complex64, 3)
	fromRealGo(dst, src)
	want := []complex64{1 + 0i, 2 + 0i, 3 + 0i}
	for i := range dst {
		if !complexClose(dst[i], want[i]) {
			t.Errorf("fromRealGo()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

// ============================================================================
// Tests for empty slice edge cases
// ============================================================================

func TestMulConj_Empty(_ *testing.T) {
	var a, b, dst []complex64
	MulConj(dst, a, b)
}

func TestScale_Empty(_ *testing.T) {
	var a, dst []complex64
	Scale(dst, a, 1+1i)
}

func TestAdd_Empty(_ *testing.T) {
	var a, b, dst []complex64
	Add(dst, a, b)
}

func TestSub_Empty(_ *testing.T) {
	var a, b, dst []complex64
	Sub(dst, a, b)
}

func TestAbs_Empty(_ *testing.T) {
	var a []complex64
	var dst []float32
	Abs(dst, a)
}

func TestAbsSq_Empty(_ *testing.T) {
	var a []complex64
	var dst []float32
	AbsSq(dst, a)
}

func TestMinLen(t *testing.T) {
	tests := []struct {
		a, b, c int
		want    int
	}{
		{1, 2, 3, 1},
		{3, 2, 1, 1},
		{2, 1, 3, 1},
		{5, 5, 5, 5},
		{0, 1, 2, 0},
	}
	for _, tt := range tests {
		got := minLen(tt.a, tt.b, tt.c)
		if got != tt.want {
			t.Errorf("minLen(%d, %d, %d) = %d, want %d", tt.a, tt.b, tt.c, got, tt.want)
		}
	}
}
