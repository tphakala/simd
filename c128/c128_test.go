package c128

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/tphakala/simd/cpu"
)

const epsilon = 1e-10

func complexClose(a, b complex128) bool {
	return math.Abs(real(a)-real(b)) < epsilon && math.Abs(imag(a)-imag(b)) < epsilon
}

func TestMul(t *testing.T) {
	t.Logf("CPU: %s", cpu.Info())

	tests := []struct {
		name string
		a, b []complex128
		want []complex128
	}{
		{"empty", nil, nil, nil},
		{"single", []complex128{1 + 2i}, []complex128{3 + 4i}, []complex128{-5 + 10i}},
		{"two", []complex128{1 + 2i, 3 + 4i}, []complex128{5 + 6i, 7 + 8i}, []complex128{-7 + 16i, -11 + 52i}},
		{"pure_real", []complex128{2 + 0i, 3 + 0i}, []complex128{4 + 0i, 5 + 0i}, []complex128{8 + 0i, 15 + 0i}},
		{"pure_imag", []complex128{0 + 2i, 0 + 3i}, []complex128{0 + 4i, 0 + 5i}, []complex128{-8 + 0i, -15 + 0i}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.a) == 0 {
				dst := make([]complex128, 0)
				Mul(dst, tt.a, tt.b)
				return
			}
			dst := make([]complex128, len(tt.a))
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
	a := make([]complex128, n)
	b := make([]complex128, n)
	for i := range n {
		a[i] = complex(float64(i+1), float64(i+2))
		b[i] = complex(float64(i+3), float64(i+4))
	}

	dst := make([]complex128, n)
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
		a, b []complex128
	}{
		{"single", []complex128{1 + 2i}, []complex128{3 + 4i}},
		{"two", []complex128{1 + 2i, 3 + 4i}, []complex128{5 + 6i, 7 + 8i}},
		{"mixed", []complex128{-1 + 2i, 3 - 4i, 5 + 0i}, []complex128{1 + 1i, 2 - 2i, 0 + 3i}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]complex128, len(tt.a))
			MulConj(dst, tt.a, tt.b)
			for i := range dst {
				expected := tt.a[i] * cmplx.Conj(tt.b[i])
				if !complexClose(dst[i], expected) {
					t.Errorf("MulConj()[%d] = %v, want %v", i, dst[i], expected)
				}
			}
		})
	}
}

func TestMulConj_Large(t *testing.T) {
	n := 100
	a := make([]complex128, n)
	b := make([]complex128, n)
	for i := range n {
		a[i] = complex(float64(i+1), float64(i+2))
		b[i] = complex(float64(i+3), float64(i+4))
	}

	dst := make([]complex128, n)
	MulConj(dst, a, b)

	for i := range n {
		expected := a[i] * cmplx.Conj(b[i])
		if !complexClose(dst[i], expected) {
			t.Errorf("MulConj_Large()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestScale(t *testing.T) {
	tests := []struct {
		name string
		a    []complex128
		s    complex128
	}{
		{"single", []complex128{1 + 2i}, 3 + 4i},
		{"two", []complex128{1 + 2i, 3 + 4i}, 2 + 0i},
		{"by_i", []complex128{1 + 0i, 0 + 1i, 1 + 1i}, 0 + 1i},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]complex128, len(tt.a))
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
	a := []complex128{1 + 2i, 3 + 4i, 5 + 6i}
	b := []complex128{7 + 8i, 9 + 10i, 11 + 12i}
	dst := make([]complex128, len(a))

	Add(dst, a, b)

	for i := range dst {
		expected := a[i] + b[i]
		if !complexClose(dst[i], expected) {
			t.Errorf("Add()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestSub(t *testing.T) {
	a := []complex128{10 + 20i, 30 + 40i, 50 + 60i}
	b := []complex128{1 + 2i, 3 + 4i, 5 + 6i}
	dst := make([]complex128, len(a))

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
		t.Run("", func(t *testing.T) {
			a := make([]complex128, size)
			b := make([]complex128, size)
			for i := range size {
				a[i] = complex(float64(i+1), float64(i+2))
				b[i] = complex(float64(i+3), float64(i+4))
			}

			dst := make([]complex128, size)
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

// Benchmarks

// benchmarkBinaryOp benchmarks a binary complex128 operation (SIMD vs Go).
func benchmarkBinaryOp(b *testing.B, size int, simdFn, goFn func(dst, a, bb []complex128)) {
	b.Helper()
	a := make([]complex128, size)
	bb := make([]complex128, size)
	dst := make([]complex128, size)
	for i := range size {
		a[i] = complex(float64(i+1), float64(i+2))
		bb[i] = complex(float64(i+3), float64(i+4))
	}

	b.Run("SIMD", func(b *testing.B) {
		b.SetBytes(int64(size * 16 * 3)) // 3 slices, 16 bytes per complex128
		for i := 0; i < b.N; i++ {
			simdFn(dst, a, bb)
		}
	})

	b.Run("Go", func(b *testing.B) {
		b.SetBytes(int64(size * 16 * 3))
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
	a := make([]complex128, size)
	dst := make([]complex128, size)
	s := complex(1.5, 2.5)
	for i := range size {
		a[i] = complex(float64(i+1), float64(i+2))
	}

	b.Run("SIMD", func(b *testing.B) {
		b.SetBytes(int64(size * 16 * 2))
		for i := 0; i < b.N; i++ {
			Scale(dst, a, s)
		}
	})

	b.Run("Go", func(b *testing.B) {
		b.SetBytes(int64(size * 16 * 2))
		for i := 0; i < b.N; i++ {
			scaleGo(dst, a, s)
		}
	})
}

// ============================================================================
// Tests for new functions: Abs, AbsSq, Conj
// ============================================================================

func TestAbs(t *testing.T) {
	t.Logf("CPU: %s", cpu.Info())

	tests := []struct {
		name string
		a    []complex128
	}{
		{"single_1_0", []complex128{1 + 0i}},
		{"single_3_4", []complex128{3 + 4i}},
		{"pair", []complex128{3 + 4i, 5 + 12i}},
		{"pure_real", []complex128{5 + 0i, 10 + 0i}},
		{"pure_imag", []complex128{0 + 5i, 0 + 10i}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			Abs(dst, tt.a)
			for i := range dst {
				expected := cmplx.Abs(tt.a[i])
				if math.Abs(dst[i]-expected) > epsilon {
					t.Errorf("Abs()[%d] = %v, want %v", i, dst[i], expected)
				}
			}
		})
	}
}

func TestAbs_Large(t *testing.T) {
	n := 100
	a := make([]complex128, n)
	for i := range n {
		a[i] = complex(float64(i+1), float64(i+2))
	}

	dst := make([]float64, n)
	Abs(dst, a)

	for i := range n {
		expected := cmplx.Abs(a[i])
		if math.Abs(dst[i]-expected) > epsilon {
			t.Errorf("Abs_Large()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestAbsSq(t *testing.T) {
	tests := []struct {
		name string
		a    []complex128
	}{
		{"single_3_4", []complex128{3 + 4i}},
		{"pair", []complex128{3 + 4i, 5 + 12i}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.a))
			AbsSq(dst, tt.a)
			for i := range dst {
				r := real(tt.a[i])
				im := imag(tt.a[i])
				expected := r*r + im*im
				if math.Abs(dst[i]-expected) > epsilon {
					t.Errorf("AbsSq()[%d] = %v, want %v", i, dst[i], expected)
				}
			}
		})
	}
}

func TestAbsSq_Large(t *testing.T) {
	n := 100
	a := make([]complex128, n)
	for i := range n {
		a[i] = complex(float64(i+1), float64(i+2))
	}

	dst := make([]float64, n)
	AbsSq(dst, a)

	for i := range n {
		r := real(a[i])
		im := imag(a[i])
		expected := r*r + im*im
		if math.Abs(dst[i]-expected) > epsilon {
			t.Errorf("AbsSq_Large()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestConj(t *testing.T) {
	tests := []struct {
		name string
		a    []complex128
		want []complex128
	}{
		{"empty", nil, nil},
		{"single", []complex128{1 + 2i}, []complex128{1 - 2i}},
		{"pair", []complex128{1 + 2i, 3 + 4i}, []complex128{1 - 2i, 3 - 4i}},
		{"pure_real", []complex128{5 + 0i}, []complex128{5 + 0i}},
		{"pure_imag", []complex128{0 + 5i}, []complex128{0 - 5i}},
		{"zero", []complex128{0 + 0i}, []complex128{0 + 0i}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.a) == 0 {
				dst := make([]complex128, 0)
				Conj(dst, tt.a)
				return
			}
			dst := make([]complex128, len(tt.a))
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
	a := make([]complex128, n)
	for i := range n {
		a[i] = complex(float64(i+1), float64(i+2))
	}

	dst := make([]complex128, n)
	Conj(dst, a)

	for i := range n {
		expected := cmplx.Conj(a[i])
		if !complexClose(dst[i], expected) {
			t.Errorf("Conj_Large()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

// Benchmarks for new functions

func benchmarkUnaryAbsOp(b *testing.B, size int, simdFn, goFn func([]float64, []complex128)) {
	b.Helper()
	a := make([]complex128, size)
	dst := make([]float64, size)
	for i := range size {
		a[i] = complex(float64(i+1), float64(i+2))
	}

	b.Run("SIMD", func(b *testing.B) {
		b.SetBytes(int64(size * 16)) // Input: complex128 (16 bytes), Output: float64 (8 bytes)
		for i := 0; i < b.N; i++ {
			simdFn(dst, a)
		}
	})

	b.Run("Go", func(b *testing.B) {
		b.SetBytes(int64(size * 16))
		for i := 0; i < b.N; i++ {
			goFn(dst, a)
		}
	})
}

func benchmarkUnaryConjOp(b *testing.B, size int, simdFn, goFn func([]complex128, []complex128)) {
	b.Helper()
	a := make([]complex128, size)
	dst := make([]complex128, size)
	for i := range size {
		a[i] = complex(float64(i+1), float64(i+2))
	}

	b.Run("SIMD", func(b *testing.B) {
		b.SetBytes(int64(size * 16 * 2))
		for i := 0; i < b.N; i++ {
			simdFn(dst, a)
		}
	})

	b.Run("Go", func(b *testing.B) {
		b.SetBytes(int64(size * 16 * 2))
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

// ============================================================================
// Tests for Go fallback implementations
// ============================================================================

func TestMulGo(t *testing.T) {
	a := []complex128{1 + 2i, 3 + 4i}
	b := []complex128{5 + 6i, 7 + 8i}
	dst := make([]complex128, 2)
	mulGo(dst, a, b)
	for i := range dst {
		expected := a[i] * b[i]
		if !complexClose(dst[i], expected) {
			t.Errorf("mulGo()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestMulConjGo(t *testing.T) {
	a := []complex128{1 + 2i, 3 + 4i}
	b := []complex128{5 + 6i, 7 + 8i}
	dst := make([]complex128, 2)
	mulConjGo(dst, a, b)
	for i := range dst {
		expected := a[i] * cmplx.Conj(b[i])
		if !complexClose(dst[i], expected) {
			t.Errorf("mulConjGo()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestScaleGo(t *testing.T) {
	a := []complex128{1 + 2i, 3 + 4i}
	s := complex128(2 + 1i)
	dst := make([]complex128, 2)
	scaleGo(dst, a, s)
	for i := range dst {
		expected := a[i] * s
		if !complexClose(dst[i], expected) {
			t.Errorf("scaleGo()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestAddGo(t *testing.T) {
	a := []complex128{1 + 2i, 3 + 4i}
	b := []complex128{5 + 6i, 7 + 8i}
	dst := make([]complex128, 2)
	addGo(dst, a, b)
	for i := range dst {
		expected := a[i] + b[i]
		if !complexClose(dst[i], expected) {
			t.Errorf("addGo()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestSubGo(t *testing.T) {
	a := []complex128{5 + 6i, 7 + 8i}
	b := []complex128{1 + 2i, 3 + 4i}
	dst := make([]complex128, 2)
	subGo(dst, a, b)
	for i := range dst {
		expected := a[i] - b[i]
		if !complexClose(dst[i], expected) {
			t.Errorf("subGo()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

func TestAbsGo(t *testing.T) {
	a := []complex128{3 + 4i, 5 + 12i}
	dst := make([]float64, 2)
	absGo(dst, a)
	want := []float64{5, 13}
	for i := range dst {
		if math.Abs(dst[i]-want[i]) > epsilon {
			t.Errorf("absGo()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestAbsSqGo(t *testing.T) {
	a := []complex128{3 + 4i, 5 + 12i}
	dst := make([]float64, 2)
	absSqGo(dst, a)
	want := []float64{25, 169}
	for i := range dst {
		if math.Abs(dst[i]-want[i]) > epsilon {
			t.Errorf("absSqGo()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestConjGo(t *testing.T) {
	a := []complex128{1 + 2i, 3 + 4i}
	dst := make([]complex128, 2)
	conjGo(dst, a)
	want := []complex128{1 - 2i, 3 - 4i}
	for i := range dst {
		if !complexClose(dst[i], want[i]) {
			t.Errorf("conjGo()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

// ============================================================================
// Tests for empty slice edge cases
// ============================================================================

func TestMulConj_Empty(_ *testing.T) {
	var a, b, dst []complex128
	MulConj(dst, a, b)
}

func TestScale_Empty(_ *testing.T) {
	var a, dst []complex128
	Scale(dst, a, 1+1i)
}

func TestAdd_Empty(_ *testing.T) {
	var a, b, dst []complex128
	Add(dst, a, b)
}

func TestSub_Empty(_ *testing.T) {
	var a, b, dst []complex128
	Sub(dst, a, b)
}

func TestAbs_Empty(_ *testing.T) {
	var a []complex128
	var dst []float64
	Abs(dst, a)
}

func TestAbsSq_Empty(_ *testing.T) {
	var a []complex128
	var dst []float64
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
