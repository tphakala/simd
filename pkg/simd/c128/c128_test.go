package c128

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/tphakala/simd/pkg/simd/cpu"
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
func BenchmarkMul(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		a := make([]complex128, size)
		bb := make([]complex128, size)
		dst := make([]complex128, size)
		for i := range size {
			a[i] = complex(float64(i+1), float64(i+2))
			bb[i] = complex(float64(i+3), float64(i+4))
		}

		b.Run("SIMD_"+string(rune('0'+size/1000))+"k", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Mul(dst, a, bb)
			}
			b.SetBytes(int64(size * 16 * 3)) // 3 slices, 16 bytes per complex128
		})

		b.Run("Go_"+string(rune('0'+size/1000))+"k", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				mulGo(dst, a, bb)
			}
			b.SetBytes(int64(size * 16 * 3))
		})
	}
}

func BenchmarkMulConj(b *testing.B) {
	size := 1024
	a := make([]complex128, size)
	bb := make([]complex128, size)
	dst := make([]complex128, size)
	for i := range size {
		a[i] = complex(float64(i+1), float64(i+2))
		bb[i] = complex(float64(i+3), float64(i+4))
	}

	b.Run("SIMD", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			MulConj(dst, a, bb)
		}
		b.SetBytes(int64(size * 16 * 3))
	})

	b.Run("Go", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			mulConjGo(dst, a, bb)
		}
		b.SetBytes(int64(size * 16 * 3))
	})
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
		for i := 0; i < b.N; i++ {
			Scale(dst, a, s)
		}
		b.SetBytes(int64(size * 16 * 2))
	})

	b.Run("Go", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			scaleGo(dst, a, s)
		}
		b.SetBytes(int64(size * 16 * 2))
	})
}

func BenchmarkAdd(b *testing.B) {
	size := 1024
	a := make([]complex128, size)
	bb := make([]complex128, size)
	dst := make([]complex128, size)
	for i := range size {
		a[i] = complex(float64(i+1), float64(i+2))
		bb[i] = complex(float64(i+3), float64(i+4))
	}

	b.Run("SIMD", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			Add(dst, a, bb)
		}
		b.SetBytes(int64(size * 16 * 3))
	})

	b.Run("Go", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			addGo(dst, a, bb)
		}
		b.SetBytes(int64(size * 16 * 3))
	})
}
