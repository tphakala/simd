// Package f32 provides SIMD-accelerated operations on float32 slices.
//
// All functions automatically select the optimal implementation based on
// runtime CPU feature detection. Functions gracefully fall back to pure Go
// implementations on unsupported architectures.
//
// Thread Safety: All functions are safe for concurrent use.
// Memory: All functions are zero-allocation (no heap allocations).
package f32

import "math"

// DotProduct computes the dot product of two float32 slices.
// Returns sum(a[i] * b[i]) for i in 0..min(len(a), len(b)).
//
// Uses AVX+FMA on AMD64 (8x float32), NEON on ARM64 (4x float32).
func DotProduct(a, b []float32) float32 {
	n := min(len(b), len(a))
	if n == 0 {
		return 0
	}
	return dotProduct(a[:n], b[:n])
}

// Add computes element-wise addition: dst[i] = a[i] + b[i].
func Add(dst, a, b []float32) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	add(dst[:n], a[:n], b[:n])
}

// Sub computes element-wise subtraction: dst[i] = a[i] - b[i].
func Sub(dst, a, b []float32) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	sub(dst[:n], a[:n], b[:n])
}

// Mul computes element-wise multiplication: dst[i] = a[i] * b[i].
func Mul(dst, a, b []float32) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	mul(dst[:n], a[:n], b[:n])
}

// Div computes element-wise division: dst[i] = a[i] / b[i].
func Div(dst, a, b []float32) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	div(dst[:n], a[:n], b[:n])
}

// Scale multiplies each element by a scalar: dst[i] = a[i] * s.
func Scale(dst, a []float32, s float32) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	scale(dst[:n], a[:n], s)
}

// AddScalar adds a scalar to each element: dst[i] = a[i] + s.
func AddScalar(dst, a []float32, s float32) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	addScalar(dst[:n], a[:n], s)
}

// Sum returns the sum of all elements.
func Sum(a []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return sum(a)
}

// Min returns the minimum value.
func Min(a []float32) float32 {
	if len(a) == 0 {
		return posInf
	}
	return min32(a)
}

// Max returns the maximum value.
func Max(a []float32) float32 {
	if len(a) == 0 {
		return negInf
	}
	return max32(a)
}

// Abs computes element-wise absolute value: dst[i] = |a[i]|.
func Abs(dst, a []float32) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	abs32(dst[:n], a[:n])
}

// Neg computes element-wise negation: dst[i] = -a[i].
func Neg(dst, a []float32) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	neg32(dst[:n], a[:n])
}

// FMA computes fused multiply-add: dst[i] = a[i] * b[i] + c[i].
func FMA(dst, a, b, c []float32) {
	n := min(len(c), minLen(len(dst), len(a), len(b)))
	if n == 0 {
		return
	}
	fma32(dst[:n], a[:n], b[:n], c[:n])
}

// Clamp clamps each element to [min, max].
func Clamp(dst, a []float32, minVal, maxVal float32) {
	n := min(len(a), len(dst))
	if n == 0 {
		return
	}
	clamp32(dst[:n], a[:n], minVal, maxVal)
}

func minLen(a, b, c int) int {
	if b < a {
		a = b
	}
	if c < a {
		a = c
	}
	return a
}

// DotProductBatch computes multiple dot products against the same vector.
// results[i] = DotProduct(rows[i], vec) for each row.
// This is more cache-efficient than calling DotProduct in a loop because
// vec stays hot in L1 cache across all dot products.
func DotProductBatch(results []float32, rows [][]float32, vec []float32) {
	n := min(len(results), len(rows))
	if n == 0 || len(vec) == 0 {
		return
	}
	dotProductBatch32(results[:n], rows[:n], vec)
}

// ConvolveValid computes valid convolution of signal with kernel.
// dst[i] = sum(signal[i+j] * kernel[j]) for j in 0..len(kernel)-1.
// Output length is len(signal) - len(kernel) + 1.
//
// This is equivalent to applying a FIR filter without zero-padding.
func ConvolveValid(dst, signal, kernel []float32) {
	if len(kernel) == 0 || len(signal) < len(kernel) {
		return
	}
	validLen := len(signal) - len(kernel) + 1
	n := min(len(dst), validLen)
	if n == 0 {
		return
	}
	convolveValid32(dst[:n], signal, kernel)
}

// AccumulateAdd adds src to dst starting at offset: dst[offset:offset+len(src)] += src.
// This is a key primitive for overlap-add in FFT-based convolution.
//
// Panics if offset+len(src) > len(dst) or if offset < 0.
func AccumulateAdd(dst, src []float32, offset int) {
	if offset < 0 {
		panic("simd: negative offset")
	}
	n := len(src)
	if n == 0 {
		return
	}
	if offset+n > len(dst) {
		panic("simd: offset+len(src) exceeds len(dst)")
	}
	accumulateAdd32(dst[offset:offset+n], src)
}

var (
	posInf = float32(math.Inf(1))
	negInf = float32(math.Inf(-1))
)
