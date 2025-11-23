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

// Interleave2 interleaves two slices: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
// Processes min(len(a), len(b), len(dst)/2) pairs.
// This is useful for converting separate channels to interleaved stereo audio.
func Interleave2(dst, a, b []float32) {
	n := min(len(dst)/interleave2Channels, min(len(a), len(b)))
	if n == 0 {
		return
	}
	interleave2_32(dst[:n*interleave2Channels], a[:n], b[:n])
}

// Deinterleave2 deinterleaves a slice: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
// Processes min(len(a), len(b), len(src)/2) pairs.
// This is the inverse of Interleave2, useful for splitting stereo audio to channels.
func Deinterleave2(a, b, src []float32) {
	n := min(len(src)/interleave2Channels, min(len(a), len(b)))
	if n == 0 {
		return
	}
	deinterleave2_32(a[:n], b[:n], src[:n*interleave2Channels])
}

const interleave2Channels = 2

// ConvolveValidMulti applies multiple kernels to the same signal in one pass.
// dsts[k][i] = sum(signal[i+j] * kernels[k][j]) for each kernel k.
// All kernels must have the same length.
//
// This is more cache-efficient than calling ConvolveValid in a loop because
// the signal data stays hot in cache across all kernel applications.
//
// Panics if kernels have different lengths or if dsts/kernels lengths don't match.
func ConvolveValidMulti(dsts [][]float32, signal []float32, kernels [][]float32) {
	numKernels := len(kernels)
	if numKernels == 0 || len(dsts) < numKernels {
		return
	}

	// Validate all kernels have the same length
	kLen := len(kernels[0])
	if kLen == 0 || len(signal) < kLen {
		return
	}
	for i := 1; i < numKernels; i++ {
		if len(kernels[i]) != kLen {
			panic("simd: all kernels must have the same length")
		}
	}

	validLen := len(signal) - kLen + 1

	// Determine actual output length based on smallest dst
	n := validLen
	for i := range numKernels {
		if len(dsts[i]) < n {
			n = len(dsts[i])
		}
	}
	if n <= 0 {
		return
	}

	convolveValidMulti32(dsts, signal, kernels, n, kLen)
}
