// Package c64 provides SIMD-accelerated operations on complex64 slices.
//
// These operations are designed to accelerate FFT-based DSP pipelines
// using float32 precision for higher throughput:
// - Mul: Complex multiplication for frequency-domain convolution
// - MulConj: Multiply by conjugate for correlation
// - Scale: Scale by complex scalar
// - Add/Sub: Complex addition/subtraction for FFT butterflies
// - AbsSq: Magnitude squared for power spectrum computation
// - FromReal: Convert real float32 to complex64
//
// Complex64 uses float32 internally, providing 2x the throughput of complex128
// operations on SIMD registers (8 complex64 per AVX-512 vs 4 complex128).
//
// All functions automatically select the optimal implementation based on
// runtime CPU feature detection. Functions gracefully fall back to pure Go
// implementations on unsupported architectures.
//
// Thread Safety: All functions are safe for concurrent use.
// Memory: All functions are zero-allocation (no heap allocations).
package c64

// Mul computes element-wise complex multiplication: dst[i] = a[i] * b[i].
// Processes min(len(dst), len(a), len(b)) elements.
//
// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
//
// This is the core operation for FFT-based convolution in frequency domain.
func Mul(dst, a, b []complex64) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	mul64(dst[:n], a[:n], b[:n])
}

// MulConj computes element-wise multiplication by conjugate: dst[i] = a[i] * conj(b[i]).
// Processes min(len(dst), len(a), len(b)) elements.
//
// MulConj: (a + bi)(c - di) = (ac + bd) + (bc - ad)i
//
// This is used for cross-correlation in frequency domain.
func MulConj(dst, a, b []complex64) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	mulConj64(dst[:n], a[:n], b[:n])
}

// Scale multiplies each element by a complex scalar: dst[i] = a[i] * s.
// Processes min(len(dst), len(a)) elements.
func Scale(dst, a []complex64, s complex64) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	scale64(dst[:n], a[:n], s)
}

// Add computes element-wise complex addition: dst[i] = a[i] + b[i].
// Processes min(len(dst), len(a), len(b)) elements.
//
// Essential for FFT butterfly operations.
func Add(dst, a, b []complex64) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	add64(dst[:n], a[:n], b[:n])
}

// Sub computes element-wise complex subtraction: dst[i] = a[i] - b[i].
// Processes min(len(dst), len(a), len(b)) elements.
//
// Essential for FFT butterfly operations.
func Sub(dst, a, b []complex64) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	sub64(dst[:n], a[:n], b[:n])
}

// Abs computes element-wise complex magnitude: dst[i] = |a[i]|.
// Processes min(len(dst), len(a)) elements.
//
// Magnitude: |a + bi| = sqrt(a^2 + b^2)
func Abs(dst []float32, a []complex64) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	abs64(dst[:n], a[:n])
}

// AbsSq computes element-wise magnitude squared: dst[i] = |a[i]|^2.
// Processes min(len(dst), len(a)) elements.
//
// Magnitude squared: |a + bi|^2 = a^2 + b^2
//
// This is faster than Abs (no sqrt) and is the core operation for
// power spectrum computation in spectrograms.
func AbsSq(dst []float32, a []complex64) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	absSq64(dst[:n], a[:n])
}

// Conj computes element-wise complex conjugate: dst[i] = conj(a[i]).
// Processes min(len(dst), len(a)) elements.
//
// Conjugate: conj(a + bi) = a - bi
func Conj(dst, a []complex64) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	conj64(dst[:n], a[:n])
}

// FromReal converts real float32 values to complex64: dst[i] = complex(src[i], 0).
// Processes min(len(dst), len(src)) elements.
//
// This is used to convert real-valued signals to complex for FFT input.
func FromReal(dst []complex64, src []float32) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	fromReal64(dst[:n], src[:n])
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
