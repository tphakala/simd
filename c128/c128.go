// Package c128 provides SIMD-accelerated operations on complex128 slices.
//
// These operations are designed to accelerate FFT-based convolution pipelines:
// - Mul: Complex multiplication for frequency-domain convolution
// - MulConj: Multiply by conjugate for correlation
// - Scale: Scale by complex scalar
// - FromReal: Convert real float64 to complex128 (FFT input preparation)
//
// All functions automatically select the optimal implementation based on
// runtime CPU feature detection. Functions gracefully fall back to pure Go
// implementations on unsupported architectures.
//
// Thread Safety: All functions are safe for concurrent use.
// Memory: All functions are zero-allocation (no heap allocations).
package c128

// Mul computes element-wise complex multiplication: dst[i] = a[i] * b[i].
// Processes min(len(dst), len(a), len(b)) elements.
//
// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
//
// This is the core operation for FFT-based convolution in frequency domain.
func Mul(dst, a, b []complex128) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	mul128(dst[:n], a[:n], b[:n])
}

// MulConj computes element-wise multiplication by conjugate: dst[i] = a[i] * conj(b[i]).
// Processes min(len(dst), len(a), len(b)) elements.
//
// MulConj: (a + bi)(c - di) = (ac + bd) + (bc - ad)i
//
// This is used for cross-correlation in frequency domain.
func MulConj(dst, a, b []complex128) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	mulConj128(dst[:n], a[:n], b[:n])
}

// Scale multiplies each element by a complex scalar: dst[i] = a[i] * s.
// Processes min(len(dst), len(a)) elements.
func Scale(dst, a []complex128, s complex128) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	scale128(dst[:n], a[:n], s)
}

// Add computes element-wise complex addition: dst[i] = a[i] + b[i].
// Processes min(len(dst), len(a), len(b)) elements.
func Add(dst, a, b []complex128) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	add128(dst[:n], a[:n], b[:n])
}

// Sub computes element-wise complex subtraction: dst[i] = a[i] - b[i].
// Processes min(len(dst), len(a), len(b)) elements.
func Sub(dst, a, b []complex128) {
	n := minLen(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	sub128(dst[:n], a[:n], b[:n])
}

// Abs computes element-wise complex magnitude: dst[i] = |a[i]|.
// Processes min(len(dst), len(a)) elements.
//
// Magnitude: |a + bi| = sqrt(a² + b²)
//
// This is the core operation for spectrograms and frequency-domain analysis.
func Abs(dst []float64, a []complex128) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	abs128(dst[:n], a[:n])
}

// AbsSq computes element-wise magnitude squared: dst[i] = |a[i]|².
// Processes min(len(dst), len(a)) elements.
//
// Magnitude squared: |a + bi|² = a² + b²
//
// This is faster than Abs (no sqrt) and useful for power spectrograms.
func AbsSq(dst []float64, a []complex128) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	absSq128(dst[:n], a[:n])
}

// Conj computes element-wise complex conjugate: dst[i] = conj(a[i]).
// Processes min(len(dst), len(a)) elements.
//
// Conjugate: conj(a + bi) = a - bi
func Conj(dst, a []complex128) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	conj128(dst[:n], a[:n])
}


// FromReal converts real float64 values to complex128: dst[i] = complex(src[i], 0).
// Processes min(len(dst), len(src)) elements.
//
// This is used to convert real-valued signals to complex for FFT input.
func FromReal(dst []complex128, src []float64) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	fromReal128(dst[:n], src[:n])
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
