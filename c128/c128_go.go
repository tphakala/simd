package c128

import "math"

// Pure Go implementations - used as fallback on all architectures.
// Each fallback explicitly bounds-checks input slices once at entry; this lets
// the compiler hoist per-iteration bounds checks out of the hot loop.

// mulGo computes element-wise complex multiplication.
// (a + bi)(c + di) = (ac - bd) + (ad + bc)i
func mulGo(dst, a, b []complex128) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

// mulConjGo computes element-wise multiplication by conjugate.
// (a + bi)(c - di) = (ac + bd) + (bc - ad)i
func mulConjGo(dst, a, b []complex128) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		ar, ai := real(a[i]), imag(a[i])
		br, bi := real(b[i]), imag(b[i])
		// conj(b) = br - bi*i
		// a * conj(b) = (ar + ai*i)(br - bi*i)
		//             = ar*br + ai*bi + (ai*br - ar*bi)*i
		dst[i] = complex(ar*br+ai*bi, ai*br-ar*bi)
	}
}

// scaleGo multiplies each element by a complex scalar.
func scaleGo(dst, a []complex128, s complex128) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] * s
	}
}

// addGo computes element-wise complex addition.
func addGo(dst, a, b []complex128) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

// subGo computes element-wise complex subtraction.
func subGo(dst, a, b []complex128) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

// absGo computes element-wise complex magnitude: |a + bi| = sqrt(a² + b²).
func absGo(dst []float64, a []complex128) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		r := real(a[i])
		im := imag(a[i])
		dst[i] = math.Hypot(r, im)
	}
}

// absSqGo computes element-wise magnitude squared: |a + bi|² = a² + b².
func absSqGo(dst []float64, a []complex128) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		r := real(a[i])
		im := imag(a[i])
		dst[i] = r*r + im*im
	}
}

// conjGo computes element-wise complex conjugate: conj(a + bi) = a - bi.
func conjGo(dst, a []complex128) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = complex(real(a[i]), -imag(a[i]))
	}
}

// fromRealGo converts real float64 values to complex128: dst[i] = complex(src[i], 0).
// This is the trusted reference and the path that runs on architectures without
// SIMD.
func fromRealGo(dst []complex128, src []float64) {
	if len(dst) == 0 {
		return
	}
	_ = src[len(dst)-1]
	for i := range dst {
		dst[i] = complex(src[i], 0)
	}
}

// dotProductGo computes the complex dot product sum(a[i]*b[i]) over
// min(len(a), len(b)) elements, accumulating the real and imaginary parts in
// float64 (the native element precision).
func dotProductGo(a, b []complex128) complex128 {
	n := min(len(a), len(b))
	var sumRe, sumIm float64
	for i := range n {
		ar, ai := real(a[i]), imag(a[i])
		br, bi := real(b[i]), imag(b[i])
		// a*b = (ar*br - ai*bi) + (ar*bi + ai*br)i
		sumRe += ar*br - ai*bi
		sumIm += ar*bi + ai*br
	}
	return complex(sumRe, sumIm)
}

// dotProductConjGo computes the conjugated dot product sum(a[i]*conj(b[i])) over
// min(len(a), len(b)) elements, accumulating in float64. This is the standard
// Hermitian inner product used for correlation and matched filtering.
func dotProductConjGo(a, b []complex128) complex128 {
	n := min(len(a), len(b))
	var sumRe, sumIm float64
	for i := range n {
		ar, ai := real(a[i]), imag(a[i])
		br, bi := real(b[i]), imag(b[i])
		// a*conj(b) = (ar*br + ai*bi) + (ai*br - ar*bi)i
		sumRe += ar*br + ai*bi
		sumIm += ai*br - ar*bi
	}
	return complex(sumRe, sumIm)
}
