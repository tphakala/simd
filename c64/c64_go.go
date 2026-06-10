package c64

import "math"

// Pure Go implementations - used as fallback on all architectures.
// Each fallback bounds-checks input slices once at entry so the compiler can
// hoist per-iteration bounds checks out of the hot loop.

// mulGo computes element-wise complex multiplication.
// (a + bi)(c + di) = (ac - bd) + (ad + bc)i
func mulGo(dst, a, b []complex64) {
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
func mulConjGo(dst, a, b []complex64) {
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
func scaleGo(dst, a []complex64, s complex64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] * s
	}
}

// addGo computes element-wise complex addition.
func addGo(dst, a, b []complex64) {
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
func subGo(dst, a, b []complex64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

// absGo computes element-wise complex magnitude: |a + bi| = sqrt(a^2 + b^2).
func absGo(dst []float32, a []complex64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		r := float64(real(a[i]))
		im := float64(imag(a[i]))
		dst[i] = float32(math.Hypot(r, im))
	}
}

// absSqGo computes element-wise magnitude squared: |a + bi|^2 = a^2 + b^2.
func absSqGo(dst []float32, a []complex64) {
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
func conjGo(dst, a []complex64) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	for i := range dst {
		dst[i] = complex(real(a[i]), -imag(a[i]))
	}
}

// fromRealGo converts real float32 values to complex64.
func fromRealGo(dst []complex64, src []float32) {
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
// float32. Like f32.DotProduct, the float32 accumulation trades a little
// precision for throughput; widen to complex128 if the vectors are long and
// accuracy-critical.
func dotProductGo(a, b []complex64) complex64 {
	n := min(len(a), len(b))
	var sumRe, sumIm float32
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
// min(len(a), len(b)) elements, accumulating in float32. This is the standard
// Hermitian inner product used for correlation and matched filtering.
func dotProductConjGo(a, b []complex64) complex64 {
	n := min(len(a), len(b))
	var sumRe, sumIm float32
	for i := range n {
		ar, ai := real(a[i]), imag(a[i])
		br, bi := real(b[i]), imag(b[i])
		// a*conj(b) = (ar*br + ai*bi) + (ai*br - ar*bi)i
		sumRe += ar*br + ai*bi
		sumIm += ai*br - ar*bi
	}
	return complex(sumRe, sumIm)
}
