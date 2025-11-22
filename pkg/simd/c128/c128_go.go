package c128

// Pure Go implementations - used as fallback on all architectures

// mulGo computes element-wise complex multiplication.
// (a + bi)(c + di) = (ac - bd) + (ad + bc)i
func mulGo(dst, a, b []complex128) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

// mulConjGo computes element-wise multiplication by conjugate.
// (a + bi)(c - di) = (ac + bd) + (bc - ad)i
func mulConjGo(dst, a, b []complex128) {
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
	for i := range dst {
		dst[i] = a[i] * s
	}
}

// addGo computes element-wise complex addition.
func addGo(dst, a, b []complex128) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

// subGo computes element-wise complex subtraction.
func subGo(dst, a, b []complex128) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}
