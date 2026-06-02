package i32

// Pure-Go reference implementations.
//
// These are the source of truth for behavior: every SIMD kernel is validated
// for bit-exact parity against the functions here. They are compiled on every
// architecture and used directly as the fallback when no SIMD path applies.

func interleave2Go(dst, a, b []int32) {
	for i := range a {
		dst[i*2] = a[i]
		dst[i*2+1] = b[i]
	}
}

func deinterleave2Go(a, b, src []int32) {
	for i := range a {
		a[i] = src[i*2]
		b[i] = src[i*2+1]
	}
}
