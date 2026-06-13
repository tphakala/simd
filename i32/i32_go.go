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

func addGo(dst, a, b []int32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func subGo(dst, a, b []int32) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

// minMaxGo returns the smallest and largest int32 in res via a single signed
// scan. res must be non-empty (the public MinMax guards the empty case); it is
// the bit-exact source of truth the SIMD MinMax kernels are validated against.
func minMaxGo(res []int32) (minVal, maxVal int32) {
	lo, hi := res[0], res[0]
	for _, r := range res[1:] {
		if r < lo {
			lo = r
		}
		if r > hi {
			hi = r
		}
	}
	return lo, hi
}
