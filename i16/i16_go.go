package i16

// Pure-Go reference implementations.
//
// These are the source of truth for behavior: every SIMD kernel is validated
// for bit-exact parity against the functions here. They are compiled on every
// architecture and used directly as the fallback when no SIMD path applies.

func interleave2Go(dst, a, b []int16) {
	for i := range a {
		dst[i*2] = a[i]
		dst[i*2+1] = b[i]
	}
}

func deinterleave2Go(a, b, src []int16) {
	for i := range a {
		a[i] = src[i*2]
		b[i] = src[i*2+1]
	}
}

// dotGo computes the widening int16 dot product, accumulating each int32
// product into an int32 running sum with two's-complement wraparound. It
// clamps to min(len(a), len(b)) itself so the unsafe entry point can hand it
// mismatched slices, and it is the bit-exact source of truth the SIMD Dot
// kernels are validated against.
//
// Wrapping (rather than saturating) accumulation is the contract: wrapping
// addition is associative and commutative modulo 2^32, so the SIMD lane
// grouping and horizontal reduction produce bit-identical results to this
// sequential loop for every input, including forced overflow.
func dotGo(a, b []int16) int32 {
	var s int32
	for i := range min(len(a), len(b)) {
		s += int32(a[i]) * int32(b[i])
	}
	return s
}
