package i32

import "math"

// float32SignBitPos is the IEEE-754 float32 sign bit position. An arithmetic
// right shift of the raw bits by this amount broadcasts the sign bit across a
// full int32 lane (all-ones when set, zero otherwise).
const float32SignBitPos = 31

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

// sumGo accumulates a into int32 with two's-complement wraparound. Wrapping
// addition is associative and commutative modulo 2^32, so the SIMD lane split
// and horizontal reduction are bit-identical to this sequential loop for every
// input; it is the source of truth the Sum kernels are validated against.
func sumGo(a []int32) int32 {
	var s int32
	for _, v := range a {
		s += v
	}
	return s
}

// absGo writes the wrapping absolute value: int32 negation wraps in Go, so
// abs(MinInt32) stays MinInt32, exactly what a 32-bit ABS/VPABSD lane
// computes.
func absGo(dst, a []int32) {
	for i := range dst {
		if a[i] < 0 {
			dst[i] = -a[i]
		} else {
			dst[i] = a[i]
		}
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

// negWhereNegGo writes the branchless conditional negate that is NegWhereNeg's
// source of truth. For each lane it broadcasts sign[i]'s IEEE-754 sign bit into
// a full-width int32 mask m via an arithmetic right shift (m = -1 when the sign
// bit is set, including -0.0, -Inf and -NaN; m = 0 otherwise) and computes
// (mag[i] ^ m) - m: that is -mag[i] when m = -1 and mag[i] when m = 0, with the
// two's-complement wrap of int32 negation so a MinInt32 magnitude maps back to
// MinInt32 (matching Abs). mag and sign are indexed in lockstep with dst, so
// dst may alias mag. dst may be empty; the len-0 guard below protects the
// len(dst)-1 BCE hints.
func negWhereNegGo(dst, mag []int32, sign []float32) {
	if len(dst) == 0 {
		return
	}
	_ = mag[len(dst)-1]  // BCE hint: mag is indexed [0, len(dst))
	_ = sign[len(dst)-1] // BCE hint: sign is indexed [0, len(dst))
	for i := range dst {
		m := int32(math.Float32bits(sign[i])) >> float32SignBitPos
		dst[i] = (mag[i] ^ m) - m
	}
}
