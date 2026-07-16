package i16

// DotProduct returns sum_i int32(a[i]) * int32(b[i]) over i in [0, n),
// n = min(len(a), len(b)), accumulated in int32 with two's-complement
// wraparound. An empty operand returns 0. a and b are read-only; the call
// allocates nothing.
//
// The accumulator wraps rather than saturates, and that is a guarantee callers
// may rely on: wrapping addition is associative and commutative modulo 2^32,
// so every SIMD lane grouping and horizontal reduction below is bit-identical
// to the scalar loop for all inputs, including operands engineered to overflow.
// This is what makes the result reproducible across architectures, and it is
// the property fixed-point codecs (Opus/CELT, FLAC LPC) need to stay bit-exact.
// A saturating accumulator would not be associative and could not be vectorized
// this way.
//
// This is the inner loop of fixed-point correlation and filtering. On ARM64 it
// uses SMLAL/SMLAL2 (4 widening int16 multiply-accumulates per instruction); on
// AMD64 it uses PMADDWD, which is SSE2 and therefore always present on the
// GOAMD64=v1 baseline, with an AVX2 path at twice the width.
func DotProduct(a, b []int16) int32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	return dotI16(a, b)
}

// DotProductUnsafe computes the dot product without empty-slice checks.
// It skips the len==0 guard in [DotProduct] but is otherwise identical:
// the underlying SIMD kernels and Go fallback clamp to min(len(a), len(b))
// internally, so mismatched lengths do not cause out-of-bounds access.
//
// The precondition is deliberately conservative: the kernels and the Go
// reference all handle n==0 safely today, so an empty operand would in fact
// return 0 here. Requiring non-empty inputs keeps that an implementation
// detail rather than a promise.
//
// PRECONDITIONS (caller must ensure):
//   - len(a) > 0 && len(b) > 0
func DotProductUnsafe(a, b []int16) int32 {
	return dotI16(a, b)
}
