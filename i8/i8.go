// Package i8 provides SIMD-accelerated operations on int8 slices.
//
// int8 is the 8-bit signed integer workhorse of quantized numeric pipelines.
// Its narrow range (-128..127) makes element-wise arithmetic overflow almost
// immediately, so this package does not mirror the wrapping arithmetic of the
// i16/i32 packages one-to-one. Instead it ships the operations that are
// genuinely high-impact and well-defined at 8-bit width:
//
//   - Saturating arithmetic (AddSaturate, SubSaturate): single hardware
//     instructions (PADDSB/PSUBSB, SQADD/SQSUB) that clamp to [-128, 127]
//     instead of wrapping, which is what 8-bit arithmetic almost always wants.
//   - int32-accumulated reductions (Sum, DotProduct): widen to int32 so the
//     running total has headroom. DotProduct is the inner loop of quantized
//     matmul/conv; it uses ARM64 SDOT (FEAT_DotProd) where available and AVX2
//     VPMADDWD otherwise.
//   - Signed min/max (MinMax).
//   - Sign-extending widening (ToInt16, ToInt32) to hand off to the wider
//     integer or float packages.
//
// Sum and DotProduct accumulate in int32 with two's-complement wraparound,
// exactly like their pure-Go references. int32 wrapping addition is associative
// and commutative modulo 2^32, so the lane-parallel SIMD reductions are
// bit-identical to the scalar reference regardless of summation order. The
// intermediate products never overflow their SIMD lane (|int8 * int8| <= 16384),
// so only the final running total can wrap, and it wraps identically.
//
// All functions automatically select the optimal implementation based on
// runtime CPU feature detection and fall back to a pure-Go implementation on
// unsupported architectures.
//
// Thread Safety: All functions are safe for concurrent use.
// Memory: All functions are zero-allocation (no heap allocations).
package i8

// AddSaturate writes dst[i] = clamp(int(a[i]) + int(b[i]), -128, 127) for i in
// [0, n), n = min(len(dst), len(a), len(b)). The add saturates to the int8
// range instead of wrapping, so 100 + 100 = 127 and -100 + -100 = -128. Any
// trailing capacity in dst is left untouched.
func AddSaturate(dst, a, b []int8) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	addSatI8(dst[:n], a[:n], b[:n])
}

// SubSaturate writes dst[i] = clamp(int(a[i]) - int(b[i]), -128, 127) for i in
// [0, n), n = min(len(dst), len(a), len(b)). The subtract saturates to the int8
// range instead of wrapping. Any trailing capacity in dst is left untouched.
func SubSaturate(dst, a, b []int8) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	subSatI8(dst[:n], a[:n], b[:n])
}

// ToInt16 sign-extends src into dst: dst[i] = int16(src[i]) for i in [0, n),
// n = min(len(dst), len(src)). It is exact (int8 fits in int16). Any trailing
// capacity in dst is left untouched.
func ToInt16(dst []int16, src []int8) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	toI16(dst[:n], src[:n])
}

// ToInt32 sign-extends src into dst: dst[i] = int32(src[i]) for i in [0, n),
// n = min(len(dst), len(src)). It is exact (int8 fits in int32). Any trailing
// capacity in dst is left untouched.
func ToInt32(dst []int32, src []int8) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	toI32(dst[:n], src[:n])
}

// Sum returns the sum of all elements of a, accumulated in int32 with
// two's-complement wraparound. An empty a returns 0. a is read-only; the call
// allocates nothing.
func Sum(a []int8) int32 {
	if len(a) == 0 {
		return 0
	}
	return sumI8(a)
}

// DotProduct returns sum_i int32(a[i]) * int32(b[i]) over i in [0, n),
// n = min(len(a), len(b)), accumulated in int32 with two's-complement
// wraparound. An empty operand returns 0. a and b are read-only; the call
// allocates nothing.
//
// This is the inner loop of quantized matmul/convolution. On ARM64 with
// FEAT_DotProd it uses SDOT (16 int8 multiply-accumulates per instruction); on
// AVX2 it widens with VPMOVSXBW and reduces with VPMADDWD.
func DotProduct(a, b []int8) int32 {
	n := min(len(a), len(b))
	if n == 0 {
		return 0
	}
	return dotI8(a[:n], b[:n])
}

// MinMax returns the smallest and largest int8 in a:
//
//	minVal = min_i a[i],  maxVal = max_i a[i]
//
// Both are signed comparisons. An empty a returns (0, 0). a is read-only; the
// call allocates nothing.
func MinMax(a []int8) (minVal, maxVal int8) {
	if len(a) == 0 {
		return 0, 0
	}
	return minMaxI8(a)
}
