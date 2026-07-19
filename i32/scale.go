package i32

// Fixed-point scale-by-scalar on int32 slices.
//
// ScaleQ31 and ScaleQ15 multiply every int32 sample by a single fixed-point
// coefficient and shift the product back into int32 range, the integer form of
// libopus MULT32_32_Q31 and MULT16_32_Q15. Both are truncating (an arithmetic
// right shift, no rounding) and wrap in int32 rather than saturating, so the
// SIMD and pure-Go paths are bit-identical across the full int32 range and there
// is no relaxed tier. The product is always formed in a 64-bit intermediate, so
// it never overflows before the shift; only the final int32() cast wraps.

// ScaleQ31 writes dst[i] = int32(int64(a[i]) * int64(k) >> 31) for i in [0, n),
// n = min(len(dst), len(a)). It is the integer MULT32_32_Q31: a and k are both
// interpreted as signed 32-bit values, multiplied into a 64-bit product, then
// arithmetically shifted right by 31 (truncating toward -inf, no rounding). Any
// trailing capacity in dst past n is left untouched.
//
// The result wraps in int32 rather than saturating. The extreme case
// a[i] = k = MinInt32 gives a product of 2^62, whose arithmetic shift by 31 is
// 2^31; that does not fit int32 and wraps to MinInt32, exactly as the pure-Go
// reference computes. Every result is bit-identical across all backends.
//
// dst may alias a exactly (element for element): each lane reads a[i] before its
// own dst[i] store and the forward iteration never revisits a written lane, so
// the samples can be scaled in place. dst must not otherwise overlap a: the SIMD
// kernels load a whole block of a before storing the block of dst, so a shifted
// dst/a overlay could clobber input lanes a later iteration has not read yet.
func ScaleQ31(dst, a []int32, k int32) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	scaleQ31I32(dst[:n], a[:n], k)
}

// ScaleQ15 writes dst[i] = int32(int64(k) * int64(a[i]) >> 15) for i in [0, n),
// n = min(len(dst), len(a)). It is the integer MULT16_32_Q15: the coefficient k
// is a signed 16-bit Q15 value, each a[i] a full signed int32 sample, and the
// 64-bit product is arithmetically shifted right by 15 (truncating toward -inf,
// no rounding). Any trailing capacity in dst past n is left untouched.
//
// The result wraps in int32 rather than saturating; the product |k * a[i]| is at
// most 2^46 so it never overflows the int64 intermediate, and only the final
// int32() cast can wrap. Every result is bit-identical across all backends.
//
// dst may alias a exactly (element for element), scaling the samples in place,
// under the same rule and caveat as [ScaleQ31].
func ScaleQ15(dst, a []int32, k int16) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	scaleQ15I32(dst[:n], a[:n], k)
}
