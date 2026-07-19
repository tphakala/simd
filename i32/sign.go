package i32

// Sign-directed integer transforms on int32 slices.
//
// NegWhereNeg conditionally negates each int32 magnitude according to the sign
// bit of a parallel float32 stream. It is the integer counterpart of applying a
// sign: a decoder that carries magnitudes as int32 and their signs as float32
// (for example a reconstructed spectrum whose signs live in a float mask) folds
// the two back together in one pass. Like Abs it wraps in int32 rather than
// saturating, so the SIMD and pure-Go paths are bit-identical across the full
// int32 range and there is no relaxed tier.

// NegWhereNeg writes dst[i] = -mag[i] when sign[i] has its IEEE-754 sign bit set
// and dst[i] = mag[i] otherwise, for i in [0, n), n = min(len(dst), len(mag),
// len(sign)). Any trailing capacity in dst past n is left untouched.
//
// The predicate is purely the sign bit of sign[i], so a negative zero (-0.0),
// -Inf and every negative NaN all negate, while +0.0 and positive values do
// not; the float32 magnitude itself is irrelevant. The negation wraps in int32
// exactly as [Abs] does: -MinInt32 does not fit int32, so a MinInt32 magnitude
// under a negative sign maps back to MinInt32. Every result is bit-identical to
// the pure-Go reference on all backends.
//
// dst may alias mag exactly (element for element): each lane reads mag[i] before
// its own dst[i] store and the forward iteration never revisits a written lane,
// so the magnitudes can be conditionally negated in place. dst must not otherwise
// overlap mag or sign. The SIMD kernels load a whole block of sign (and mag)
// before storing the block of dst, so a shifted dst/sign or dst/mag overlay could
// clobber input lanes a later iteration has not read yet.
func NegWhereNeg(dst, mag []int32, sign []float32) {
	n := min(len(dst), len(mag), len(sign))
	if n == 0 {
		return
	}
	negWhereNegI32(dst[:n], mag[:n], sign[:n])
}
