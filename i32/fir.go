package i32

// FIRValidQ15 writes the int32 valid convolution of x with the Q15 taps into
// dst, computing min(len(dst), len(x)-len(taps)+1) outputs in correlation
// orientation. It is the integer analog of the f32 ConvolveValidMaxAbs
// orientation and covers libopus combFilterConst.
//
// For output i it computes:
//
//	dst[i] = sum over j in [0, len(taps)) of int32(int64(taps[j]) * int64(x[i+j]) >> 15)
//
// which is a "valid" convolution in correlation (non-flipped) orientation:
// taps[j] multiplies x[i+j], not x[i+len(taps)-1-j]. The full valid output has
// len(x)-len(taps)+1 samples; the count is clamped to len(dst) and any trailing
// capacity in dst past that is left untouched.
//
// Each tap product is Q15-TRUNCATED before it is added: the 64-bit product
// int64(taps[j])*int64(x[i+j]) is arithmetically shifted right by 15 (toward
// -inf, no rounding constant), matching go-opus MULT16_32_Q15 bit-for-bit. The
// truncation is per product, NOT a single truncation of the final sum, so a
// fault that quantized only the final accumulator instead would diverge. The running accumulator is
// two's-complement wrapping int32: intermediate and final sums wrap rather than
// saturating, so the SIMD and pure-Go paths are bit-identical across the full
// int32 range and there is no relaxed tier. This variant is NON-saturating; a
// saturating variant (FIRValidQ15Sat) can follow for the comb-filter sigSat
// consumer that clamps to int16 range.
//
// Empty taps or an x shorter than taps produces no output and writes nothing
// (the valid-output count would otherwise underflow). When len(x) == len(taps)
// there is exactly one output.
//
// dst must NOT overlap x. The SIMD kernels load whole sliding windows of x ahead
// of the stores into dst, so an overlapping dst could clobber input lanes a later
// output has not yet read. x and taps are read-only; the call allocates nothing.
func FIRValidQ15(dst, x []int32, taps []int16) {
	// Guard first: an empty taps or an x shorter than taps makes the
	// len(x)-len(taps)+1 valid-output count underflow, so reject both before it
	// is computed.
	if len(taps) == 0 || len(x) < len(taps) {
		return
	}
	outLen := len(x) - len(taps) + 1
	n := min(len(dst), outLen)
	if n == 0 {
		return
	}
	// x and taps are passed whole: the kernel reads the sliding window x[i+j] up
	// to index n-1+len(taps)-1 <= len(x)-1, so x must not be pre-sliced to n.
	firValidQ15I32(dst[:n], x, taps)
}

// FIRSymValidQ15 writes the int32 valid convolution of x with a SYMMETRIC Q15 tap
// set: a center tap plus K = len(pairs) mirror pairs (window length 2K+1), in
// correlation orientation. For output i, with the center sample at c = i+K:
//
//	dst[i] = int32(int64(center) * int64(x[c]) >> 15)
//	       + sum over k in [1,K] of int32(int64(pairs[k-1]) * int64(x[c-k]+x[c+k]) >> 15)
//
// The full valid output has len(x)-2K samples; the count is clamped to len(dst)
// and any trailing capacity in dst past that is left untouched.
//
// The load-bearing difference from FIRValidQ15 is the mirror FOLD: the two mirror
// samples x[c-k] and x[c+k] are summed with a two's-complement WRAPPING int32 add
// BEFORE the single Q15 truncation, so each pair contributes exactly ONE
// truncation, not two. This is what makes the result bit-identical to libopus
// comb_filter_const_c (FIXED_POINT), which pre-adds each mirror pair with ADD32
// and then does one MULT16_32_Q15. Because trunc((p+q)>>15) != trunc(p>>15) +
// trunc(q>>15) in general, running FIRValidQ15 over a folded tap set is NOT a
// bit-exact comb filter, so this variant exists to fold the pairs the way the
// codec does. Each product's int64(coeff)*int64(sample) is arithmetically shifted
// right by 15 (toward -inf, no rounding constant) and the accumulator is wrapping
// int32; the SIMD and pure-Go paths are bit-identical and there is no relaxed
// tier. This variant is NON-saturating; a saturating companion
// (FIRSymValidQ15Sat) can follow for the comb-filter sigSat consumer.
//
// K == 0 (empty pairs) degenerates to a pure center scale over the whole window
// (window length 1, outLen = len(x)), equivalent to ScaleQ15 with k = center. An
// x shorter than the 2K+1 window produces no output and writes nothing; when
// len(x) == 2K+1 there is exactly one output.
//
// dst must NOT overlap x. The SIMD kernels load whole sliding windows of x ahead
// of the stores into dst, so an overlapping dst could clobber input lanes a later
// output has not yet read. x, center and pairs are read-only; the call allocates
// nothing.
func FIRSymValidQ15(dst, x []int32, center int16, pairs []int16) {
	// Guard first: an x shorter than the 2K+1 window makes the len(x)-2K
	// valid-output count non-positive, so reject it before clamping. 2*K+1 never
	// underflows because K = len(pairs) >= 0.
	k := len(pairs)
	if len(x) < mirrorSamplesPerPair*k+1 {
		return
	}
	outLen := len(x) - mirrorSamplesPerPair*k
	n := min(len(dst), outLen)
	if n == 0 {
		return
	}
	// x is passed whole: the kernel reads the sliding window x[i+K-k .. i+K+k] up
	// to index n-1+2K <= len(x)-1, so x must not be pre-sliced to n.
	firSymValidQ15I32(dst[:n], x, center, pairs)
}
