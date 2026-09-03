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

// maxAbsGo is MaxAbs's source of truth: the peak magnitude with celtMaxabs32
// semantics. It runs the signed min/max scan, then returns max(hi, -lo), where
// -lo is a WRAPPING int32 negate (-MinInt32 == MinInt32). This is NOT per-lane
// abs-then-max: for a = [MinInt32, -3] it returns max(-3, MinInt32) = -3, not 3.
// minMaxGo requires a non-empty a, so the public MaxAbs guards the len==0 case
// before dispatch.
func maxAbsGo(a []int32) int32 {
	lo, hi := minMaxGo(a) // reuse the signed min/max scan
	neg := -lo            // wrapping: -MinInt32 == MinInt32
	if hi > neg {
		return hi
	}
	return neg
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

// butterflyGo writes the in-place radix-2 butterfly that is Butterfly's source
// of truth: lo[i], hi[i] = lo[i]+hi[i], lo[i]-hi[i]. Both operations wrap in
// int32 (addition and subtraction modulo 2^32), matching the VPADDD/VPSUBD and
// ADD/SUB .4S lanes exactly. Each lane forms the sum and difference into
// temporaries before writing either output, so lo[i] is read for the difference
// before its own store overwrites it and the in-place update is well defined.
// lo and hi must not overlap each other. lo may be empty; the len-0 guard
// protects the len(lo)-1 BCE hint.
func butterflyGo(lo, hi []int32) {
	if len(lo) == 0 {
		return
	}
	_ = hi[len(lo)-1] // BCE hint: hi is indexed [0, len(lo))
	for i := range lo {
		s := lo[i] + hi[i] // wrapping int32 sum
		d := lo[i] - hi[i] // wrapping int32 difference
		lo[i] = s
		hi[i] = d
	}
}

// scaleQ31Shift is the fixed-point position of the Q31 result: the full 64-bit
// product int64(a)*int64(k) is arithmetically shifted right by 31 to land the
// Q31 fractional scale back in int32 range.
const scaleQ31Shift = 31

// scaleQ15Shift is the Q15 counterpart: MULT16_32_Q15 multiplies a full int32 by
// an int16 Q15 coefficient and arithmetically shifts right by 15.
const scaleQ15Shift = 15

// scaleQ31Go writes the wrapping truncating Q31 scale that is ScaleQ31's source
// of truth: dst[i] = int32(int64(a[i]) * int64(k) >> 31). The product is formed
// in int64 (|a|,|k| <= 2^31 so |product| <= 2^62, never overflowing int64), the
// shift is an arithmetic (sign-preserving) right shift that truncates toward
// -inf, and the int32() cast wraps rather than saturates: a[i]=k=MinInt32 gives
// 2^62 >> 31 = 2^31, which wraps to MinInt32. dst may alias a exactly (each lane
// reads a[i] before its own dst[i] store). dst may be empty; the len-0 guard
// protects the len(dst)-1 BCE hint.
func scaleQ31Go(dst, a []int32, k int32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1] // BCE hint: a is indexed [0, len(dst))
	for i := range dst {
		dst[i] = int32(int64(a[i]) * int64(k) >> scaleQ31Shift)
	}
}

// scaleQ15Go writes the wrapping truncating Q15 scale that is ScaleQ15's source
// of truth: dst[i] = int32(int64(k) * int64(a[i]) >> 15). k is a signed int16
// coefficient and a[i] a full int32 sample, so |product| <= 2^15 * 2^31 = 2^46,
// far inside int64; the arithmetic shift truncates toward -inf and the int32()
// cast wraps. dst may alias a exactly. dst may be empty; the len-0 guard protects
// the len(dst)-1 BCE hint.
func scaleQ15Go(dst, a []int32, k int16) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1] // BCE hint: a is indexed [0, len(dst))
	for i := range dst {
		dst[i] = int32(int64(k) * int64(a[i]) >> scaleQ15Shift)
	}
}

// gainShiftMax is the largest valid preShift/postShift for GainQ31: a 32-bit lane
// admits shift counts 0..31. Public GainQ31 rejects anything outside [0, gainShiftMax];
// gainQ31Go reuses it as the low-5-bit mask (31 == 2^5-1) so the compiler can prove
// the per-element variable shifts take a count in [0,31] and drop the
// runtime.panicshift and range-clamp guards from the hot loop.
const gainShiftMax = 31

// gainQ31Go writes the fused Q31 gain that is GainQ31's source of truth:
// dst[i] = PSHR32(MULT32_32_Q31(SHL32(a[i], preShift), gain), postShift), the
// gain-application inner loop of a fixed-point audio decoder in a single pass.
// Each stage wraps in int32 exactly as the SIMD lanes do:
//
//   - SHL32:  x = int32(uint32(a[i]) << preShift), a wrapping left shift. It runs
//     on the int32 sample before the widen, so |x| <= 2^31 and the product below
//     never overflows int64, just like scaleQ31Go.
//   - MULT32_32_Q31: p = int32(int64(x) * int64(gain) >> 31), the ScaleQ31 core:
//     an arithmetic shift that truncates toward -inf, the int32 cast wrapping.
//   - PSHR32: (p + bias) >> postShift with bias = (int32(1)<<postShift)>>1, a
//     round-half-up arithmetic right shift. bias is 0 when postShift == 0 (no
//     rounding), and the p+bias addition is int32 and is meant to wrap, exactly as
//     libopus PSHR32 does.
//
// preShift and postShift must be in [0, 31]. dst may alias a exactly (each lane
// reads a[i] before its own dst[i] store). dst may be empty; the len-0 guard
// protects the len(dst)-1 BCE hint.
func gainQ31Go(dst, a []int32, gain int32, preShift, postShift int) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1] // BCE hint: a is indexed [0, len(dst))
	// Mask the shift counts into [0,31] once. Within the documented precondition
	// this is a no-op, but it lets the compiler prove the per-element variable
	// shifts cannot take a negative count, lifting runtime.panicshift and the gain
	// spill out of the hot loop (~1.6x). Out-of-range counts are unspecified either
	// way (the SIMD kernels do not fault on them either).
	ps := uint(preShift) & gainShiftMax
	qs := uint(postShift) & gainShiftMax
	bias := (int32(1) << qs) >> 1
	for i := range dst {
		x := int32(uint32(a[i]) << ps)
		p := int32(int64(x) * int64(gain) >> scaleQ31Shift)
		dst[i] = (p + bias) >> qs
	}
}

// sumSqShiftedQ31Go is SumSqShiftedQ31's source of truth: the wrapping int32 sum
// over a of MULT32_32_Q31(a[i]<<shift, a[i]<<shift). Each term applies SHL32 (a
// wrapping int32 left shift) to the sample BEFORE the widen, so the square is of
// the already-truncated int32 value; the 64-bit square is at most (2^31)^2 = 2^62
// and never overflows its int64 intermediate, and only the final int32() cast
// wraps (a shifted value of MinInt32 gives 2^62 >> 31 = 2^31, wrapping to
// MinInt32). The square is non-negative, so the arithmetic shift here is a plain
// truncating >> 31.
//
// The accumulator wraps in int32: two's-complement addition is associative and
// commutative modulo 2^32, so the SIMD lane grouping and horizontal reduction are
// bit-identical to this sequential loop for every input, including forced
// overflow. That reproducibility is the contract that lets the kernels vectorize.
//
// shift is masked into [0,31] once (a no-op within the documented precondition,
// which public SumSqShiftedQ31 enforces) so the compiler can prove the per-element
// shift count is in range and drop runtime.panicshift from the hot loop, exactly
// as gainQ31Go does.
func sumSqShiftedQ31Go(a []int32, shift int) int32 {
	s := uint(shift) & gainShiftMax
	var sum int32
	for _, v := range a {
		x := int32(uint32(v) << s)
		sum += int32(int64(x) * int64(x) >> scaleQ31Shift)
	}
	return sum
}

// firValidQ15Shift is the Q15 fixed-point position of each tap product: the
// 64-bit int64(taps[j])*int64(x[i+j]) is arithmetically shifted right by 15 to
// land the Q15 fractional scale back in int32 range, per product, before it is
// accumulated (MULT16_32_Q15, a truncating shift with no rounding constant). It
// is tied to scaleQ15Shift so the two Q15 kernels cannot drift out of lockstep.
const firValidQ15Shift = scaleQ15Shift

// firValidQ15Go writes the int32 valid convolution in correlation orientation
// that is FIRValidQ15's source of truth:
//
//	dst[i] = sum_j int32(int64(taps[j]) * int64(x[i+j]) >> 15)
//
// for i in [0, n), n = min(len(dst), len(x)-len(taps)+1). Each tap product is
// Q15-truncated (an arithmetic right shift, no rounding) BEFORE it is added, and
// the accumulator is two's-complement wrapping int32, matching MULT16_32_Q15 and
// a wrapping accumulate exactly. The product |taps[j] * x[i+j]| <= 2^15 * 2^31 =
// 2^46 stays inside int64; only the per-product int32() cast and the running add
// wrap. The public FIRValidQ15 applies the empty-taps / short-x guard and clamps
// dst, but this function is self-guarding (it recomputes the safe output count),
// so a direct fallback call cannot read x out of range. dst must not overlap x.
func firValidQ15Go(dst, x []int32, taps []int16) {
	if len(taps) == 0 || len(x) < len(taps) {
		return
	}
	outLen := len(x) - len(taps) + 1
	n := min(len(dst), outLen)
	if n == 0 {
		return
	}
	kl := len(taps)
	for i := range n {
		// One slice bound per output instead of a per-element x[i+j] check:
		// i+kl <= (n-1)+kl <= len(x) because n <= len(x)-kl+1, so w has length kl
		// and w[j] is provably in range for every j in [0, kl).
		w := x[i : i+kl]
		var acc int32 // wrapping int32 accumulator
		for j := range taps {
			p := int32(int64(taps[j]) * int64(w[j]) >> firValidQ15Shift)
			acc += p // two's-complement wrapping add
		}
		dst[i] = acc
	}
}

// mirrorSamplesPerPair is the number of samples a symmetric mirror pair spans
// away from the center: one on the left and one on the right. A symmetric window
// with K = len(pairs) pairs therefore holds mirrorSamplesPerPair*K + 1 samples (K
// each side plus the center tap), and the valid-output count of a length-len(x)
// input is len(x) - mirrorSamplesPerPair*K.
const mirrorSamplesPerPair = 2

// firSymValidQ15Go writes the int32 valid convolution of a SYMMETRIC Q15 tap set
// that is FIRSymValidQ15's source of truth: a center tap plus K = len(pairs)
// mirror pairs, window length 2K+1, in correlation orientation. For output i,
// with the center sample at c = i+K:
//
//	dst[i] = int32(int64(center) * int64(x[c]) >> 15)
//	       + sum over k in [1,K] of int32(int64(pairs[k-1]) * int64(x[c-k]+x[c+k]) >> 15)
//
// for i in [0, n), n = min(len(dst), len(x)-2K). The load-bearing difference from
// firValidQ15Go is the mirror fold: the two mirror samples x[c-k] and x[c+k] are
// summed with a two's-complement WRAPPING int32 add BEFORE the single Q15
// truncation, so each pair contributes exactly ONE truncation, not two. This
// matches libopus comb_filter_const_c (FIXED_POINT), which pre-adds each mirror
// pair with ADD32 and then does one MULT16_32_Q15; because trunc((p+q)>>15) !=
// trunc(p>>15)+trunc(q>>15) in general, a per-product FIR (FIRValidQ15 over a
// folded tap set) is NOT bit-identical to the codec. The addends stay int32 (Go
// wraps the +) and only widen to int64 for the multiply, so the pre-truncation
// wrap is preserved; the running accumulator is likewise wrapping int32. The
// product |pairs[k-1] * (x[c-k]+x[c+k])| <= 2^15 * 2^31 = 2^46 stays inside int64.
//
// When K == 0 (no pairs) this degenerates to a pure center scale over the whole
// window (window length 1, n = min(len(dst), len(x))), matching ScaleQ15. The
// public FIRSymValidQ15 applies the short-x guard and clamps dst, but this
// function is self-guarding (it recomputes the safe output count), so a direct
// fallback call cannot read x out of range. dst must not overlap x.
func firSymValidQ15Go(dst, x []int32, center int16, pairs []int16) {
	k := len(pairs)
	if len(x) < mirrorSamplesPerPair*k+1 {
		return
	}
	outLen := len(x) - mirrorSamplesPerPair*k
	n := min(len(dst), outLen)
	if n == 0 {
		return
	}
	for i := range n {
		// One slice bound per output hoists the x[c-k .. c+k] checks: the window
		// w = x[i : i+2k+1] has length 2k+1 and w[k] is the center, w[k-p] and
		// w[k+p] the mirror samples for pair p. i+2k+1 <= (n-1)+2k+1 <= len(x)
		// because n <= len(x)-2k, so every index below is provably in range.
		w := x[i : i+mirrorSamplesPerPair*k+1]
		acc := int32(int64(center) * int64(w[k]) >> firValidQ15Shift) // wrapping int32
		for p := 1; p <= k; p++ {
			sum := w[k-p] + w[k+p]                                           // wrapping int32 add BEFORE truncation
			acc += int32(int64(pairs[p-1]) * int64(sum) >> firValidQ15Shift) // wrapping accumulate
		}
		dst[i] = acc
	}
}
