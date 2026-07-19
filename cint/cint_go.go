package cint

// Pure-Go reference implementations.
//
// These are the source of truth for behavior: every SIMD kernel is validated
// for bit-exact parity against the functions here. They are compiled on every
// architecture and used directly as the fallback when no SIMD path applies.
//
// The complex data is interleaved int32, [r0,i0,r1,i1,...], and the twiddle is
// interleaved Q15 int16, [r0,i0,r1,i1,...]. sMul is the truncating Q15 multiply
// S_MUL; every add and subtract wraps in int32 (two's complement, no
// saturation), matching the SIMD lanes exactly across the full int32 range.

// sMulShift is the Q15 fixed-point position: the 64-bit product of an int32
// sample and an int16 Q15 coefficient is arithmetically shifted right by 15 to
// land the result back in int32 range.
const sMulShift = 15

// sMul is the truncating fixed-point multiply S_MUL(x, c) = int32(int64(x) *
// int64(c) >> 15). The product is formed in int64 (|x| <= 2^31, |c| <= 2^15, so
// |product| <= 2^46, never overflowing int64), the shift is an arithmetic
// (sign-preserving) right shift that truncates toward -inf with NO rounding
// constant, and the int32() cast wraps rather than saturates: x = MinInt32,
// c = MinInt16 gives 2^46 >> 15 = 2^31, which wraps to MinInt32. It is the
// integer MULT16_32_Q15 of libopus, matching go-opus fixed_generic.go bit for
// bit.
func sMul(x int32, c int16) int32 {
	return int32(int64(x) * int64(c) >> sMulShift)
}

// addGo writes the wrapping flat int32 add dst[j] = a[j] + b[j] over every lane.
// Complex addition is a lanewise real add on the interleaved layout, so the flat
// int32 sum over the lanes is the complex sum; the add wraps in int32 (modulo
// 2^32), matching a VPADDD or ADD .4S lane. dst may be empty; the len-0 guard
// protects the len(dst)-1 BCE hints.
func addGo(dst, a, b []int32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

// subGo writes the wrapping flat int32 sub dst[j] = a[j] - b[j] over every lane,
// the complex difference on the interleaved layout. The subtract wraps in int32,
// matching a VPSUBD or SUB .4S lane. dst may be empty; the len-0 guard protects
// the len(dst)-1 BCE hints.
func subGo(dst, a, b []int32) {
	if len(dst) == 0 {
		return
	}
	_ = a[len(dst)-1]
	_ = b[len(dst)-1]
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

// mulByScalarGo scales every int32 lane in place by the Q15 scalar s:
// a[j] = S_MUL(a[j], s). It is the flat ScaleQ15 over the interleaved layout, so
// both the real and imaginary part of each complex sample are scaled. The
// truncating Q15 arithmetic and the int32 wrap are exactly sMul's.
func mulByScalarGo(a []int32, s int16) {
	for i := range a {
		a[i] = sMul(a[i], s)
	}
}

// mulGo writes the C_MUL complex multiply of a by the twiddle tw, over whole
// complex pairs of the interleaved layout:
//
//	dst[2k]   = S_MUL(ar, br) - S_MUL(ai, bi)   // real
//	dst[2k+1] = S_MUL(ar, bi) + S_MUL(ai, br)   // imag
//
// where ar,ai = a[2k],a[2k+1] and br,bi = tw[2k],tw[2k+1]. Each S_MUL is
// truncated to int32 before the wrapping int32 add/subtract, matching the SIMD
// lanes. The k+1 < n loop bound processes only whole pairs, so a trailing
// half-complex (odd length) is never read. dst may alias a exactly: both a lanes
// of a pair are read into locals before either dst lane is stored, so the
// in-place update is well defined pair by pair.
func mulGo(dst, a []int32, tw []int16) {
	n := len(dst)
	if n == 0 {
		return
	}
	_ = a[n-1]
	_ = tw[n-1]
	for k := 0; k+1 < n; k += 2 {
		ar, ai := a[k], a[k+1]
		br, bi := tw[k], tw[k+1]
		dst[k] = sMul(ar, br) - sMul(ai, bi)
		dst[k+1] = sMul(ar, bi) + sMul(ai, br)
	}
}

// mulConjGo writes the C_MUL complex multiply of a by the conjugated twiddle
// conj(tw), over whole complex pairs:
//
//	dst[2k]   = S_MUL(ar, br) + S_MUL(ai, bi)   // real
//	dst[2k+1] = S_MUL(ai, br) - S_MUL(ar, bi)   // imag
//
// Same truncating Q15 arithmetic, int32 wrap and in-place-alias discipline as
// mulGo; only the two combine signs differ (the conjugate negates bi).
func mulConjGo(dst, a []int32, tw []int16) {
	n := len(dst)
	if n == 0 {
		return
	}
	_ = a[n-1]
	_ = tw[n-1]
	for k := 0; k+1 < n; k += 2 {
		ar, ai := a[k], a[k+1]
		br, bi := tw[k], tw[k+1]
		dst[k] = sMul(ar, br) + sMul(ai, bi)
		dst[k+1] = sMul(ai, br) - sMul(ar, bi)
	}
}
