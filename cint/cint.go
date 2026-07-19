// Package cint provides SIMD-accelerated fixed-point complex arithmetic on
// interleaved int32 slices, the integer FFT butterfly kernels of libopus
// kiss_fft.
//
// # Data model
//
// Complex data is a []int32 of length 2*N laid out interleaved as
// [r0,i0,r1,i1,...], one int32 per real or imaginary part. A twiddle factor is a
// []int16 of length 2*N in Q15, laid out the same interleaved way. This matches
// the C kiss_fft_cpx arrays libopus operates on.
//
// # Fixed-point semantics
//
// The scalar multiply is S_MUL(x, c) = int32(int64(x) * int64(c) >> 15): a
// TRUNCATING Q15 multiply (an arithmetic right shift, NO rounding constant),
// exactly the integer MULT16_32_Q15 of libopus. Every add and subtract wraps in
// int32 (two's complement, no saturation). The products are always formed in a
// 64-bit intermediate that never overflows (|int32 * int16| <= 2^46), so only the
// final int32 cast wraps: the extreme MinInt32 * MinInt16 = 2^46 shifts to 2^31
// and wraps to MinInt32. The SIMD and pure-Go paths are bit-identical across the
// full int32 range; there is no relaxed tier.
//
// # Length handling
//
// Every function clamps to n = min of the applicable slice lengths and then masks
// n down to a whole number of complex pairs (an even lane count, n &^= 1). This
// is not a claimed precondition the caller must meet: a mismatched or odd-length
// slice is not an error, it simply bounds the work to the leading whole complex
// pairs both (or all) slices cover, and a trailing lone real lane is left
// untouched. Masking to even also guarantees Mul and MulConj never read the
// second half a[2k+1] / tw[2k+1] of a pair the slice does not fully contain.
// Empty inputs write nothing.
//
// # Aliasing
//
// Mul, MulConj and MulByScalar may be used fully in place: dst may alias a
// exactly, element for element (and MulByScalar is defined in place). Each SIMD
// block reads its whole complex block of inputs into registers before storing any
// output lane, so an exact overlay is well defined block by block. dst must not
// otherwise overlap a at a shifted offset: a SIMD load pulls a block ahead of the
// stores, so a shifted overlay could clobber input lanes a later iteration has
// not yet read. Add and Sub write dst from separate a and b; pass a dst that does
// not shift-overlap either input.
//
// # Guarantees
//
// All functions are zero-allocation (they write into caller-provided slices or
// in place) and safe for concurrent use on non-overlapping slices.
package cint

// Add writes the wrapping complex sum dst[k] = a[k] + b[k] over the leading whole
// complex pairs of dst, a and b. Complex addition is a lanewise real add on the
// interleaved layout, so this is the flat int32 add over n = (min of the three
// lengths) masked to an even lane count; each add wraps in int32 rather than
// saturating. Any lane past n, including a trailing lone real, is left untouched.
func Add(dst, a, b []int32) {
	n := min(len(dst), len(a), len(b)) &^ 1
	if n == 0 {
		return
	}
	addCint(dst[:n], a[:n], b[:n])
}

// Sub writes the wrapping complex difference dst[k] = a[k] - b[k] over the
// leading whole complex pairs of dst, a and b, the flat int32 subtract over the
// even-masked min length. Each subtract wraps in int32. Any lane past n is left
// untouched.
func Sub(dst, a, b []int32) {
	n := min(len(dst), len(a), len(b)) &^ 1
	if n == 0 {
		return
	}
	subCint(dst[:n], a[:n], b[:n])
}

// MulByScalar scales the complex data a in place by the Q15 scalar s:
// a[j] = S_MUL(a[j], s) for every int32 lane, scaling both the real and the
// imaginary part of each complex sample. It is the integer MULT16_32_Q15 applied
// flat over the interleaved layout, truncating and wrapping in int32. The length
// is masked to a whole number of complex pairs, so a trailing lone real lane in
// an odd-length slice is left unscaled.
func MulByScalar(a []int32, s int16) {
	n := len(a) &^ 1
	if n == 0 {
		return
	}
	mulByScalarCint(a[:n], s)
}

// Mul writes the C_MUL complex multiply of a by the twiddle tw:
//
//	dst[2k]   = S_MUL(ar, br) - S_MUL(ai, bi)   // real
//	dst[2k+1] = S_MUL(ar, bi) + S_MUL(ai, br)   // imag
//
// with ar,ai = a[2k],a[2k+1] and br,bi = tw[2k],tw[2k+1]. It runs over n = (min
// of len(dst), len(a), len(tw)) masked to a whole number of complex pairs; a
// trailing half-complex is never read or written. Each S_MUL truncates to int32
// before the wrapping int32 combine. dst may alias a exactly for an in-place
// transform; see the package doc on aliasing.
func Mul(dst, a []int32, tw []int16) {
	n := min(len(dst), len(a), len(tw)) &^ 1
	if n == 0 {
		return
	}
	mulCint(dst[:n], a[:n], tw[:n])
}

// MulConj writes the C_MUL complex multiply of a by the conjugated twiddle
// conj(tw):
//
//	dst[2k]   = S_MUL(ar, br) + S_MUL(ai, bi)   // real
//	dst[2k+1] = S_MUL(ai, br) - S_MUL(ar, bi)   // imag
//
// the correlation counterpart of Mul (the conjugate negates the twiddle
// imaginary part). Same even-masked length handling, truncating Q15 arithmetic,
// int32 wrap and in-place-alias support as Mul.
func MulConj(dst, a []int32, tw []int16) {
	n := min(len(dst), len(a), len(tw)) &^ 1
	if n == 0 {
		return
	}
	mulConjCint(dst[:n], a[:n], tw[:n])
}
