package i32

// SumSqShiftedQ31 returns the wrapping int32 sum over a of
// MULT32_32_Q31(a[i]<<shift, a[i]<<shift), where
//
//	MULT32_32_Q31(p, q) = int32(int64(p) * int64(q) >> 31)
//
// and a[i]<<shift is a wrapping int32 left shift (SHL32). It is the band-energy
// reduction of a fixed-point Opus/CELT encoder's compute_band_energies: for each
// band the caller derives a per-band shift from the band peak, sums the shifted
// squares with this primitive, then takes a Q-domain sqrt. Only the inner
// reduction lives here; the shift derivation and the sqrt stay in the caller.
//
// The shift is applied to the int32 sample BEFORE the widening multiply, so the
// square is of the already-truncated int32 value: a[i]<<shift wraps in 32 bits
// (it can change sign or overflow), and the square is of that wrapped value. The
// 64-bit product is at most (2^31)^2 = 2^62 and never overflows its int64
// intermediate; only the final int32() cast wraps, so a shifted sample equal to
// MinInt32 contributes 2^62 >> 31 = 2^31, which wraps back to MinInt32.
//
// The accumulator wraps in int32 rather than saturating. Two's-complement
// addition is associative and commutative modulo 2^32, so every SIMD lane
// grouping and horizontal reduction order yields the same bits as the sequential
// loop, including on inputs engineered to overflow; that reproducibility is the
// contract that lets the kernels vectorize. Every result is bit-identical across
// the AVX2, NEON and pure-Go backends (there is no relaxed tier).
//
// shift must be in [0, 31]; SumSqShiftedQ31 panics otherwise, since an
// out-of-range count would diverge across backends (the NEON SSHL reads it as a
// signed per-lane shift, the amd64 scalar tail masks it to 5 bits, and the vector
// VPSLLD saturates to zero). An empty a returns 0. a is read-only; the call
// allocates nothing.
func SumSqShiftedQ31(a []int32, shift int) int32 {
	if shift < 0 || shift > gainShiftMax {
		panic("i32.SumSqShiftedQ31: shift must be in [0, 31]")
	}
	if len(a) == 0 {
		return 0
	}
	return sumSqShiftedQ31I32(a, shift)
}
