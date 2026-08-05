package i32

// Fused fixed-point gain-by-scalar on int32 slices.
//
// GainQ31 applies a single Q31 gain with a per-call input pre-shift and a
// rounding output post-shift, fused into one pass. It fuses the ScaleQ31 core
// (MULT32_32_Q31) with the input SHL32 and the rounding requant PSHR32 that a
// fixed-point audio decoder's gain-application hot path applies around that core,
// so the three stages that would otherwise be three separate passes over the
// buffer run as a single sweep. Like ScaleQ31 it wraps in int32 rather than saturating,
// so the SIMD and pure-Go paths are bit-identical and there is no relaxed tier.

// GainQ31 writes, for i in [0, n) with n = min(len(dst), len(a)):
//
//	dst[i] = PSHR32(MULT32_32_Q31(SHL32(a[i], preShift), g), postShift)
//
// It is the integer denormalise-bands requant: the ScaleQ31 core
// MULT32_32_Q31(x, g) = int32(int64(x) * int64(g) >> 31) applied to a
// pre-shifted sample and then rounded back down. The three stages each wrap in
// int32, so the result is bit-identical to the ScaleQ31 + shift composition and
// identical across all backends:
//
//   - SHL32(a[i], preShift) is the wrapping left shift int32(uint32(a[i])<<preShift).
//     It runs on the int32 sample before the widen, so the shifted value stays in
//     int32 range and the Q31 product below never overflows its int64 intermediate.
//   - MULT32_32_Q31 forms the product in int64 and arithmetically shifts right by
//     31, truncating toward -inf (no rounding); the int32 cast wraps.
//   - PSHR32(_, postShift) adds the rounding bias (int32(1)<<postShift)>>1 and
//     arithmetically shifts right by postShift (round half up). postShift == 0
//     adds no bias and leaves the value unchanged. The bias addition is int32 and
//     wraps, exactly as libopus PSHR32 does.
//
// preShift and postShift must each be in [0, 31]; GainQ31 panics otherwise, since
// an out-of-range count would otherwise diverge across backends (the NEON SSHL
// reads it as a signed per-lane shift, the amd64 scalar tail masks it to 5 bits,
// and the vector VPSLLD saturates to zero). g, preShift and postShift are per-call
// scalar constants. Any trailing capacity in dst past n is left untouched.
//
// dst may alias a exactly (element for element): each lane reads a[i] before its
// own dst[i] store and the forward iteration never revisits a written lane, so
// the samples can be processed in place. dst must not otherwise overlap a: the
// SIMD kernels load a whole block of a before storing the block of dst, so a
// shifted dst/a overlay could clobber input lanes a later iteration has not read
// yet. This is the same aliasing rule and caveat as [ScaleQ31].
func GainQ31(dst, a []int32, g int32, preShift, postShift int) {
	if preShift < 0 || preShift > gainShiftMax || postShift < 0 || postShift > gainShiftMax {
		panic("i32.GainQ31: preShift and postShift must be in [0, 31]")
	}
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	gainQ31I32(dst[:n], a[:n], g, preShift, postShift)
}
