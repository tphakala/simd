package f32

// Layout of one biquad section in the coeffs and state slices of BiquadCascade.
const (
	biquadCoeffs   = 5 // b0, b1, b2, a1, a2
	biquadStateLen = 4 // x1, x2, y1, y2
)

// BiquadCascade filters src through a cascade of Direct Form I biquad sections
// and keeps the filter state between calls, so a stream can be processed in
// chunks of any length.
//
// coeffs holds five float32 per section, in order [b0 b1 b2 a1 a2] with a0
// normalized to 1. state holds four float32 per section, in order [x1 x2 y1 y2]:
// the section's previous two inputs and previous two outputs. Section k reads
// section k-1's output for the same sample; the first section reads src and the
// last one writes dst. state is read and written: on return it holds each
// section's history after the last processed sample, and a zero state is a
// silent history.
//
// Each section computes
//
//	y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2
//
// from left to right, then shifts x2=x1, x1=x, y2=y1, y1=y. Every product and
// every sum is rounded to float32 separately: nothing is fused into a
// multiply-add, and subnormals are never flushed to zero. The result is
// therefore independent of how the work is dispatched:
//   - Every output and state value that is not NaN is bit-identical on every
//     architecture, on every kernel and in the pure-Go path, and for any way of
//     splitting a stream into calls (including empty and one-sample chunks):
//     processing a stream in chunks gives exactly the bits of one call over the
//     whole stream.
//   - NaN appears exactly where the recurrence produces NaN (a NaN input, an
//     invalid operation such as Inf-Inf or 0*Inf, and everything downstream of
//     one). Which NaN bit pattern appears is not specified and differs between
//     architectures (the default NaN is 0xFFC00000 on amd64 and 0x7FC00000 on
//     arm64).
//
// Because subnormals are computed, a cascade fed silence after a signal does not
// have to decay to exactly zero: it can settle at a nonzero subnormal state. On
// x86 CPUs arithmetic on subnormal values can be much slower than on normal
// values, so a long silent tail can cost many times more than the signal that
// preceded it.
//
// Processes min(len(dst), len(src)) samples, leaves the rest of dst untouched
// and advances state by exactly that many samples. dst may be the same slice as
// src (the filter then runs in place) but must not otherwise overlap it. coeffs
// and state must not overlap dst, src or each other.
//
// BiquadCascade panics unless len(coeffs) is a positive multiple of 5 and
// len(state) is 4 per section; the check runs even when there is nothing to
// process.
//
// Uses AVX2 on AMD64 (8 sections per pass; AVX alone gives passes of 4) and NEON
// on ARM64 (8 sections per pass). A longer cascade runs as several passes, each
// over dst. Chunks too short to pay for a kernel call, and other architectures,
// use the Go path. For several channels sharing coefficients, see
// [BiquadCascadeN].
func BiquadCascade(dst, src, coeffs, state []float32) {
	sections := biquadSectionCount(coeffs)
	if len(state) != sections*biquadStateLen {
		panic("simd: len(state) must be 4 per section")
	}
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	biquadCascade32(dst[:n], src[:n], coeffs, state)
}

// BiquadCascadeN filters several channels through the same cascade, each with
// its own state: channel c reads src[c], writes dst[c] and advances state[c].
// coeffs and each state[c] have BiquadCascade's layout, and coeffs is shared by
// every channel. Each channel's output and final state are exactly what
// BiquadCascade(dst[c], src[c], coeffs, state[c]) produces, including
// BiquadCascade's NaN exception, so a channel's result does not depend on the
// other channels, on their lengths, or on the channel order. A stream can
// therefore switch between BiquadCascade and BiquadCascadeN at any chunk
// boundary.
//
// Channel c processes min(len(dst[c]), len(src[c])) samples, leaves the rest of
// dst[c] untouched and advances state[c] by exactly that many samples; channels
// may have different lengths. dst[c] may be the same slice as src[c] (the
// channel then runs in place); otherwise no dst[c] may overlap any src or dst
// slice of any channel. src slices may overlap each other. coeffs and the state
// slices must not overlap dst, src or each other.
//
// BiquadCascadeN panics unless len(src) and len(state) equal len(dst),
// len(coeffs) is a positive multiple of 5 and every state[c] holds 4 per
// section. All checks run before any channel is processed, even when there is
// nothing to process.
//
// For a cascade of one or two sections, channels 0 and 1, 2 and 3, and so on run
// in pairs on one kernel that holds both channels (AVX on AMD64, NEON on ARM64).
// An odd last channel, longer cascades, and pairs too short to pay for the
// kernel run each channel as BiquadCascade does.
func BiquadCascadeN(dst, src [][]float32, coeffs []float32, state [][]float32) {
	sections := biquadSectionCount(coeffs)
	if len(src) != len(dst) || len(state) != len(dst) {
		panic("simd: dst, src and state must have the same number of channels")
	}
	for _, st := range state {
		if len(st) != sections*biquadStateLen {
			panic("simd: len(state[c]) must be 4 per section in every channel")
		}
	}
	biquadCascadeN32(dst, src, coeffs, state)
}

// biquadSectionCount is the number of sections in coeffs. It panics unless
// len(coeffs) is a positive multiple of 5.
func biquadSectionCount(coeffs []float32) int {
	if len(coeffs) == 0 || len(coeffs)%biquadCoeffs != 0 {
		panic("simd: len(coeffs) must be a positive multiple of 5")
	}
	return len(coeffs) / biquadCoeffs
}
