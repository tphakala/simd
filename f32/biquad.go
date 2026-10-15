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
// use the Go path.
func BiquadCascade(dst, src, coeffs, state []float32) {
	if len(coeffs) == 0 || len(coeffs)%biquadCoeffs != 0 {
		panic("simd: len(coeffs) must be a positive multiple of 5")
	}
	if len(state) != len(coeffs)/biquadCoeffs*biquadStateLen {
		panic("simd: len(state) must be 4 per section")
	}
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	biquadCascade32(dst[:n], src[:n], coeffs, state)
}
