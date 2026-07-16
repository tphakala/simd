// Package i16 provides SIMD-accelerated operations on int16 slices.
//
// It is the 16-bit integer counterpart to the i32 package and serves two kinds
// of hot loop: raw-PCM shuffling (for example a pure-Go FLAC codec), where the
// cheapest place to vectorize is the channel (de)interleaving that happens
// before samples are widened to int32; and fixed-point DSP, where int16 inputs
// are multiplied and accumulated into int32.
//
// Widening semantics: an operation that multiplies int16 inputs accumulates
// into int32 with two's-complement wraparound, never saturation. This is a
// guarantee, not an implementation detail. Wrapping addition is associative and
// commutative modulo 2^32, so any lane grouping and any horizontal reduction
// order yields bit-identical results to the scalar loop; a saturating
// accumulator is not associative and could not be vectorized without changing
// results. Fixed-point codecs (Opus/CELT, FLAC LPC) depend on that
// reproducibility, which is the whole reason they use integer arithmetic.
//
// Element-wise int16 arithmetic still belongs in the i32 package, because
// inter-channel decorrelation can exceed the source bit depth by one bit. What
// this package adds at 16-bit width is the widening direction, where the narrow
// input is the point.
//
// All functions automatically select the optimal implementation based on
// runtime CPU feature detection and fall back to a pure-Go implementation on
// unsupported architectures.
//
// Thread Safety: All functions are safe for concurrent use.
// Memory: All functions are zero-allocation (no heap allocations).
package i16

// interleave2Channels is the number of channels handled by Interleave2 and
// Deinterleave2 (stereo).
const interleave2Channels = 2

// Interleave2 interleaves two slices: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
// Processes min(len(a), len(b), len(dst)/2) pairs; any trailing capacity in dst
// and ragged tails of a or b are left untouched.
//
// This is useful for packing separate channels into interleaved stereo PCM.
func Interleave2(dst, a, b []int16) {
	n := min(len(dst)/interleave2Channels, min(len(a), len(b)))
	if n == 0 {
		return
	}
	interleave2I16(dst[:n*interleave2Channels], a[:n], b[:n])
}

// Deinterleave2 deinterleaves a slice: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
// Processes min(len(a), len(b), len(src)/2) pairs; any trailing capacity in a or
// b and a ragged tail of src are left untouched.
//
// This is the inverse of Interleave2, useful for splitting interleaved stereo
// PCM into separate channels before widening to int32.
func Deinterleave2(a, b, src []int16) {
	n := min(len(src)/interleave2Channels, min(len(a), len(b)))
	if n == 0 {
		return
	}
	deinterleave2I16(a[:n], b[:n], src[:n*interleave2Channels])
}
