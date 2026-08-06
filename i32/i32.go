// Package i32 provides SIMD-accelerated operations on int32 slices.
//
// It is the integer counterpart to the f32/f64 packages, covering the
// element-wise integer arithmetic, the signed min/max and wrapping-sum
// reductions, and the channel (de)interleaving that integer-domain DSP hot
// loops need where the per-sample work is integer arithmetic rather than
// floating-point math.
//
// All functions automatically select the optimal implementation based on
// runtime CPU feature detection and fall back to a pure-Go implementation on
// unsupported architectures.
//
// Thread Safety: All functions are safe for concurrent use.
// Memory: All functions are zero-allocation (no heap allocations).
//
// # Aliasing
//
// The element-wise operations may be used fully in place: the destination may
// alias an input exactly, element for element. Abs, ScaleQ31, ScaleQ15 and
// GainQ31 accept dst equal to their source; Add and Sub accept dst equal to a, to
// b, or to both; NegWhereNeg accepts dst equal to mag (its sign input is a
// separate float32 slice that cannot alias the int32 destination). Each SIMD
// block reads its whole block of inputs into registers before storing any output
// lane, and the scalar tail reads each element before it writes that element, so
// an exact overlay is well defined on the AVX2 and NEON kernels and the pure-Go
// fallback.
//
// A destination must not overlap an input at a shifted offset: a SIMD load pulls
// a whole block of an input ahead of the stores, so a shifted overlay clobbers
// input lanes a later iteration has not yet read; the resulting corruption is
// undefined and varies with kernel width and length.
//
// Some operations do not take an element-for-element overlay. Interleave2 and
// Deinterleave2 have a destination whose length differs from their inputs (twice,
// or half). FIRValidQ15 writes a valid-convolution output shorter than its input
// and reads a sliding window ahead of each output, so its dst must be distinct
// from x. Butterfly rewrites its two operands in place, so lo and hi must not
// overlap each other. The reductions (Sum, MaxAbs, MinMax) write no output slice,
// so aliasing does not apply to them.
package i32

// interleave2Channels is the number of channels handled by Interleave2 and
// Deinterleave2 (stereo).
const interleave2Channels = 2

// Interleave2 interleaves two slices: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
// Processes min(len(a), len(b), len(dst)/2) pairs; any trailing capacity in dst
// and ragged tails of a or b are left untouched.
//
// This is useful for packing separate channels into interleaved stereo PCM.
func Interleave2(dst, a, b []int32) {
	n := min(len(dst)/interleave2Channels, min(len(a), len(b)))
	if n == 0 {
		return
	}
	interleave2I32(dst[:n*interleave2Channels], a[:n], b[:n])
}

// Deinterleave2 deinterleaves a slice: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
// Processes min(len(a), len(b), len(src)/2) pairs; any trailing capacity in a or
// b and a ragged tail of src are left untouched.
//
// This is the inverse of Interleave2, useful for splitting interleaved stereo
// PCM into separate channels (for example raw 16-bit input widened to int32).
func Deinterleave2(a, b, src []int32) {
	n := min(len(src)/interleave2Channels, min(len(a), len(b)))
	if n == 0 {
		return
	}
	deinterleave2I32(a[:n], b[:n], src[:n*interleave2Channels])
}
