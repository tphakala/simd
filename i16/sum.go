package i16

// Sum returns the sum of all elements of a, widened to and accumulated in int32
// with two's-complement wraparound, never saturation. This is the same contract
// as [DotProduct]: wrapping addition is associative and commutative modulo 2^32,
// so any SIMD lane grouping and any horizontal reduction order yields the same
// bits as the sequential loop, including on inputs engineered to overflow. That
// reproducibility is what lets the kernels vectorize at all.
//
// The accumulator is int32, so a run of same-signed samples wraps once |sum|
// passes 2^31 (about 65536 samples at full scale). A caller that needs a
// wider running total over a long buffer should widen first (or reduce in
// blocks); the wrap here is defined, not an error. An empty a returns 0. a is
// read-only; the call allocates nothing.
func Sum(a []int16) int32 {
	if len(a) == 0 {
		return 0
	}
	return sumI16(a)
}
