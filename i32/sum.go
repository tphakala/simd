package i32

// Sum returns the sum of all elements of a, accumulated in int32 with
// two's-complement wraparound, never saturation. Wrapping addition is
// associative and commutative modulo 2^32, so any SIMD lane grouping and any
// horizontal reduction order yields the same bits as the sequential loop,
// including on inputs engineered to overflow; that reproducibility is the
// contract that lets the kernels vectorize at all. An empty a returns 0. a is
// read-only; the call allocates nothing.
func Sum(a []int32) int32 {
	if len(a) == 0 {
		return 0
	}
	return sumI32(a)
}
