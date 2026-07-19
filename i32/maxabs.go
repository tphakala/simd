package i32

// MaxAbs returns the peak magnitude of a as a single signed min/max scan, the
// exact integer form of libopus celtMaxabs32.
//
// It is defined as:
//
//	minVal, maxVal := signed min and max over a
//	return max(maxVal, -minVal)
//
// where -minVal is a WRAPPING int32 negate. This is deliberately NOT a per-lane
// absolute-value-then-max: the two differ when a MinInt32 element sits beside a
// negative one. For a = [MinInt32, -3], MaxAbs computes max(-3, -MinInt32) =
// max(-3, MinInt32) = -3, whereas abs-then-max would give 3. MaxAbs follows the
// max(maxVal, -minVal) form of the consumer rather than the intuitive
// absolute-value peak. (This example lives in the MinInt32 overflow regime below,
// which the mirrored CELT data never reaches; every representable input agrees.)
//
// The result type is int32, matching celtMaxabs32's opus_val32 return. The
// magnitude of MinInt32 (2^31) is not representable in int32, so a slice whose
// peak is driven by a MinInt32 element returns a wrapped (negative) value:
// -MinInt32 wraps back to MinInt32. The CELT data this mirrors never reaches
// that input; the wrap is documented, not hidden. Every result is bit-identical
// across the AVX2, NEON and pure-Go backends (there is no relaxed tier), because
// signed min/max has no accumulation order.
//
// An empty a returns 0. a is read-only; the call allocates nothing.
func MaxAbs(a []int32) int32 {
	if len(a) == 0 {
		return 0
	}
	return maxAbsI32(a)
}
