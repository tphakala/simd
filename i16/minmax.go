package i16

// MinMax returns the smallest and largest int16 in a:
//
//	minVal = min_i a[i],  maxVal = max_i a[i]
//
// Both are signed comparisons. Unlike [MaxAbs], the result needs no widening: the
// minimum and maximum of a set of int16 values are themselves int16, so this is
// the signed range probe (peak and trough), distinct from the abs-max headroom
// probe. An empty a returns (0, 0). a is read-only; the call allocates nothing.
//
// The SIMD fast path uses signed-min/max vector instructions (VPMINSW/VPMAXSW on
// amd64, SMIN/SMAX with single-instruction SMINV/SMAXV folds on arm64). Signed
// min/max has no accumulation order, so the SIMD paths are bit-identical to the
// pure-Go reference by construction.
func MinMax(a []int16) (minVal, maxVal int16) {
	if len(a) == 0 {
		return 0, 0
	}
	return minMaxI16(a)
}
