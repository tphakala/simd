package i32

// MinMax returns the smallest and largest int32 in res:
//
//	minVal = min_i res[i],  maxVal = max_i res[i]
//
// Both are signed comparisons. An empty res returns (0, 0). res is read-only;
// the call allocates nothing.
//
// The SIMD fast path uses signed-min/max vector instructions (VPMINSD/VPMAXSD on
// amd64, SMIN/SMAX with single-instruction SMINV/SMAXV folds on arm64) with a
// scalar tail. Since signed min/max has no accumulation order, the SIMD paths
// are bit-identical to the pure-Go reference by construction.
func MinMax(res []int32) (minVal, maxVal int32) {
	if len(res) == 0 {
		return 0, 0
	}
	return minMaxI32(res)
}
