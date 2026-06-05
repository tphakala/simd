package i32

// MinMax returns the smallest and largest int32 in res:
//
//	minVal = min_i res[i],  maxVal = max_i res[i]
//
// Both are signed comparisons. An empty res returns (0, 0). res is read-only;
// the call allocates nothing.
//
// This is the reduction the FLAC Rice planner runs per finest partition to find
// each partition's largest-magnitude residual (the escape-code cost input): the
// largest zigzag fold over a partition is reached at its most-negative or
// most-positive sample, so max(zigzag(minVal), zigzag(maxVal)) is that extreme.
// The SIMD fast path uses signed-min/max vector instructions (VPMINSD/VPMAXSD on
// amd64, SMIN/SMAX on arm64) with a scalar tail; both are bit-exact against the
// pure-Go reference.
func MinMax(res []int32) (minVal, maxVal int32) {
	if len(res) == 0 {
		return 0, 0
	}
	return minMaxI32(res)
}
