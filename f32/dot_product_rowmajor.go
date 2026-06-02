package f32

// Shared scaffolding for the row-major batch dot APIs (DotProductIndexed /
// DotProductStrided). The per-architecture dispatch (f32_amd64.go, f32_arm64.go)
// owns the SIMD kernel selection and the batch-of-4 loop; everything here is the
// portable glue both arches lean on: the SIMD-vs-fallback eligibility gate, the
// full-row range math, and the ragged-safe scalar fallback/tail paths.

const (
	batchDotRows    = 4
	batchDotMinDims = 64

	// SIMD-vs-fallback gate. Above these dim/row sizes the batched kernel stops
	// beating the per-row fallback on the hardware measured so far, so very
	// large shapes stay on the scalar path. These are deliberately conservative
	// monotone thresholds, not per-machine tuned cliffs; re-tune against a
	// benchmark matrix before enabling more shapes.
	batchDotLargeDims    = 768
	batchDotLargeMaxRows = 256
	batchDotHugeDims     = 2048
	batchDotHugeMaxRows  = 64
)

func fullRowMaxIndex(baseLen, dims, stride int) int {
	if dims <= 0 || stride <= 0 || baseLen < dims {
		return -1
	}
	return (baseLen - dims) / stride
}

func rowIDInFullRange(rowID uint32, maxRow int) bool {
	return maxRow >= 0 && uint64(rowID) <= uint64(maxRow)
}

func dotProductIndexedFallback(dst, base, query []float32, rowIDs []uint32, dims int) {
	n := min(len(dst), len(rowIDs))
	if n == 0 {
		return
	}
	if dims <= 0 || len(query) == 0 {
		clear(dst[:n])
		return
	}
	queryN := min(dims, len(query))
	queryFull := query[:queryN]
	maxRow := fullRowMaxIndex(len(base), queryN, dims)
	for i := range n {
		rowID := rowIDs[i]
		if rowIDInFullRange(rowID, maxRow) {
			off := int(rowID) * dims
			dst[i] = dotProduct(base[off:off+queryN], queryFull)
			continue
		}
		dst[i] = dotProductIndexedOneGo(base, query, rowID, dims)
	}
}

func dotProductStridedFallback(dst, base, query []float32, rowCount, dims, stride int) {
	if rowCount <= 0 || len(dst) == 0 {
		return
	}
	n := min(len(dst), rowCount)
	if dims <= 0 || stride <= 0 || len(query) == 0 {
		clear(dst[:n])
		return
	}
	queryN := min(dims, len(query))
	queryFull := query[:queryN]
	maxRow := fullRowMaxIndex(len(base), queryN, stride)
	if n-1 <= maxRow {
		for i, off := 0, 0; i < n; i, off = i+1, off+stride {
			dst[i] = dotProduct(base[off:off+queryN], queryFull)
		}
		return
	}
	for i := range n {
		if i <= maxRow {
			off := i * stride
			dst[i] = dotProduct(base[off:off+queryN], queryFull)
			continue
		}
		dst[i] = dotProductStridedOneGo(base, query, i, dims, stride)
	}
}

// dotProductIndexedTail scores one row for the mixed-block and trailing-tail
// paths. The CPU capability and queryLen >= dims are already verified by the
// caller, so any in-range row can take the optimized single-row dotProduct;
// out-of-range rows score zero via the ragged-safe Go path.
func dotProductIndexedTail(base, query, queryFull []float32, rowID uint32, dims, maxRow int) float32 {
	if rowIDInFullRange(rowID, maxRow) {
		off := int(rowID) * dims
		return dotProduct(base[off:off+dims], queryFull)
	}
	return dotProductIndexedOneGo(base, query, rowID, dims)
}

func dotProductStridedTail(base, query, queryFull []float32, row, dims, stride, maxRow int) float32 {
	if row >= 0 && row <= maxRow {
		off := row * stride
		return dotProduct(base[off:off+dims], queryFull)
	}
	return dotProductStridedOneGo(base, query, row, dims, stride)
}

func batchDotIndexedSIMDEligible(rows, dims, queryLen int) bool {
	if rows < batchDotRows || dims < batchDotMinDims || queryLen < dims {
		return false
	}
	return batchDotRowsWithinGate(rows, dims)
}

func batchDotStridedSIMDEligible(rows, dims, stride, queryLen int) bool {
	if rows < batchDotRows || dims < batchDotMinDims || stride <= 0 || queryLen < dims {
		return false
	}
	return batchDotRowsWithinGate(rows, dims)
}

// batchDotRowsWithinGate reports whether a (rows, dims) shape is small enough
// that the batched SIMD kernel is expected to beat the per-row fallback. The
// gate is monotone in both dimensions: larger shapes fall back first.
func batchDotRowsWithinGate(rows, dims int) bool {
	switch {
	case dims >= batchDotHugeDims:
		return rows < batchDotHugeMaxRows
	case dims >= batchDotLargeDims:
		return rows < batchDotLargeMaxRows
	default:
		return true
	}
}
