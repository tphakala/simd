//go:build amd64

package f32

import (
	"unsafe"

	"github.com/tphakala/simd/cpu"
)

// Minimum number of float32 elements required for SIMD operations.
// AVX processes 8 float32 values per 256-bit register.
// AVX-512 processes 16 float32 values per 512-bit register.
const (
	minAVXElements    = 8
	minAVX512Elements = 16
)

// minSIMDElements is set at init time based on which SIMD implementation is selected.
// Used by min32/max32 to determine when to fall back to scalar code.
var minSIMDElements = minAVXElements

// Function pointer types for SIMD operations
type (
	dotProductFunc       func(a, b []float32) float32
	binaryOpFunc         func(dst, a, b []float32)
	scaleFunc            func(dst, a []float32, s float32)
	unaryOpFunc          func(dst, a []float32)
	reduceFunc           func(a []float32) float32
	reduceIdxFunc        func(a []float32) int
	fmaFunc              func(dst, a, b, c []float32)
	clampFunc            func(dst, a []float32, minVal, maxVal float32)
	addScaledFunc        func(dst []float32, alpha float32, s []float32)
	convolveDecimateFunc func(dst, signal, kernel []float32, factor, phase int)
)

// Function pointers - assigned at init time based on CPU features
var (
	dotProductImpl       dotProductFunc
	addImpl              binaryOpFunc
	subImpl              binaryOpFunc
	mulImpl              binaryOpFunc
	divImpl              binaryOpFunc
	scaleImpl            scaleFunc
	addScalarImpl        scaleFunc
	sumImpl              reduceFunc
	minImpl              reduceFunc
	maxImpl              reduceFunc
	absImpl              unaryOpFunc
	negImpl              unaryOpFunc
	sqrtImpl             unaryOpFunc
	reciprocalImpl       unaryOpFunc
	fmaImpl              fmaFunc
	clampImpl            clampFunc
	minIdxImpl           reduceIdxFunc
	maxIdxImpl           reduceIdxFunc
	addScaledImpl        addScaledFunc
	convolveDecimateImpl convolveDecimateFunc
)

func init() {
	// Select optimal implementation based on CPU features
	// Priority: AVX-512 > AVX+FMA > SSE2 > Go
	switch {
	case cpu.X86.AVX512F && cpu.X86.AVX512VL:
		initAVX512()
	case cpu.X86.AVX && cpu.X86.FMA:
		initAVX()
	case cpu.X86.SSE2:
		initSSE()
	default:
		initGo()
	}
}

func initAVX512() {
	minSIMDElements = minAVX512Elements
	dotProductImpl = dotProductAVX512
	addImpl = addAVX512
	subImpl = subAVX512
	mulImpl = mulAVX512
	divImpl = divAVX512
	scaleImpl = scaleAVX512
	addScalarImpl = addScalarAVX512
	sumImpl = sumAVX512
	minImpl = minAVX512
	maxImpl = maxAVX512
	absImpl = absAVX512
	negImpl = negAVX512
	sqrtImpl = sqrtAVX512
	reciprocalImpl = reciprocalAVX512
	fmaImpl = fmaAVX512
	clampImpl = clampAVX512
	minIdxImpl = minIdxGo
	maxIdxImpl = maxIdxGo
	addScaledImpl = addScaledAVX512
	convolveDecimateImpl = convolveDecimateAVX512
}

func initAVX() {
	dotProductImpl = dotProductAVX
	addImpl = addAVX
	subImpl = subAVX
	mulImpl = mulAVX
	divImpl = divAVX
	scaleImpl = scaleAVX
	addScalarImpl = addScalarAVX
	sumImpl = sumAVX
	minImpl = minAVX
	maxImpl = maxAVX
	absImpl = absAVX
	negImpl = negAVX
	sqrtImpl = sqrtAVX
	reciprocalImpl = reciprocalAVX
	fmaImpl = fmaAVX
	clampImpl = clampAVX
	minIdxImpl = minIdxGo
	maxIdxImpl = maxIdxGo
	addScaledImpl = addScaledAVX
	convolveDecimateImpl = convolveDecimateAVX
}

func initSSE() {
	dotProductImpl = dotProductSSE
	addImpl = addSSE
	subImpl = subSSE
	mulImpl = mulSSE
	divImpl = divSSE
	scaleImpl = scaleSSE
	addScalarImpl = addScalarSSE
	sumImpl = sumSSE
	minImpl = minSSE
	maxImpl = maxSSE
	absImpl = absSSE
	negImpl = negSSE
	sqrtImpl = sqrtSSE
	reciprocalImpl = reciprocalSSE
	fmaImpl = fmaSSE
	clampImpl = clampSSE
	minIdxImpl = minIdxGo
	maxIdxImpl = maxIdxGo
	addScaledImpl = addScaledSSE
	convolveDecimateImpl = convolveDecimateSSE
}

func initGo() {
	dotProductImpl = dotProductGo
	addImpl = addGo
	subImpl = subGo
	mulImpl = mulGo
	divImpl = divGo
	scaleImpl = scaleGo
	addScalarImpl = addScalarGo
	sumImpl = sumGo
	minImpl = minGo
	maxImpl = maxGo
	absImpl = absGo
	negImpl = negGo
	sqrtImpl = sqrt32Go
	reciprocalImpl = reciprocal32Go
	fmaImpl = fmaGo
	clampImpl = clampGo
	minIdxImpl = minIdxGo
	maxIdxImpl = maxIdxGo
	addScaledImpl = addScaledGo
	convolveDecimateImpl = convolveDecimate32Go
}

// Dispatch functions - call function pointers (zero overhead after init)

func dotProduct(a, b []float32) float32 {
	return dotProductImpl(a, b)
}

func add(dst, a, b []float32) {
	addImpl(dst, a, b)
}

func sub(dst, a, b []float32) {
	subImpl(dst, a, b)
}

func mul(dst, a, b []float32) {
	mulImpl(dst, a, b)
}

func div(dst, a, b []float32) {
	divImpl(dst, a, b)
}

func scale(dst, a []float32, s float32) {
	scaleImpl(dst, a, s)
}

func addScalar(dst, a []float32, s float32) {
	addScalarImpl(dst, a, s)
}

func sum(a []float32) float32 {
	return sumImpl(a)
}

func min32(a []float32) float32 {
	// AVX/AVX-512 requires at least 8/16 elements for initial vector load
	// Fall back to Go for small slices to avoid reading beyond bounds
	if len(a) < minSIMDElements {
		return minGo(a)
	}
	return minImpl(a)
}

func max32(a []float32) float32 {
	// AVX/AVX-512 requires at least 8/16 elements for initial vector load
	// Fall back to Go for small slices to avoid reading beyond bounds
	if len(a) < minSIMDElements {
		return maxGo(a)
	}
	return maxImpl(a)
}

func abs32(dst, a []float32) {
	absImpl(dst, a)
}

func neg32(dst, a []float32) {
	negImpl(dst, a)
}

func fma32(dst, a, b, c []float32) {
	fmaImpl(dst, a, b, c)
}

func clamp32(dst, a []float32, minVal, maxVal float32) {
	clampImpl(dst, a, minVal, maxVal)
}

func sqrt32(dst, a []float32) {
	sqrtImpl(dst, a)
}

func reciprocal32(dst, a []float32) {
	reciprocalImpl(dst, a)
}

func minIdx32(a []float32) int {
	return minIdxImpl(a)
}

func maxIdx32(a []float32) int {
	return maxIdxImpl(a)
}

func addScaled32(dst []float32, alpha float32, s []float32) {
	addScaledImpl(dst, alpha, s)
}

func cumulativeSum32(dst, a []float32) {
	// CumulativeSum is inherently sequential
	cumulativeSum32Go(dst, a)
}

func dotProductBatch32(results []float32, rows [][]float32, vec []float32) {
	vecLen := len(vec)
	if vecLen == 0 {
		for i := range rows {
			results[i] = 0
		}
		return
	}
	if cpu.X86.AVX512F && cpu.X86.AVX512VL && len(rows) >= 4 && vecLen >= minAVX512Elements {
		i := 0
		for i+3 < len(rows) {
			row0, row1, row2, row3 := rows[i], rows[i+1], rows[i+2], rows[i+3]
			if len(row0) >= vecLen && len(row1) >= vecLen && len(row2) >= vecLen && len(row3) >= vecLen {
				dotProduct4AVX512(
					(*float32)(unsafe.Pointer(&results[i])),
					(*float32)(unsafe.Pointer(&row0[0])),
					(*float32)(unsafe.Pointer(&row1[0])),
					(*float32)(unsafe.Pointer(&row2[0])),
					(*float32)(unsafe.Pointer(&row3[0])),
					(*float32)(unsafe.Pointer(&vec[0])),
					vecLen,
				)
				i += 4
				continue
			}
			for j := range 4 {
				row := rows[i+j]
				n := min(len(row), vecLen)
				if n == 0 {
					results[i+j] = 0
				} else {
					results[i+j] = dotProduct(row[:n], vec[:n])
				}
			}
			i += 4
		}
		for ; i < len(rows); i++ {
			row := rows[i]
			n := min(len(row), vecLen)
			if n == 0 {
				results[i] = 0
			} else {
				results[i] = dotProduct(row[:n], vec[:n])
			}
		}
		return
	}
	for i, row := range rows {
		n := min(len(row), vecLen)
		if n == 0 {
			results[i] = 0
			continue
		}
		results[i] = dotProduct(row[:n], vec[:n])
	}
}

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

// dotProduct4Batch scores four full rows (base[off0..off3], each dims long)
// against query and writes the four results starting at dst[di]. It centralizes
// the unsafe pointer setup and the AVX-512/AVX kernel selection shared by the
// indexed and strided batch loops.
func dotProduct4Batch(useAVX512 bool, dst []float32, di int, base []float32, off0, off1, off2, off3 int, query []float32, dims int) {
	results := (*float32)(unsafe.Pointer(&dst[di]))
	r0 := (*float32)(unsafe.Pointer(&base[off0]))
	r1 := (*float32)(unsafe.Pointer(&base[off1]))
	r2 := (*float32)(unsafe.Pointer(&base[off2]))
	r3 := (*float32)(unsafe.Pointer(&base[off3]))
	q := (*float32)(unsafe.Pointer(&query[0]))
	if useAVX512 {
		dotProduct4AVX512(results, r0, r1, r2, r3, q, dims)
	} else {
		dotProduct4AVX(results, r0, r1, r2, r3, q, dims)
	}
}

func dotProductIndexed(dst, base, query []float32, rowIDs []uint32, dims int) bool {
	n := min(len(dst), len(rowIDs))
	if n == 0 {
		return false
	}
	if !batchDotIndexedSIMDEligible(n, dims, len(query)) {
		dotProductIndexedFallback(dst[:n], base, query, rowIDs[:n], dims)
		return false
	}
	maxRow := fullRowMaxIndex(len(base), dims, dims)
	if maxRow < 0 {
		dotProductIndexedFallback(dst[:n], base, query, rowIDs[:n], dims)
		return false
	}
	useAVX512 := cpu.X86.AVX512F && cpu.X86.AVX512VL
	useAVX := !useAVX512 && cpu.X86.AVX2 && cpu.X86.FMA
	if !useAVX512 && !useAVX {
		dotProductIndexedFallback(dst[:n], base, query, rowIDs[:n], dims)
		return false
	}

	queryFull := query[:dims]
	usedSIMD := false
	i := 0
	for ; i+batchDotRows-1 < n; i += batchDotRows {
		id0, id1, id2, id3 := rowIDs[i], rowIDs[i+1], rowIDs[i+2], rowIDs[i+3]
		if rowIDInFullRange(id0, maxRow) && rowIDInFullRange(id1, maxRow) && rowIDInFullRange(id2, maxRow) && rowIDInFullRange(id3, maxRow) {
			off0 := int(id0) * dims
			off1 := int(id1) * dims
			off2 := int(id2) * dims
			off3 := int(id3) * dims
			dotProduct4Batch(useAVX512, dst, i, base, off0, off1, off2, off3, queryFull, dims)
			usedSIMD = true
			continue
		}
		for j := range batchDotRows {
			dst[i+j] = dotProductIndexedTail(base, query, queryFull, rowIDs[i+j], dims, maxRow)
		}
	}
	for ; i < n; i++ {
		dst[i] = dotProductIndexedTail(base, query, queryFull, rowIDs[i], dims, maxRow)
	}
	return usedSIMD
}

func dotProductStrided(dst, base, query []float32, rowCount, dims, stride int) bool {
	if rowCount <= 0 || len(dst) == 0 {
		return false
	}
	n := min(len(dst), rowCount)
	if !batchDotStridedSIMDEligible(n, dims, stride, len(query)) {
		dotProductStridedFallback(dst[:n], base, query, n, dims, stride)
		return false
	}
	maxRow := fullRowMaxIndex(len(base), dims, stride)
	if maxRow < 0 {
		dotProductStridedFallback(dst[:n], base, query, n, dims, stride)
		return false
	}
	useAVX512 := cpu.X86.AVX512F && cpu.X86.AVX512VL
	useAVX := !useAVX512 && cpu.X86.AVX2 && cpu.X86.FMA
	if !useAVX512 && !useAVX {
		dotProductStridedFallback(dst[:n], base, query, n, dims, stride)
		return false
	}

	queryFull := query[:dims]
	usedSIMD := false
	i := 0
	for ; i+batchDotRows-1 < n; i += batchDotRows {
		if i+batchDotRows-1 <= maxRow {
			off0 := i * stride
			off1 := off0 + stride
			off2 := off1 + stride
			off3 := off2 + stride
			dotProduct4Batch(useAVX512, dst, i, base, off0, off1, off2, off3, queryFull, dims)
			usedSIMD = true
			continue
		}
		for j := range batchDotRows {
			dst[i+j] = dotProductStridedTail(base, query, queryFull, i+j, dims, stride, maxRow)
		}
	}
	for ; i < n; i++ {
		dst[i] = dotProductStridedTail(base, query, queryFull, i, dims, stride, maxRow)
	}
	return usedSIMD
}

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

func convolveValid32(dst, signal, kernel []float32) {
	kLen := len(kernel)
	for i := range dst {
		dst[i] = dotProduct(signal[i:i+kLen], kernel)
	}
}

func convolveDecimate32(dst, signal, kernel []float32, factor, phase int) {
	convolveDecimateImpl(dst, signal, kernel, factor, phase)
}

func accumulateAdd32(dst, src []float32) {
	// AccumulateAdd is dst += src, which is the same as add(dst, dst, src)
	addImpl(dst, dst, src)
}

func interleave2_32(dst, a, b []float32) {
	// Need at least 8 pairs for SIMD to be worthwhile (AVX processes 8 at a time)
	if len(a) >= minAVXElements {
		interleave2AVX(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2_32(a, b, src []float32) {
	if len(a) >= minAVXElements {
		deinterleave2AVX(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

// interleaveN32 interleaves nc = len(srcs) planar streams into dst. N == 2
// reuses the existing Interleave2 SIMD; other small N use shuffle-based
// transposes (added incrementally); the rest fall back to the generic Go path.
// Stream counts with dedicated AMD64 SIMD interleave/deinterleave kernels (the
// 2-stream path reuses interleave2Channels). interleave4BlockMask/8 align a
// frame count down to a whole SIMD block; the caller handles the remainder.
const (
	interleave4Streams   = 4
	interleave8Streams   = 8
	interleave4BlockMask = interleave4Streams - 1
	interleave8BlockMask = interleave8Streams - 1
)

func interleaveN32(dst []float32, srcs [][]float32, n int) {
	switch len(srcs) {
	case interleave2Channels:
		interleave2_32(dst[:n*interleave2Channels], srcs[0][:n], srcs[1][:n])
	case interleave4Streams:
		if cpu.X86.AVX && n >= interleave4Streams {
			blk := n &^ interleave4BlockMask
			interleave4AVX(dst, srcs[0], srcs[1], srcs[2], srcs[3], blk)
			interleaveNTailGo(dst, srcs, blk, n)
			return
		}
		interleaveNGo(dst, srcs, n)
	case interleave8Streams:
		if cpu.X86.AVX && n >= interleave8Streams {
			blk := n &^ interleave8BlockMask
			interleave8AVX(dst, srcs, blk)
			interleaveNTailGo(dst, srcs, blk, n)
			return
		}
		interleaveNGo(dst, srcs, n)
	default:
		interleaveNGo(dst, srcs, n)
	}
}

// interleaveNTailGo writes the trailing frames [from, n) of an N-stream
// interleave that an asm kernel left after processing whole SIMD blocks. It
// reslices nothing in srcs, so it stays allocation-free.
func interleaveNTailGo(dst []float32, srcs [][]float32, from, n int) {
	nc := len(srcs)
	for i := from; i < n; i++ {
		base := i * nc
		for c := range nc {
			dst[base+c] = srcs[c][i]
		}
	}
}

// deinterleaveNTailGo writes the trailing frames [from, n) of an N-stream
// deinterleave, allocation-free.
func deinterleaveNTailGo(dsts [][]float32, src []float32, from, n int) {
	nc := len(dsts)
	for i := from; i < n; i++ {
		base := i * nc
		for c := range nc {
			dsts[c][i] = src[base+c]
		}
	}
}

// deinterleaveN32 splits src into nc = len(dsts) planar streams. N == 2 reuses
// the existing Deinterleave2 SIMD; the rest fall back to the generic Go path.
func deinterleaveN32(dsts [][]float32, src []float32, n int) {
	switch len(dsts) {
	case interleave2Channels:
		deinterleave2_32(dsts[0][:n], dsts[1][:n], src[:n*interleave2Channels])
	case interleave4Streams:
		if cpu.X86.AVX && n >= interleave4Streams {
			blk := n &^ interleave4BlockMask
			deinterleave4AVX(dsts[0], dsts[1], dsts[2], dsts[3], src, blk)
			deinterleaveNTailGo(dsts, src, blk, n)
			return
		}
		deinterleaveNGo(dsts, src, n)
	case interleave8Streams:
		if cpu.X86.AVX && n >= interleave8Streams {
			blk := n &^ interleave8BlockMask
			deinterleave8AVX(dsts, src, blk)
			deinterleaveNTailGo(dsts, src, blk, n)
			return
		}
		deinterleaveNGo(dsts, src, n)
	default:
		deinterleaveNGo(dsts, src, n)
	}
}

func convolveValidMulti32(dsts [][]float32, signal []float32, kernels [][]float32, n, _ int) {
	// Kernel-major loop order: each kernel stays hot in cache for entire signal pass
	for k, kernel := range kernels {
		convolveValid32(dsts[k][:n], signal, kernel)
	}
}

// AVX+FMA assembly function declarations (8x float32 per iteration)
//
//go:noescape
func dotProductAVX(a, b []float32) float32

//go:noescape
func convolveDecimateAVX(dst, signal, kernel []float32, factor, phase int)

//go:noescape
func convolveDecimateAVX512(dst, signal, kernel []float32, factor, phase int)

//go:noescape
func convolveDecimateSSE(dst, signal, kernel []float32, factor, phase int)

//go:noescape
func dotProduct4AVX(results, row0, row1, row2, row3, vec *float32, n int)

//go:noescape
func addAVX(dst, a, b []float32)

//go:noescape
func subAVX(dst, a, b []float32)

//go:noescape
func mulAVX(dst, a, b []float32)

//go:noescape
func divAVX(dst, a, b []float32)

//go:noescape
func scaleAVX(dst, a []float32, s float32)

//go:noescape
func addScalarAVX(dst, a []float32, s float32)

//go:noescape
func sumAVX(a []float32) float32

//go:noescape
func minAVX(a []float32) float32

//go:noescape
func maxAVX(a []float32) float32

//go:noescape
func absAVX(dst, a []float32)

//go:noescape
func negAVX(dst, a []float32)

//go:noescape
func fmaAVX(dst, a, b, c []float32)

//go:noescape
func clampAVX(dst, a []float32, minVal, maxVal float32)

//go:noescape
func clampScaleAVX(dst, src []float32, minVal, maxVal, scale float32)

//go:noescape
func sqrtAVX(dst, a []float32)

//go:noescape
func reciprocalAVX(dst, a []float32)

//go:noescape
func addScaledAVX(dst []float32, alpha float32, s []float32)

// AVX-512 assembly function declarations (16x float32 per iteration)
//
//go:noescape
func dotProductAVX512(a, b []float32) float32

//go:noescape
func dotProduct4AVX512(results, row0, row1, row2, row3, vec *float32, n int)

//go:noescape
func addAVX512(dst, a, b []float32)

//go:noescape
func subAVX512(dst, a, b []float32)

//go:noescape
func mulAVX512(dst, a, b []float32)

//go:noescape
func divAVX512(dst, a, b []float32)

//go:noescape
func scaleAVX512(dst, a []float32, s float32)

//go:noescape
func addScalarAVX512(dst, a []float32, s float32)

//go:noescape
func sumAVX512(a []float32) float32

//go:noescape
func minAVX512(a []float32) float32

//go:noescape
func maxAVX512(a []float32) float32

//go:noescape
func absAVX512(dst, a []float32)

//go:noescape
func negAVX512(dst, a []float32)

//go:noescape
func fmaAVX512(dst, a, b, c []float32)

//go:noescape
func clampAVX512(dst, a []float32, minVal, maxVal float32)

//go:noescape
func sqrtAVX512(dst, a []float32)

//go:noescape
func reciprocalAVX512(dst, a []float32)

//go:noescape
func addScaledAVX512(dst []float32, alpha float32, s []float32)

// SSE assembly function declarations (4x float32 per iteration)
//
//go:noescape
func dotProductSSE(a, b []float32) float32

//go:noescape
func addSSE(dst, a, b []float32)

//go:noescape
func subSSE(dst, a, b []float32)

//go:noescape
func mulSSE(dst, a, b []float32)

//go:noescape
func divSSE(dst, a, b []float32)

//go:noescape
func scaleSSE(dst, a []float32, s float32)

//go:noescape
func addScalarSSE(dst, a []float32, s float32)

//go:noescape
func sumSSE(a []float32) float32

//go:noescape
func minSSE(a []float32) float32

//go:noescape
func maxSSE(a []float32) float32

//go:noescape
func absSSE(dst, a []float32)

//go:noescape
func negSSE(dst, a []float32)

//go:noescape
func fmaSSE(dst, a, b, c []float32)

//go:noescape
func clampSSE(dst, a []float32, minVal, maxVal float32)

//go:noescape
func sqrtSSE(dst, a []float32)

//go:noescape
func reciprocalSSE(dst, a []float32)

//go:noescape
func addScaledSSE(dst []float32, alpha float32, s []float32)

// Interleave/Deinterleave assembly function declarations
//
//go:noescape
func interleave2AVX(dst, a, b []float32)

//go:noescape
func deinterleave2AVX(a, b, src []float32)

//go:noescape
func interleave4AVX(dst, s0, s1, s2, s3 []float32, n int)

//go:noescape
func deinterleave4AVX(d0, d1, d2, d3, src []float32, n int)

//go:noescape
func interleave8AVX(dst []float32, srcs [][]float32, n int)

//go:noescape
func deinterleave8AVX(dsts [][]float32, src []float32, n int)

func variance32(a []float32, mean float32) float32 {
	return variance32Go(a, mean)
}

func euclideanDistance32(a, b []float32) float32 {
	return euclideanDistance32Go(a, b)
}

func cubicInterpDot32(hist, a, b, c, d []float32, x float32) float32 {
	// Use AVX+FMA if available and have enough elements
	if cpu.X86.AVX && cpu.X86.FMA && len(hist) >= minAVXElements {
		return cubicInterpDotAVX(hist, a, b, c, d, x)
	}
	return cubicInterpDotGo(hist, a, b, c, d, x)
}

// CubicInterpDot assembly function declaration
//
//go:noescape
func cubicInterpDotAVX(hist, a, b, c, d []float32, x float32) float32

func sigmoid32(dst, src []float32) {
	// Requires AVX2: sigmoidAVX reconstructs 2^k with 256-bit YMM integer ops
	// (VCVTPS2DQ/VPSLLD/VPADDD) that do not exist on AVX1-only CPUs. Gating on
	// AVX+FMA let AVX1+FMA parts (e.g. AMD Piledriver) reach an AVX2-only kernel
	// and fault with SIGILL. FMA is not used here, so AVX2 alone is correct.
	if cpu.X86.AVX2 && len(dst) >= minAVXElements && len(src) >= minAVXElements {
		sigmoidAVX(dst, src)
		return
	}
	sigmoid32Go(dst, src)
}

// Sigmoid assembly function declaration
//
//go:noescape
func sigmoidAVX(dst, src []float32)

func relu32(dst, src []float32) {
	if cpu.X86.AVX && len(dst) >= minAVXElements && len(src) >= minAVXElements {
		reluAVX(dst, src)
		return
	}
	relu32Go(dst, src)
}

//go:noescape
func reluAVX(dst, src []float32)

func clampScale32(dst, src []float32, minVal, maxVal, scale float32) {
	if cpu.X86.AVX && len(dst) >= minAVXElements && len(src) >= minAVXElements {
		clampScaleAVX(dst, src, minVal, maxVal, scale)
		return
	}
	clampScale32Go(dst, src, minVal, maxVal, scale)
}

func tanh32(dst, src []float32) {
	// Requires AVX2: tanhAVX reconstructs 2^k with 256-bit YMM integer ops
	// (VCVTPS2DQ/VPSLLD/VPADDD) that do not exist on AVX1-only CPUs. Gating on
	// plain AVX let AVX1 parts reach an AVX2-only kernel and fault with SIGILL.
	// FMA is not used here, so AVX2 alone is correct.
	if cpu.X86.AVX2 && len(dst) >= minAVXElements && len(src) >= minAVXElements {
		tanhAVX(dst, src)
		return
	}
	tanh32Go(dst, src)
}

//go:noescape
func tanhAVX(dst, src []float32)

func exp32(dst, src []float32) {
	// Requires AVX2: the 2^k reconstruction uses 256-bit integer ops
	// (VPSLLD/VPADDD/VCVTPS2DQ on YMM) that are not available on AVX1-only
	// CPUs. FMA is not used, so AVX2 alone is the correct guard.
	if cpu.X86.AVX2 && len(dst) >= minAVXElements && len(src) >= minAVXElements {
		expAVX(dst, src)
		return
	}
	exp32Go(dst, src)
}

//go:noescape
func expAVX(dst, src []float32)

func int32ToFloat32Scale(dst []float32, src []int32, scale float32) {
	// Use AVX if available and have enough elements
	if cpu.X86.AVX && len(dst) >= minAVXElements {
		int32ToFloat32ScaleAVX(dst, src, scale)
		return
	}
	int32ToFloat32ScaleGo(dst, src, scale)
}

//go:noescape
func int32ToFloat32ScaleAVX(dst []float32, src []int32, scale float32)

func int16ToFloat32Scale(dst []float32, src []int16, scale float32) {
	// AVX2 is required because widening int16 to int32 uses VPMOVSXWD (ymm form).
	if cpu.X86.AVX2 && len(dst) >= minAVXElements {
		int16ToFloat32ScaleAVX(dst, src, scale)
		return
	}
	int16ToFloat32ScaleGo(dst, src, scale)
}

//go:noescape
func int16ToFloat32ScaleAVX(dst []float32, src []int16, scale float32)

func float32ToInt16Scale(dst []int16, src []float32, scale float32) {
	// AVX2 path; the saturating pack (VPACKSSDW) needs AVX2's VEXTRACTF128 split.
	if cpu.X86.AVX2 && len(dst) >= minAVXElements {
		float32ToInt16ScaleAVX(dst, src, scale)
		return
	}
	float32ToInt16ScaleGo(dst, src, scale)
}

//go:noescape
func float32ToInt16ScaleAVX(dst []int16, src []float32, scale float32)

// ============================================================================
// SPLIT-FORMAT COMPLEX OPERATIONS
// ============================================================================

func mulComplex32(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	// Use AVX+FMA if available and have enough elements
	if cpu.X86.AVX && cpu.X86.FMA && len(dstRe) >= minAVXElements {
		mulComplexAVX(dstRe, dstIm, aRe, aIm, bRe, bIm)
		return
	}
	mulComplex32Go(dstRe, dstIm, aRe, aIm, bRe, bIm)
}

func mulConjComplex32(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	// Use AVX+FMA if available and have enough elements
	if cpu.X86.AVX && cpu.X86.FMA && len(dstRe) >= minAVXElements {
		mulConjComplexAVX(dstRe, dstIm, aRe, aIm, bRe, bIm)
		return
	}
	mulConjComplex32Go(dstRe, dstIm, aRe, aIm, bRe, bIm)
}

func absSqComplex32(dst, aRe, aIm []float32) {
	// Use AVX+FMA if available and have enough elements
	if cpu.X86.AVX && cpu.X86.FMA && len(dst) >= minAVXElements {
		absSqComplexAVX(dst, aRe, aIm)
		return
	}
	absSqComplex32Go(dst, aRe, aIm)
}

func butterflyComplex32(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm []float32) {
	// Use AVX+FMA if available and have enough elements
	if cpu.X86.AVX && cpu.X86.FMA && len(upperRe) >= minAVXElements {
		butterflyComplexAVX(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)
		return
	}
	butterflyComplex32Go(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)
}

func realFFTUnpack32(outRe, outIm, zRe, zIm, twRe, twIm []float32, n int) {
	// Use AVX+FMA if available and have enough elements
	// Need at least 9 elements: process k=1..n-1 where n>=9 gives 8+ iterations
	if cpu.X86.AVX && cpu.X86.FMA && n > minAVXElements {
		realFFTUnpackAVX(outRe, outIm, zRe, zIm, twRe, twIm, n)
		return
	}
	realFFTUnpack32Go(outRe, outIm, zRe, zIm, twRe, twIm, n)
}

// Split-format complex assembly function declarations
//
//go:noescape
func mulComplexAVX(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)

//go:noescape
func mulConjComplexAVX(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)

//go:noescape
func absSqComplexAVX(dst, aRe, aIm []float32)

//go:noescape
func butterflyComplexAVX(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm []float32)

//go:noescape
func realFFTUnpackAVX(outRe, outIm, zRe, zIm, twRe, twIm []float32, n int)

func reverse32(dst, src []float32) {
	// Use AVX if available and have enough elements
	if cpu.X86.AVX && len(dst) >= minAVXElements {
		reverseAVX(dst, src)
		return
	}
	reverse32Go(dst, src)
}

func addSub32(sumDst, diffDst, a, b []float32) {
	// Use AVX if available and have enough elements
	if cpu.X86.AVX && len(sumDst) >= minAVXElements {
		addSubAVX(sumDst, diffDst, a, b)
		return
	}
	addSub32Go(sumDst, diffDst, a, b)
}

//go:noescape
func reverseAVX(dst, src []float32)

//go:noescape
func addSubAVX(sumDst, diffDst, a, b []float32)
