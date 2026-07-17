//go:build arm64

package f32

import (
	"math"
	"unsafe"

	"github.com/tphakala/simd/cpu"
)

var (
	hasNEON = cpu.ARM64.NEON
)

func dotProduct(a, b []float32) float32 {
	if hasNEON && len(a) >= 4 {
		return dotProductNEON(a, b)
	}
	return dotProductGo(a, b)
}

func add(dst, a, b []float32) {
	if hasNEON && len(dst) >= 4 {
		addNEON(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func sub(dst, a, b []float32) {
	if hasNEON && len(dst) >= 4 {
		subNEON(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

func mul(dst, a, b []float32) {
	if hasNEON && len(dst) >= 4 {
		mulNEON(dst, a, b)
		return
	}
	mulGo(dst, a, b)
}

func div(dst, a, b []float32) {
	if hasNEON && len(dst) >= 4 {
		divNEON(dst, a, b)
		return
	}
	divGo(dst, a, b)
}

func scale(dst, a []float32, s float32) {
	if hasNEON && len(dst) >= 4 {
		scaleNEON(dst, a, s)
		return
	}
	scaleGo(dst, a, s)
}

func addScalar(dst, a []float32, s float32) {
	if hasNEON && len(dst) >= 4 {
		addScalarNEON(dst, a, s)
		return
	}
	addScalarGo(dst, a, s)
}

func sum(a []float32) float32 {
	if hasNEON && len(a) >= 4 {
		return sumNEON(a)
	}
	return sumGo(a)
}

func min32(a []float32) float32 {
	if hasNEON && len(a) >= 4 {
		return minNEON(a)
	}
	return minGo(a)
}

func max32(a []float32) float32 {
	if hasNEON && len(a) >= 4 {
		return maxNEON(a)
	}
	return maxGo(a)
}

func maxAbs32(a []float32) float32 {
	if hasNEON && len(a) >= 4 {
		return maxAbsNEON(a)
	}
	return maxAbsGo(a)
}

func abs32(dst, a []float32) {
	if hasNEON && len(dst) >= 4 {
		absNEON(dst, a)
		return
	}
	absGo(dst, a)
}

func neg32(dst, a []float32) {
	if hasNEON && len(dst) >= 4 {
		negNEON(dst, a)
		return
	}
	negGo(dst, a)
}

func subFromScalar32(dst, a []float32, s float32) {
	// Compose using already-dispatched primitives: (s - a) == (-a) + s.
	// neg32 and addScalar each gate on hasNEON internally and fall back to pure
	// Go when NEON is unavailable, so no extra guard is needed here.
	neg32(dst, a)
	addScalar(dst, dst, s)
}

func fma32(dst, a, b, c []float32) {
	if hasNEON && len(dst) >= 4 {
		fmaNEON(dst, a, b, c)
		return
	}
	fmaGo(dst, a, b, c)
}

func clamp32(dst, a []float32, minVal, maxVal float32) {
	if hasNEON && len(dst) >= 4 {
		clampNEON(dst, a, minVal, maxVal)
		return
	}
	clampGo(dst, a, minVal, maxVal)
}

func dotProductBatch32(results []float32, rows [][]float32, vec []float32) {
	vecLen := len(vec)
	if vecLen == 0 {
		for i := range rows {
			results[i] = 0
		}
		return
	}
	// Mirror dotProduct's NEON length threshold (>= 4): the batch-of-4 kernel
	// keeps the query vector resident across the group instead of reloading it
	// per row, which is the only win here since the per-row path is already NEON.
	if hasNEON && len(rows) >= batchDotRows && vecLen >= 4 {
		dotProductBatchKernel(results, rows, vec, vecLen)
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

// dotProductBatchKernel scores rows against vec in groups of four, keeping the
// query vector resident across each group via dotProduct4NEON instead of
// reloading it per row. Rows shorter than vecLen (and any tail past the last
// full group of four) fall back to the per-row dotProduct, so results stay
// anchored to the scalar contract regardless of row shape.
func dotProductBatchKernel(results []float32, rows [][]float32, vec []float32, vecLen int) {
	i := 0
	for i+3 < len(rows) {
		row0, row1, row2, row3 := rows[i], rows[i+1], rows[i+2], rows[i+3]
		if len(row0) >= vecLen && len(row1) >= vecLen && len(row2) >= vecLen && len(row3) >= vecLen {
			res := (*float32)(unsafe.Pointer(&results[i]))
			r0 := (*float32)(unsafe.Pointer(&row0[0]))
			r1 := (*float32)(unsafe.Pointer(&row1[0]))
			r2 := (*float32)(unsafe.Pointer(&row2[0]))
			r3 := (*float32)(unsafe.Pointer(&row3[0]))
			q := (*float32)(unsafe.Pointer(&vec[0]))
			dotProduct4NEON(res, r0, r1, r2, r3, q, vecLen)
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
}

func dotProductIndexed(dst, base, query []float32, rowIDs []uint32, dims int) bool {
	n := min(len(dst), len(rowIDs))
	if n == 0 {
		return false
	}
	if !hasNEON || !batchDotIndexedSIMDEligible(n, dims, len(query)) {
		dotProductIndexedFallback(dst[:n], base, query, rowIDs[:n], dims)
		return false
	}
	maxRow := fullRowMaxIndex(len(base), dims, dims)
	if maxRow < 0 {
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
			dotProduct4Batch(dst, i, base, off0, off1, off2, off3, queryFull, dims)
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
	if !hasNEON || !batchDotStridedSIMDEligible(n, dims, stride, len(query)) {
		dotProductStridedFallback(dst[:n], base, query, n, dims, stride)
		return false
	}
	maxRow := fullRowMaxIndex(len(base), dims, stride)
	if maxRow < 0 {
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
			dotProduct4Batch(dst, i, base, off0, off1, off2, off3, queryFull, dims)
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

// dotProduct4Batch scores four full rows (base[off0..off3], each dims long)
// against query and writes the four results starting at dst[di], centralizing
// the unsafe pointer setup shared by the indexed and strided batch loops.
func dotProduct4Batch(dst []float32, di int, base []float32, off0, off1, off2, off3 int, query []float32, dims int) {
	results := (*float32)(unsafe.Pointer(&dst[di]))
	r0 := (*float32)(unsafe.Pointer(&base[off0]))
	r1 := (*float32)(unsafe.Pointer(&base[off1]))
	r2 := (*float32)(unsafe.Pointer(&base[off2]))
	r3 := (*float32)(unsafe.Pointer(&base[off3]))
	q := (*float32)(unsafe.Pointer(&query[0]))
	dotProduct4NEON(results, r0, r1, r2, r3, q, dims)
}

func convolveValid32(dst, signal, kernel []float32) {
	kLen := len(kernel)
	for i := range dst {
		dst[i] = dotProduct(signal[i:i+kLen], kernel)
	}
}

func convolveDecimate32(dst, signal, kernel []float32, factor, phase int) {
	// Mirror dotProduct's NEON length threshold (>= 4) so the fused kernel and a
	// per-window DotProductUnsafe pick the same backend, keeping results identical.
	if hasNEON && len(kernel) >= 4 {
		convolveDecimateNEON(dst, signal, kernel, factor, phase)
		return
	}
	convolveDecimate32Go(dst, signal, kernel, factor, phase)
}

func convolveValidMaxAbs32(signal, kernel []float32) float32 {
	// Mirror dotProduct's NEON threshold (>= 4) so the fused kernel and the
	// per-window dotProduct in ConvolveValid pick the same backend, keeping the
	// peak bit-identical.
	if hasNEON && len(kernel) >= 4 {
		return convolveValidMaxAbsNEON(signal, kernel)
	}
	return convolveValidMaxAbsGo(signal, kernel)
}

//go:noescape
func convolveValidMaxAbsNEON(signal, kernel []float32) float32

//go:noescape
func convolveDecimateNEON(dst, signal, kernel []float32, factor, phase int)

func accumulateAdd32(dst, src []float32) {
	// AccumulateAdd is dst += src, use add with dst as both operands
	if hasNEON && len(dst) >= 4 {
		addNEON(dst, dst, src)
		return
	}
	accumulateAdd32Go(dst, src)
}

//go:noescape
func dotProductNEON(a, b []float32) float32

//go:noescape
func dotProduct4NEON(results, row0, row1, row2, row3, vec *float32, n int)

//go:noescape
func addNEON(dst, a, b []float32)

//go:noescape
func subNEON(dst, a, b []float32)

//go:noescape
func mulNEON(dst, a, b []float32)

//go:noescape
func divNEON(dst, a, b []float32)

//go:noescape
func scaleNEON(dst, a []float32, s float32)

//go:noescape
func addScalarNEON(dst, a []float32, s float32)

//go:noescape
func sumNEON(a []float32) float32

//go:noescape
func minNEON(a []float32) float32

//go:noescape
func maxNEON(a []float32) float32

//go:noescape
func maxAbsNEON(a []float32) float32

//go:noescape
func absNEON(dst, a []float32)

//go:noescape
func negNEON(dst, a []float32)

//go:noescape
func fmaNEON(dst, a, b, c []float32)

//go:noescape
func clampNEON(dst, a []float32, minVal, maxVal float32)

func interleave2_32(dst, a, b []float32) {
	if hasNEON && len(a) >= 4 {
		interleave2NEON(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2_32(a, b, src []float32) {
	if hasNEON && len(a) >= 4 {
		deinterleave2NEON(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

// interleaveN32 interleaves nc = len(srcs) planar streams into dst. N == 2
// reuses the existing Interleave2 NEON; N in {3,4} use the NEON ST3/ST4
// structured stores (added incrementally); the rest fall back to generic Go.
// Stream counts with dedicated ARM64 NEON interleave/deinterleave kernels (the
// 2-stream path reuses interleave2Channels). neonInterleaveBlock is the NEON
// structured load/store block size in frames (4 float32 per .4S register).
const (
	interleave3Streams  = 3
	interleave4Streams  = 4
	interleave6Streams  = 6
	interleave8Streams  = 8
	neonInterleaveBlock = 4
)

func interleaveN32(dst []float32, srcs [][]float32, n int) {
	switch len(srcs) {
	case interleave2Channels:
		interleave2_32(dst[:n*interleave2Channels], srcs[0][:n], srcs[1][:n])
	case interleave3Streams:
		if hasNEON && n >= neonInterleaveBlock {
			interleave3NEON(dst, srcs[0], srcs[1], srcs[2], n)
			return
		}
		interleaveNGo(dst, srcs, n)
	case interleave4Streams:
		if hasNEON && n >= neonInterleaveBlock {
			interleave4NEON(dst, srcs[0], srcs[1], srcs[2], srcs[3], n)
			return
		}
		interleaveNGo(dst, srcs, n)
	case interleave6Streams:
		if hasNEON && n >= neonInterleaveBlock {
			interleave6NEON(dst, srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], n)
			return
		}
		interleaveNGo(dst, srcs, n)
	case interleave8Streams:
		if hasNEON && n >= neonInterleaveBlock {
			interleave8NEON(dst, srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], srcs[6], srcs[7], n)
			return
		}
		interleaveNGo(dst, srcs, n)
	default:
		interleaveNGo(dst, srcs, n)
	}
}

// deinterleaveN32 splits src into nc = len(dsts) planar streams. N == 2 reuses
// the existing Deinterleave2 NEON; N in {3,4} use the NEON LD3/LD4 structured
// loads (added incrementally); the rest fall back to generic Go.
func deinterleaveN32(dsts [][]float32, src []float32, n int) {
	switch len(dsts) {
	case interleave2Channels:
		deinterleave2_32(dsts[0][:n], dsts[1][:n], src[:n*interleave2Channels])
	case interleave3Streams:
		if hasNEON && n >= neonInterleaveBlock {
			deinterleave3NEON(dsts[0], dsts[1], dsts[2], src, n)
			return
		}
		deinterleaveNGo(dsts, src, n)
	case interleave4Streams:
		if hasNEON && n >= neonInterleaveBlock {
			deinterleave4NEON(dsts[0], dsts[1], dsts[2], dsts[3], src, n)
			return
		}
		deinterleaveNGo(dsts, src, n)
	case interleave6Streams:
		if hasNEON && n >= neonInterleaveBlock {
			deinterleave6NEON(dsts[0], dsts[1], dsts[2], dsts[3], dsts[4], dsts[5], src, n)
			return
		}
		deinterleaveNGo(dsts, src, n)
	case interleave8Streams:
		if hasNEON && n >= neonInterleaveBlock {
			deinterleave8NEON(dsts[0], dsts[1], dsts[2], dsts[3], dsts[4], dsts[5], dsts[6], dsts[7], src, n)
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

//go:noescape
func interleave2NEON(dst, a, b []float32)

//go:noescape
func deinterleave2NEON(a, b, src []float32)

//go:noescape
func interleave3NEON(dst, s0, s1, s2 []float32, n int)

//go:noescape
func deinterleave3NEON(d0, d1, d2, src []float32, n int)

//go:noescape
func interleave4NEON(dst, s0, s1, s2, s3 []float32, n int)

//go:noescape
func deinterleave4NEON(d0, d1, d2, d3, src []float32, n int)

//go:noescape
func interleave6NEON(dst, s0, s1, s2, s3, s4, s5 []float32, n int)

//go:noescape
func deinterleave6NEON(d0, d1, d2, d3, d4, d5, src []float32, n int)

//go:noescape
func interleave8NEON(dst, s0, s1, s2, s3, s4, s5, s6, s7 []float32, n int)

//go:noescape
func deinterleave8NEON(d0, d1, d2, d3, d4, d5, d6, d7, src []float32, n int)

func sqrt32(dst, a []float32) {
	if hasNEON && len(dst) >= 4 {
		sqrtNEON(dst, a)
		return
	}
	sqrt32Go(dst, a)
}

func round32(dst, src []float32) {
	if hasNEON && len(dst) >= 4 {
		roundNEON(dst, src)
		return
	}
	round32Go(dst, src)
}

func reciprocal32(dst, a []float32) {
	if hasNEON && len(dst) >= 4 {
		reciprocalNEON(dst, a)
		return
	}
	reciprocal32Go(dst, a)
}

func minIdx32(a []float32) int {
	return minIdxGo(a)
}

func maxIdx32(a []float32) int {
	return maxIdxGo(a)
}

// minIdxOfSumRows32 routes the sliding-window shapes (slide +1 and -1) through
// the NEON block kernel, four rows per lane-per-row block, and composes any
// remainder rows with the scalar path. Non-unit slides and a missing NEON stay
// on the Go reference. All paths are bit-identical (see minIdxOfSumRows4NEON).
func minIdxOfSumRows32(vals []float32, idxs []int32, a, k []float32, base, slide int) {
	if hasNEON && (slide == 1 || slide == -1) {
		r := 0
		for ; r+4 <= len(vals); r += 4 {
			off := base + r*slide
			if slide == 1 {
				minIdxOfSumRows4NEON(vals[r:r+4], idxs[r:r+4], a, k[off:], 0)
			} else {
				minIdxOfSumRows4NEON(vals[r:r+4], idxs[r:r+4], a, k[off-3:], 1)
			}
		}
		for ; r < len(vals); r++ {
			off := base + r*slide
			i, v := minIdxOfSumGo(a, k[off:off+len(a)])
			vals[r] = v
			idxs[r] = int32(i)
		}
		return
	}
	minIdxOfSumRowsGo(vals, idxs, a, k, base, slide)
}

//go:noescape
func minIdxOfSumRows4NEON(vals []float32, idxs []int32, a, k []float32, rev int)

func addScaled32(dst []float32, alpha float32, s []float32) {
	if hasNEON && len(dst) >= 4 {
		addScaledNEON(dst, alpha, s)
		return
	}
	addScaledGo(dst, alpha, s)
}

func cumulativeSum32(dst, a []float32) {
	cumulativeSum32Go(dst, a)
}

//go:noescape
func sqrtNEON(dst, a []float32)

//go:noescape
func roundNEON(dst, src []float32)

//go:noescape
func reciprocalNEON(dst, a []float32)

//go:noescape
func addScaledNEON(dst []float32, alpha float32, s []float32)

func variance32(a []float32, mean float32) float32 {
	if hasNEON && len(a) >= 4 {
		return varianceNEON32(a, mean)
	}
	return variance32Go(a, mean)
}

func euclideanDistance32(a, b []float32) float32 {
	if hasNEON && len(a) >= 4 {
		return euclideanDistanceNEON32(a, b)
	}
	return euclideanDistance32Go(a, b)
}

//go:noescape
func varianceNEON32(a []float32, mean float32) float32

//go:noescape
func euclideanDistanceNEON32(a, b []float32) float32

func cubicInterpDot32(hist, a, b, c, d []float32, x float32) float32 {
	if hasNEON && len(hist) >= 4 {
		return cubicInterpDotNEON(hist, a, b, c, d, x)
	}
	return cubicInterpDotGo(hist, a, b, c, d, x)
}

//go:noescape
func cubicInterpDotNEON(hist, a, b, c, d []float32, x float32) float32

func sigmoid32(dst, src []float32) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 4 {
		sigmoidNEON(dst, src)
		return
	}
	sigmoid32Go(dst, src)
}

//go:noescape
func sigmoidNEON(dst, src []float32)

func relu32(dst, src []float32) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 4 {
		reluNEON(dst, src)
		return
	}
	relu32Go(dst, src)
}

//go:noescape
func reluNEON(dst, src []float32)

func clampScale32(dst, src []float32, minVal, maxVal, scale float32) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 4 {
		clampScaleNEON(dst, src, minVal, maxVal, scale)
		return
	}
	clampScale32Go(dst, src, minVal, maxVal, scale)
}

//go:noescape
func clampScaleNEON(dst, src []float32, minVal, maxVal, scale float32)

func tanh32(dst, src []float32) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 4 {
		tanhNEON(dst, src)
		return
	}
	tanh32Go(dst, src)
}

//go:noescape
func tanhNEON(dst, src []float32)

func exp32(dst, src []float32) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 4 {
		expNEON(dst, src)
		return
	}
	exp32Go(dst, src)
}

//go:noescape
func expNEON(dst, src []float32)

func log32(dst, src []float32) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 4 {
		logNEON32(dst, src, logLn2Hi32, logLn2Lo32, 1.0)
		return
	}
	logGo(dst, src)
}

func log2_32(dst, src []float32) {
	// log2(x) = e + ln(m)*log2(e): the e term is exact, so no hi/lo split is
	// needed and exact powers of two come out exact.
	if hasNEON && len(dst) >= 4 {
		logNEON32(dst, src, 1.0, 0.0, logLog2E32)
		return
	}
	log2Go(dst, src)
}

func log10_32(dst, src []float32) {
	if hasNEON && len(dst) >= 4 {
		logNEON32(dst, src, logL102Hi32, logL102Lo32, logLog10E32)
		return
	}
	log10Go(dst, src)
}

func pow32(dst, src []float32, exp float32) {
	// A zero or non-finite exponent has whole-slice math.Pow semantics
	// (for example Pow(x, 0) = 1 even for NaN x); keep those exact.
	e := float64(exp)
	if hasNEON && len(dst) >= 4 && exp != 0 && !math.IsNaN(e) && !math.IsInf(e, 0) &&
		powSIMDOK32(src[:len(dst)]) {
		powNEON32(dst, src, exp)
		return
	}
	powGo(dst, src, exp)
}

func powElem32(dst, base, exp []float32) {
	if hasNEON && len(dst) >= 4 && powSIMDOK32(base[:len(dst)]) && allFinite32(exp[:len(dst)]) {
		powElemNEON32(dst, base, exp)
		return
	}
	powElemGo(dst, base, exp)
}

//go:noescape
func logNEON32(dst, src []float32, k1hi, k1lo, k2 float32)

//go:noescape
func powNEON32(dst, src []float32, exp float32)

//go:noescape
func powElemNEON32(dst, base, exp []float32)

func int32ToFloat32Scale(dst []float32, src []int32, scale float32) {
	if hasNEON && len(dst) >= 4 {
		int32ToFloat32ScaleNEON(dst, src, scale)
		return
	}
	int32ToFloat32ScaleGo(dst, src, scale)
}

//go:noescape
func int32ToFloat32ScaleNEON(dst []float32, src []int32, scale float32)

func int16ToFloat32Scale(dst []float32, src []int16, scale float32) {
	if hasNEON && len(dst) >= 4 {
		int16ToFloat32ScaleNEON(dst, src, scale)
		return
	}
	int16ToFloat32ScaleGo(dst, src, scale)
}

//go:noescape
func int16ToFloat32ScaleNEON(dst []float32, src []int16, scale float32)

func float32ToInt16Scale(dst []int16, src []float32, scale float32) {
	if hasNEON && len(dst) >= 4 {
		float32ToInt16ScaleNEON(dst, src, scale)
		return
	}
	float32ToInt16ScaleGo(dst, src, scale)
}

//go:noescape
func float32ToInt16ScaleNEON(dst []int16, src []float32, scale float32)

// ============================================================================
// SPLIT-FORMAT COMPLEX OPERATIONS
// ============================================================================

func mulComplex32(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	if hasNEON && len(dstRe) >= 4 {
		mulComplexNEON(dstRe, dstIm, aRe, aIm, bRe, bIm)
		return
	}
	mulComplex32Go(dstRe, dstIm, aRe, aIm, bRe, bIm)
}

func mulConjComplex32(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	if hasNEON && len(dstRe) >= 4 {
		mulConjComplexNEON(dstRe, dstIm, aRe, aIm, bRe, bIm)
		return
	}
	mulConjComplex32Go(dstRe, dstIm, aRe, aIm, bRe, bIm)
}

func absSqComplex32(dst, aRe, aIm []float32) {
	if hasNEON && len(dst) >= 4 {
		absSqComplexNEON(dst, aRe, aIm)
		return
	}
	absSqComplex32Go(dst, aRe, aIm)
}

func butterflyComplex32(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm []float32) {
	if hasNEON && len(upperRe) >= 4 {
		butterflyComplexNEON(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)
		return
	}
	butterflyComplex32Go(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)
}

func realFFTUnpack32(outRe, outIm, zRe, zIm, twRe, twIm []float32, n int) {
	// Use NEON if available and have enough elements
	// Need at least 5 elements: process k=1..n-1 where n>=5 gives 4+ iterations
	if hasNEON && n > 4 {
		realFFTUnpackNEON(outRe, outIm, zRe, zIm, twRe, twIm, n)
		return
	}
	realFFTUnpack32Go(outRe, outIm, zRe, zIm, twRe, twIm, n)
}

//go:noescape
func mulComplexNEON(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)

//go:noescape
func mulConjComplexNEON(dstRe, dstIm, aRe, aIm, bRe, bIm []float32)

//go:noescape
func absSqComplexNEON(dst, aRe, aIm []float32)

//go:noescape
func butterflyComplexNEON(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm []float32)

//go:noescape
func realFFTUnpackNEON(outRe, outIm, zRe, zIm, twRe, twIm []float32, n int)

func reverse32(dst, src []float32) {
	if hasNEON && len(dst) >= 4 {
		reverseNEON(dst, src)
		return
	}
	reverse32Go(dst, src)
}

func addSub32(sumDst, diffDst, a, b []float32) {
	if hasNEON && len(sumDst) >= 4 {
		addSubNEON(sumDst, diffDst, a, b)
		return
	}
	addSub32Go(sumDst, diffDst, a, b)
}

//go:noescape
func reverseNEON(dst, src []float32)

//go:noescape
func addSubNEON(sumDst, diffDst, a, b []float32)
