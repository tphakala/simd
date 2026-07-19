//go:build amd64

package f32

import (
	"math"
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
	dotProductFunc          func(a, b []float32) float32
	binaryOpFunc            func(dst, a, b []float32)
	scaleFunc               func(dst, a []float32, s float32)
	unaryOpFunc             func(dst, a []float32)
	reduceFunc              func(a []float32) float32
	reduceIdxFunc           func(a []float32) int
	fmaFunc                 func(dst, a, b, c []float32)
	clampFunc               func(dst, a []float32, minVal, maxVal float32)
	varianceFunc            func(a []float32, mean float32) float32
	euclideanDistanceFunc   func(a, b []float32) float32
	addScaledFunc           func(dst []float32, alpha float32, s []float32)
	convolveDecimateFunc    func(dst, signal, kernel []float32, factor, phase int)
	convolveValidMaxAbsFunc func(signal, kernel []float32) float32
	interleave2Func         func(dst, a, b []float32)
	deinterleave2Func       func(a, b, src []float32)
)

// Function pointers - assigned at init time based on CPU features
var (
	dotProductImpl          dotProductFunc
	addImpl                 binaryOpFunc
	subImpl                 binaryOpFunc
	mulImpl                 binaryOpFunc
	divImpl                 binaryOpFunc
	scaleImpl               scaleFunc
	addScalarImpl           scaleFunc
	sumImpl                 reduceFunc
	minImpl                 reduceFunc
	maxImpl                 reduceFunc
	maxAbsImpl              reduceFunc
	absImpl                 unaryOpFunc
	negImpl                 unaryOpFunc
	sqrtImpl                unaryOpFunc
	reciprocalImpl          unaryOpFunc
	roundImpl               unaryOpFunc
	fmaImpl                 fmaFunc
	clampImpl               clampFunc
	varianceImpl            varianceFunc
	euclideanDistanceImpl   euclideanDistanceFunc
	minIdxImpl              reduceIdxFunc
	maxIdxImpl              reduceIdxFunc
	addScaledImpl           addScaledFunc
	convolveDecimateImpl    convolveDecimateFunc
	convolveValidMaxAbsImpl convolveValidMaxAbsFunc
	interleave2Impl         interleave2Func
	deinterleave2Impl       deinterleave2Func
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
	// AVX-512 reuses the AVX2 MaxAbs kernel (no AVX-512 hardware to verify a
	// dedicated VL kernel; see #138), mirroring variance/euclidean above.
	maxAbsImpl = maxAbsAVX
	absImpl = absAVX512
	negImpl = negAVX512
	sqrtImpl = sqrtAVX512
	reciprocalImpl = reciprocalAVX512
	roundImpl = roundAVX
	fmaImpl = fmaAVX512
	clampImpl = clampAVX512
	// AVX-512 variance/euclidean kernels are out of scope (no AVX-512 hardware to
	// verify them; see #75/#96); reuse the AVX kernels so the tier still benefits.
	varianceImpl = varianceAVX
	euclideanDistanceImpl = euclideanDistanceAVX
	minIdxImpl = minIdxGo
	maxIdxImpl = maxIdxGo
	addScaledImpl = addScaledAVX512
	convolveDecimateImpl = convolveDecimateAVX512
	// AVX-512 keeps the Go-level fusion over the 16-wide dotProductAVX512: the
	// fused kernel is 8-wide (AVX2) and its dot summation order would diverge from
	// ConvolveValid here. A dedicated 16-wide fused kernel needs AVX-512 hardware to
	// validate (none available; see #138), so the AVX-512 tier forgoes the fused
	// speedup but stays bit-identical to ConvolveValid.
	convolveValidMaxAbsImpl = convolveValidMaxAbsGo
	// AVX-512 CPUs also support AVX, so the AVX1 interleave kernels are safe.
	interleave2Impl, deinterleave2Impl = interleave2Kernels(true)
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
	maxAbsImpl = maxAbsAVX
	absImpl = absAVX
	negImpl = negAVX
	sqrtImpl = sqrtAVX
	reciprocalImpl = reciprocalAVX
	roundImpl = roundAVX
	fmaImpl = fmaAVX
	clampImpl = clampAVX
	varianceImpl = varianceAVX
	euclideanDistanceImpl = euclideanDistanceAVX
	minIdxImpl = minIdxGo
	maxIdxImpl = maxIdxGo
	addScaledImpl = addScaledAVX
	convolveDecimateImpl = convolveDecimateAVX
	convolveValidMaxAbsImpl = convolveValidMaxAbsAVX
	interleave2Impl, deinterleave2Impl = interleave2Kernels(true)
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
	maxAbsImpl = maxAbsSSE
	absImpl = absSSE
	negImpl = negSSE
	sqrtImpl = sqrtSSE
	reciprocalImpl = reciprocalSSE
	roundImpl = round32Go
	fmaImpl = fmaSSE
	clampImpl = clampSSE
	varianceImpl = varianceSSE
	euclideanDistanceImpl = euclideanDistanceSSE
	minIdxImpl = minIdxGo
	maxIdxImpl = maxIdxGo
	addScaledImpl = addScaledSSE
	convolveDecimateImpl = convolveDecimateSSE
	convolveValidMaxAbsImpl = convolveValidMaxAbsSSE
	// No SSE2 interleave2 kernel exists; the AVX kernel would SIGILL here, so
	// fall back to the scalar Go path on AVX-less CPUs (birdnet-go issue #3353).
	interleave2Impl, deinterleave2Impl = interleave2Kernels(false)
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
	maxAbsImpl = maxAbsGo
	absImpl = absGo
	negImpl = negGo
	sqrtImpl = sqrt32Go
	reciprocalImpl = reciprocal32Go
	roundImpl = round32Go
	fmaImpl = fmaGo
	clampImpl = clampGo
	varianceImpl = variance32Go
	euclideanDistanceImpl = euclideanDistance32Go
	minIdxImpl = minIdxGo
	maxIdxImpl = maxIdxGo
	addScaledImpl = addScaledGo
	convolveDecimateImpl = convolveDecimate32Go
	convolveValidMaxAbsImpl = convolveValidMaxAbsGo
	interleave2Impl, deinterleave2Impl = interleave2Kernels(false)
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

func maxAbs32(a []float32) float32 {
	// The SIMD kernels do a full-width initial vector load; fall back to Go for
	// small slices to avoid reading beyond bounds (mirrors min32/max32).
	if len(a) < minSIMDElements {
		return maxAbsGo(a)
	}
	return maxAbsImpl(a)
}

func abs32(dst, a []float32) {
	absImpl(dst, a)
}

func neg32(dst, a []float32) {
	negImpl(dst, a)
}

// copySign32 dispatches CopySign directly to a //go:noescape kernel (rather than
// through a function-pointer table) so the caller's slices do not escape to the
// heap: an indirect call through a func value defeats //go:noescape and forces a
// per-call allocation, breaking the zero-allocation contract. The direct-dispatch
// shape mirrors absPow34_32. CopySign is plain bit manipulation (VANDPS/VORPS),
// so plain AVX gates the VEX kernel (AVX-512 CPUs also have AVX and run it
// bit-identically) and SSE2 covers the rest. Like abs/neg, no minimum-length
// guard is needed: the kernels compute the block count with a shift and fall
// through to a scalar tail, so a full-width load is never issued out of bounds.
func copySign32(dst, mag, sign []float32) {
	switch {
	case cpu.X86.AVX:
		copySignAVX(dst, mag, sign)
	case cpu.X86.SSE2:
		copySignSSE(dst, mag, sign)
	default:
		copySign32Go(dst, mag, sign)
	}
}

func subFromScalar32(dst, a []float32, s float32) {
	// Compose using already-dispatched primitives: (s - a) == (-a) + s.
	// Each step is internally vectorized or falls back to Go via the global impl
	// pointers, so this works on every supported CPU without an extra guard.
	neg32(dst, a)
	addScalar(dst, dst, s)
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

// absPow34_32 dispatches AbsPow34 directly to a //go:noescape kernel (rather than
// through a function-pointer table) so the caller's dst/src do not escape to the
// heap: an indirect call through a func value defeats //go:noescape and forces a
// per-call allocation, which would break the zero-allocation contract. The
// direct-dispatch shape mirrors the exp/relu/pow family in this file. AbsPow34
// needs no FMA, so plain AVX gates the VEX kernel (AVX-512 CPUs also have AVX and
// run it bit-identically). Unlike exp/relu/pow, no minimum-length guard is needed:
// like the element-wise abs/neg/sqrt kernels, these compute the block count with a
// shift and fall through to a scalar tail for short inputs, so a full-width load is
// never issued out of bounds.
func absPow34_32(dst, src []float32) {
	switch {
	case cpu.X86.AVX:
		absPow34AVX(dst, src)
	case cpu.X86.SSE2:
		absPow34SSE(dst, src)
	default:
		absPow34Go(dst, src)
	}
}

func reciprocal32(dst, a []float32) {
	reciprocalImpl(dst, a)
}

func round32(dst, src []float32) {
	roundImpl(dst, src)
}

func minIdx32(a []float32) int {
	return minIdxImpl(a)
}

func maxIdx32(a []float32) int {
	return maxIdxImpl(a)
}

// minIdxOfSumRows32 routes the sliding-window shapes (slide +1 and -1) through
// the AVX2 block kernels, eight then four rows per lane-per-row block, and
// composes any remainder rows with the scalar path. Non-unit slides and a
// missing AVX2 stay on the Go reference. All paths are bit-identical (see
// minIdxOfSumRows8AVX2). SSE2-only hosts take the Go path deliberately.
// Row-block widths for the MinIdxOfSumRows AVX2 dispatcher: minIdxOfSumRows8AVX2
// covers 8 rows per call, minIdxOfSumRows4AVX2 covers 4. The dispatcher blocks by
// these widths and covers any remainder with one overlapping block of the same
// width, so they double as the minimum m each block can serve.
const (
	minIdxRows8 = 8
	minIdxRows4 = 4
)

func minIdxOfSumRows32(vals []float32, idxs []int32, a, k []float32, base, slide int) {
	if cpu.X86.AVX2 && (slide == 1 || slide == -1) {
		r := 0
		for ; r+minIdxRows8 <= len(vals); r += minIdxRows8 {
			off := base + r*slide
			if slide == 1 {
				minIdxOfSumRows8AVX2(vals[r:r+minIdxRows8], idxs[r:r+minIdxRows8], a, k[off:], 0)
			} else {
				minIdxOfSumRows8AVX2(vals[r:r+minIdxRows8], idxs[r:r+minIdxRows8], a, k[off-7:], 1)
			}
		}
		if r+minIdxRows4 <= len(vals) {
			off := base + r*slide
			if slide == 1 {
				minIdxOfSumRows4AVX2(vals[r:r+minIdxRows4], idxs[r:r+minIdxRows4], a, k[off:], 0)
			} else {
				minIdxOfSumRows4AVX2(vals[r:r+minIdxRows4], idxs[r:r+minIdxRows4], a, k[off-3:], 1)
			}
			r += minIdxRows4
		}
		// The 0-3 leftover rows (leftover == m mod 4 after the 8- and 4-wide
		// blocks) go through SIMD via one overlapping 4-wide block ONLY when there
		// are at least 2 of them. The 4-wide block's cost is dominated by its fixed
		// column loop and per-row argmin reduction, not by how many rows it
		// recomputes, so it costs about one scalar row regardless; it therefore
		// only pays off when it replaces 2 or 3 scalar rows. A single leftover row
		// is cheaper computed scalar: #161 measured a +10% regression at 17x17
		// (= 2*8+1, so exactly 1 leftover row) when a lone remainder row was
		// covered by a full block, and narrowing the block from 8- to 4-wide did
		// not move it. Recomputing is bit-identical (each row is independent and
		// the kernel is pure) and the wrapper already range-validated every row's
		// window, so the overlap reads stay in bounds. A single leftover row, or
		// m<4 (no wide-enough block), stays scalar; a leftover of 0 leaves r == m
		// and skips this block entirely. Same rule as the NEON dispatcher.
		if r < len(vals) {
			m := len(vals)
			if m >= minIdxRows4 && m-r >= 2 {
				off := base + (m-minIdxRows4)*slide
				if slide == 1 {
					minIdxOfSumRows4AVX2(vals[m-minIdxRows4:], idxs[m-minIdxRows4:], a, k[off:], 0)
				} else {
					minIdxOfSumRows4AVX2(vals[m-minIdxRows4:], idxs[m-minIdxRows4:], a, k[off-3:], 1)
				}
			} else {
				minIdxOfSumRowsGo(vals[r:], idxs[r:], a, k, base+r*slide, slide)
			}
		}
		return
	}
	minIdxOfSumRowsGo(vals, idxs, a, k, base, slide)
}

//go:noescape
func minIdxOfSumRows8AVX2(vals []float32, idxs []int32, a, k []float32, rev int)

//go:noescape
func minIdxOfSumRows4AVX2(vals []float32, idxs []int32, a, k []float32, rev int)

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
	switch {
	case cpu.X86.AVX512F && cpu.X86.AVX512VL && len(rows) >= 4 && vecLen >= minAVX512Elements:
		dotProductBatchKernel(true, results, rows, vec, vecLen)
	case cpu.X86.AVX && cpu.X86.FMA && len(rows) >= 4 && vecLen >= minAVXElements:
		dotProductBatchKernel(false, results, rows, vec, vecLen)
	default:
		for i, row := range rows {
			n := min(len(row), vecLen)
			if n == 0 {
				results[i] = 0
				continue
			}
			results[i] = dotProduct(row[:n], vec[:n])
		}
	}
}

// dotProductBatchKernel scores every row in rows against vec, writing result i
// to results[i]. Rows are processed in groups of four so the query vector stays
// resident in registers across the group instead of being re-loaded per row;
// useAVX512 selects the AVX-512 kernel over the AVX+FMA one. A group whose four
// rows are each at least vecLen long takes the fused 4-row kernel; any shorter
// or trailing row falls back to the per-row dotProduct dispatch (scoring 0 when
// empty). The caller guarantees the selected ISA is available, len(rows) >= 4,
// and vecLen meets the kernel's minimum width.
func dotProductBatchKernel(useAVX512 bool, results []float32, rows [][]float32, vec []float32, vecLen int) {
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
			if useAVX512 {
				dotProduct4AVX512(res, r0, r1, r2, r3, q, vecLen)
			} else {
				dotProduct4AVX(res, r0, r1, r2, r3, q, vecLen)
			}
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
	useAVX := !useAVX512 && cpu.X86.AVX && cpu.X86.FMA
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
	useAVX := !useAVX512 && cpu.X86.AVX && cpu.X86.FMA
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

func convolveValid32(dst, signal, kernel []float32) {
	kLen := len(kernel)
	for i := range dst {
		dst[i] = dotProduct(signal[i:i+kLen], kernel)
	}
}

func convolveDecimate32(dst, signal, kernel []float32, factor, phase int) {
	convolveDecimateImpl(dst, signal, kernel, factor, phase)
}

func convolveValidMaxAbs32(signal, kernel []float32) float32 {
	return convolveValidMaxAbsImpl(signal, kernel)
}

func accumulateAdd32(dst, src []float32) {
	// AccumulateAdd is dst += src, which is the same as add(dst, dst, src)
	addImpl(dst, dst, src)
}

// interleave2Kernels returns the interleave2 / deinterleave2 implementations for
// a CPU with (avx == true) or without (avx == false) AVX support. The kernels
// are VEX-encoded AVX1 (VUNPCKLPS / VPERM2F128 / VSHUFPS) and there is no SSE2
// variant, so an AVX-less CPU must use the scalar Go fallback rather than the
// AVX kernel (which would SIGILL). init() picks the impl once via this helper,
// the same way every other f32 kernel is selected; routing interleave2 through
// the same selection is what fixes birdnet-go issue #3353. Splitting it out lets
// the gate test pin the selection without re-running init or mutating globals.
func interleave2Kernels(avx bool) (interleave2Func, deinterleave2Func) {
	if avx {
		return interleave2AVX, deinterleave2AVX
	}
	return interleave2Go, deinterleave2Go
}

func interleave2_32(dst, a, b []float32) {
	// Need at least 8 pairs for SIMD to be worthwhile (AVX processes 8 at a time).
	// interleave2Impl is the AVX kernel only on AVX-capable CPUs (chosen in init,
	// like every other f32 kernel); on AVX-less CPUs it is interleave2Go, so this
	// never issues an AVX instruction on a CPU without AVX (birdnet-go #3353).
	if len(a) >= minAVXElements {
		interleave2Impl(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2_32(a, b, src []float32) {
	if len(a) >= minAVXElements {
		deinterleave2Impl(a, b, src)
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
	interleave3Streams   = 3
	interleave4Streams   = 4
	interleave6Streams   = 6
	interleave8Streams   = 8
	interleave3BlockMask = interleave8Streams - 1 // N=3 gathers 8 frames per block
	interleave4BlockMask = interleave4Streams - 1
	interleave6BlockMask = interleave8Streams - 1 // N=6 zips pairs, 8 frames per block
	interleave8BlockMask = interleave8Streams - 1
)

func interleaveN32(dst []float32, srcs [][]float32, n int) {
	switch len(srcs) {
	case interleave2Channels:
		interleave2_32(dst[:n*interleave2Channels], srcs[0][:n], srcs[1][:n])
	case interleave3Streams:
		if cpu.X86.AVX2 && n >= interleave8Streams {
			blk := n &^ interleave3BlockMask
			interleave3AVX(dst, srcs[0], srcs[1], srcs[2], blk)
			interleaveNTailGo(dst, srcs, blk, n)
			return
		}
		interleaveNGo(dst, srcs, n)
	case interleave4Streams:
		if cpu.X86.AVX && n >= interleave4Streams {
			blk := n &^ interleave4BlockMask
			interleave4AVX(dst, srcs[0], srcs[1], srcs[2], srcs[3], blk)
			interleaveNTailGo(dst, srcs, blk, n)
			return
		}
		interleaveNGo(dst, srcs, n)
	case interleave6Streams:
		if cpu.X86.AVX2 && n >= interleave8Streams {
			blk := n &^ interleave6BlockMask
			interleave6AVX(dst, srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], blk)
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
	case interleave3Streams:
		if cpu.X86.AVX2 && n >= interleave8Streams {
			blk := n &^ interleave3BlockMask
			deinterleave3AVX(dsts[0], dsts[1], dsts[2], src, blk)
			deinterleaveNTailGo(dsts, src, blk, n)
			return
		}
		deinterleaveNGo(dsts, src, n)
	case interleave4Streams:
		if cpu.X86.AVX && n >= interleave4Streams {
			blk := n &^ interleave4BlockMask
			deinterleave4AVX(dsts[0], dsts[1], dsts[2], dsts[3], src, blk)
			deinterleaveNTailGo(dsts, src, blk, n)
			return
		}
		deinterleaveNGo(dsts, src, n)
	case interleave6Streams:
		if cpu.X86.AVX2 && n >= interleave8Streams {
			blk := n &^ interleave6BlockMask
			deinterleave6AVX(dsts[0], dsts[1], dsts[2], dsts[3], dsts[4], dsts[5], src, blk)
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
func convolveValidMaxAbsAVX(signal, kernel []float32) float32

//go:noescape
func convolveValidMaxAbsSSE(signal, kernel []float32) float32

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
func maxAbsAVX(a []float32) float32

//go:noescape
func absAVX(dst, a []float32)

//go:noescape
func negAVX(dst, a []float32)

//go:noescape
func copySignAVX(dst, mag, sign []float32)

//go:noescape
func fmaAVX(dst, a, b, c []float32)

//go:noescape
func clampAVX(dst, a []float32, minVal, maxVal float32)

//go:noescape
func clampScaleAVX(dst, src []float32, minVal, maxVal, scale float32)

//go:noescape
func sqrtAVX(dst, a []float32)

//go:noescape
func absPow34AVX(dst, src []float32)

//go:noescape
func roundAVX(dst, src []float32)

//go:noescape
func reciprocalAVX(dst, a []float32)

//go:noescape
func addScaledAVX(dst []float32, alpha float32, s []float32)

// Variance and Euclidean-distance reductions. sum((a[i]-mean)^2)/n and
// sqrt(sum((a[i]-b[i])^2)), accumulated in float32 to match the Go references.
//
//go:noescape
func varianceSSE(a []float32, mean float32) float32

//go:noescape
func varianceAVX(a []float32, mean float32) float32

//go:noescape
func euclideanDistanceSSE(a, b []float32) float32

//go:noescape
func euclideanDistanceAVX(a, b []float32) float32

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
func maxAbsSSE(a []float32) float32

//go:noescape
func absSSE(dst, a []float32)

//go:noescape
func negSSE(dst, a []float32)

//go:noescape
func copySignSSE(dst, mag, sign []float32)

//go:noescape
func fmaSSE(dst, a, b, c []float32)

//go:noescape
func clampSSE(dst, a []float32, minVal, maxVal float32)

//go:noescape
func sqrtSSE(dst, a []float32)

//go:noescape
func absPow34SSE(dst, src []float32)

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
func interleave3AVX(dst, s0, s1, s2 []float32, n int)

//go:noescape
func deinterleave3AVX(d0, d1, d2, src []float32, n int)

//go:noescape
func interleave4AVX(dst, s0, s1, s2, s3 []float32, n int)

//go:noescape
func deinterleave4AVX(d0, d1, d2, d3, src []float32, n int)

//go:noescape
func interleave6AVX(dst, s0, s1, s2, s3, s4, s5 []float32, n int)

//go:noescape
func deinterleave6AVX(d0, d1, d2, d3, d4, d5, src []float32, n int)

//go:noescape
func interleave8AVX(dst []float32, srcs [][]float32, n int)

//go:noescape
func deinterleave8AVX(dsts [][]float32, src []float32, n int)

func variance32(a []float32, mean float32) float32 {
	return varianceImpl(a, mean)
}

func euclideanDistance32(a, b []float32) float32 {
	return euclideanDistanceImpl(a, b)
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

// logSIMDOK32 reports whether the AVX log/pow kernels can run: they need
// AVX2 (YMM integer ops for the exponent extraction) and FMA (polynomial
// evaluation). AVX1-only or FMA-less CPUs use the accurate Go path.
func logSIMDOK32(n int) bool {
	return cpu.X86.AVX2 && cpu.X86.FMA && n >= minAVXElements
}

func log32(dst, src []float32) {
	if logSIMDOK32(len(dst)) {
		logAVX(dst, src, logLn2Hi32, logLn2Lo32, 1.0)
		return
	}
	logGo(dst, src)
}

func log2_32(dst, src []float32) {
	// log2(x) = e + ln(m)*log2(e): the e term is exact, so no hi/lo split is
	// needed and exact powers of two come out exact.
	if logSIMDOK32(len(dst)) {
		logAVX(dst, src, 1.0, 0.0, logLog2E32)
		return
	}
	log2Go(dst, src)
}

func log10_32(dst, src []float32) {
	if logSIMDOK32(len(dst)) {
		logAVX(dst, src, logL102Hi32, logL102Lo32, logLog10E32)
		return
	}
	log10Go(dst, src)
}

func pow32(dst, src []float32, exp float32) {
	// A zero or non-finite exponent has whole-slice math.Pow semantics
	// (for example Pow(x, 0) = 1 even for NaN x); keep those exact.
	e := float64(exp)
	if logSIMDOK32(len(dst)) && exp != 0 && !math.IsNaN(e) && !math.IsInf(e, 0) &&
		powSIMDOK32(src[:len(dst)]) {
		powAVX(dst, src, exp)
		return
	}
	powGo(dst, src, exp)
}

func powElem32(dst, base, exp []float32) {
	if logSIMDOK32(len(dst)) && powSIMDOK32(base[:len(dst)]) && allFinite32(exp[:len(dst)]) {
		powElemAVX(dst, base, exp)
		return
	}
	powElemGo(dst, base, exp)
}

//go:noescape
func logAVX(dst, src []float32, k1hi, k1lo, k2 float32)

//go:noescape
func powAVX(dst, src []float32, exp float32)

//go:noescape
func powElemAVX(dst, base, exp []float32)

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

func float32ToInt32ScaleClamp(dst []int32, src []float32, scale, offset, minV, maxV float32) {
	// Pure AVX (VEX.256 float ops + VCVTTPS2DQ); the int32 output needs no
	// saturating pack, so unlike float32ToInt16Scale it does not require AVX2.
	if cpu.X86.AVX && len(dst) >= minAVXElements {
		float32ToInt32ScaleClampAVX(dst, src, scale, offset, minV, maxV)
		return
	}
	float32ToInt32ScaleClampGo(dst, src, scale, offset, minV, maxV)
}

//go:noescape
func float32ToInt32ScaleClampAVX(dst []int32, src []float32, scale, offset, minV, maxV float32)

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
