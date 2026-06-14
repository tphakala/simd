//go:build amd64

package f64

import (
	"math"
	"unsafe"

	"github.com/tphakala/simd/cpu"
)

// Minimum number of float64 elements required for SIMD operations.
// AVX processes 4 float64 values per 256-bit register.
// AVX-512 processes 8 float64 values per 512-bit register.
const (
	minAVXElements    = 4
	minAVX512Elements = 8
)

// minSIMDElements is set at init time based on which SIMD implementation is selected.
// Used by min64/max64 to determine when to fall back to scalar code.
var minSIMDElements = minAVXElements

// hasAVX2 gates the autocorrelation kernel, which needs VPERMPD (AVX2). It uses
// a direct dispatch rather than the init-time function pointers above.
var hasAVX2 = cpu.X86.AVX2

// Function pointer types for SIMD operations
type (
	dotProductFunc          func(a, b []float64) float64
	binaryOpFunc            func(dst, a, b []float64)
	scaleFunc               func(dst, a []float64, s float64)
	unaryOpFunc             func(dst, a []float64)
	reduceFunc              func(a []float64) float64
	fmaFunc                 func(dst, a, b, c []float64)
	clampFunc               func(dst, a []float64, minVal, maxVal float64)
	varianceFunc            func(a []float64, mean float64) float64
	euclideanDistanceFunc   func(a, b []float64) float64
	interleave2Func         func(dst, a, b []float64)
	deinterleave2Func       func(a, b, src []float64)
	addScaledFunc           func(dst []float64, alpha float64, s []float64)
	convolveDecimateFunc    func(dst, signal, kernel []float64, factor, phase int)
	convolveValidMaxAbsFunc func(signal, kernel []float64) float64
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
	interleave2Impl         interleave2Func
	deinterleave2Impl       deinterleave2Func
	addScaledImpl           addScaledFunc
	convolveDecimateImpl    convolveDecimateFunc
	convolveValidMaxAbsImpl convolveValidMaxAbsFunc
)

func init() {
	// Select optimal implementation based on CPU features.
	// Priority: AVX-512 > AVX+FMA > AVX (no FMA) > SSE2 > Go
	switch {
	case cpu.X86.AVX512F && cpu.X86.AVX512VL:
		initAVX512()
	case cpu.X86.AVX && cpu.X86.FMA:
		initAVX()
	case cpu.X86.AVX:
		initAVXNoFMA()
	case cpu.X86.SSE2:
		initSSE2()
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
	maxAbsImpl = maxAbsAVX
	absImpl = absAVX512
	negImpl = negAVX512
	sqrtImpl = sqrtAVX512
	reciprocalImpl = reciprocalAVX512
	roundImpl = roundAVX
	fmaImpl = fmaAVX512
	clampImpl = clampAVX512
	varianceImpl = varianceAVX512
	euclideanDistanceImpl = euclideanDistanceAVX512
	interleave2Impl = interleave2AVX
	deinterleave2Impl = deinterleave2AVX
	addScaledImpl = addScaledAVX512
	convolveDecimateImpl = convolveDecimateAVX512
	convolveValidMaxAbsImpl = convolveValidMaxAbsGo
}

func initAVX() {
	minSIMDElements = minAVXElements
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
	interleave2Impl = interleave2AVX
	deinterleave2Impl = deinterleave2AVX
	addScaledImpl = addScaledAVX
	convolveDecimateImpl = convolveDecimateAVX
	convolveValidMaxAbsImpl = convolveValidMaxAbsGo
}

// initAVXNoFMA runs on AVX-capable CPUs that lack FMA (rare but possible:
// some early Sandy/Ivy Bridge generations, certain Atom/Pentium SKUs).
// AVX kernels that don't depend on FMA stay on AVX paths; FMA-dependent kernels
// fall back to SSE2 variants which use scalar/SSE multiply-add sequences.
func initAVXNoFMA() {
	minSIMDElements = minAVXElements
	dotProductImpl = dotProductSSE2
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
	fmaImpl = fmaSSE2
	clampImpl = clampAVX
	varianceImpl = varianceSSE2
	euclideanDistanceImpl = euclideanDistanceSSE2
	interleave2Impl = interleave2AVX
	deinterleave2Impl = deinterleave2AVX
	addScaledImpl = addScaledSSE2
	convolveDecimateImpl = convolveDecimateSSE2
	convolveValidMaxAbsImpl = convolveValidMaxAbsGo
}

func initSSE2() {
	dotProductImpl = dotProductSSE2
	addImpl = addSSE2
	subImpl = subSSE2
	mulImpl = mulSSE2
	divImpl = divSSE2
	scaleImpl = scaleSSE2
	addScalarImpl = addScalarSSE2
	sumImpl = sumSSE2
	minImpl = minSSE2
	maxImpl = maxSSE2
	maxAbsImpl = maxAbsSSE2
	absImpl = absSSE2
	negImpl = negSSE2
	sqrtImpl = sqrtSSE2
	reciprocalImpl = reciprocalSSE2
	roundImpl = round64Go
	fmaImpl = fmaSSE2
	clampImpl = clampSSE2
	varianceImpl = varianceSSE2
	euclideanDistanceImpl = euclideanDistanceSSE2
	interleave2Impl = interleave2SSE2
	deinterleave2Impl = deinterleave2SSE2
	addScaledImpl = addScaledSSE2
	convolveDecimateImpl = convolveDecimateSSE2
	convolveValidMaxAbsImpl = convolveValidMaxAbsGo
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
	sqrtImpl = sqrt64Go
	reciprocalImpl = reciprocal64Go
	roundImpl = round64Go
	fmaImpl = fmaGo
	clampImpl = clampGo
	varianceImpl = variance64Go
	euclideanDistanceImpl = euclideanDistance64Go
	interleave2Impl = interleave2Go
	deinterleave2Impl = deinterleave2Go
	addScaledImpl = addScaledGo64
	convolveDecimateImpl = convolveDecimate64Go
	convolveValidMaxAbsImpl = convolveValidMaxAbsGo
}

// Dispatch functions - call function pointers (zero overhead after init)

func dotProduct(a, b []float64) float64 {
	return dotProductImpl(a, b)
}

func add(dst, a, b []float64) {
	addImpl(dst, a, b)
}

func sub(dst, a, b []float64) {
	subImpl(dst, a, b)
}

func mul(dst, a, b []float64) {
	mulImpl(dst, a, b)
}

func div(dst, a, b []float64) {
	divImpl(dst, a, b)
}

func scale(dst, a []float64, s float64) {
	scaleImpl(dst, a, s)
}

func addScalar(dst, a []float64, s float64) {
	addScalarImpl(dst, a, s)
}

func subFromScalar64(dst, a []float64, s float64) {
	// Compose using already-dispatched primitives: (s - a) == (-a) + s.
	// Each step is internally vectorized or falls back to Go via the global impl
	// pointers, so this works on every supported CPU without an extra guard.
	neg64(dst, a)
	addScalar(dst, dst, s)
}

func sum(a []float64) float64 {
	return sumImpl(a)
}

func min64(a []float64) float64 {
	// AVX/AVX-512 requires at least 4/8 elements for initial vector load
	// Fall back to Go for small slices to avoid reading beyond bounds
	if len(a) < minSIMDElements {
		return minGo(a)
	}
	return minImpl(a)
}

func max64(a []float64) float64 {
	// AVX/AVX-512 requires at least 4/8 elements for initial vector load
	// Fall back to Go for small slices to avoid reading beyond bounds
	if len(a) < minSIMDElements {
		return maxGo(a)
	}
	return maxImpl(a)
}

func maxAbs64(a []float64) float64 {
	// The SIMD kernels do a full-width initial vector load; fall back to Go for
	// small slices to avoid reading beyond bounds (mirrors min64/max64).
	if len(a) < minSIMDElements {
		return maxAbsGo(a)
	}
	return maxAbsImpl(a)
}

func abs64(dst, a []float64) {
	absImpl(dst, a)
}

func neg64(dst, a []float64) {
	negImpl(dst, a)
}

func fma64(dst, a, b, c []float64) {
	fmaImpl(dst, a, b, c)
}

func clamp64(dst, a []float64, minVal, maxVal float64) {
	clampImpl(dst, a, minVal, maxVal)
}

func round64(dst, src []float64) {
	roundImpl(dst, src)
}

func sqrt64(dst, a []float64) {
	sqrtImpl(dst, a)
}

func reciprocal64(dst, a []float64) {
	reciprocalImpl(dst, a)
}

func variance64(a []float64, mean float64) float64 {
	return varianceImpl(a, mean)
}

func euclideanDistance64(a, b []float64) float64 {
	return euclideanDistanceImpl(a, b)
}

func cumulativeSum64(dst, a []float64) {
	// CumulativeSum is inherently sequential
	cumulativeSum64Go(dst, a)
}

func dotProductBatch64(results []float64, rows [][]float64, vec []float64) {
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

// dotProductBatchKernel scores rows against vec in groups of four, keeping the
// query vector resident across each group via dotProduct4AVX/dotProduct4AVX512
// instead of reloading it per row. Rows shorter than vecLen (and any tail past
// the last full group of four) fall back to the per-row dotProduct, so results
// stay anchored to the scalar contract regardless of row shape.
func dotProductBatchKernel(useAVX512 bool, results []float64, rows [][]float64, vec []float64, vecLen int) {
	i := 0
	for i+3 < len(rows) {
		row0, row1, row2, row3 := rows[i], rows[i+1], rows[i+2], rows[i+3]
		if len(row0) >= vecLen && len(row1) >= vecLen && len(row2) >= vecLen && len(row3) >= vecLen {
			res := (*float64)(unsafe.Pointer(&results[i]))
			r0 := (*float64)(unsafe.Pointer(&row0[0]))
			r1 := (*float64)(unsafe.Pointer(&row1[0]))
			r2 := (*float64)(unsafe.Pointer(&row2[0]))
			r3 := (*float64)(unsafe.Pointer(&row3[0]))
			q := (*float64)(unsafe.Pointer(&vec[0]))
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

// autocorrelate64 computes autoc[lag] for lag in 0..maxLag. On AVX2 it
// vectorizes ACROSS lags: four consecutive lags share one accumulator register,
// each lane summing its lag's x[i]*x[i-lag] terms in increasing-i order with
// separate VMULPD+VADDPD (never FMA), so the result is bit-identical to
// autocorrelateGo. The triangular prologue (i < pmax) is seeded in scalar Go;
// the kernel then sweeps the rectangular steady region i in pmax..n-1 where
// every per-lag window load is in bounds. Without AVX2, or when the block is
// too short to have a steady region, it falls back to the scalar reference.
func autocorrelate64(autoc, x []float64, maxLag int) {
	const lanes = 4
	n := len(x)
	groups := (maxLag + lanes) / lanes // ceil((maxLag+1)/lanes)
	pmax := groups*lanes - 1           // steady-region start = padded max lag
	if !hasAVX2 || n <= pmax {
		autocorrelateGo(autoc, x, maxLag)
		return
	}
	autocorrTriangularSeed(autoc, x, maxLag, pmax)
	count := n - pmax
	bcast := &x[pmax]
	for g := range groups {
		base := g * lanes
		// window points at x[pmax-base-(lanes-1)]; the kernel loads `lanes`
		// ascending elements there and reverses them so lane j accumulates lag
		// base+j. Both pmax-base-(lanes-1) >= 0 and the final read index
		// n-1-base stay in bounds for every group.
		window := &x[pmax-base-(lanes-1)]
		if base+lanes-1 <= maxLag {
			autocorrStep4AVX(&autoc[base], bcast, window, count)
			continue
		}
		// Final partial group (maxLag+1 not a multiple of lanes): run the kernel
		// over a padded stack buffer and copy back only the real lags. The pad
		// lanes accumulate valid-but-unused lags and are discarded.
		var buf [lanes]float64
		realLanes := maxLag + 1 - base
		copy(buf[:realLanes], autoc[base:maxLag+1])
		autocorrStep4AVX(&buf[0], bcast, window, count)
		copy(autoc[base:maxLag+1], buf[:realLanes])
	}
}

func convolveValid64(dst, signal, kernel []float64) {
	kLen := len(kernel)
	for i := range dst {
		dst[i] = dotProduct(signal[i:i+kLen], kernel)
	}
}

func convolveDecimate64(dst, signal, kernel []float64, factor, phase int) {
	convolveDecimateImpl(dst, signal, kernel, factor, phase)
}

func convolveValidMaxAbs64(signal, kernel []float64) float64 {
	return convolveValidMaxAbsImpl(signal, kernel)
}

func accumulateAdd64(dst, src []float64) {
	// AccumulateAdd is dst += src, which is the same as add(dst, dst, src)
	addImpl(dst, dst, src)
}

func interleave2_64(dst, a, b []float64) {
	// Need at least 2 pairs for SIMD to be worthwhile (AVX processes 4 at a time)
	if len(a) >= minAVXElements {
		interleave2Impl(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2_64(a, b, src []float64) {
	// Need at least 2 pairs for SIMD to be worthwhile
	if len(a) >= minAVXElements {
		deinterleave2Impl(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

// interleaveN64 interleaves nc = len(srcs) planar streams into dst. N == 2
// reuses the existing Interleave2 SIMD; other small N use shuffle-based
// transposes (added incrementally); the rest fall back to the generic Go path.
// Stream counts with dedicated AMD64 SIMD interleave/deinterleave kernels (the
// 2-stream path reuses interleave2Channels). Every f64 YMM kernel (N=3, N=4,
// N=8) processes interleaveBlockFrames frames per block because a YMM holds 4
// float64; interleaveBlockMask aligns a frame count down to a whole block and
// the caller handles the tail.
const (
	interleave3Streams = 3
	interleave4Streams = 4
	interleave6Streams = 6
	interleave8Streams = 8

	interleaveBlockFrames = 4
	interleaveBlockMask   = interleaveBlockFrames - 1
)

func interleaveN64(dst []float64, srcs [][]float64, n int) {
	switch len(srcs) {
	case interleave2Channels:
		interleave2_64(dst[:n*interleave2Channels], srcs[0][:n], srcs[1][:n])
	case interleave3Streams:
		if cpu.X86.AVX2 && n >= interleaveBlockFrames {
			blk := n &^ interleaveBlockMask
			interleave3AVX(dst, srcs[0], srcs[1], srcs[2], blk)
			interleaveNTailGo(dst, srcs, blk, n)
			return
		}
		interleaveNGo(dst, srcs, n)
	case interleave4Streams:
		if cpu.X86.AVX && n >= interleaveBlockFrames {
			blk := n &^ interleaveBlockMask
			interleave4AVX(dst, srcs[0], srcs[1], srcs[2], srcs[3], blk)
			interleaveNTailGo(dst, srcs, blk, n)
			return
		}
		interleaveNGo(dst, srcs, n)
	case interleave6Streams:
		if cpu.X86.AVX2 && n >= interleaveBlockFrames {
			blk := n &^ interleaveBlockMask
			interleave6AVX(dst, srcs[0], srcs[1], srcs[2], srcs[3], srcs[4], srcs[5], blk)
			interleaveNTailGo(dst, srcs, blk, n)
			return
		}
		interleaveNGo(dst, srcs, n)
	case interleave8Streams:
		if cpu.X86.AVX && n >= interleaveBlockFrames {
			blk := n &^ interleaveBlockMask
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
// interleave that an asm kernel left after processing whole SIMD blocks,
// allocation-free.
func interleaveNTailGo(dst []float64, srcs [][]float64, from, n int) {
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
func deinterleaveNTailGo(dsts [][]float64, src []float64, from, n int) {
	nc := len(dsts)
	for i := from; i < n; i++ {
		base := i * nc
		for c := range nc {
			dsts[c][i] = src[base+c]
		}
	}
}

// deinterleaveN64 splits src into nc = len(dsts) planar streams. N == 2 reuses
// the existing Deinterleave2 SIMD; the rest fall back to the generic Go path.
func deinterleaveN64(dsts [][]float64, src []float64, n int) {
	switch len(dsts) {
	case interleave2Channels:
		deinterleave2_64(dsts[0][:n], dsts[1][:n], src[:n*interleave2Channels])
	case interleave3Streams:
		if cpu.X86.AVX2 && n >= interleaveBlockFrames {
			blk := n &^ interleaveBlockMask
			deinterleave3AVX(dsts[0], dsts[1], dsts[2], src, blk)
			deinterleaveNTailGo(dsts, src, blk, n)
			return
		}
		deinterleaveNGo(dsts, src, n)
	case interleave4Streams:
		if cpu.X86.AVX && n >= interleaveBlockFrames {
			blk := n &^ interleaveBlockMask
			deinterleave4AVX(dsts[0], dsts[1], dsts[2], dsts[3], src, blk)
			deinterleaveNTailGo(dsts, src, blk, n)
			return
		}
		deinterleaveNGo(dsts, src, n)
	case interleave6Streams:
		if cpu.X86.AVX2 && n >= interleaveBlockFrames {
			blk := n &^ interleaveBlockMask
			deinterleave6AVX(dsts[0], dsts[1], dsts[2], dsts[3], dsts[4], dsts[5], src, blk)
			deinterleaveNTailGo(dsts, src, blk, n)
			return
		}
		deinterleaveNGo(dsts, src, n)
	case interleave8Streams:
		if cpu.X86.AVX && n >= interleaveBlockFrames {
			blk := n &^ interleaveBlockMask
			deinterleave8AVX(dsts, src, blk)
			deinterleaveNTailGo(dsts, src, blk, n)
			return
		}
		deinterleaveNGo(dsts, src, n)
	default:
		deinterleaveNGo(dsts, src, n)
	}
}

func convolveValidMulti64(dsts [][]float64, signal []float64, kernels [][]float64, n, _ int) {
	// Kernel-major loop order: each kernel stays hot in cache for entire signal pass
	for k, kernel := range kernels {
		convolveValid64(dsts[k][:n], signal, kernel)
	}
}

func minIdx64(a []float64) int {
	return minIdxGo64(a)
}

func maxIdx64(a []float64) int {
	return maxIdxGo64(a)
}

func addScaled64(dst []float64, alpha float64, s []float64) {
	addScaledImpl(dst, alpha, s)
}

func cubicInterpDot64(hist, a, b, c, d []float64, x float64) float64 {
	// Use AVX+FMA if available and have enough elements
	if cpu.X86.AVX && cpu.X86.FMA && len(hist) >= minAVXElements {
		return cubicInterpDotAVX(hist, a, b, c, d, x)
	}
	return cubicInterpDotGo(hist, a, b, c, d, x)
}

func sigmoid64(dst, src []float64) {
	// Requires AVX2: sigmoidAVX reconstructs 2^k with 256-bit YMM integer ops
	// (VCVTTPD2DQ/VPMOVSXDQ/VPSLLQ/VPADDQ) that do not exist on AVX1-only CPUs.
	// Gating on plain AVX would let AVX1 parts reach an AVX2-only kernel and
	// fault with SIGILL (same fix class as the f32 sigmoid32/tanh32 gating).
	// FMA is not used here, so AVX2 alone is the correct guard; AVX1-only CPUs
	// fall back to the accurate Go path.
	if cpu.X86.AVX2 && len(dst) >= minAVXElements {
		sigmoidAVX(dst, src)
		return
	}
	sigmoid64Go(dst, src)
}

func relu64(dst, src []float64) {
	if cpu.X86.AVX && len(dst) >= minAVXElements {
		reluAVX(dst, src)
		return
	}
	relu64Go(dst, src)
}

//go:noescape
func reluAVX(dst, src []float64)

func clampScale64(dst, src []float64, minVal, maxVal, scale float64) {
	if cpu.X86.AVX && len(dst) >= minAVXElements && len(src) >= minAVXElements {
		clampScaleAVX(dst, src, minVal, maxVal, scale)
		return
	}
	clampScale64Go(dst, src, minVal, maxVal, scale)
}

func tanh64(dst, src []float64) {
	// Requires AVX2: tanhAVX reconstructs 2^k with 256-bit YMM integer ops
	// (VCVTTPD2DQ/VPMOVSXDQ/VPSLLQ/VPADDQ) that do not exist on AVX1-only CPUs.
	// FMA is not used here, so AVX2 alone is the correct guard; AVX1-only CPUs
	// fall back to the accurate Go path.
	if cpu.X86.AVX2 && len(dst) >= minAVXElements {
		tanhAVX(dst, src)
		return
	}
	tanh64Go(dst, src)
}

//go:noescape
func tanhAVX(dst, src []float64)

func exp64(dst, src []float64) {
	// Requires AVX2: expAVX reconstructs 2^k with 256-bit YMM integer ops
	// (VCVTTPD2DQ/VPMOVSXDQ/VPSLLQ/VPADDQ) that do not exist on AVX1-only CPUs.
	// FMA is not used here, so AVX2 alone is the correct guard; AVX1-only CPUs
	// fall back to the accurate Go path.
	if cpu.X86.AVX2 && len(dst) >= minAVXElements {
		expAVX(dst, src)
		return
	}
	exp64Go(dst, src)
}

// logSIMDOK reports whether the AVX log/pow kernels can run: they need AVX2
// (YMM integer ops for the exponent extraction) and FMA (polynomial
// evaluation). AVX1-only or FMA-less CPUs use the accurate Go path.
func logSIMDOK(n int) bool {
	return cpu.X86.AVX2 && cpu.X86.FMA && n >= minAVXElements
}

func log64(dst, src []float64) {
	if logSIMDOK(len(dst)) {
		logAVX(dst, src, logLn2Hi64, logLn2Lo64, 1.0)
		return
	}
	logGo(dst, src)
}

func log2_64(dst, src []float64) {
	// log2(x) = e + ln(m)*log2(e): the e term is exact, so no hi/lo split is
	// needed and exact powers of two come out exact.
	if logSIMDOK(len(dst)) {
		logAVX(dst, src, 1.0, 0.0, logLog2E64)
		return
	}
	log2Go(dst, src)
}

func log10_64(dst, src []float64) {
	if logSIMDOK(len(dst)) {
		logAVX(dst, src, logL102Hi64, logL102Lo64, logLog10E64)
		return
	}
	log10Go(dst, src)
}

func pow64(dst, src []float64, exp float64) {
	// A zero or non-finite exponent has whole-slice math.Pow semantics
	// (for example Pow(x, 0) = 1 even for NaN x); keep those exact.
	if logSIMDOK(len(dst)) && exp != 0 && !math.IsNaN(exp) && !math.IsInf(exp, 0) &&
		powSIMDOK64(src[:len(dst)]) {
		powAVX(dst, src, exp)
		return
	}
	powGo(dst, src, exp)
}

func powElem64(dst, base, exp []float64) {
	if logSIMDOK(len(dst)) && powSIMDOK64(base[:len(dst)]) && allFinite64(exp[:len(dst)]) {
		powElemAVX(dst, base, exp)
		return
	}
	powElemGo(dst, base, exp)
}

//go:noescape
func logAVX(dst, src []float64, k1hi, k1lo, k2 float64)

//go:noescape
func powAVX(dst, src []float64, exp float64)

//go:noescape
func powElemAVX(dst, base, exp []float64)

//go:noescape
func expAVX(dst, src []float64)

//go:noescape
func sigmoidAVX(dst, src []float64)

// AVX+FMA assembly function declarations (4x float64 per iteration)
//
//go:noescape
func dotProductAVX(a, b []float64) float64

//go:noescape
func dotProduct4AVX(results, row0, row1, row2, row3, vec *float64, n int)

// autocorrStep4AVX accumulates the steady region of four autocorrelation lags.
// acc points at four contiguous seeded accumulators (lags base..base+3);
// broadcast at x[pmax] and window at x[pmax-base-3] advance one element per
// iteration for count iterations. Each step does acc += x[i] * reverse(window)
// with separate VMULPD+VADDPD so the per-lag sum order matches the scalar
// reference exactly. See autocorrelate64.
//
//go:noescape
func autocorrStep4AVX(acc, broadcast, window *float64, count int)

//go:noescape
func convolveDecimateAVX(dst, signal, kernel []float64, factor, phase int)

//go:noescape
func convolveDecimateAVX512(dst, signal, kernel []float64, factor, phase int)

//go:noescape
func convolveDecimateSSE2(dst, signal, kernel []float64, factor, phase int)

//go:noescape
func addAVX(dst, a, b []float64)

//go:noescape
func subAVX(dst, a, b []float64)

//go:noescape
func mulAVX(dst, a, b []float64)

//go:noescape
func divAVX(dst, a, b []float64)

//go:noescape
func scaleAVX(dst, a []float64, s float64)

//go:noescape
func addScalarAVX(dst, a []float64, s float64)

//go:noescape
func sumAVX(a []float64) float64

//go:noescape
func minAVX(a []float64) float64

//go:noescape
func maxAVX(a []float64) float64

//go:noescape
func maxAbsAVX(a []float64) float64

//go:noescape
func absAVX(dst, a []float64)

//go:noescape
func negAVX(dst, a []float64)

//go:noescape
func roundAVX(dst, src []float64)

//go:noescape
func fmaAVX(dst, a, b, c []float64)

//go:noescape
func clampAVX(dst, a []float64, minVal, maxVal float64)

//go:noescape
func clampScaleAVX(dst, src []float64, minVal, maxVal, scale float64)

//go:noescape
func sqrtAVX(dst, a []float64)

//go:noescape
func reciprocalAVX(dst, a []float64)

//go:noescape
func varianceAVX(a []float64, mean float64) float64

//go:noescape
func euclideanDistanceAVX(a, b []float64) float64

//go:noescape
func addScaledAVX(dst []float64, alpha float64, s []float64)

// AVX-512 assembly function declarations (8x float64 per iteration)
//
//go:noescape
func dotProductAVX512(a, b []float64) float64

//go:noescape
func dotProduct4AVX512(results, row0, row1, row2, row3, vec *float64, n int)

//go:noescape
func addAVX512(dst, a, b []float64)

//go:noescape
func subAVX512(dst, a, b []float64)

//go:noescape
func mulAVX512(dst, a, b []float64)

//go:noescape
func divAVX512(dst, a, b []float64)

//go:noescape
func scaleAVX512(dst, a []float64, s float64)

//go:noescape
func addScalarAVX512(dst, a []float64, s float64)

//go:noescape
func sumAVX512(a []float64) float64

//go:noescape
func minAVX512(a []float64) float64

//go:noescape
func maxAVX512(a []float64) float64

//go:noescape
func absAVX512(dst, a []float64)

//go:noescape
func negAVX512(dst, a []float64)

//go:noescape
func fmaAVX512(dst, a, b, c []float64)

//go:noescape
func clampAVX512(dst, a []float64, minVal, maxVal float64)

//go:noescape
func sqrtAVX512(dst, a []float64)

//go:noescape
func reciprocalAVX512(dst, a []float64)

//go:noescape
func varianceAVX512(a []float64, mean float64) float64

//go:noescape
func euclideanDistanceAVX512(a, b []float64) float64

//go:noescape
func addScaledAVX512(dst []float64, alpha float64, s []float64)

// SSE2 assembly function declarations (2x float64 per iteration)
//
//go:noescape
func dotProductSSE2(a, b []float64) float64

//go:noescape
func addSSE2(dst, a, b []float64)

//go:noescape
func subSSE2(dst, a, b []float64)

//go:noescape
func mulSSE2(dst, a, b []float64)

//go:noescape
func divSSE2(dst, a, b []float64)

//go:noescape
func scaleSSE2(dst, a []float64, s float64)

//go:noescape
func addScalarSSE2(dst, a []float64, s float64)

//go:noescape
func sumSSE2(a []float64) float64

//go:noescape
func minSSE2(a []float64) float64

//go:noescape
func maxSSE2(a []float64) float64

//go:noescape
func maxAbsSSE2(a []float64) float64

//go:noescape
func absSSE2(dst, a []float64)

//go:noescape
func negSSE2(dst, a []float64)

//go:noescape
func fmaSSE2(dst, a, b, c []float64)

//go:noescape
func clampSSE2(dst, a []float64, minVal, maxVal float64)

//go:noescape
func sqrtSSE2(dst, a []float64)

//go:noescape
func reciprocalSSE2(dst, a []float64)

//go:noescape
func varianceSSE2(a []float64, mean float64) float64

//go:noescape
func euclideanDistanceSSE2(a, b []float64) float64

//go:noescape
func addScaledSSE2(dst []float64, alpha float64, s []float64)

// Interleave/Deinterleave assembly function declarations
//
//go:noescape
func interleave2AVX(dst, a, b []float64)

//go:noescape
func deinterleave2AVX(a, b, src []float64)

//go:noescape
func interleave3AVX(dst, s0, s1, s2 []float64, n int)

//go:noescape
func deinterleave3AVX(d0, d1, d2, src []float64, n int)

//go:noescape
func interleave4AVX(dst, s0, s1, s2, s3 []float64, n int)

//go:noescape
func deinterleave4AVX(d0, d1, d2, d3, src []float64, n int)

//go:noescape
func interleave6AVX(dst, s0, s1, s2, s3, s4, s5 []float64, n int)

//go:noescape
func deinterleave6AVX(d0, d1, d2, d3, d4, d5, src []float64, n int)

//go:noescape
func interleave8AVX(dst []float64, srcs [][]float64, n int)

//go:noescape
func deinterleave8AVX(dsts [][]float64, src []float64, n int)

//go:noescape
func interleave2SSE2(dst, a, b []float64)

//go:noescape
func deinterleave2SSE2(a, b, src []float64)

// CubicInterpDot assembly function declaration
//
//go:noescape
func cubicInterpDotAVX(hist, a, b, c, d []float64, x float64) float64
