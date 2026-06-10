//go:build arm64

package f64

import (
	"math"
	"unsafe"

	"github.com/tphakala/simd/cpu"
)

var (
	hasNEON = cpu.ARM64.NEON
)

func dotProduct(a, b []float64) float64 {
	if hasNEON && len(a) >= 2 {
		return dotProductNEON(a, b)
	}
	return dotProductGo(a, b)
}

func add(dst, a, b []float64) {
	if hasNEON && len(dst) >= 2 {
		addNEON(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func sub(dst, a, b []float64) {
	if hasNEON && len(dst) >= 2 {
		subNEON(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

func mul(dst, a, b []float64) {
	if hasNEON && len(dst) >= 2 {
		mulNEON(dst, a, b)
		return
	}
	mulGo(dst, a, b)
}

func div(dst, a, b []float64) {
	if hasNEON && len(dst) >= 2 {
		divNEON(dst, a, b)
		return
	}
	divGo(dst, a, b)
}

func scale(dst, a []float64, s float64) {
	if hasNEON && len(dst) >= 2 {
		scaleNEON(dst, a, s)
		return
	}
	scaleGo(dst, a, s)
}

func addScalar(dst, a []float64, s float64) {
	if hasNEON && len(dst) >= 2 {
		addScalarNEON(dst, a, s)
		return
	}
	addScalarGo(dst, a, s)
}

func subFromScalar64(dst, a []float64, s float64) {
	// Compose using already-dispatched primitives: (s - a) == (-a) + s.
	// neg64 and addScalar each gate on hasNEON internally and fall back to
	// pure Go when NEON is unavailable, so no extra guard is needed here.
	neg64(dst, a)
	addScalar(dst, dst, s)
}

func sum(a []float64) float64 {
	if hasNEON && len(a) >= 2 {
		return sumNEON(a)
	}
	return sumGo(a)
}

func min64(a []float64) float64 {
	if hasNEON && len(a) >= 2 {
		return minNEON(a)
	}
	return minGo(a)
}

func max64(a []float64) float64 {
	if hasNEON && len(a) >= 2 {
		return maxNEON(a)
	}
	return maxGo(a)
}

func abs64(dst, a []float64) {
	if hasNEON && len(dst) >= 2 {
		absNEON(dst, a)
		return
	}
	absGo(dst, a)
}

func neg64(dst, a []float64) {
	if hasNEON && len(dst) >= 2 {
		negNEON(dst, a)
		return
	}
	negGo(dst, a)
}

func fma64(dst, a, b, c []float64) {
	if hasNEON && len(dst) >= 2 {
		fmaNEON(dst, a, b, c)
		return
	}
	fmaGo(dst, a, b, c)
}

func clamp64(dst, a []float64, minVal, maxVal float64) {
	if hasNEON && len(dst) >= 2 {
		clampNEON(dst, a, minVal, maxVal)
		return
	}
	clampGo(dst, a, minVal, maxVal)
}

func sqrt64(dst, a []float64) {
	if hasNEON && len(dst) >= 2 {
		sqrtNEON(dst, a)
		return
	}
	sqrt64Go(dst, a)
}

func round64(dst, src []float64) {
	if hasNEON && len(dst) >= 2 {
		roundNEON(dst, src)
		return
	}
	round64Go(dst, src)
}

func reciprocal64(dst, a []float64) {
	if hasNEON && len(dst) >= 2 {
		reciprocalNEON(dst, a)
		return
	}
	reciprocal64Go(dst, a)
}

func variance64(a []float64, mean float64) float64 {
	if hasNEON && len(a) >= 2 {
		return varianceNEON(a, mean)
	}
	return variance64Go(a, mean)
}

func euclideanDistance64(a, b []float64) float64 {
	if hasNEON && len(a) >= 2 {
		return euclideanDistanceNEON(a, b)
	}
	return euclideanDistance64Go(a, b)
}

func cumulativeSum64(dst, a []float64) {
	// CumulativeSum is inherently sequential
	cumulativeSum64Go(dst, a)
}

// dotProductBatch64 scores rows against vec in groups of four, keeping the query
// vector resident across each group via dotProduct4NEON instead of reloading it
// per row. Rows shorter than vecLen (and any tail past the last full group of
// four) fall back to the per-row dotProduct, so results stay anchored to the
// scalar contract regardless of row shape. f64 packs half the lanes of f32, so
// the query-reuse win is smaller; benchmarks on the Raspberry Pi 5 justify
// keeping the kernel.
func dotProductBatch64(results []float64, rows [][]float64, vec []float64) {
	vecLen := len(vec)
	i := 0
	if hasNEON && vecLen > 0 {
		for i+3 < len(rows) {
			row0, row1, row2, row3 := rows[i], rows[i+1], rows[i+2], rows[i+3]
			if len(row0) >= vecLen && len(row1) >= vecLen && len(row2) >= vecLen && len(row3) >= vecLen {
				res := (*float64)(unsafe.Pointer(&results[i]))
				r0 := (*float64)(unsafe.Pointer(&row0[0]))
				r1 := (*float64)(unsafe.Pointer(&row1[0]))
				r2 := (*float64)(unsafe.Pointer(&row2[0]))
				r3 := (*float64)(unsafe.Pointer(&row3[0]))
				q := (*float64)(unsafe.Pointer(&vec[0]))
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
	}
	for ; i < len(rows); i++ {
		row := rows[i]
		n := min(len(row), vecLen)
		if n == 0 {
			results[i] = 0
			continue
		}
		results[i] = dotProduct(row[:n], vec[:n])
	}
}

// autocorrelate64 computes autoc[lag] for lag in 0..maxLag. On NEON it
// vectorizes ACROSS lags (two consecutive lags per accumulator register), each
// lane summing its lag's x[i]*x[i-lag] terms in increasing-i order. The kernel
// uses fused FMLA because Go's arm64 compiler fuses the scalar reference's
// multiply-add into FMADDD, so FMLA is what reproduces autocorrelateGo
// bit-for-bit on arm64 (the amd64 AVX2 path instead uses separate mul+add to
// match its own non-fused fallback). The triangular prologue (i < pmax) is
// seeded in scalar Go; the kernel sweeps the rectangular steady region
// i in pmax..n-1. Mirrors the amd64 orchestrator with two lanes instead of four.
func autocorrelate64(autoc, x []float64, maxLag int) {
	const lanes = 2
	n := len(x)
	groups := (maxLag + lanes) / lanes // ceil((maxLag+1)/lanes)
	pmax := groups*lanes - 1           // steady-region start = padded max lag
	if !hasNEON || n <= pmax {
		autocorrelateGo(autoc, x, maxLag)
		return
	}
	autocorrTriangularSeed(autoc, x, maxLag, pmax)
	count := n - pmax
	bcast := &x[pmax]
	for g := range groups {
		base := g * lanes
		window := &x[pmax-base-(lanes-1)]
		if base+lanes-1 <= maxLag {
			autocorrStep2NEON(&autoc[base], bcast, window, count)
			continue
		}
		// Final partial group (odd lag count): run the kernel over a padded
		// stack buffer; the pad lane accumulates a valid-but-unused lag.
		var buf [lanes]float64
		realLanes := maxLag + 1 - base
		copy(buf[:realLanes], autoc[base:maxLag+1])
		autocorrStep2NEON(&buf[0], bcast, window, count)
		copy(autoc[base:maxLag+1], buf[:realLanes])
	}
}

//go:noescape
func autocorrStep2NEON(acc, broadcast, window *float64, count int)

func convolveValid64(dst, signal, kernel []float64) {
	// Convolution as sliding dot products
	kLen := len(kernel)
	for i := range dst {
		dst[i] = dotProduct(signal[i:i+kLen], kernel)
	}
}

func convolveDecimate64(dst, signal, kernel []float64, factor, phase int) {
	// Mirror dotProduct's NEON length threshold (>= 2) so the fused kernel and a
	// per-window DotProductUnsafe pick the same backend, keeping results identical.
	if hasNEON && len(kernel) >= 2 {
		convolveDecimateNEON(dst, signal, kernel, factor, phase)
		return
	}
	convolveDecimate64Go(dst, signal, kernel, factor, phase)
}

//go:noescape
func convolveDecimateNEON(dst, signal, kernel []float64, factor, phase int)

func accumulateAdd64(dst, src []float64) {
	// AccumulateAdd is dst += src, use add with dst as both operands
	if hasNEON && len(dst) >= 2 {
		addNEON(dst, dst, src)
		return
	}
	accumulateAdd64Go(dst, src)
}

//go:noescape
func dotProductNEON(a, b []float64) float64

//go:noescape
func dotProduct4NEON(results, row0, row1, row2, row3, vec *float64, n int)

//go:noescape
func addNEON(dst, a, b []float64)

//go:noescape
func subNEON(dst, a, b []float64)

//go:noescape
func mulNEON(dst, a, b []float64)

//go:noescape
func divNEON(dst, a, b []float64)

//go:noescape
func scaleNEON(dst, a []float64, s float64)

//go:noescape
func addScalarNEON(dst, a []float64, s float64)

//go:noescape
func sumNEON(a []float64) float64

//go:noescape
func minNEON(a []float64) float64

//go:noescape
func maxNEON(a []float64) float64

//go:noescape
func absNEON(dst, a []float64)

//go:noescape
func negNEON(dst, a []float64)

//go:noescape
func fmaNEON(dst, a, b, c []float64)

//go:noescape
func clampNEON(dst, a []float64, minVal, maxVal float64)

//go:noescape
func sqrtNEON(dst, a []float64)

//go:noescape
func reciprocalNEON(dst, a []float64)

//go:noescape
func roundNEON(dst, src []float64)

//go:noescape
func varianceNEON(a []float64, mean float64) float64

//go:noescape
func euclideanDistanceNEON(a, b []float64) float64

func interleave2_64(dst, a, b []float64) {
	if hasNEON && len(a) >= 2 {
		interleave2NEON(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2_64(a, b, src []float64) {
	if hasNEON && len(a) >= 2 {
		deinterleave2NEON(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

// interleaveN64 interleaves nc = len(srcs) planar streams into dst. N == 2
// reuses the existing Interleave2 NEON; N in {3,4} use the NEON ST3/ST4
// structured stores (added incrementally); the rest fall back to generic Go.
// Stream counts with dedicated ARM64 NEON interleave/deinterleave kernels (the
// 2-stream path reuses interleave2Channels). neonInterleaveBlock is the NEON
// structured load/store block size in frames (2 float64 per .2D register).
const (
	interleave3Streams  = 3
	interleave4Streams  = 4
	neonInterleaveBlock = 2
)

func interleaveN64(dst []float64, srcs [][]float64, n int) {
	switch len(srcs) {
	case interleave2Channels:
		interleave2_64(dst[:n*interleave2Channels], srcs[0][:n], srcs[1][:n])
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
	default:
		interleaveNGo(dst, srcs, n)
	}
}

// deinterleaveN64 splits src into nc = len(dsts) planar streams. N == 2 reuses
// the existing Deinterleave2 NEON; N in {3,4} use the NEON LD3/LD4 structured
// loads (added incrementally); the rest fall back to generic Go.
func deinterleaveN64(dsts [][]float64, src []float64, n int) {
	switch len(dsts) {
	case interleave2Channels:
		deinterleave2_64(dsts[0][:n], dsts[1][:n], src[:n*interleave2Channels])
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

//go:noescape
func interleave2NEON(dst, a, b []float64)

//go:noescape
func deinterleave2NEON(a, b, src []float64)

//go:noescape
func interleave3NEON(dst, s0, s1, s2 []float64, n int)

//go:noescape
func deinterleave3NEON(d0, d1, d2, src []float64, n int)

//go:noescape
func interleave4NEON(dst, s0, s1, s2, s3 []float64, n int)

//go:noescape
func deinterleave4NEON(d0, d1, d2, d3, src []float64, n int)

func minIdx64(a []float64) int {
	return minIdxGo64(a)
}

func maxIdx64(a []float64) int {
	return maxIdxGo64(a)
}

func addScaled64(dst []float64, alpha float64, s []float64) {
	if hasNEON && len(dst) >= 2 {
		addScaledNEON(dst, alpha, s)
		return
	}
	addScaledGo64(dst, alpha, s)
}

//go:noescape
func addScaledNEON(dst []float64, alpha float64, s []float64)

func cubicInterpDot64(hist, a, b, c, d []float64, x float64) float64 {
	if hasNEON && len(hist) >= 2 {
		return cubicInterpDotNEON(hist, a, b, c, d, x)
	}
	return cubicInterpDotGo(hist, a, b, c, d, x)
}

//go:noescape
func cubicInterpDotNEON(hist, a, b, c, d []float64, x float64) float64

func sigmoid64(dst, src []float64) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 2 {
		sigmoidNEON64(dst, src)
		return
	}
	sigmoid64Go(dst, src)
}

func relu64(dst, src []float64) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 2 {
		reluNEON64(dst, src)
		return
	}
	relu64Go(dst, src)
}

func clampScale64(dst, src []float64, minVal, maxVal, scale float64) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 2 {
		clampScaleNEON64(dst, src, minVal, maxVal, scale)
		return
	}
	clampScale64Go(dst, src, minVal, maxVal, scale)
}

//go:noescape
func clampScaleNEON64(dst, src []float64, minVal, maxVal, scale float64)

func tanh64(dst, src []float64) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 2 {
		tanhNEON64(dst, src)
		return
	}
	tanh64Go(dst, src)
}

func exp64(dst, src []float64) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 2 {
		expNEON64(dst, src)
		return
	}
	exp64Go(dst, src)
}

func log64(dst, src []float64) {
	// Assumes len(src) >= len(dst); caller ensures this via public API
	if hasNEON && len(dst) >= 2 {
		logNEON64(dst, src, logLn2Hi64, logLn2Lo64, 1.0)
		return
	}
	logGo(dst, src)
}

func log2_64(dst, src []float64) {
	// log2(x) = e + ln(m)*log2(e): the e term is exact, so no hi/lo split is
	// needed and exact powers of two come out exact.
	if hasNEON && len(dst) >= 2 {
		logNEON64(dst, src, 1.0, 0.0, logLog2E64)
		return
	}
	log2Go(dst, src)
}

func log10_64(dst, src []float64) {
	if hasNEON && len(dst) >= 2 {
		logNEON64(dst, src, logL102Hi64, logL102Lo64, logLog10E64)
		return
	}
	log10Go(dst, src)
}

func pow64(dst, src []float64, exp float64) {
	// A zero or non-finite exponent has whole-slice math.Pow semantics
	// (for example Pow(x, 0) = 1 even for NaN x); keep those exact.
	if hasNEON && len(dst) >= 2 && exp != 0 && !math.IsNaN(exp) && !math.IsInf(exp, 0) &&
		powSIMDOK64(src[:len(dst)]) {
		powNEON64(dst, src, exp)
		return
	}
	powGo(dst, src, exp)
}

func powElem64(dst, base, exp []float64) {
	if hasNEON && len(dst) >= 2 && powSIMDOK64(base[:len(dst)]) && allFinite64(exp[:len(dst)]) {
		powElemNEON64(dst, base, exp)
		return
	}
	powElemGo(dst, base, exp)
}

//go:noescape
func logNEON64(dst, src []float64, k1hi, k1lo, k2 float64)

//go:noescape
func powNEON64(dst, src []float64, exp float64)

//go:noescape
func powElemNEON64(dst, base, exp []float64)

//go:noescape
func expNEON64(dst, src []float64)

//go:noescape
func sigmoidNEON64(dst, src []float64)

//go:noescape
func reluNEON64(dst, src []float64)

//go:noescape
func tanhNEON64(dst, src []float64)
