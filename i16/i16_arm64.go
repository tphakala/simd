//go:build arm64

package i16

import "github.com/tphakala/simd/cpu"

// NEON processes 8 int16 pairs (one .8H register) per iteration.
const minNEONElements = 8

// The dot kernel has an 8-wide block before its scalar tail, so it shares the
// 8-element threshold. Eight is where the kernel stops losing and starts
// winning, which is also where the short CELT bands this primitive exists for
// land.
//
// Measured kernel against Go reference, both called through the same indirect
// pointer so neither gets a call-depth advantage (that is the comparison the
// dispatch decision turns on, since dotI16 calls both directly). Below 8 the
// reference wins on both microarchitectures tested: at n=4, 1.8 vs 2.5 ns on
// Apple M and 8.3 vs 10.7 ns on Cortex-A76. At n=8 the kernel is 1.3x ahead on
// Cortex-A76 and level on Apple M. From n=10 up it wins on both, reaching 1.8x
// (Apple M) and 2.0x (Cortex-A76) by n=16.
//
// Retuning note: the committed benchmarks compare the public DotProduct against
// dotGo called directly, so they charge dispatch overhead to the SIMD side only
// and understate the kernel at small n. They measure what a caller sees, not
// what this constant should be; reproduce the numbers above before moving it.
//
// This is an independent literal rather than an alias of minNEONElements, which
// it happens to equal, so retuning the interleave kernels cannot silently move
// the dot threshold.
const minNEONDot = 8 // the kernel's 8-wide block, below its 16-wide body

var hasNEON = cpu.ARM64.NEON

func interleave2I16(dst, a, b []int16) {
	if hasNEON && len(a) >= minNEONElements {
		interleave2NEON(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2I16(a, b, src []int16) {
	if hasNEON && len(a) >= minNEONElements {
		deinterleave2NEON(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

// XCorr vectorizes once x is long enough that reusing each x load across four
// lags pays for the kernel call. Below that the Go reference runs.
//
// This is set to minNEONDot because the dot product is the per-lag work and the
// 4-lag amortisation should move the break-even down from there, not up. That
// is a heuristic rather than an implication: the minNEONDot measurement put
// both sides behind an indirect call so neither got a call-depth advantage,
// whereas here the Go side inlines fully while the kernel cannot, so the two
// break-evens are not directly comparable.
//
// Measured at the boundary (public XCorr against xcorrGo, 16 lags): at x=8 the
// kernel is 3.5x ahead on Apple M (12.5 vs 43.8 ns) and 2.8x on Cortex-A76
// (58.6 vs 164.3 ns), so 8 is safe and in fact conservative; the lag reuse
// dominates well before the SIMD width does. The committed benchmarks start at
// x=240 and do not cover this, so re-measure before moving it.
const minNEONXCorr = 8

// The block loop is inlined here rather than shared with amd64 through a driver
// taking the kernel as a parameter: an indirect call defeats escape analysis
// and forces every caller of XCorr to heap-allocate. See xcorrWindow in
// i16_go.go. TestXCorr_AllocFree catches it if that refactor returns.
func xcorrI16(dst []int32, x, y []int16) {
	if !hasNEON || len(x) < minNEONXCorr {
		xcorrGo(dst, x, y)
		return
	}
	m := xcorrLags(dst, x, y)
	k := 0
	for ; k+xcorrLagBlock <= m; k += xcorrLagBlock {
		xcorr4NEON(dst[k:k+xcorrLagBlock], x, xcorrWindow(x, y, k))
	}
	// Remainder lags go through the dot dispatcher, not dotNEON directly, so
	// they still honour minNEONDot if minNEONXCorr is ever lowered below it.
	for ; k < m; k++ {
		dst[k] = dotI16(x, y[k:k+len(x)])
	}
}

func dotI16(a, b []int16) int32 {
	if hasNEON && min(len(a), len(b)) >= minNEONDot {
		return dotNEON(a, b)
	}
	return dotGo(a, b)
}

// Tier-3 thresholds: one 8-wide (.8H) vector block each. Like minNEONDot they
// are independent literals rather than aliases of minNEONElements, so retuning
// one op can never silently move another's cut. All three kernels are correct
// at any length (each falls through to a scalar tail), so these are
// performance cuts only, never a safety requirement.
const (
	minNEONMulQ15 = 8
	minNEONAbs    = 8
	minNEONMaxAbs = 8
)

func mulQ15I16(dst, a, b []int16) {
	if hasNEON && len(dst) >= minNEONMulQ15 {
		mulQ15NEON(dst, a, b)
		return
	}
	mulQ15Go(dst, a, b)
}

func absI16(dst, a []int16) {
	if hasNEON && len(dst) >= minNEONAbs {
		absNEON(dst, a)
		return
	}
	absGo(dst, a)
}

func maxAbsI16(a []int16) int {
	if hasNEON && len(a) >= minNEONMaxAbs {
		return maxAbsNEON(a)
	}
	return maxAbsGo(a)
}

//go:noescape
func mulQ15NEON(dst, a, b []int16)

//go:noescape
func absNEON(dst, a []int16)

//go:noescape
func maxAbsNEON(a []int16) int

//go:noescape
func dotNEON(a, b []int16) int32

//go:noescape
func xcorr4NEON(dst []int32, x, y []int16)

//go:noescape
func interleave2NEON(dst, a, b []int16)

//go:noescape
func deinterleave2NEON(a, b, src []int16)
