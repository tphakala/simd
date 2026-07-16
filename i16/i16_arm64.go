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
// Unlike minNEONDot above, this value is inherited rather than measured: it is
// set to minNEONDot because the dot product is the per-lag work, and the 4-lag
// amortisation can only move the break-even down from there, never up. It has
// not been measured at the boundary, and the committed benchmarks start at
// x=240, so they do not cover it either. Measure before lowering it.
const minNEONXCorr = 8

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

//go:noescape
func dotNEON(a, b []int16) int32

//go:noescape
func xcorr4NEON(dst []int32, x, y []int16)

//go:noescape
func interleave2NEON(dst, a, b []int16)

//go:noescape
func deinterleave2NEON(a, b, src []int16)
