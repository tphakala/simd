//go:build amd64

package i16

import "github.com/tphakala/simd/cpu"

// Block sizes: the AVX2 kernels process 16 int16 pairs (one 256-bit register)
// per iteration, the SSE2 kernels 8 (one 128-bit register). Below the chosen
// block size each kernel falls through to a scalar tail, so these constants are
// just the dispatch thresholds, not a correctness requirement of the kernels.
const (
	minAVX2Elements = 16
	minSSE2Elements = 8
)

// Dispatch priority mirrors f32/f64: AVX2 > SSE2 > Go, with XCorr adding an
// AVX-VNNI tier above AVX2 (see xcorrI16). The kernels gate on the CPU feature
// explicitly (rather than relying on length alone) so the package is correct on
// every amd64 baseline.
//
// SSE2 is part of the amd64 baseline, so for the ops that ship an SSE2 tier
// (interleave, dot, xcorr) the pure-Go fallback is effectively a non-amd64
// safety net. That is not true package-wide: the tier-3 ops below (MulQ15, Abs,
// MaxAbs) are AVX2-or-Go, so on a pre-AVX2 amd64 host their Go reference is a
// live, reachable path rather than a formality.
var (
	hasAVXVNNI = cpu.X86.AVXVNNI
	hasAVX2    = cpu.X86.AVX2
	hasSSE2    = cpu.X86.SSE2
)

// Dot dispatch thresholds. They are independent literals rather than aliases of
// the interleave block sizes they happen to equal, so retuning the interleave
// kernels cannot silently retune the dot dispatch.
//
// The dot kernels are correct at any n (each falls through to a scalar tail), so
// these are performance cuts only, never a safety requirement. Measured kernel
// against Go reference at n=8, SSE2 wins 2.2x, so unlike NEON there is no
// break-even region to step around here.
//
// Both kernels vectorize from 8 (see dotAVX2's 8-wide block), so the AVX2 cut at
// 16 is not about AVX2 needing 16 elements to vectorize. It is that AVX2's fold
// pays a VEXTRACTI128 and a VZEROUPPER that SSE2's identical one-XMM block does
// not, a flat cost only the 16-wide body amortizes: measured kernel-direct, at
// n=8-15 AVX2 is 7-20% slower than SSE2, and at n >= 16 it is never slower.
const (
	minSSE2Dot = 8  // PMADDWL retires 8 int16 pairs per iteration
	minAVX2Dot = 16 // below this AVX2's fold overhead outweighs its width
)

// XCorr vectorizes once x is long enough that reusing each x load across four
// lags pays for the kernel call, and below 8 the Go reference runs.
//
// Both kernels vectorize from 8 (see xcorr4AVX2's 8-wide block), so the cut at
// 16 is not about AVX2 needing 16 elements to vectorize. It is that AVX2's fold
// pays four extra VEXTRACTI128 plus VZEROUPPER, a flat cost only the 16-wide
// body amortizes: measured at len(x) 8-15 AVX2 is 5-13% slower than SSE2, and at
// len(x) >= 16 it is never slower.
const (
	minSSE2XCorr = 8
	minAVX2XCorr = 16
	// The AVX-VNNI kernel shares xcorr4AVX2's structure below the 16-wide loop
	// (identical 8- and 4-wide blocks, identical fold), so it carries the same
	// fold overhead and the same break-even against SSE2: its cut is 16 too, an
	// independent literal per the dot precedent above rather than an alias.
	minAVXVNNIXCorr = 16
)

// The two block loops below are identical except for the kernel they call, and
// extracting that difference into a shared driver taking the kernel as a
// parameter is exactly the refactor that must not happen: an indirect call
// defeats escape analysis and forces every caller of XCorr to heap-allocate.
// See xcorrWindow in i16_go.go. TestXCorr_AllocFree catches it if it returns.
func xcorrI16(dst []int32, x, y []int16) {
	m := xcorrLags(dst, x, y)
	k := 0
	switch {
	case hasAVXVNNI && len(x) >= minAVXVNNIXCorr:
		for ; k+xcorrLagBlock <= m; k += xcorrLagBlock {
			xcorr4AVXVNNI(dst[k:k+xcorrLagBlock], x, xcorrWindow(x, y, k))
		}
	case hasAVX2 && len(x) >= minAVX2XCorr:
		for ; k+xcorrLagBlock <= m; k += xcorrLagBlock {
			xcorr4AVX2(dst[k:k+xcorrLagBlock], x, xcorrWindow(x, y, k))
		}
	case hasSSE2 && len(x) >= minSSE2XCorr:
		for ; k+xcorrLagBlock <= m; k += xcorrLagBlock {
			xcorr4SSE2(dst[k:k+xcorrLagBlock], x, xcorrWindow(x, y, k))
		}
	default:
		xcorrGo(dst, x, y)
		return
	}
	for ; k < m; k++ {
		dst[k] = dotI16(x, y[k:k+len(x)])
	}
}

func dotI16(a, b []int16) int32 {
	n := min(len(a), len(b))
	switch {
	case hasAVX2 && n >= minAVX2Dot:
		return dotAVX2(a, b)
	case hasSSE2 && n >= minSSE2Dot:
		return dotSSE2(a, b)
	default:
		return dotGo(a, b)
	}
}

func interleave2I16(dst, a, b []int16) {
	switch {
	case hasAVX2 && len(a) >= minAVX2Elements:
		interleave2AVX2(dst, a, b)
	case hasSSE2 && len(a) >= minSSE2Elements:
		interleave2SSE2(dst, a, b)
	default:
		interleave2Go(dst, a, b)
	}
}

func deinterleave2I16(a, b, src []int16) {
	switch {
	case hasAVX2 && len(a) >= minAVX2Elements:
		deinterleave2AVX2(a, b, src)
	case hasSSE2 && len(a) >= minSSE2Elements:
		deinterleave2SSE2(a, b, src)
	default:
		deinterleave2Go(a, b, src)
	}
}

// Tier-3 thresholds: one 16-wide (256-bit) vector block each, independent
// literals per the dot precedent above. These ops are AVX2-or-Go: unlike
// dot/xcorr/interleave they ship no SSE2 tier, matching i8 and the i32
// arithmetic, because cheap store-bound element work does not earn a third
// tier's maintenance. Below AVX2 the Go reference runs.
//
// The kernels are correct at any length (each falls through to a scalar tail),
// so these are performance cuts only, never a safety requirement. Nothing here
// keeps a kernel in bounds: the tier-3 kernels carry no in-assembly clamp, and
// the public wrappers are what reconcile the operand lengths.
const (
	minAVX2MulQ15 = 16
	minAVX2Abs    = 16
	minAVX2MaxAbs = 16
)

func mulQ15I16(dst, a, b []int16) {
	if hasAVX2 && len(dst) >= minAVX2MulQ15 {
		mulQ15AVX2(dst, a, b)
		return
	}
	mulQ15Go(dst, a, b)
}

func absI16(dst, a []int16) {
	if hasAVX2 && len(dst) >= minAVX2Abs {
		absAVX2(dst, a)
		return
	}
	absGo(dst, a)
}

func maxAbsI16(a []int16) int {
	if hasAVX2 && len(a) >= minAVX2MaxAbs {
		return maxAbsAVX2(a)
	}
	return maxAbsGo(a)
}

//go:noescape
func mulQ15AVX2(dst, a, b []int16)

//go:noescape
func absAVX2(dst, a []int16)

//go:noescape
func maxAbsAVX2(a []int16) int

//go:noescape
func xcorr4AVXVNNI(dst []int32, x, y []int16)

//go:noescape
func xcorr4AVX2(dst []int32, x, y []int16)

//go:noescape
func xcorr4SSE2(dst []int32, x, y []int16)

//go:noescape
func dotAVX2(a, b []int16) int32

//go:noescape
func dotSSE2(a, b []int16) int32

//go:noescape
func interleave2AVX2(dst, a, b []int16)

//go:noescape
func deinterleave2AVX2(a, b, src []int16)

//go:noescape
func interleave2SSE2(dst, a, b []int16)

//go:noescape
func deinterleave2SSE2(a, b, src []int16)
