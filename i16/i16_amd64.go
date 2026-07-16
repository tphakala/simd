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

// Dispatch priority mirrors f32/f64: AVX2 > SSE2 > Go. The kernels gate on the
// CPU feature explicitly (rather than relying on length alone) so the package is
// correct on every amd64 baseline. SSE2 is part of the amd64 baseline, so the
// pure-Go fallback below is effectively a non-amd64 safety net here.
var (
	hasAVX2 = cpu.X86.AVX2
	hasSSE2 = cpu.X86.SSE2
)

// Dot dispatch thresholds: one vector block each. They are independent literals
// rather than aliases of the interleave block sizes they happen to equal, so
// retuning the interleave kernels cannot silently retune the dot dispatch.
//
// The dot kernels are correct at any n (each falls through to a scalar tail), so
// these are performance cuts only, never a safety requirement. Measured kernel
// against Go reference at n=8, SSE2 wins 2.2x, so unlike NEON there is no
// break-even region to step around here.
const (
	minSSE2Dot = 8  // PMADDWL retires 8 int16 pairs per iteration
	minAVX2Dot = 16 // VPMADDWD retires 16
)

// XCorr vectorizes once x is long enough that reusing each x load across four
// lags pays for the kernel call. The AVX2 body needs 16 elements; below that
// SSE2's 8 still wins, and below 8 the Go reference runs.
const (
	minSSE2XCorr = 8
	minAVX2XCorr = 16
)

func xcorrI16(dst []int32, x, y []int16) {
	switch {
	case hasAVX2 && len(x) >= minAVX2XCorr:
		xcorrBlocked(dst, x, y, xcorr4AVX2, dotI16)
	case hasSSE2 && len(x) >= minSSE2XCorr:
		xcorrBlocked(dst, x, y, xcorr4SSE2, dotI16)
	default:
		xcorrGo(dst, x, y)
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
