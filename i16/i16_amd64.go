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

// Dot thresholds are named separately from the interleave block sizes they
// currently equal, so retuning one op cannot silently retune the other. The
// values are one vector block each: the dot kernels are correct at any n (they
// fall through to a scalar tail), so these are performance cuts only, never a
// safety requirement. Measured kernel against Go reference at n=8: SSE2 wins
// 2.2x, so unlike NEON there is no break-even region to avoid here.
const (
	minSSE2Dot = minSSE2Elements
	minAVX2Dot = minAVX2Elements
)

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
