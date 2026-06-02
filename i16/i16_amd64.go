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
func interleave2AVX2(dst, a, b []int16)

//go:noescape
func deinterleave2AVX2(a, b, src []int16)

//go:noescape
func interleave2SSE2(dst, a, b []int16)

//go:noescape
func deinterleave2SSE2(a, b, src []int16)
