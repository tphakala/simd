//go:build amd64

package i32

import "github.com/tphakala/simd/cpu"

// Minimum number of int32 pairs before the AVX kernel beats the scalar loop.
// AVX processes 8 pairs (8 int32 per 256-bit register) per iteration.
const minAVXElements = 8

// hasAVX gates the SIMD kernels. The interleave kernels use only AVX1
// instructions (VUNPCKLPS / VPERM2F128 / VSHUFPS), so AVX without AVX2 is
// sufficient. Unlike f32's interleave path this checks the CPU feature
// explicitly rather than relying on length alone, so the package is safe on the
// (now rare) AVX-less amd64 baseline.
var hasAVX = cpu.X86.AVX

func interleave2I32(dst, a, b []int32) {
	if hasAVX && len(a) >= minAVXElements {
		interleave2AVX(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2I32(a, b, src []int32) {
	if hasAVX && len(a) >= minAVXElements {
		deinterleave2AVX(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

//go:noescape
func interleave2AVX(dst, a, b []int32)

//go:noescape
func deinterleave2AVX(a, b, src []int32)
