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

// The arithmetic / decorrelation / fixed-predictor kernels operate on 256-bit
// integer lanes (VPADDD/VPSUBD/VPSRAD/...), which require AVX2 rather than the
// AVX1 that suffices for the float-shuffle interleave kernels above. They gate
// on AVX2 explicitly and fall back to the pure-Go reference otherwise.
var hasAVX2 = cpu.X86.AVX2

func addI32(dst, a, b []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		addAVX2(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func subI32(dst, a, b []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		subAVX2(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

func midSideEncodeI32(mid, side, left, right []int32) {
	if hasAVX2 && len(mid) >= minAVXElements {
		midSideEncodeAVX2(mid, side, left, right)
		return
	}
	midSideEncodeGo(mid, side, left, right)
}

func midSideDecodeI32(left, right, mid, side []int32) {
	if hasAVX2 && len(left) >= minAVXElements {
		midSideDecodeAVX2(left, right, mid, side)
		return
	}
	midSideDecodeGo(left, right, mid, side)
}

func diff1I32(dst, src []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		diff1AVX2(dst, src)
		return
	}
	diff1Go(dst, src)
}

func diff2I32(dst, src []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		diff2AVX2(dst, src)
		return
	}
	diff2Go(dst, src)
}

func diff3I32(dst, src []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		diff3AVX2(dst, src)
		return
	}
	diff3Go(dst, src)
}

func diff4I32(dst, src []int32) {
	if hasAVX2 && len(dst) >= minAVXElements {
		diff4AVX2(dst, src)
		return
	}
	diff4Go(dst, src)
}

//go:noescape
func addAVX2(dst, a, b []int32)

//go:noescape
func subAVX2(dst, a, b []int32)

//go:noescape
func midSideEncodeAVX2(mid, side, left, right []int32)

//go:noescape
func midSideDecodeAVX2(left, right, mid, side []int32)

func cumsumI32(a []int32) {
	if hasAVX2 && len(a) >= minAVXElements {
		cumsumAVX2(a)
		return
	}
	cumsumGo(a)
}

//go:noescape
func cumsumAVX2(a []int32)

//go:noescape
func diff1AVX2(dst, src []int32)

//go:noescape
func diff2AVX2(dst, src []int32)

//go:noescape
func diff3AVX2(dst, src []int32)

//go:noescape
func diff4AVX2(dst, src []int32)
