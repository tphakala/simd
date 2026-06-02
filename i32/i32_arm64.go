//go:build arm64

package i32

import "github.com/tphakala/simd/cpu"

// NEON processes 4 int32 pairs (one .4S register) per iteration.
const minNEONElements = 4

var hasNEON = cpu.ARM64.NEON

func interleave2I32(dst, a, b []int32) {
	if hasNEON && len(a) >= minNEONElements {
		interleave2NEON(dst, a, b)
		return
	}
	interleave2Go(dst, a, b)
}

func deinterleave2I32(a, b, src []int32) {
	if hasNEON && len(a) >= minNEONElements {
		deinterleave2NEON(a, b, src)
		return
	}
	deinterleave2Go(a, b, src)
}

//go:noescape
func interleave2NEON(dst, a, b []int32)

//go:noescape
func deinterleave2NEON(a, b, src []int32)

func addI32(dst, a, b []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		addNEON(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func subI32(dst, a, b []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		subNEON(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

func midSideEncodeI32(mid, side, left, right []int32) {
	if hasNEON && len(mid) >= minNEONElements {
		midSideEncodeNEON(mid, side, left, right)
		return
	}
	midSideEncodeGo(mid, side, left, right)
}

func midSideDecodeI32(left, right, mid, side []int32) {
	if hasNEON && len(left) >= minNEONElements {
		midSideDecodeNEON(left, right, mid, side)
		return
	}
	midSideDecodeGo(left, right, mid, side)
}

func diff1I32(dst, src []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		diff1NEON(dst, src)
		return
	}
	diff1Go(dst, src)
}

func diff2I32(dst, src []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		diff2NEON(dst, src)
		return
	}
	diff2Go(dst, src)
}

func diff3I32(dst, src []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		diff3NEON(dst, src)
		return
	}
	diff3Go(dst, src)
}

func diff4I32(dst, src []int32) {
	if hasNEON && len(dst) >= minNEONElements {
		diff4NEON(dst, src)
		return
	}
	diff4Go(dst, src)
}

//go:noescape
func addNEON(dst, a, b []int32)

//go:noescape
func subNEON(dst, a, b []int32)

//go:noescape
func midSideEncodeNEON(mid, side, left, right []int32)

//go:noescape
func midSideDecodeNEON(left, right, mid, side []int32)

func cumsumI32(a []int32) {
	if hasNEON && len(a) >= minNEONElements {
		cumsumNEON(a)
		return
	}
	cumsumGo(a)
}

//go:noescape
func cumsumNEON(a []int32)

//go:noescape
func diff1NEON(dst, src []int32)

//go:noescape
func diff2NEON(dst, src []int32)

//go:noescape
func diff3NEON(dst, src []int32)

//go:noescape
func diff4NEON(dst, src []int32)
