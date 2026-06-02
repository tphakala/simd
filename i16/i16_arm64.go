//go:build arm64

package i16

import "github.com/tphakala/simd/cpu"

// NEON processes 8 int16 pairs (one .8H register) per iteration.
const minNEONElements = 8

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

//go:noescape
func interleave2NEON(dst, a, b []int16)

//go:noescape
func deinterleave2NEON(a, b, src []int16)
