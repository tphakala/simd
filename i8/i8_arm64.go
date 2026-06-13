//go:build arm64

package i8

import "github.com/tphakala/simd/cpu"

// NEON processes 16 int8 (one .16B register) per iteration; ToInt32 widens 8.
const (
	minNEON16 = 16
	minNEON8  = 8
)

var hasNEON = cpu.ARM64.NEON

// hasDotProd selects the SDOT DotProduct kernel. FEAT_DotProd is a NEON
// extension, so it implies NEON, but require both explicitly to be safe.
var hasDotProd = cpu.ARM64.NEON && cpu.ARM64.DOTPROD

func addSatI8(dst, a, b []int8) {
	if hasNEON && len(dst) >= minNEON16 {
		addSatNEON(dst, a, b)
		return
	}
	addSatGo(dst, a, b)
}

func subSatI8(dst, a, b []int8) {
	if hasNEON && len(dst) >= minNEON16 {
		subSatNEON(dst, a, b)
		return
	}
	subSatGo(dst, a, b)
}

func toI16(dst []int16, src []int8) {
	if hasNEON && len(src) >= minNEON16 {
		toI16NEON(dst, src)
		return
	}
	toI16Go(dst, src)
}

func toI32(dst []int32, src []int8) {
	if hasNEON && len(src) >= minNEON8 {
		toI32NEON(dst, src)
		return
	}
	toI32Go(dst, src)
}

func sumI8(a []int8) int32 {
	if hasNEON && len(a) >= minNEON16 {
		return sumNEON(a)
	}
	return sumGo(a)
}

func dotI8(a, b []int8) int32 {
	switch {
	case hasDotProd && len(a) >= minNEON16:
		return dotSDOT(a, b)
	case hasNEON && len(a) >= minNEON16:
		return dotNEON(a, b)
	default:
		return dotGo(a, b)
	}
}

func minMaxI8(a []int8) (minVal, maxVal int8) {
	if hasNEON && len(a) >= minNEON16 {
		return minMaxNEON(a)
	}
	return minMaxGo(a)
}

//go:noescape
func addSatNEON(dst, a, b []int8)

//go:noescape
func subSatNEON(dst, a, b []int8)

//go:noescape
func toI16NEON(dst []int16, src []int8)

//go:noescape
func toI32NEON(dst []int32, src []int8)

//go:noescape
func sumNEON(a []int8) int32

//go:noescape
func dotNEON(a, b []int8) int32

//go:noescape
func dotSDOT(a, b []int8) int32

//go:noescape
func minMaxNEON(a []int8) (minVal, maxVal int8)
