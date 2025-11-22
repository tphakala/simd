//go:build arm64

package c128

import "github.com/tphakala/simd/pkg/simd/cpu"

var (
	hasNEON = cpu.ARM64.NEON
)

func mul128(dst, a, b []complex128) {
	if hasNEON && len(dst) >= 1 {
		mulNEON(dst, a, b)
		return
	}
	mulGo(dst, a, b)
}

func mulConj128(dst, a, b []complex128) {
	if hasNEON && len(dst) >= 1 {
		mulConjNEON(dst, a, b)
		return
	}
	mulConjGo(dst, a, b)
}

func scale128(dst, a []complex128, s complex128) {
	if hasNEON && len(dst) >= 1 {
		scaleNEON(dst, a, s)
		return
	}
	scaleGo(dst, a, s)
}

func add128(dst, a, b []complex128) {
	if hasNEON && len(dst) >= 1 {
		addNEON(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func sub128(dst, a, b []complex128) {
	if hasNEON && len(dst) >= 1 {
		subNEON(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

//go:noescape
func mulNEON(dst, a, b []complex128)

//go:noescape
func mulConjNEON(dst, a, b []complex128)

//go:noescape
func scaleNEON(dst, a []complex128, s complex128)

//go:noescape
func addNEON(dst, a, b []complex128)

//go:noescape
func subNEON(dst, a, b []complex128)
