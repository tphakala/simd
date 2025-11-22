//go:build arm64

package c128

import "github.com/tphakala/simd/cpu"

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

func abs128(dst []float64, a []complex128) {
	if hasNEON && len(dst) >= 1 {
		absNEON(dst, a)
		return
	}
	absGo(dst, a)
}

func absSq128(dst []float64, a []complex128) {
	if hasNEON && len(dst) >= 1 {
		absSqNEON(dst, a)
		return
	}
	absSqGo(dst, a)
}

func phase128(dst []float64, a []complex128) {
	if hasNEON && len(dst) >= 1 {
		phaseNEON(dst, a)
		return
	}
	phaseGo(dst, a)
}

func conj128(dst, a []complex128) {
	if hasNEON && len(dst) >= 1 {
		conjNEON(dst, a)
		return
	}
	conjGo(dst, a)
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

//go:noescape
func absNEON(dst []float64, a []complex128)

//go:noescape
func absSqNEON(dst []float64, a []complex128)

//go:noescape
func phaseNEON(dst []float64, a []complex128)

//go:noescape
func conjNEON(dst, a []complex128)
