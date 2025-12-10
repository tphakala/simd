//go:build arm64

package c64

import "github.com/tphakala/simd/cpu"

var hasNEON = cpu.ARM64.NEON

func mul64(dst, a, b []complex64) {
	if hasNEON {
		mulNEON(dst, a, b)
		return
	}
	mulGo(dst, a, b)
}

func mulConj64(dst, a, b []complex64) {
	if hasNEON {
		mulConjNEON(dst, a, b)
		return
	}
	mulConjGo(dst, a, b)
}

func scale64(dst, a []complex64, s complex64) {
	if hasNEON {
		scaleNEON(dst, a, s)
		return
	}
	scaleGo(dst, a, s)
}

func add64(dst, a, b []complex64) {
	if hasNEON {
		addNEON(dst, a, b)
		return
	}
	addGo(dst, a, b)
}

func sub64(dst, a, b []complex64) {
	if hasNEON {
		subNEON(dst, a, b)
		return
	}
	subGo(dst, a, b)
}

func abs64(dst []float32, a []complex64) {
	if hasNEON {
		absNEON(dst, a)
		return
	}
	absGo(dst, a)
}

func absSq64(dst []float32, a []complex64) {
	if hasNEON {
		absSqNEON(dst, a)
		return
	}
	absSqGo(dst, a)
}

func conj64(dst, a []complex64) {
	if hasNEON {
		conjNEON(dst, a)
		return
	}
	conjGo(dst, a)
}

func fromReal64(dst []complex64, src []float32) {
	if hasNEON {
		fromRealNEON(dst, src)
		return
	}
	fromRealGo(dst, src)
}

//go:noescape
func mulNEON(dst, a, b []complex64)

//go:noescape
func mulConjNEON(dst, a, b []complex64)

//go:noescape
func scaleNEON(dst, a []complex64, s complex64)

//go:noescape
func addNEON(dst, a, b []complex64)

//go:noescape
func subNEON(dst, a, b []complex64)

//go:noescape
func absNEON(dst []float32, a []complex64)

//go:noescape
func absSqNEON(dst []float32, a []complex64)

//go:noescape
func conjNEON(dst, a []complex64)

//go:noescape
func fromRealNEON(dst []complex64, src []float32)
