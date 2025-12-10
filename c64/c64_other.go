//go:build !amd64 && !arm64

package c64

// Fallback implementations for unsupported architectures

func mul64(dst, a, b []complex64)               { mulGo(dst, a, b) }
func mulConj64(dst, a, b []complex64)           { mulConjGo(dst, a, b) }
func scale64(dst, a []complex64, s complex64)   { scaleGo(dst, a, s) }
func add64(dst, a, b []complex64)               { addGo(dst, a, b) }
func sub64(dst, a, b []complex64)               { subGo(dst, a, b) }
func abs64(dst []float32, a []complex64)        { absGo(dst, a) }
func absSq64(dst []float32, a []complex64)      { absSqGo(dst, a) }
func conj64(dst, a []complex64)                 { conjGo(dst, a) }
func fromReal64(dst []complex64, src []float32) { fromRealGo(dst, src) }
