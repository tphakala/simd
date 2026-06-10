//go:build !amd64 && !arm64

package c128

// Fallback implementations for unsupported architectures

func mul128(dst, a, b []complex128)                  { mulGo(dst, a, b) }
func mulConj128(dst, a, b []complex128)              { mulConjGo(dst, a, b) }
func dotProduct128(a, b []complex128) complex128     { return dotProductGo(a, b) }
func dotProductConj128(a, b []complex128) complex128 { return dotProductConjGo(a, b) }
func scale128(dst, a []complex128, s complex128)     { scaleGo(dst, a, s) }
func add128(dst, a, b []complex128)                  { addGo(dst, a, b) }
func sub128(dst, a, b []complex128)                  { subGo(dst, a, b) }
func abs128(dst []float64, a []complex128)           { absGo(dst, a) }
func absSq128(dst []float64, a []complex128)         { absSqGo(dst, a) }
func conj128(dst, a []complex128)                    { conjGo(dst, a) }
func fromReal128(dst []complex128, src []float64)    { fromRealGo(dst, src) }
