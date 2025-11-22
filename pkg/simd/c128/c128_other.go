//go:build !amd64 && !arm64

package c128

// Fallback implementations for unsupported architectures

func mul128(dst, a, b []complex128)              { mulGo(dst, a, b) }
func mulConj128(dst, a, b []complex128)          { mulConjGo(dst, a, b) }
func scale128(dst, a []complex128, s complex128) { scaleGo(dst, a, s) }
func add128(dst, a, b []complex128)              { addGo(dst, a, b) }
func sub128(dst, a, b []complex128)              { subGo(dst, a, b) }
