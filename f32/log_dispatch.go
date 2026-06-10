//go:build amd64 || arm64

package f32

import "math"

// Reconstruction constants for the shared SIMD log core (logAVX / logNEON32).
// The core computes, per lane, result = e*k1hi + ((z + lnmLo)*k2 + e*k1lo),
// where x = m * 2^e with m in [sqrt(2)/2, sqrt(2)), z = m-1, and
// z + lnmLo = ln(m) (Cephes logf polynomial). The hi/lo pairs keep the
// e*ln(2) (resp. e*log10(2)) term accurate: the hi part has its low mantissa
// bits zeroed (Cephes logf splitting), so the product with the integer-valued
// e is exact, and the remainder rides in the lo term.
const (
	logLn2Hi32  = 0.693359375    // 0x3f318000
	logLn2Lo32  = -2.12194440e-4 // 0xb95e8083
	logL102Hi32 = 0.301025390625 // 0x3e9a2000, log10(2) hi
	logL102Lo32 = 4.6050389811952e-6
	logLog2E32  = 1.4426950408889634 // 1/ln(2), rounds to 0x3fb8aa3b
	logLog10E32 = 0.4342944819032518 // 1/ln(10), rounds to 0x3ede5bd9
)

// powSIMDOK32 reports whether every element is positive and finite, the
// precondition for the fused exp(p*ln(x)) kernels. Lanes outside that range
// have math.Pow edge semantics the kernel does not implement (negative bases
// with integer exponents, signed zeros, infinities, NaN propagation), so the
// whole call falls back to the exact scalar path.
func powSIMDOK32(src []float32) bool {
	for _, x := range src {
		if !(x > 0 && x <= math.MaxFloat32) {
			return false
		}
	}
	return true
}

// allFinite32 reports whether every element is finite (PowElem exponents).
func allFinite32(s []float32) bool {
	for _, x := range s {
		d := float64(x)
		if math.IsNaN(d) || math.IsInf(d, 0) {
			return false
		}
	}
	return true
}
