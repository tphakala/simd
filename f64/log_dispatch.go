//go:build amd64 || arm64

package f64

import "math"

// Reconstruction constants for the shared SIMD log core (logAVX / logNEON64).
// The core computes, per lane, result = e*k1hi + (lnm*k2 + e*k1lo), where
// x = m * 2^e with m in [sqrt(2)/2, sqrt(2)) and lnm = ln(m). The hi/lo pairs
// keep the e*ln(2) (resp. e*log10(2)) term exact: the hi part has its low
// mantissa bits zeroed (fdlibm splittings), so the product with the
// integer-valued e introduces no rounding, and the remainder rides in the lo
// term.
const (
	logLn2Hi64  = 6.93147180369123816490e-01 // 0x3fe62e42fee00000
	logLn2Lo64  = 1.90821492927058770002e-10 // 0x3dea39ef35793c76
	logL102Hi64 = 3.01029995663611771306e-01 // 0x3fd34413509f6000, log10(2) hi
	logL102Lo64 = 3.69423907715893078616e-13 // 0x3d59fef311f12b36, log10(2) lo
	logLog2E64  = 1 / math.Ln2               // 0x3ff71547652b82fe
	logLog10E64 = 1 / math.Ln10              // 0x3fdbcb7b1526e50e
)

// powSIMDOK64 reports whether every element is positive and finite, the
// precondition for the fused exp(p*ln(x)) kernels. Lanes outside that range
// have math.Pow edge semantics the kernel does not implement (negative bases
// with integer exponents, signed zeros, infinities, NaN propagation), so the
// whole call falls back to the exact scalar path.
func powSIMDOK64(src []float64) bool {
	for _, x := range src {
		if !(x > 0 && x <= math.MaxFloat64) {
			return false
		}
	}
	return true
}

// allFinite64 reports whether every element is finite (PowElem exponents).
func allFinite64(s []float64) bool {
	for _, x := range s {
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}
