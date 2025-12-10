//go:build !amd64 && !arm64

package f32

func dotProduct(a, b []float32) float32                { return dotProductGo(a, b) }
func add(dst, a, b []float32)                          { addGo(dst, a, b) }
func sub(dst, a, b []float32)                          { subGo(dst, a, b) }
func mul(dst, a, b []float32)                          { mulGo(dst, a, b) }
func div(dst, a, b []float32)                          { divGo(dst, a, b) }
func scale(dst, a []float32, s float32)                { scaleGo(dst, a, s) }
func addScalar(dst, a []float32, s float32)            { addScalarGo(dst, a, s) }
func sum(a []float32) float32                          { return sumGo(a) }
func min32(a []float32) float32                        { return minGo(a) }
func max32(a []float32) float32                        { return maxGo(a) }
func abs32(dst, a []float32)                           { absGo(dst, a) }
func neg32(dst, a []float32)                           { negGo(dst, a) }
func fma32(dst, a, b, c []float32)                     { fmaGo(dst, a, b, c) }
func clamp32(dst, a []float32, minVal, maxVal float32) { clampGo(dst, a, minVal, maxVal) }
func dotProductBatch32(results []float32, rows [][]float32, vec []float32) {
	dotProductBatch32Go(results, rows, vec)
}
func convolveValid32(dst, signal, kernel []float32)         { convolveValid32Go(dst, signal, kernel) }
func accumulateAdd32(dst, src []float32)                    { accumulateAdd32Go(dst, src) }
func interleave2_32(dst, a, b []float32)                    { interleave2Go(dst, a, b) }
func deinterleave2_32(a, b, src []float32)                  { deinterleave2Go(a, b, src) }
func sqrt32(dst, a []float32)                               { sqrt32Go(dst, a) }
func reciprocal32(dst, a []float32)                         { reciprocal32Go(dst, a) }
func minIdx32(a []float32) int                              { return minIdxGo(a) }
func maxIdx32(a []float32) int                              { return maxIdxGo(a) }
func addScaled32(dst []float32, alpha float32, s []float32) { addScaledGo(dst, alpha, s) }
func cumulativeSum32(dst, a []float32)                      { cumulativeSum32Go(dst, a) }
func convolveValidMulti32(dsts [][]float32, signal []float32, kernels [][]float32, n, kLen int) {
	convolveValidMultiGo(dsts, signal, kernels, n, kLen)
}
func variance32(a []float32, mean float32) float32 { return variance32Go(a, mean) }
func euclideanDistance32(a, b []float32) float32   { return euclideanDistance32Go(a, b) }
func cubicInterpDot32(hist, a, b, c, d []float32, x float32) float32 {
	return cubicInterpDotGo(hist, a, b, c, d, x)
}
func sigmoid32(dst, src []float32) { sigmoid32Go(dst, src) }
func relu32(dst, src []float32)    { relu32Go(dst, src) }
func clampScale32(dst, src []float32, minVal, maxVal, scale float32) {
	clampScale32Go(dst, src, minVal, maxVal, scale)
}
func tanh32(dst, src []float32)                                 { tanh32Go(dst, src) }
func exp32(dst, src []float32)                                  { exp32Go(dst, src) }
func int32ToFloat32Scale(dst []float32, src []int32, s float32) { int32ToFloat32ScaleGo(dst, src, s) }

// Split-format complex operations
func mulComplex32(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	mulComplex32Go(dstRe, dstIm, aRe, aIm, bRe, bIm)
}
func mulConjComplex32(dstRe, dstIm, aRe, aIm, bRe, bIm []float32) {
	mulConjComplex32Go(dstRe, dstIm, aRe, aIm, bRe, bIm)
}
func absSqComplex32(dst, aRe, aIm []float32) { absSqComplex32Go(dst, aRe, aIm) }
func butterflyComplex32(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm []float32) {
	butterflyComplex32Go(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)
}
