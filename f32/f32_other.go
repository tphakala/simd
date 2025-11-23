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
