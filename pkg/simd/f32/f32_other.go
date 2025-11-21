//go:build !amd64 && !arm64

package f32

func dotProduct(a, b []float32) float32 { return dotProductGo(a, b) }
func add(dst, a, b []float32)           { addGo(dst, a, b) }
func sub(dst, a, b []float32)           { subGo(dst, a, b) }
func mul(dst, a, b []float32)           { mulGo(dst, a, b) }
func div(dst, a, b []float32)           { divGo(dst, a, b) }
func scale(dst, a []float32, s float32) { scaleGo(dst, a, s) }
func addScalar(dst, a []float32, s float32) { addScalarGo(dst, a, s) }
func sum(a []float32) float32           { return sumGo(a) }
func min32(a []float32) float32         { return minGo(a) }
func max32(a []float32) float32         { return maxGo(a) }
func abs32(dst, a []float32)            { absGo(dst, a) }
func neg32(dst, a []float32)            { negGo(dst, a) }
func fma32(dst, a, b, c []float32)      { fmaGo(dst, a, b, c) }
func clamp32(dst, a []float32, minVal, maxVal float32) { clampGo(dst, a, minVal, maxVal) }
func dotProductBatch32(results []float32, rows [][]float32, vec []float32) { dotProductBatch32Go(results, rows, vec) }
func convolveValid32(dst, signal, kernel []float32) { convolveValid32Go(dst, signal, kernel) }
