//go:build !amd64 && !arm64

package f64

// Fallback implementations for unsupported architectures

func dotProduct(a, b []float64) float64 { return dotProductGo(a, b) }
func add(dst, a, b []float64)           { addGo(dst, a, b) }
func sub(dst, a, b []float64)           { subGo(dst, a, b) }
func mul(dst, a, b []float64)           { mulGo(dst, a, b) }
func div(dst, a, b []float64)           { divGo(dst, a, b) }
func scale(dst, a []float64, s float64) { scaleGo(dst, a, s) }
func addScalar(dst, a []float64, s float64) { addScalarGo(dst, a, s) }
func sum(a []float64) float64           { return sumGo(a) }
func min64(a []float64) float64         { return minGo(a) }
func max64(a []float64) float64         { return maxGo(a) }
func abs64(dst, a []float64)            { absGo(dst, a) }
func neg64(dst, a []float64)            { negGo(dst, a) }
func fma64(dst, a, b, c []float64)      { fmaGo(dst, a, b, c) }
func clamp64(dst, a []float64, minVal, maxVal float64) { clampGo(dst, a, minVal, maxVal) }
func sqrt64(dst, a []float64) { sqrt64Go(dst, a) }
func reciprocal64(dst, a []float64) { reciprocal64Go(dst, a) }
func variance64(a []float64, mean float64) float64 { return variance64Go(a, mean) }
func euclideanDistance64(a, b []float64) float64 { return euclideanDistance64Go(a, b) }
func cumulativeSum64(dst, a []float64) { cumulativeSum64Go(dst, a) }
func dotProductBatch64(results []float64, rows [][]float64, vec []float64) { dotProductBatch64Go(results, rows, vec) }
func convolveValid64(dst, signal, kernel []float64) { convolveValid64Go(dst, signal, kernel) }
