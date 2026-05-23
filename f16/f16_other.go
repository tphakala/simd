//go:build !arm64

package f16

func toFloat32(h Float16) float32 {
	return toFloat32Go(h)
}

func fromFloat32(f float32) Float16 {
	return fromFloat32Go(f)
}

func toFloat32Slice(dst []float32, src []Float16) {
	toFloat32SliceGo(dst, src)
}

func fromFloat32Slice(dst []Float16, src []float32) {
	fromFloat32SliceGo(dst, src)
}

func dotProduct(a, b []Float16) float32 {
	return dotProductGo(a, b)
}

func dotProductF32(a, b []Float16) float32 {
	return dotProductGo(a, b)
}

func add(dst, a, b []Float16) {
	addGo(dst, a, b)
}

func sub(dst, a, b []Float16) {
	subGo(dst, a, b)
}

func mul(dst, a, b []Float16) {
	mulGo(dst, a, b)
}

func scale(dst, a []Float16, s Float16) {
	scaleGo(dst, a, s)
}

func fma16(dst, a, b, c []Float16) {
	fmaGo(dst, a, b, c)
}

func sum(a []Float16) float32 {
	return sumGo(a)
}

func abs16(dst, a []Float16) {
	absGo(dst, a)
}

func neg16(dst, a []Float16) {
	negGo(dst, a)
}

func relu16(dst, src []Float16) {
	reluGo(dst, src)
}

func sigmoid16(dst, src []Float16) {
	sigmoidGo(dst, src)
}

func min16(a []Float16) Float16 {
	return minGo(a)
}

func max16(a []Float16) Float16 {
	return maxGo(a)
}

func div16(dst, a, b []Float16) {
	divGo(dst, a, b)
}

func addScalar16(dst, a []Float16, s Float16) {
	addScalarGo(dst, a, s)
}

func clamp16(dst, a []Float16, minVal, maxVal Float16) {
	clampGo(dst, a, minVal, maxVal)
}

func sqrt16(dst, a []Float16) {
	sqrtGo(dst, a)
}

func reciprocal16(dst, a []Float16) {
	reciprocalGo(dst, a)
}

func exp16(dst, src []Float16) {
	expGo(dst, src)
}

func tanh16(dst, src []Float16) {
	tanhGo(dst, src)
}

func minIdx16(a []Float16) int {
	return minIdxGo(a)
}

func maxIdx16(a []Float16) int {
	return maxIdxGo(a)
}

func addScaled16(dst []Float16, alpha Float16, s []Float16) {
	addScaledGo(dst, alpha, s)
}

func euclideanDistance16(a, b []Float16) float32 {
	return euclideanDistanceGo(a, b)
}

func variance16(a []Float16, mean float32) float32 {
	return varianceGo(a, mean)
}

func cumulativeSum16(dst, a []Float16) {
	cumulativeSumGo(dst, a)
}

func dotProductBatch16(results []float32, rows [][]Float16, vec []Float16) {
	dotProductBatchGo(results, rows, vec)
}

func accumulateAdd16(dst, src []Float16) {
	accumulateAddGo(dst, src)
}

func convolveValid16(dst, signal, kernel []Float16) {
	convolveValidGo(dst, signal, kernel)
}

func interleave2_16(dst, a, b []Float16) {
	interleave2Go(dst, a, b)
}

func deinterleave2_16(a, b, src []Float16) {
	deinterleave2Go(a, b, src)
}

func clampScale16(dst, src []Float16, minVal, maxVal, scale Float16) {
	clampScaleGo(dst, src, minVal, maxVal, scale)
}
