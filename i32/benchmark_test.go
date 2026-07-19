package i32

import "testing"

// Benchmarks pair the dispatched (SIMD) path with the pure-Go baseline so the
// speedup is visible directly. SetBytes counts the three buffers touched
// (a + b read, dst written), 4 bytes per int32.

const benchN = 1000

func BenchmarkInterleave2_1000(b *testing.B) {
	a := make([]int32, benchN)
	c := make([]int32, benchN)
	dst := make([]int32, benchN*2)
	for i := range a {
		a[i] = int32(i)
		c[i] = int32(i + benchN)
	}
	b.SetBytes(benchN * 4 * 3)
	for b.Loop() {
		Interleave2(dst, a, c)
	}
}

func BenchmarkInterleave2Go_1000(b *testing.B) {
	a := make([]int32, benchN)
	c := make([]int32, benchN)
	dst := make([]int32, benchN*2)
	for i := range a {
		a[i] = int32(i)
		c[i] = int32(i + benchN)
	}
	b.SetBytes(benchN * 4 * 3)
	for b.Loop() {
		interleave2Go(dst, a, c)
	}
}

func BenchmarkDeinterleave2_1000(b *testing.B) {
	src := make([]int32, benchN*2)
	a := make([]int32, benchN)
	c := make([]int32, benchN)
	for i := range src {
		src[i] = int32(i)
	}
	b.SetBytes(benchN * 4 * 3)
	for b.Loop() {
		Deinterleave2(a, c, src)
	}
}

func BenchmarkDeinterleave2Go_1000(b *testing.B) {
	src := make([]int32, benchN*2)
	a := make([]int32, benchN)
	c := make([]int32, benchN)
	for i := range src {
		src[i] = int32(i)
	}
	b.SetBytes(benchN * 4 * 3)
	for b.Loop() {
		deinterleave2Go(a, c, src)
	}
}

func benchAB() (a, b, dst []int32) {
	a = make([]int32, benchN)
	b = make([]int32, benchN)
	dst = make([]int32, benchN)
	for i := range a {
		a[i] = int32(i)
		b[i] = int32(i - benchN/2)
	}
	return a, b, dst
}

func BenchmarkAdd_1000(b *testing.B) {
	x, y, dst := benchAB()
	b.SetBytes(benchN * 4 * 3)
	for b.Loop() {
		Add(dst, x, y)
	}
}

func BenchmarkAddGo_1000(b *testing.B) {
	x, y, dst := benchAB()
	b.SetBytes(benchN * 4 * 3)
	for b.Loop() {
		addGo(dst, x, y)
	}
}

func BenchmarkSub_1000(b *testing.B) {
	x, y, dst := benchAB()
	b.SetBytes(benchN * 4 * 3)
	for b.Loop() {
		Sub(dst, x, y)
	}
}

func BenchmarkSubGo_1000(b *testing.B) {
	x, y, dst := benchAB()
	b.SetBytes(benchN * 4 * 3)
	for b.Loop() {
		subGo(dst, x, y)
	}
}

func BenchmarkMinMax_1000(b *testing.B) {
	res, _, _ := benchAB()
	b.SetBytes(benchN * 4)
	for b.Loop() {
		MinMax(res)
	}
}

func BenchmarkMinMaxGo_1000(b *testing.B) {
	res, _, _ := benchAB()
	b.SetBytes(benchN * 4)
	for b.Loop() {
		minMaxGo(res)
	}
}

// Tier-3 benchmarks. 1000 is a multiple of both vector widths; 25 and 1003
// are deliberate non-multiples of 4 and 8, so the scalar tails are always on
// the clock rather than structurally idle. Sum SetBytes counts a; Abs counts
// a read and dst written.

func benchmarkSum(b *testing.B, n int, fn func(a []int32) int32) {
	b.Helper()
	a := make([]int32, n)
	for i := range a {
		a[i] = int32(i*7 - 3000)
	}
	b.SetBytes(int64(n) * 4)
	for b.Loop() {
		_ = fn(a)
	}
}

func BenchmarkSum_25(b *testing.B)   { benchmarkSum(b, 25, Sum) }
func BenchmarkSum_1000(b *testing.B) { benchmarkSum(b, 1000, Sum) }
func BenchmarkSum_1003(b *testing.B) { benchmarkSum(b, 1003, Sum) }

func BenchmarkSumGo_25(b *testing.B)   { benchmarkSum(b, 25, sumGo) }
func BenchmarkSumGo_1000(b *testing.B) { benchmarkSum(b, 1000, sumGo) }
func BenchmarkSumGo_1003(b *testing.B) { benchmarkSum(b, 1003, sumGo) }

func benchmarkAbs(b *testing.B, n int, fn func(dst, a []int32)) {
	b.Helper()
	a := make([]int32, n)
	dst := make([]int32, n)
	for i := range a {
		a[i] = int32(i*7 - 3000)
	}
	b.SetBytes(int64(n) * 4 * 2)
	for b.Loop() {
		fn(dst, a)
	}
}

func BenchmarkAbs_25(b *testing.B)   { benchmarkAbs(b, 25, Abs) }
func BenchmarkAbs_1000(b *testing.B) { benchmarkAbs(b, 1000, Abs) }
func BenchmarkAbs_1003(b *testing.B) { benchmarkAbs(b, 1003, Abs) }

func BenchmarkAbsGo_25(b *testing.B)   { benchmarkAbs(b, 25, absGo) }
func BenchmarkAbsGo_1000(b *testing.B) { benchmarkAbs(b, 1000, absGo) }
func BenchmarkAbsGo_1003(b *testing.B) { benchmarkAbs(b, 1003, absGo) }

func benchmarkNegWhereNeg(b *testing.B, n int, fn func(dst, mag []int32, sign []float32)) {
	b.Helper()
	mag := make([]int32, n)
	sign := make([]float32, n)
	dst := make([]int32, n)
	for i := range mag {
		mag[i] = int32(i*7 - 3000)
		sign[i] = float32(i%3 - 1) // mix of -1, 0, 1 so both the negate and keep branches run
	}
	b.SetBytes(int64(n) * 4 * 3)
	for b.Loop() {
		fn(dst, mag, sign)
	}
}

func BenchmarkNegWhereNeg_25(b *testing.B)   { benchmarkNegWhereNeg(b, 25, NegWhereNeg) }
func BenchmarkNegWhereNeg_1000(b *testing.B) { benchmarkNegWhereNeg(b, 1000, NegWhereNeg) }
func BenchmarkNegWhereNeg_1003(b *testing.B) { benchmarkNegWhereNeg(b, 1003, NegWhereNeg) }

func BenchmarkNegWhereNegGo_25(b *testing.B)   { benchmarkNegWhereNeg(b, 25, negWhereNegGo) }
func BenchmarkNegWhereNegGo_1000(b *testing.B) { benchmarkNegWhereNeg(b, 1000, negWhereNegGo) }
func BenchmarkNegWhereNegGo_1003(b *testing.B) { benchmarkNegWhereNeg(b, 1003, negWhereNegGo) }

func benchmarkScaleQ31(b *testing.B, n int, fn func(dst, a []int32, k int32)) {
	b.Helper()
	a := make([]int32, n)
	dst := make([]int32, n)
	for i := range a {
		a[i] = int32(i*7 - 3000)
	}
	const k = int32(0x40000000) // 0.5 in Q31
	b.SetBytes(int64(n) * 4 * 2)
	for b.Loop() {
		fn(dst, a, k)
	}
}

func BenchmarkScaleQ31_25(b *testing.B)   { benchmarkScaleQ31(b, 25, ScaleQ31) }
func BenchmarkScaleQ31_1000(b *testing.B) { benchmarkScaleQ31(b, 1000, ScaleQ31) }
func BenchmarkScaleQ31_1003(b *testing.B) { benchmarkScaleQ31(b, 1003, ScaleQ31) }

func BenchmarkScaleQ31Go_25(b *testing.B)   { benchmarkScaleQ31(b, 25, scaleQ31Go) }
func BenchmarkScaleQ31Go_1000(b *testing.B) { benchmarkScaleQ31(b, 1000, scaleQ31Go) }
func BenchmarkScaleQ31Go_1003(b *testing.B) { benchmarkScaleQ31(b, 1003, scaleQ31Go) }

func benchmarkScaleQ15(b *testing.B, n int, fn func(dst, a []int32, k int16)) {
	b.Helper()
	a := make([]int32, n)
	dst := make([]int32, n)
	for i := range a {
		a[i] = int32(i*7 - 3000)
	}
	const k = int16(0x4000) // 0.5 in Q15
	b.SetBytes(int64(n) * 4 * 2)
	for b.Loop() {
		fn(dst, a, k)
	}
}

func BenchmarkScaleQ15_25(b *testing.B)   { benchmarkScaleQ15(b, 25, ScaleQ15) }
func BenchmarkScaleQ15_1000(b *testing.B) { benchmarkScaleQ15(b, 1000, ScaleQ15) }
func BenchmarkScaleQ15_1003(b *testing.B) { benchmarkScaleQ15(b, 1003, ScaleQ15) }

func BenchmarkScaleQ15Go_25(b *testing.B)   { benchmarkScaleQ15(b, 25, scaleQ15Go) }
func BenchmarkScaleQ15Go_1000(b *testing.B) { benchmarkScaleQ15(b, 1000, scaleQ15Go) }
func BenchmarkScaleQ15Go_1003(b *testing.B) { benchmarkScaleQ15(b, 1003, scaleQ15Go) }
