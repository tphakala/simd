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
