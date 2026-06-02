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

func benchStereo() (left, right, p, q []int32) {
	left = make([]int32, benchN)
	right = make([]int32, benchN)
	p = make([]int32, benchN)
	q = make([]int32, benchN)
	for i := range left {
		left[i] = int32(i*3 - benchN)
		right[i] = int32(benchN - i)
	}
	return left, right, p, q
}

func BenchmarkMidSideEncode_1000(b *testing.B) {
	left, right, mid, side := benchStereo()
	b.SetBytes(benchN * 4 * 4)
	for b.Loop() {
		MidSideEncode(mid, side, left, right)
	}
}

func BenchmarkMidSideEncodeGo_1000(b *testing.B) {
	left, right, mid, side := benchStereo()
	b.SetBytes(benchN * 4 * 4)
	for b.Loop() {
		midSideEncodeGo(mid, side, left, right)
	}
}

func BenchmarkMidSideDecode_1000(b *testing.B) {
	mid, side, left, right := benchStereo()
	b.SetBytes(benchN * 4 * 4)
	for b.Loop() {
		MidSideDecode(left, right, mid, side)
	}
}

func BenchmarkMidSideDecodeGo_1000(b *testing.B) {
	mid, side, left, right := benchStereo()
	b.SetBytes(benchN * 4 * 4)
	for b.Loop() {
		midSideDecodeGo(left, right, mid, side)
	}
}

func benchSrcDst() (src, dst []int32) {
	src = make([]int32, benchN)
	dst = make([]int32, benchN)
	for i := range src {
		src[i] = int32(i * i)
	}
	return src, dst
}

func BenchmarkDiff1_1000(b *testing.B) {
	src, dst := benchSrcDst()
	b.SetBytes(benchN * 4 * 2)
	for b.Loop() {
		Diff1(dst, src)
	}
}

func BenchmarkDiff1Go_1000(b *testing.B) {
	src, dst := benchSrcDst()
	b.SetBytes(benchN * 4 * 2)
	for b.Loop() {
		diff1Go(dst, src)
	}
}

func BenchmarkDiff2_1000(b *testing.B) {
	src, dst := benchSrcDst()
	b.SetBytes(benchN * 4 * 2)
	for b.Loop() {
		Diff2(dst, src)
	}
}

func BenchmarkDiff2Go_1000(b *testing.B) {
	src, dst := benchSrcDst()
	b.SetBytes(benchN * 4 * 2)
	for b.Loop() {
		diff2Go(dst, src)
	}
}

func BenchmarkDiff3_1000(b *testing.B) {
	src, dst := benchSrcDst()
	b.SetBytes(benchN * 4 * 2)
	for b.Loop() {
		Diff3(dst, src)
	}
}

func BenchmarkDiff3Go_1000(b *testing.B) {
	src, dst := benchSrcDst()
	b.SetBytes(benchN * 4 * 2)
	for b.Loop() {
		diff3Go(dst, src)
	}
}

func BenchmarkDiff4_1000(b *testing.B) {
	src, dst := benchSrcDst()
	b.SetBytes(benchN * 4 * 2)
	for b.Loop() {
		Diff4(dst, src)
	}
}

func BenchmarkDiff4Go_1000(b *testing.B) {
	src, dst := benchSrcDst()
	b.SetBytes(benchN * 4 * 2)
	for b.Loop() {
		diff4Go(dst, src)
	}
}
