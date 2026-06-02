package i16

import "testing"

// Benchmarks pair the dispatched (SIMD) path with the pure-Go baseline so the
// speedup is visible directly. SetBytes counts every byte moved: a and b (benchN
// int16 each) plus the double-width interleaved buffer (dst for interleave, src
// for deinterleave, 2*benchN int16), so 4*benchN int16 = 8*benchN bytes at 2
// bytes per int16.

const benchN = 1000

func BenchmarkInterleave2_1000(b *testing.B) {
	a := make([]int16, benchN)
	c := make([]int16, benchN)
	dst := make([]int16, benchN*2)
	for i := range a {
		a[i] = int16(i)
		c[i] = int16(i + benchN)
	}
	b.SetBytes(benchN * 2 * 4)
	for b.Loop() {
		Interleave2(dst, a, c)
	}
}

func BenchmarkInterleave2Go_1000(b *testing.B) {
	a := make([]int16, benchN)
	c := make([]int16, benchN)
	dst := make([]int16, benchN*2)
	for i := range a {
		a[i] = int16(i)
		c[i] = int16(i + benchN)
	}
	b.SetBytes(benchN * 2 * 4)
	for b.Loop() {
		interleave2Go(dst, a, c)
	}
}

func BenchmarkDeinterleave2_1000(b *testing.B) {
	src := make([]int16, benchN*2)
	a := make([]int16, benchN)
	c := make([]int16, benchN)
	for i := range src {
		src[i] = int16(i)
	}
	b.SetBytes(benchN * 2 * 4)
	for b.Loop() {
		Deinterleave2(a, c, src)
	}
}

func BenchmarkDeinterleave2Go_1000(b *testing.B) {
	src := make([]int16, benchN*2)
	a := make([]int16, benchN)
	c := make([]int16, benchN)
	for i := range src {
		src[i] = int16(i)
	}
	b.SetBytes(benchN * 2 * 4)
	for b.Loop() {
		deinterleave2Go(a, c, src)
	}
}
