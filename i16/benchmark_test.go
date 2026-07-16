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

// Dot benchmarks sweep the lengths fixed-point codecs actually call at: n=8 is
// a short CELT band (where call overhead is the concern), n=240 a 20 ms frame
// at 12 kHz, n=480 one at 24 kHz. SetBytes counts both operands: 2*n int16.

func benchmarkDot(b *testing.B, n int, fn func(a, c []int16) int32) {
	b.Helper()
	a := make([]int16, n)
	c := make([]int16, n)
	for i := range a {
		a[i] = int16(i*7 - 3000)
		c[i] = int16(i*-5 + 2000)
	}
	b.SetBytes(int64(n) * 2 * 2)
	for b.Loop() {
		_ = fn(a, c)
	}
}

func BenchmarkDotProduct_8(b *testing.B)    { benchmarkDot(b, 8, DotProduct) }
func BenchmarkDotProduct_64(b *testing.B)   { benchmarkDot(b, 64, DotProduct) }
func BenchmarkDotProduct_240(b *testing.B)  { benchmarkDot(b, 240, DotProduct) }
func BenchmarkDotProduct_480(b *testing.B)  { benchmarkDot(b, 480, DotProduct) }
func BenchmarkDotProduct_4096(b *testing.B) { benchmarkDot(b, 4096, DotProduct) }

func BenchmarkDotProductGo_8(b *testing.B)    { benchmarkDot(b, 8, dotGo) }
func BenchmarkDotProductGo_64(b *testing.B)   { benchmarkDot(b, 64, dotGo) }
func BenchmarkDotProductGo_240(b *testing.B)  { benchmarkDot(b, 240, dotGo) }
func BenchmarkDotProductGo_480(b *testing.B)  { benchmarkDot(b, 480, dotGo) }
func BenchmarkDotProductGo_4096(b *testing.B) { benchmarkDot(b, 4096, dotGo) }

// XCorr benchmarks use the shape pitch analysis calls at: a 240-element frame
// correlated over a sweep of lags. SetBytes counts x once plus the y span the
// lags cover.

func benchmarkXCorr(b *testing.B, xn, lags int, fn func(dst []int32, x, y []int16)) {
	b.Helper()
	x := make([]int16, xn)
	y := make([]int16, xn+lags-1)
	for i := range x {
		x[i] = int16(i*7 - 3000)
	}
	for i := range y {
		y[i] = int16(i*-5 + 2000)
	}
	dst := make([]int32, lags)
	b.SetBytes(int64(xn+len(y)) * 2)
	for b.Loop() {
		fn(dst, x, y)
	}
}

func BenchmarkXCorr_240x4(b *testing.B)   { benchmarkXCorr(b, 240, 4, XCorr) }
func BenchmarkXCorr_240x64(b *testing.B)  { benchmarkXCorr(b, 240, 64, XCorr) }
func BenchmarkXCorr_240x288(b *testing.B) { benchmarkXCorr(b, 240, 288, XCorr) }

// 480 is a 20 ms frame at 24 kHz, and covers a second x length through the
// 16-wide AVX2 body.
func BenchmarkXCorr_480x64(b *testing.B) { benchmarkXCorr(b, 480, 64, XCorr) }

func BenchmarkXCorrGo_240x4(b *testing.B)   { benchmarkXCorr(b, 240, 4, xcorrGo) }
func BenchmarkXCorrGo_240x64(b *testing.B)  { benchmarkXCorr(b, 240, 64, xcorrGo) }
func BenchmarkXCorrGo_240x288(b *testing.B) { benchmarkXCorr(b, 240, 288, xcorrGo) }
func BenchmarkXCorrGo_480x64(b *testing.B)  { benchmarkXCorr(b, 480, 64, xcorrGo) }
