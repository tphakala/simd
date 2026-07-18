package i16

import "testing"

// Benchmarks pair the dispatched (SIMD) path with the pure-Go baseline so the
// speedup is visible directly. SetBytes counts every byte moved: a and b (n
// int16 each) plus the double-width interleaved buffer (dst for interleave, src
// for deinterleave, 2*n int16), so 4*n int16 = 8*n bytes at 2 bytes per int16.
//
// n=24 is the regression guard for the interleave/deinterleave 8-wide AVX2 block
// (#149): 24 % 16 = 8, the residue where the block runs, and it was AVX2's worst
// case (up to 1.55x slower than SSE2 before the block, measured on the i7-1260P).
// 1000 % 16 = 8 too but at a high block count, where the residue amortizes; a
// parity test cannot see a performance defect, so only a benchmark at a length
// the block serves can. n=1000 keeps the historical name for benchstat history.

func benchmarkInterleave2(b *testing.B, n int, fn func(dst, a, c []int16)) {
	b.Helper()
	a := make([]int16, n)
	c := make([]int16, n)
	dst := make([]int16, n*2)
	for i := range a {
		a[i] = int16(i)
		c[i] = int16(i + n)
	}
	b.SetBytes(int64(n) * 2 * 4)
	for b.Loop() {
		fn(dst, a, c)
	}
}

func BenchmarkInterleave2_24(b *testing.B)   { benchmarkInterleave2(b, 24, Interleave2) }
func BenchmarkInterleave2_1000(b *testing.B) { benchmarkInterleave2(b, 1000, Interleave2) }

func BenchmarkInterleave2Go_24(b *testing.B)   { benchmarkInterleave2(b, 24, interleave2Go) }
func BenchmarkInterleave2Go_1000(b *testing.B) { benchmarkInterleave2(b, 1000, interleave2Go) }

func benchmarkDeinterleave2(b *testing.B, n int, fn func(a, c, src []int16)) {
	b.Helper()
	src := make([]int16, n*2)
	a := make([]int16, n)
	c := make([]int16, n)
	for i := range src {
		src[i] = int16(i)
	}
	b.SetBytes(int64(n) * 2 * 4)
	for b.Loop() {
		fn(a, c, src)
	}
}

func BenchmarkDeinterleave2_24(b *testing.B)   { benchmarkDeinterleave2(b, 24, Deinterleave2) }
func BenchmarkDeinterleave2_1000(b *testing.B) { benchmarkDeinterleave2(b, 1000, Deinterleave2) }

func BenchmarkDeinterleave2Go_24(b *testing.B)   { benchmarkDeinterleave2(b, 24, deinterleave2Go) }
func BenchmarkDeinterleave2Go_1000(b *testing.B) { benchmarkDeinterleave2(b, 1000, deinterleave2Go) }

// Dot benchmarks sweep the lengths fixed-point codecs actually call at: n=8 is
// a short CELT band (where call overhead is the concern), n=240 a 20 ms frame
// at 12 kHz, n=480 one at 24 kHz. SetBytes counts both operands: 2*n int16.
//
// 24 and 248 are the regression guard for the 8-wide AVX2 block, and they are
// here because every other length above is a multiple of 16 (8 aside), which is
// exactly how #149's sawtooth stayed invisible: a parity test cannot see a
// performance defect, so only a benchmark at a length the block serves can. 24
// is the worst case #149 measured (AVX2 was 1.41x slower than SSE2 there) and
// 248 is the same residue at a high block count.

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
func BenchmarkDotProduct_24(b *testing.B)   { benchmarkDot(b, 24, DotProduct) }
func BenchmarkDotProduct_64(b *testing.B)   { benchmarkDot(b, 64, DotProduct) }
func BenchmarkDotProduct_240(b *testing.B)  { benchmarkDot(b, 240, DotProduct) }
func BenchmarkDotProduct_248(b *testing.B)  { benchmarkDot(b, 248, DotProduct) }
func BenchmarkDotProduct_480(b *testing.B)  { benchmarkDot(b, 480, DotProduct) }
func BenchmarkDotProduct_4096(b *testing.B) { benchmarkDot(b, 4096, DotProduct) }

func BenchmarkDotProductGo_8(b *testing.B)    { benchmarkDot(b, 8, dotGo) }
func BenchmarkDotProductGo_24(b *testing.B)   { benchmarkDot(b, 24, dotGo) }
func BenchmarkDotProductGo_64(b *testing.B)   { benchmarkDot(b, 64, dotGo) }
func BenchmarkDotProductGo_240(b *testing.B)  { benchmarkDot(b, 240, dotGo) }
func BenchmarkDotProductGo_248(b *testing.B)  { benchmarkDot(b, 248, dotGo) }
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

// The lengths above are all multiples of 16 and 4, which is precisely why they
// hide things. 240 % 16 == 0 and 480 % 16 == 0, so the AVX2 kernel's scalar
// tail never runs in any of them; 4, 64 and 288 are all multiples of
// xcorrLagBlock, so the remainder-lag path never runs either. The cases below
// exist to make both visible.
//
// x=25 and x=248 leave len(x) % 16 in 8..15, the remainder #151 gave xcorr4AVX2
// an 8-wide block to absorb; AVX2 had been up to 1.75x slower than SSE2 there.
// x=248 is that shape at a realistic length.
//
// lags=61 is the pitch-analysis count from the motivating caller, and it is not
// a multiple of xcorrLagBlock, so it leaves one remainder lag. That lag runs
// through dotI16 rather than xcorr4, which makes 248x61 the only case here that
// reaches dotAVX2's own 8-wide block (#149): 240x61 takes the same remainder-lag
// path but hands dotI16 a residue of 0, so the block never runs.
func BenchmarkXCorr_25x64(b *testing.B)  { benchmarkXCorr(b, 25, 64, XCorr) }
func BenchmarkXCorr_248x64(b *testing.B) { benchmarkXCorr(b, 248, 64, XCorr) }
func BenchmarkXCorr_240x61(b *testing.B) { benchmarkXCorr(b, 240, 61, XCorr) }
func BenchmarkXCorr_248x61(b *testing.B) { benchmarkXCorr(b, 248, 61, XCorr) }

func BenchmarkXCorrGo_25x64(b *testing.B)  { benchmarkXCorr(b, 25, 64, xcorrGo) }
func BenchmarkXCorrGo_248x64(b *testing.B) { benchmarkXCorr(b, 248, 64, xcorrGo) }
func BenchmarkXCorrGo_240x61(b *testing.B) { benchmarkXCorr(b, 240, 61, xcorrGo) }
func BenchmarkXCorrGo_248x61(b *testing.B) { benchmarkXCorr(b, 248, 61, xcorrGo) }

func BenchmarkXCorrGo_240x4(b *testing.B)   { benchmarkXCorr(b, 240, 4, xcorrGo) }
func BenchmarkXCorrGo_240x64(b *testing.B)  { benchmarkXCorr(b, 240, 64, xcorrGo) }
func BenchmarkXCorrGo_240x288(b *testing.B) { benchmarkXCorr(b, 240, 288, xcorrGo) }
func BenchmarkXCorrGo_480x64(b *testing.B)  { benchmarkXCorr(b, 480, 64, xcorrGo) }

// Tier-3 benchmarks. These ops are store-bound (far cheaper per element than
// dot), so the SIMD-vs-Go crossover at small n is what the 8 case exists to
// measure: 8 is one NEON block and half an AVX2 one, so it sits exactly at the
// NEON dispatch cut and below the AVX2 one.
//
// 240 is a multiple of both vector widths; 8 is a multiple of the NEON width
// only, which is the previous paragraph's point. 25 and 1003 are deliberate
// non-multiples of 8 and 16, so the scalar tails are always on the clock (the
// original XCorr lengths were all multiples of 16 and structurally hid a
// tail-cost defect; see BenchmarkXCorr_25x64 above). 1000 is a multiple of the
// NEON width but not the AVX2 one (1000 mod 16 = 8), so it also charges an
// AVX2 tail.
//
// MulQ15 SetBytes counts a + b read and dst written; Abs counts a + dst;
// MaxAbs counts a.

func benchmarkMulQ15(b *testing.B, n int, fn func(dst, a, c []int16)) {
	b.Helper()
	a := make([]int16, n)
	c := make([]int16, n)
	dst := make([]int16, n)
	for i := range a {
		a[i] = int16(i*7 - 3000)
		c[i] = int16(i*-5 + 2000)
	}
	b.SetBytes(int64(n) * 2 * 3)
	for b.Loop() {
		fn(dst, a, c)
	}
}

func BenchmarkMulQ15_8(b *testing.B)    { benchmarkMulQ15(b, 8, MulQ15) }
func BenchmarkMulQ15_25(b *testing.B)   { benchmarkMulQ15(b, 25, MulQ15) }
func BenchmarkMulQ15_240(b *testing.B)  { benchmarkMulQ15(b, 240, MulQ15) }
func BenchmarkMulQ15_1003(b *testing.B) { benchmarkMulQ15(b, 1003, MulQ15) }

func BenchmarkMulQ15Go_8(b *testing.B)    { benchmarkMulQ15(b, 8, mulQ15Go) }
func BenchmarkMulQ15Go_25(b *testing.B)   { benchmarkMulQ15(b, 25, mulQ15Go) }
func BenchmarkMulQ15Go_240(b *testing.B)  { benchmarkMulQ15(b, 240, mulQ15Go) }
func BenchmarkMulQ15Go_1003(b *testing.B) { benchmarkMulQ15(b, 1003, mulQ15Go) }

func benchmarkAbs(b *testing.B, n int, fn func(dst, a []int16)) {
	b.Helper()
	a := make([]int16, n)
	dst := make([]int16, n)
	for i := range a {
		a[i] = int16(i*7 - 3000)
	}
	b.SetBytes(int64(n) * 2 * 2)
	for b.Loop() {
		fn(dst, a)
	}
}

func BenchmarkAbs_8(b *testing.B)    { benchmarkAbs(b, 8, Abs) }
func BenchmarkAbs_25(b *testing.B)   { benchmarkAbs(b, 25, Abs) }
func BenchmarkAbs_1000(b *testing.B) { benchmarkAbs(b, 1000, Abs) }
func BenchmarkAbs_1003(b *testing.B) { benchmarkAbs(b, 1003, Abs) }

func BenchmarkAbsGo_8(b *testing.B)    { benchmarkAbs(b, 8, absGo) }
func BenchmarkAbsGo_25(b *testing.B)   { benchmarkAbs(b, 25, absGo) }
func BenchmarkAbsGo_1000(b *testing.B) { benchmarkAbs(b, 1000, absGo) }
func BenchmarkAbsGo_1003(b *testing.B) { benchmarkAbs(b, 1003, absGo) }

func benchmarkMaxAbs(b *testing.B, n int, fn func(a []int16) int) {
	b.Helper()
	a := make([]int16, n)
	for i := range a {
		a[i] = int16(i*7 - 3000)
	}
	b.SetBytes(int64(n) * 2)
	for b.Loop() {
		_ = fn(a)
	}
}

func BenchmarkMaxAbs_8(b *testing.B)    { benchmarkMaxAbs(b, 8, MaxAbs) }
func BenchmarkMaxAbs_25(b *testing.B)   { benchmarkMaxAbs(b, 25, MaxAbs) }
func BenchmarkMaxAbs_1000(b *testing.B) { benchmarkMaxAbs(b, 1000, MaxAbs) }
func BenchmarkMaxAbs_1003(b *testing.B) { benchmarkMaxAbs(b, 1003, MaxAbs) }

func BenchmarkMaxAbsGo_8(b *testing.B)    { benchmarkMaxAbs(b, 8, maxAbsGo) }
func BenchmarkMaxAbsGo_25(b *testing.B)   { benchmarkMaxAbs(b, 25, maxAbsGo) }
func BenchmarkMaxAbsGo_1000(b *testing.B) { benchmarkMaxAbs(b, 1000, maxAbsGo) }
func BenchmarkMaxAbsGo_1003(b *testing.B) { benchmarkMaxAbs(b, 1003, maxAbsGo) }
