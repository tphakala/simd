package i8

import (
	"fmt"
	"testing"
)

const benchN = 4096

func BenchmarkAddSaturate(b *testing.B) {
	a, c := genI8(benchN, 1), genI8(benchN, 2)
	dst := make([]int8, benchN)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		AddSaturate(dst, a, c)
	}
}

func BenchmarkSubSaturate(b *testing.B) {
	a, c := genI8(benchN, 1), genI8(benchN, 2)
	dst := make([]int8, benchN)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		SubSaturate(dst, a, c)
	}
}

func BenchmarkToInt16(b *testing.B) {
	src := genI8(benchN, 1)
	dst := make([]int16, benchN)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		ToInt16(dst, src)
	}
}

func BenchmarkToInt32(b *testing.B) {
	src := genI8(benchN, 1)
	dst := make([]int32, benchN)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		ToInt32(dst, src)
	}
}

func BenchmarkSum(b *testing.B) {
	a := genI8(benchN, 1)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		_ = Sum(a)
	}
}

func BenchmarkDotProduct(b *testing.B) {
	a, c := genI8(benchN, 1), genI8(benchN, 2)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		_ = DotProduct(a, c)
	}
}

// BenchmarkSum_N and BenchmarkDotProduct_N guard the 8-wide AVX2 tail block
// (#149): n%16 in 8..15 exercises the block, aligned n (16, 32) does not. The
// fixed 4096-byte benchmarks above are all n%16==0 and cannot see the tail
// sawtooth, so residue lengths (24, 40, 248) must be measured explicitly; n=16
// is the aligned sentinel where the block never runs.
func BenchmarkSum_N(b *testing.B) {
	// 16 is the aligned sentinel (block skipped); 24/40/248 are residue 8 (block,
	// empty scalar tail); 31 is residue 15 (block plus the full 7-element scalar
	// tail), so the guard times the block in isolation and alongside a tail.
	for _, n := range []int{16, 24, 31, 40, 248} {
		a := genI8(n, 1)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				_ = Sum(a)
			}
		})
	}
}

func BenchmarkDotProduct_N(b *testing.B) {
	for _, n := range []int{16, 24, 31, 40, 248} {
		a, c := genI8(n, 1), genI8(n, 2)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				_ = DotProduct(a, c)
			}
		})
	}
}

func BenchmarkMinMax(b *testing.B) {
	a := genI8(benchN, 1)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		_, _ = MinMax(a)
	}
}

func BenchmarkMin(b *testing.B) {
	a, c := genI8(benchN, 1), genI8(benchN, 2)
	dst := make([]int8, benchN)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		Min(dst, a, c)
	}
}

func BenchmarkMax(b *testing.B) {
	a, c := genI8(benchN, 1), genI8(benchN, 2)
	dst := make([]int8, benchN)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		Max(dst, a, c)
	}
}

func BenchmarkClamp(b *testing.B) {
	a := genI8(benchN, 1)
	dst := make([]int8, benchN)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		Clamp(dst, a, -64, 64)
	}
}

func BenchmarkAbs(b *testing.B) {
	a := genI8(benchN, 1)
	dst := make([]int8, benchN)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		Abs(dst, a)
	}
}

func BenchmarkNeg(b *testing.B) {
	a := genI8(benchN, 1)
	dst := make([]int8, benchN)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		Neg(dst, a)
	}
}

func BenchmarkMaxAbs(b *testing.B) {
	a := genI8(benchN, 1)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		_ = MaxAbs(a)
	}
}

func BenchmarkAbsDiff(b *testing.B) {
	a, c := genI8(benchN, 1), genI8(benchN, 2)
	dst := make([]int8, benchN)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		AbsDiff(dst, a, c)
	}
}

func BenchmarkAddScalarSaturate(b *testing.B) {
	a := genI8(benchN, 1)
	dst := make([]int8, benchN)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		AddScalarSaturate(dst, a, 7)
	}
}

func BenchmarkSubScalarSaturate(b *testing.B) {
	a := genI8(benchN, 1)
	dst := make([]int8, benchN)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		SubScalarSaturate(dst, a, 7)
	}
}

func BenchmarkSumAbs(b *testing.B) {
	a := genI8(benchN, 1)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		_ = SumAbs(a)
	}
}

func BenchmarkSAD(b *testing.B) {
	a, c := genI8(benchN, 1), genI8(benchN, 2)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		_ = SAD(a, c)
	}
}

// BenchmarkSumAbs_N and BenchmarkSAD_N guard the 16- and 8-wide AVX2 tail blocks
// (#149) on the 32-wide reduction bodies: a residue of 1..31 bytes was served by
// a serial scalar accumulator chain, so a worst-case residue of 31 cost up to 31
// dependent adds. The fixed 4096-byte benchmarks above are all n%32==0 and cannot
// see the sawtooth, so residue lengths must be measured explicitly. 32 is the
// aligned sentinel (both blocks skipped); 40 is residue 8 (only the 8-wide block);
// 48 is residue 16 (only the 16-wide block); 63 is residue 31 (both blocks plus
// the full 7-element scalar tail, the worst case); 95 is residue 31 with more
// 32-wide blocks; 248 is a realistic length (residue 24, both blocks).
func BenchmarkSumAbs_N(b *testing.B) {
	for _, n := range []int{32, 40, 48, 63, 95, 248} {
		a := genI8(n, 1)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				_ = SumAbs(a)
			}
		})
	}
}

func BenchmarkSAD_N(b *testing.B) {
	for _, n := range []int{32, 40, 48, 63, 95, 248} {
		a, c := genI8(n, 1), genI8(n, 2)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				_ = SAD(a, c)
			}
		})
	}
}
