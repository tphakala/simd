package i8

import "testing"

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

func BenchmarkMinMax(b *testing.B) {
	a := genI8(benchN, 1)
	b.SetBytes(benchN)
	b.ResetTimer()
	for b.Loop() {
		_, _ = MinMax(a)
	}
}
