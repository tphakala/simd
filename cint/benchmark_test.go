package cint

import "testing"

// Benchmarks pair the dispatched (SIMD) path with the pure-Go baseline so the
// speedup is visible directly. benchN is the int32 lane count (benchN/2 complex).

const benchN = 2048

func benchData() (dst, a []int32, tw []int16) {
	dst = make([]int32, benchN)
	a = genI32(benchN, 1)
	tw = genI16(benchN, 2)
	return
}

func BenchmarkMul(b *testing.B) {
	dst, a, tw := benchData()
	b.SetBytes(benchN * (4 + 4 + 2)) // a read (int32), dst write (int32), tw read (int16)
	for b.Loop() {
		Mul(dst, a, tw)
	}
}

func BenchmarkMulGo(b *testing.B) {
	dst, a, tw := benchData()
	b.SetBytes(benchN * (4 + 4 + 2))
	for b.Loop() {
		mulGo(dst, a, tw)
	}
}

func BenchmarkMulConj(b *testing.B) {
	dst, a, tw := benchData()
	b.SetBytes(benchN * (4 + 4 + 2))
	for b.Loop() {
		MulConj(dst, a, tw)
	}
}

func BenchmarkMulConjGo(b *testing.B) {
	dst, a, tw := benchData()
	b.SetBytes(benchN * (4 + 4 + 2))
	for b.Loop() {
		mulConjGo(dst, a, tw)
	}
}

func BenchmarkMulByScalar(b *testing.B) {
	a := genI32(benchN, 3)
	b.SetBytes(benchN * 4)
	for b.Loop() {
		MulByScalar(a, 0x1234)
	}
}

func BenchmarkMulByScalarGo(b *testing.B) {
	a := genI32(benchN, 3)
	b.SetBytes(benchN * 4)
	for b.Loop() {
		mulByScalarGo(a, 0x1234)
	}
}

func BenchmarkAdd(b *testing.B) {
	dst := make([]int32, benchN)
	a := genI32(benchN, 4)
	c := genI32(benchN, 5)
	b.SetBytes(benchN * 4 * 3)
	for b.Loop() {
		Add(dst, a, c)
	}
}

func BenchmarkAddGo(b *testing.B) {
	dst := make([]int32, benchN)
	a := genI32(benchN, 4)
	c := genI32(benchN, 5)
	b.SetBytes(benchN * 4 * 3)
	for b.Loop() {
		addGo(dst, a, c)
	}
}

func BenchmarkSub(b *testing.B) {
	dst := make([]int32, benchN)
	a := genI32(benchN, 4)
	c := genI32(benchN, 5)
	b.SetBytes(benchN * 4 * 3)
	for b.Loop() {
		Sub(dst, a, c)
	}
}
