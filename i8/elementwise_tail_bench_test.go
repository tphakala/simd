package i8

import (
	"fmt"
	"testing"
)

// tailBenchLengths exercises the 16-wide and 8-wide AVX2 pre-blocks (#149) added
// to the element-wise kernels. Each length is past the blockSat32 (32) dispatch
// threshold so the SIMD path runs, and the residue n%32 sweeps the pre-block
// combinations: 32 is aligned (both blocks skipped, empty scalar tail); 40 is
// residue 8 (8-wide block only); 48 is residue 16 (16-wide block only); 63 is
// residue 31 (16+8 blocks plus a full 7-element scalar tail); 95 is residue 31
// with more full 32-blocks; 248 is residue 24 (16+8 blocks, empty scalar tail).
var tailBenchLengths = []int{32, 40, 48, 63, 95, 248}

func BenchmarkAbs_N(b *testing.B) {
	for _, n := range tailBenchLengths {
		a := genI8(n, 1)
		dst := make([]int8, n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				Abs(dst, a)
			}
		})
	}
}

func BenchmarkNeg_N(b *testing.B) {
	for _, n := range tailBenchLengths {
		a := genI8(n, 1)
		dst := make([]int8, n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				Neg(dst, a)
			}
		})
	}
}

func BenchmarkAddSaturate_N(b *testing.B) {
	for _, n := range tailBenchLengths {
		a, c := genI8(n, 1), genI8(n, 2)
		dst := make([]int8, n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				AddSaturate(dst, a, c)
			}
		})
	}
}

func BenchmarkSubSaturate_N(b *testing.B) {
	for _, n := range tailBenchLengths {
		a, c := genI8(n, 1), genI8(n, 2)
		dst := make([]int8, n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				SubSaturate(dst, a, c)
			}
		})
	}
}

func BenchmarkMin_N(b *testing.B) {
	for _, n := range tailBenchLengths {
		a, c := genI8(n, 1), genI8(n, 2)
		dst := make([]int8, n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				Min(dst, a, c)
			}
		})
	}
}

func BenchmarkMax_N(b *testing.B) {
	for _, n := range tailBenchLengths {
		a, c := genI8(n, 1), genI8(n, 2)
		dst := make([]int8, n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				Max(dst, a, c)
			}
		})
	}
}

func BenchmarkClamp_N(b *testing.B) {
	for _, n := range tailBenchLengths {
		src := genI8(n, 1)
		dst := make([]int8, n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				Clamp(dst, src, -64, 64)
			}
		})
	}
}

func BenchmarkAbsDiff_N(b *testing.B) {
	for _, n := range tailBenchLengths {
		a, c := genI8(n, 1), genI8(n, 2)
		dst := make([]int8, n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				AbsDiff(dst, a, c)
			}
		})
	}
}

func BenchmarkAddScalarSaturate_N(b *testing.B) {
	for _, n := range tailBenchLengths {
		a := genI8(n, 1)
		dst := make([]int8, n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				AddScalarSaturate(dst, a, 37)
			}
		})
	}
}

func BenchmarkSubScalarSaturate_N(b *testing.B) {
	for _, n := range tailBenchLengths {
		a := genI8(n, 1)
		dst := make([]int8, n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				SubScalarSaturate(dst, a, 37)
			}
		})
	}
}
