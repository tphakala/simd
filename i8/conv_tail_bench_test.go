package i8

import (
	"fmt"
	"testing"
)

func BenchmarkToInt16_N(b *testing.B) {
	for _, n := range []int{16, 31, 240, 255} {
		src := genI8(n, 1)
		dst := make([]int16, n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				ToInt16(dst, src)
			}
		})
	}
}

func BenchmarkToInt32_N(b *testing.B) {
	for _, n := range []int{8, 15, 240, 255} {
		src := genI8(n, 1)
		dst := make([]int32, n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				ToInt32(dst, src)
			}
		})
	}
}
