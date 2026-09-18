//go:build amd64

package i8

import (
	"fmt"
	"testing"
)

// BenchmarkDotProduct4Kernels drives the AVX2 and AVX-VNNI 4-row kernels
// head-to-head (one full group) across the small compute-bound dims #304 targets,
// isolating the kernel cost from the batch dispatch. Only kernels the host
// supports run. Ragged dims (63, 127, 255) exercise the scalar tail too.
func BenchmarkDotProduct4Kernels(b *testing.B) {
	for _, dims := range []int{32, 40, 48, 56, 63, 64, 127, 128, 255, 256} {
		r0 := genI8(dims, 1)
		r1 := genI8(dims, 2)
		r2 := genI8(dims, 3)
		r3 := genI8(dims, 4)
		vec := genI8(dims, 5)
		res := make([]int32, 4)
		for _, k := range dotProduct4Kernels() {
			if !k.available {
				continue
			}
			b.Run(fmt.Sprintf("%s/dims%d", k.name, dims), func(b *testing.B) {
				b.SetBytes(int64(dims * 4))
				b.ResetTimer()
				for b.Loop() {
					k.fn(res, r0, r1, r2, r3, vec)
				}
			})
		}
	}
}
