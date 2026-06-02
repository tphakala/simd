//go:build amd64

package f32

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// kernelBatchModes returns the SIMD batch-of-4 modes that the current CPU can
// actually execute. dotProductBatchKernel(false, ...) runs the AVX+FMA kernel
// and dotProductBatchKernel(true, ...) the AVX-512 one, so calling either on a
// CPU without the matching ISA would fault. Tests use this to exercise every
// kernel the host supports, including the AVX+FMA path that issue #64 adds.
func kernelBatchModes() []bool {
	var modes []bool
	if cpu.X86.AVX && cpu.X86.FMA {
		modes = append(modes, false)
	}
	if cpu.X86.AVX512F && cpu.X86.AVX512VL {
		modes = append(modes, true)
	}
	return modes
}

func TestDotProductBatchKernelMatchesScalar(t *testing.T) {
	modes := kernelBatchModes()
	if len(modes) == 0 {
		t.Skip("no AVX2/FMA or AVX-512 batch kernel available on this CPU")
	}
	// dims span the per-vector body plus assorted tails; rowCounts cross the
	// batch-of-4 boundary so both the fused kernel and the scalar remainder run.
	dims := []int{8, 15, 16, 31, 32, 64, 127, 128, 255, 256, 768}
	rowCounts := []int{4, 5, 7, 8, 16, 17}
	for _, useAVX512 := range modes {
		for _, dim := range dims {
			for _, rowCount := range rowCounts {
				vec := deterministicF32Vector(11, dim)
				rows := make([][]float32, rowCount)
				for i := range rows {
					rows[i] = deterministicF32Vector(100+i, dim)
				}
				got := make([]float32, rowCount)
				dotProductBatchKernel(useAVX512, got, rows, vec, dim)
				for i := range rows {
					want := dotProductGo(rows[i], vec)
					if !closeFloat32(got[i], want) {
						t.Fatalf("avx512=%t dim=%d rows=%d row=%d got=%g want=%g",
							useAVX512, dim, rowCount, i, got[i], want)
					}
				}
			}
		}
	}
}

func TestDotProductBatchKernelShortRowsFallBack(t *testing.T) {
	modes := kernelBatchModes()
	if len(modes) == 0 {
		t.Skip("no AVX2/FMA or AVX-512 batch kernel available on this CPU")
	}
	const vecLen = 64
	vec := deterministicF32Vector(7, vecLen)
	// A batch-of-4 group with rows shorter than vecLen must take the per-row
	// fallback inside the group, never the fused full-row kernel. Length 0 rows
	// must score 0. The trailing group exercises the post-loop remainder.
	rowLens := []int{vecLen, 0, vecLen - 1, 3, vecLen, vecLen, vecLen + 8, 1}
	for _, useAVX512 := range modes {
		rows := make([][]float32, len(rowLens))
		for i, l := range rowLens {
			rows[i] = deterministicF32Vector(200+i, l)
		}
		got := make([]float32, len(rows))
		dotProductBatchKernel(useAVX512, got, rows, vec, vecLen)
		for i, row := range rows {
			n := min(len(row), vecLen)
			var want float32
			if n > 0 {
				want = dotProductGo(row[:n], vec[:n])
			}
			if !closeFloat32(got[i], want) {
				t.Fatalf("avx512=%t row=%d len=%d got=%g want=%g",
					useAVX512, i, len(row), got[i], want)
			}
		}
	}
}
