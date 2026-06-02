//go:build amd64

package f64

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// dotProduct4Kernel is the shared signature of the batch-of-4 query-load-reuse
// kernels: four rows scored against one resident query vector.
type dotProduct4Kernel func(results, row0, row1, row2, row3, vec *float64, n int)

// checkDotProduct4Kernel scores four deterministic rows against one query with
// the given kernel and compares each result to the pure-scalar dotProductGo
// oracle, so a buggy hand-written kernel (wrong lane count, bad reduction,
// mishandled tail) is caught rather than passing by SIMD-vs-SIMD agreement.
func checkDotProduct4Kernel(t *testing.T, kernel dotProduct4Kernel) {
	t.Helper()
	// Cover sub-vector, exact-multiple, and tail-bearing dims for both the
	// AVX (4-lane, 16/iter) and AVX-512 (8-lane, 64/iter) main loops.
	dims := []int{1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 256, 768}
	for _, dim := range dims {
		vec := deterministicF64Vector(11, dim)
		rows := [4][]float64{
			deterministicF64Vector(100, dim),
			deterministicF64Vector(101, dim),
			deterministicF64Vector(102, dim),
			deterministicF64Vector(103, dim),
		}
		results := make([]float64, 4)
		kernel(&results[0], &rows[0][0], &rows[1][0], &rows[2][0], &rows[3][0], &vec[0], dim)
		for i, row := range rows {
			want := dotProductGo(row, vec)
			if !almostEqual(results[i], want, 1e-9*(1+math.Abs(want))) {
				t.Fatalf("dim=%d row=%d got=%g want=%g", dim, i, results[i], want)
			}
		}
	}
}

func TestDotProduct4AVXKernel(t *testing.T) {
	if !cpu.X86.AVX || !cpu.X86.FMA {
		t.Skip("AVX+FMA required")
	}
	checkDotProduct4Kernel(t, dotProduct4AVX)
}

func TestDotProduct4AVX512Kernel(t *testing.T) {
	if !cpu.X86.AVX512F || !cpu.X86.AVX512VL {
		t.Skip("AVX-512F+VL required")
	}
	checkDotProduct4Kernel(t, dotProduct4AVX512)
}
