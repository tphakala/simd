//go:build arm64

package f64

import "testing"

// TestDotProduct4NEONKernel scores four deterministic rows against one query with
// the NEON batch-of-4 kernel and compares each result to the pure-scalar
// dotProductGo oracle, so a buggy hand-encoded kernel (wrong lane count, bad
// horizontal reduction, mishandled 2-element or scalar tail) is caught rather
// than passing by SIMD-vs-SIMD agreement.
func TestDotProduct4NEONKernel(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON required")
	}
	// Cover sub-vector, exact-multiple, and tail-bearing dims for the 4/iter main
	// loop, the 2-element remainder, and the scalar (odd) tail.
	dims := []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 256, 768}
	for _, dim := range dims {
		vec := deterministicF64Vector(11, dim)
		rows := [4][]float64{
			deterministicF64Vector(100, dim),
			deterministicF64Vector(101, dim),
			deterministicF64Vector(102, dim),
			deterministicF64Vector(103, dim),
		}
		results := make([]float64, 4)
		dotProduct4NEON(&results[0], &rows[0][0], &rows[1][0], &rows[2][0], &rows[3][0], &vec[0], dim)
		for i, row := range rows {
			want := dotProductGo(row, vec)
			if !closeFloat64(results[i], want) {
				t.Fatalf("dim=%d row=%d got=%g want=%g", dim, i, results[i], want)
			}
		}
	}

	// The kernel must not allocate.
	const dim = 256
	vec := deterministicF64Vector(11, dim)
	rows := [4][]float64{
		deterministicF64Vector(100, dim),
		deterministicF64Vector(101, dim),
		deterministicF64Vector(102, dim),
		deterministicF64Vector(103, dim),
	}
	results := make([]float64, 4)
	if a := testing.AllocsPerRun(100, func() {
		dotProduct4NEON(&results[0], &rows[0][0], &rows[1][0], &rows[2][0], &rows[3][0], &vec[0], dim)
	}); a != 0 {
		t.Errorf("dotProduct4NEON allocated %v times per run, want 0", a)
	}
}
