//go:build arm64

package f32

import "testing"

// TestDotProduct4NEONKernel scores four deterministic rows against one query
// with the NEON batch-of-4 kernel and compares each result to the pure-scalar
// dotProductGo oracle, so a buggy hand-encoded kernel (wrong lane count, bad
// horizontal reduction, mishandled 4-element or scalar tail) is caught rather
// than passing by SIMD-vs-SIMD agreement.
func TestDotProduct4NEONKernel(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON required")
	}
	// Cover sub-vector, exact-multiple, and tail-bearing dims for the 8/iter
	// main loop, the 4-element remainder, and the scalar tail.
	dims := []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 256, 768}
	for _, dim := range dims {
		vec := deterministicF32Vector(11, dim)
		rows := [4][]float32{
			deterministicF32Vector(100, dim),
			deterministicF32Vector(101, dim),
			deterministicF32Vector(102, dim),
			deterministicF32Vector(103, dim),
		}
		results := make([]float32, 4)
		dotProduct4NEON(&results[0], &rows[0][0], &rows[1][0], &rows[2][0], &rows[3][0], &vec[0], dim)
		for i, row := range rows {
			want := dotProductGo(row, vec)
			if !closeFloat32(results[i], want) {
				t.Fatalf("dim=%d row=%d got=%g want=%g", dim, i, results[i], want)
			}
		}
	}
}
