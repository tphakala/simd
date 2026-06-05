package f32

import "testing"

// Exercises the Go fallback for the subFromScalar composition directly, so the
// pure-Go path is covered on AVX/NEON-capable test hosts too.

func TestSubFromScalarGo(t *testing.T) {
	a := []float32{1, -2, 3.5}
	dst := make([]float32, len(a))
	subFromScalarGo(dst, a, 7.0)
	want := []float32{6, 9, 3.5}
	for i := range dst {
		if dst[i] != want[i] {
			t.Errorf("subFromScalarGo[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}
