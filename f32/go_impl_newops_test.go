package f32

import (
	"math"
	"testing"
)

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

// TestRound32Go exercises the Go fallback for Round directly, so the pure-Go
// path (and its BCE-hint bounds line) is covered on AVX/NEON-capable test hosts
// too, where Round dispatches to the assembly kernel instead.
func TestRound32Go(t *testing.T) {
	src := []float32{-2.5, -1.5, -0.5, 0.4, 0.5, 1.5, 2.5}
	dst := make([]float32, len(src))
	round32Go(dst, src)
	for i, v := range src {
		want := float32(math.Round(float64(v)))
		if dst[i] != want {
			t.Errorf("round32Go[%d](%v) = %v, want %v", i, v, dst[i], want)
		}
	}
}

// TestRound32Go_EmptyDst hits the `if len(dst) == 0 { return }` guard with a
// non-empty src argument.
func TestRound32Go_EmptyDst(_ *testing.T) {
	round32Go(nil, []float32{1, 2})
}
