package f32

import (
	"math"
	"testing"
)

// scalarMaxAbs is the independent reference: max(|a[i]|), 0 for empty.
func scalarMaxAbs(a []float32) float32 {
	var m float32
	for _, v := range a {
		if v < 0 {
			v = -v
		}
		if v > m {
			m = v
		}
	}
	return m
}

func TestMaxAbs(t *testing.T) {
	cases := [][]float32{
		nil,
		{},
		{0},
		{float32(math.Copysign(0, -1))},
		{3, -7, 2},
		{-1, -2, -3, -4, -5, -6, -7, -8, -9},
		{1.5, -1.5, 0.25, -0.25, 100, -99.9, 0, 42},
		{float32(math.Inf(1)), -3},
		{float32(math.Inf(-1)), 3},
	}
	for i, a := range cases {
		got := MaxAbs(a)
		want := scalarMaxAbs(a)
		if got != want {
			t.Errorf("case %d: MaxAbs(%v) = %v, want %v", i, a, got, want)
		}
	}
}

func TestMaxAbsParity(t *testing.T) {
	// Lengths spanning the SIMD body, the scalar tail, and the small fallback.
	for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 127, 256} {
		a := make([]float32, n)
		for i := range a {
			a[i] = float32(math.Sin(float64(i)*0.7)) * float32((i%5)-2) * 13.3
		}
		if got, want := MaxAbs(a), scalarMaxAbs(a); got != want {
			t.Errorf("n=%d: MaxAbs=%v want %v", n, got, want)
		}
	}
}

func TestMaxAbsNoAlloc(t *testing.T) {
	a := make([]float32, 1024)
	for i := range a {
		a[i] = float32(i%7) - 3
	}
	if n := testing.AllocsPerRun(100, func() { _ = MaxAbs(a) }); n != 0 {
		t.Errorf("MaxAbs allocated %v times, want 0", n)
	}
}
