package f32

import (
	"math"
	"testing"
)

// bitsEqF32 treats any NaN as equal to any NaN and otherwise requires bit
// equality, so -Inf, +Inf, and signed zeros are all compared exactly.
func bitsEqF32(a, b float32) bool {
	if a != a && b != b {
		return true
	}
	return math.Float32bits(a) == math.Float32bits(b)
}

func TestLogKnownValues(t *testing.T) {
	tests := []struct {
		name string
		fn   func(dst, src []float32)
		in   float32
		want float32
	}{
		{"Log(1)=0", Log, 1, 0},
		{"Log(e)=1", Log, float32(math.E), 1},
		{"Log2(8)=3", Log2, 8, 3},
		{"Log2(1)=0", Log2, 1, 0},
		{"Log10(100)=2", Log10, 100, 2},
		{"Log10(1)=0", Log10, 1, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, 1)
			tt.fn(dst, []float32{tt.in})
			if d := math.Abs(float64(dst[0] - tt.want)); d > 1e-6 {
				t.Errorf("%s: got %v want %v", tt.name, dst[0], tt.want)
			}
		})
	}
}

// TestLogEdgeCases pins the documented non-finite behavior (matching math.Log).
func TestLogEdgeCases(t *testing.T) {
	src := []float32{0, -1, float32(math.Copysign(0, -1)), float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}
	for _, fn := range []struct {
		name string
		log  func(dst, src []float32)
		ref  func(float64) float64
	}{
		{"Log", Log, math.Log},
		{"Log2", Log2, math.Log2},
		{"Log10", Log10, math.Log10},
	} {
		dst := make([]float32, len(src))
		fn.log(dst, src)
		for i, x := range src {
			want := float32(fn.ref(float64(x)))
			if !bitsEqF32(dst[i], want) {
				t.Errorf("%s(%v) = %v, want %v", fn.name, x, dst[i], want)
			}
		}
	}
}

func TestPowKnownAndEdge(t *testing.T) {
	tests := []struct {
		base, exp, want float32
	}{
		{2, 10, 1024},
		{9, 0.5, 3},
		{4, -1, 0.25},
		{2, 0, 1},     // anything^0 = 1
		{0, 0, 1},     // 0^0 = 1 (matches math.Pow)
		{0, 2, 0},     // 0^positive = 0
		{2, 1, 2},     // x^1 = x
		{10, 3, 1000}, // exact integer power
	}
	for _, tt := range tests {
		dst := make([]float32, 1)
		Pow(dst, []float32{tt.base}, tt.exp)
		if d := math.Abs(float64(dst[0]-tt.want)) / (1 + math.Abs(float64(tt.want))); d > 1e-6 {
			t.Errorf("Pow(%v, %v) = %v, want %v", tt.base, tt.exp, dst[0], tt.want)
		}
	}

	// Non-finite / domain-error edges must match math.Pow exactly.
	edges := []struct{ base, exp float32 }{
		{-2, 0.5},                 // negative base, non-integer exp -> NaN
		{0, -1},                   // 0^negative -> +Inf
		{float32(math.Inf(1)), 2}, // +Inf
		{float32(math.NaN()), 1},  // NaN propagation
		{2, float32(math.Inf(1))}, // 2^+Inf -> +Inf
	}
	for _, e := range edges {
		dst := make([]float32, 1)
		Pow(dst, []float32{e.base}, e.exp)
		want := float32(math.Pow(float64(e.base), float64(e.exp)))
		if !bitsEqF32(dst[0], want) {
			t.Errorf("Pow(%v, %v) = %v, want %v", e.base, e.exp, dst[0], want)
		}
	}
}

// TestLogPowParity checks the dispatched ops against the math.* reference over a
// wide finite-positive range. It is exact today (the implementation is the
// scalar reference) and becomes the parity gate once a SIMD kernel lands.
func TestLogPowParity(t *testing.T) {
	const n = 257 // odd, exercises any future vector tail
	src := make([]float32, n)
	for i := range src {
		// Logarithmically spread positives across denormal-to-large magnitudes.
		src[i] = float32(math.Ldexp(1+0.5*float64(i%7), (i%80)-40))
	}

	check := func(name string, got []float32, ref func(float64) float64) {
		for i := range src {
			want := float32(ref(float64(src[i])))
			if !bitsEqF32(got[i], want) {
				t.Errorf("%s[%d](%v) = %v, want %v", name, i, src[i], got[i], want)
			}
		}
	}
	dst := make([]float32, n)
	Log(dst, src)
	check("Log", dst, math.Log)
	Log2(dst, src)
	check("Log2", dst, math.Log2)
	Log10(dst, src)
	check("Log10", dst, math.Log10)

	for _, exp := range []float32{0.35, 0.5, 2, -1, 3.7} {
		Pow(dst, src, exp)
		for i := range src {
			want := float32(math.Pow(float64(src[i]), float64(exp)))
			if !bitsEqF32(dst[i], want) {
				t.Errorf("Pow[%d](%v, %v) = %v, want %v", i, src[i], exp, dst[i], want)
			}
		}
	}

	// PowElem against a per-element exponent.
	expv := make([]float32, n)
	for i := range expv {
		expv[i] = float32(0.1 + 0.05*float64(i%13))
	}
	PowElem(dst, src, expv)
	for i := range src {
		want := float32(math.Pow(float64(src[i]), float64(expv[i])))
		if !bitsEqF32(dst[i], want) {
			t.Errorf("PowElem[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestLogPowInPlace(t *testing.T) {
	src := []float32{1, 2, 4, 8, 16, 32, 0.5, 100}
	a := append([]float32(nil), src...)
	LogInPlace(a)
	for i := range src {
		if !bitsEqF32(a[i], float32(math.Log(float64(src[i])))) {
			t.Errorf("LogInPlace[%d] = %v", i, a[i])
		}
	}
	b := append([]float32(nil), src...)
	PowInPlace(b, 2)
	for i := range src {
		if !bitsEqF32(b[i], float32(math.Pow(float64(src[i]), 2))) {
			t.Errorf("PowInPlace[%d] = %v", i, b[i])
		}
	}
}

// TestLogPowClampsLength verifies the public ops honor min(len(dst), len(src)).
func TestLogPowClampsLength(t *testing.T) {
	dst := make([]float32, 3)
	Log(dst, []float32{1, math.E, 10, 100, 1000}) // src longer than dst: only 3 written
	for i, want := range []float32{0, 1, float32(math.Log(10))} {
		if d := math.Abs(float64(dst[i] - want)); d > 1e-6 {
			t.Errorf("Log clamp[%d] = %v want %v", i, dst[i], want)
		}
	}
	// dst shorter than needed: only len(dst) written, no panic.
	short := make([]float32, 2)
	Pow(short, []float32{2, 3, 4}, 2)
	for i, want := range []float32{4, 9} {
		if short[i] != want {
			t.Errorf("Pow clamp[%d] = %v want %v", i, short[i], want)
		}
	}
}

func TestLogPowAllocFree(t *testing.T) {
	src := make([]float32, 128)
	for i := range src {
		src[i] = float32(i) + 1
	}
	dst := make([]float32, 128)
	exp := make([]float32, 128)
	for i := range exp {
		exp[i] = 0.5
	}
	cases := []struct {
		name string
		fn   func()
	}{
		{"Log", func() { Log(dst, src) }},
		{"Log2", func() { Log2(dst, src) }},
		{"Log10", func() { Log10(dst, src) }},
		{"Pow", func() { Pow(dst, src, 0.35) }},
		{"PowElem", func() { PowElem(dst, src, exp) }},
		{"LogInPlace", func() { LogInPlace(dst) }},
		{"PowInPlace", func() { PowInPlace(dst, 2) }},
	}
	for _, c := range cases {
		if a := testing.AllocsPerRun(5, c.fn); a != 0 {
			t.Errorf("%s allocated %v times per run, want 0", c.name, a)
		}
	}
}
