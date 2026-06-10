package f64

import (
	"math"
	"testing"
)

// bitsEqF64 treats any NaN as equal to any NaN and otherwise requires bit
// equality, so -Inf, +Inf, and signed zeros are all compared exactly.
func bitsEqF64(a, b float64) bool {
	if a != a && b != b {
		return true
	}
	return math.Float64bits(a) == math.Float64bits(b)
}

func TestLogKnownValues(t *testing.T) {
	tests := []struct {
		name string
		fn   func(dst, src []float64)
		in   float64
		want float64
	}{
		{"Log(1)=0", Log, 1, 0},
		{"Log(e)=1", Log, math.E, 1},
		{"Log2(8)=3", Log2, 8, 3},
		{"Log2(1)=0", Log2, 1, 0},
		{"Log10(100)=2", Log10, 100, 2},
		{"Log10(1)=0", Log10, 1, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, 1)
			tt.fn(dst, []float64{tt.in})
			if d := math.Abs(dst[0] - tt.want); d > 1e-12 {
				t.Errorf("%s: got %v want %v", tt.name, dst[0], tt.want)
			}
		})
	}
}

// TestLogEdgeCases pins the documented non-finite behavior (matching math.Log).
func TestLogEdgeCases(t *testing.T) {
	src := []float64{0, -1, math.Copysign(0, -1), math.Inf(1), math.Inf(-1), math.NaN()}
	for _, fn := range []struct {
		name string
		log  func(dst, src []float64)
		ref  func(float64) float64
	}{
		{"Log", Log, math.Log},
		{"Log2", Log2, math.Log2},
		{"Log10", Log10, math.Log10},
	} {
		dst := make([]float64, len(src))
		fn.log(dst, src)
		for i, x := range src {
			want := fn.ref(x)
			if !bitsEqF64(dst[i], want) {
				t.Errorf("%s(%v) = %v, want %v", fn.name, x, dst[i], want)
			}
		}
	}
}

func TestPowKnownAndEdge(t *testing.T) {
	tests := []struct {
		base, exp, want float64
	}{
		{2, 10, 1024},
		{9, 0.5, 3},
		{4, -1, 0.25},
		{2, 0, 1},
		{0, 0, 1},
		{0, 2, 0},
		{2, 1, 2},
		{10, 3, 1000},
	}
	for _, tt := range tests {
		dst := make([]float64, 1)
		Pow(dst, []float64{tt.base}, tt.exp)
		if d := math.Abs(dst[0]-tt.want) / (1 + math.Abs(tt.want)); d > 1e-12 {
			t.Errorf("Pow(%v, %v) = %v, want %v", tt.base, tt.exp, dst[0], tt.want)
		}
	}

	edges := []struct{ base, exp float64 }{
		{-2, 0.5},
		{0, -1},
		{math.Inf(1), 2},
		{math.NaN(), 1},
		{2, math.Inf(1)},
	}
	for _, e := range edges {
		dst := make([]float64, 1)
		Pow(dst, []float64{e.base}, e.exp)
		want := math.Pow(e.base, e.exp)
		if !bitsEqF64(dst[0], want) {
			t.Errorf("Pow(%v, %v) = %v, want %v", e.base, e.exp, dst[0], want)
		}
	}
}

// TestLogPowParity checks the dispatched ops against the math.* reference over a
// wide finite-positive range. Exact today (the implementation is the scalar
// reference); becomes the parity gate once a SIMD kernel lands.
func TestLogPowParity(t *testing.T) {
	const n = 257
	src := make([]float64, n)
	for i := range src {
		src[i] = math.Ldexp(1+0.5*float64(i%7), (i%200)-100)
	}

	check := func(name string, got []float64, ref func(float64) float64) {
		for i := range src {
			want := ref(src[i])
			if !bitsEqF64(got[i], want) {
				t.Errorf("%s[%d](%v) = %v, want %v", name, i, src[i], got[i], want)
			}
		}
	}
	dst := make([]float64, n)
	Log(dst, src)
	check("Log", dst, math.Log)
	Log2(dst, src)
	check("Log2", dst, math.Log2)
	Log10(dst, src)
	check("Log10", dst, math.Log10)

	for _, exp := range []float64{0.35, 0.5, 2, -1, 3.7} {
		Pow(dst, src, exp)
		for i := range src {
			want := math.Pow(src[i], exp)
			if !bitsEqF64(dst[i], want) {
				t.Errorf("Pow[%d](%v, %v) = %v, want %v", i, src[i], exp, dst[i], want)
			}
		}
	}

	expv := make([]float64, n)
	for i := range expv {
		expv[i] = 0.1 + 0.05*float64(i%13)
	}
	PowElem(dst, src, expv)
	for i := range src {
		want := math.Pow(src[i], expv[i])
		if !bitsEqF64(dst[i], want) {
			t.Errorf("PowElem[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestLogPowInPlace(t *testing.T) {
	src := []float64{1, 2, 4, 8, 16, 32, 0.5, 100}
	a := append([]float64(nil), src...)
	LogInPlace(a)
	for i := range src {
		if !bitsEqF64(a[i], math.Log(src[i])) {
			t.Errorf("LogInPlace[%d] = %v", i, a[i])
		}
	}
	b := append([]float64(nil), src...)
	PowInPlace(b, 0.5)
	for i := range src {
		if !bitsEqF64(b[i], math.Pow(src[i], 0.5)) {
			t.Errorf("PowInPlace[%d] = %v", i, b[i])
		}
	}
}

func TestLogPowClampsLength(t *testing.T) {
	dst := make([]float64, 3)
	Log(dst, []float64{1, math.E, 10, 100, 1000})
	for i, want := range []float64{0, 1, math.Log(10)} {
		if d := math.Abs(dst[i] - want); d > 1e-12 {
			t.Errorf("Log clamp[%d] = %v want %v", i, dst[i], want)
		}
	}
	short := make([]float64, 2)
	Pow(short, []float64{2, 3, 4}, 2)
	for i, want := range []float64{4, 9} {
		if short[i] != want {
			t.Errorf("Pow clamp[%d] = %v want %v", i, short[i], want)
		}
	}
}

func TestLogPowAllocFree(t *testing.T) {
	src := make([]float64, 128)
	for i := range src {
		src[i] = float64(i) + 1
	}
	dst := make([]float64, 128)
	exp := make([]float64, 128)
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
