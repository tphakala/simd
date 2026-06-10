package f64

import (
	"math"
	"testing"
)

// Tolerances for the dispatched log/pow paths.
//
// The SIMD log core is an atanh-form minimax polynomial (SLEEF xlog_u1
// coefficients, 7 terms): worst-case relative error is a few float64 ulps.
// logRelTol64 leaves two orders of magnitude of headroom while still failing
// hard on any real kernel bug (a wrong coefficient or a broken reduction is
// off by many orders of magnitude). The pure-Go fallback is math.Log, which
// passes trivially.
//
// Pow feeds p*Log(x) through the Exp core, whose degree-5 polynomial bounds
// the achievable relative error at ~3e-6 (see TestExpAccuracy); powRelTol64
// mirrors the Exp tests.
const (
	logRelTol64 = 1e-13
	powRelTol64 = 1e-5
)

// bitsEqF64 treats any NaN as equal to any NaN and otherwise requires bit
// equality, so -Inf, +Inf, and signed zeros are all compared exactly.
func bitsEqF64(a, b float64) bool {
	if a != a && b != b {
		return true
	}
	return math.Float64bits(a) == math.Float64bits(b)
}

// relErrF64 returns the relative error of got vs want, falling back to the
// absolute error when want is too small for a meaningful quotient.
func relErrF64(got, want float64) float64 {
	diff := math.Abs(got - want)
	if math.Abs(want) > 1e-12 {
		return diff / math.Abs(want)
	}
	return diff
}

// refLn64 is the accuracy reference for the natural log. math.Log cannot be
// used directly: its amd64 assembly implementation extracts the exponent
// without normalizing subnormal inputs, so it returns ~-709.09 for every
// subnormal x. Normalizing by 2^54 first gives the correct value on every
// architecture.
func refLn64(x float64) float64 {
	if x > 0 && x < dblMinNormal64 {
		return math.Log(x*0x1p54) - 54*math.Ln2
	}
	return math.Log(x)
}

// refLog2_64 scales refLn64 instead of using math.Log2: the Frexp-based
// math.Log2 carries an absolute error of ~ulp(1) near x = 1 (the
// log2(frac)+exp formulation cancels), which shows up as ~4e-8 relative
// error against a relative-accurate kernel.
func refLog2_64(x float64) float64 { return refLn64(x) * (1 / math.Ln2) }

func refLog10_64(x float64) float64 { return refLn64(x) * (1 / math.Ln10) }

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

// TestLogEdgeCases pins the documented non-finite behavior (matching math.Log)
// on the dispatched path. The slice is long enough to hit the SIMD body, so
// the kernel's special-lane blends are exercised alongside the scalar tail.
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

// TestLogSubnormals checks that subnormal inputs go through the kernel's
// normalization pre-scale (or the scalar fallback) and still come out within
// tolerance of math.Log.
func TestLogSubnormals(t *testing.T) {
	src := []float64{
		5e-324,         // smallest subnormal
		1e-310, 1e-320, // assorted subnormals
		dblMinNormal64,         // DBL_MIN, smallest normal
		2.225073858507201e-308, // largest subnormal
		4e-308, 1e-300, 1,
	}
	for _, fn := range []struct {
		name string
		log  func(dst, src []float64)
		ref  func(float64) float64
	}{
		{"Log", Log, refLn64},
		{"Log2", Log2, refLog2_64},
		{"Log10", Log10, refLog10_64},
	} {
		dst := make([]float64, len(src))
		fn.log(dst, src)
		for i, x := range src {
			want := fn.ref(x)
			if re := relErrF64(dst[i], want); re > logRelTol64 {
				t.Errorf("%s(%g) = %v, want %v (relErr %g)", fn.name, x, dst[i], want, re)
			}
		}
	}
}

// TestLogAccuracy sweeps the full finite-positive range, the near-1
// neighborhood where the reduction switches sign, and the mantissa-reduction
// boundaries around sqrt(2)/2 and sqrt(2).
func TestLogAccuracy(t *testing.T) {
	src := make([]float64, 0, 4264)
	// Logarithmic sweep across the normal range.
	for e := -1022; e <= 1023; e += 3 {
		for _, m := range []float64{1.0, 1.3, 1.4142135623730950, 1.4142135623730952, 1.7, 1.9999999999999998} {
			src = append(src, math.Ldexp(m, e))
		}
	}
	// Dense near 1 (both sides), where log -> 0.
	for i := -40; i <= 40; i++ {
		src = append(src, 1+float64(i)*1e-3, 1+float64(i)*1e-9)
	}
	src = append(src, math.MaxFloat64, dblMinNormal64, math.Sqrt2, math.Sqrt2/2)

	for _, fn := range []struct {
		name string
		log  func(dst, src []float64)
		ref  func(float64) float64
	}{
		{"Log", Log, refLn64},
		{"Log2", Log2, refLog2_64},
		{"Log10", Log10, refLog10_64},
	} {
		dst := make([]float64, len(src))
		fn.log(dst, src)
		for i, x := range src {
			want := fn.ref(x)
			if re := relErrF64(dst[i], want); re > logRelTol64 {
				t.Errorf("%s(%g) = %v, want %v (relErr %g)", fn.name, x, dst[i], want, re)
			}
		}
	}
}

// TestLogLengths runs Log over every length from 1 to 33 so the scalar
// remainder loop is exercised alongside the SIMD body (the AVX body consumes
// 4 elements at a time, the NEON body 2).
func TestLogLengths(t *testing.T) {
	for n := 1; n <= 33; n++ {
		src := make([]float64, n)
		dst := make([]float64, n)
		for i := range src {
			src[i] = math.Ldexp(1+0.13*float64(i), 2*i-n)
		}
		Log(dst, src)
		for i, x := range src {
			want := refLn64(x)
			if re := relErrF64(dst[i], want); re > logRelTol64 {
				t.Errorf("len=%d Log(%v)[%d] = %v, want %v (relErr %g)", n, x, i, dst[i], want, re)
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
		if d := math.Abs(dst[0]-tt.want) / (1 + math.Abs(tt.want)); d > powRelTol64 {
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

// TestPowEdgeLanesSIMDLength repeats the math.Pow edge cases at a length that
// engages the SIMD body, so the dispatch-level scalar fallback for slices
// containing non-positive or non-finite bases is exercised.
func TestPowEdgeLanesSIMDLength(t *testing.T) {
	src := []float64{2, -2, 0, math.Inf(1), math.NaN(), 4, 0.5, 9, 16, 25}
	for _, p := range []float64{0.5, 2, -1, 0, math.NaN(), math.Inf(1)} {
		dst := make([]float64, len(src))
		Pow(dst, src, p)
		for i, x := range src {
			want := math.Pow(x, p)
			if math.IsNaN(want) || math.IsInf(want, 0) || want == 0 || x <= 0 {
				if !bitsEqF64(dst[i], want) {
					t.Errorf("Pow(%v, %v) = %v, want %v (exact class)", x, p, dst[i], want)
				}
				continue
			}
			if re := relErrF64(dst[i], want); re > powRelTol64 {
				t.Errorf("Pow(%v, %v) = %v, want %v (relErr %g)", x, p, dst[i], want, re)
			}
		}
	}

	// PowElem with non-finite exponent lanes falls back per the same rule.
	base := []float64{2, 3, 4, 5, 6, 7, 8, 9}
	expv := []float64{1, 2, math.NaN(), math.Inf(1), math.Inf(-1), 0.5, 0, -2}
	dst := make([]float64, len(base))
	PowElem(dst, base, expv)
	for i := range base {
		want := math.Pow(base[i], expv[i])
		if math.IsNaN(want) || math.IsInf(want, 0) || want == 0 {
			if !bitsEqF64(dst[i], want) {
				t.Errorf("PowElem(%v, %v) = %v, want %v (exact class)", base[i], expv[i], dst[i], want)
			}
			continue
		}
		if re := relErrF64(dst[i], want); re > powRelTol64 {
			t.Errorf("PowElem(%v, %v) = %v, want %v (relErr %g)", base[i], expv[i], dst[i], want, re)
		}
	}
}

// TestPowOverflowClasses pins the overflow/underflow behavior of the SIMD
// path: for positive finite bases, results that overflow in math.Pow must
// come out as +Inf and results that underflow must come out as 0, matching
// the scalar reference classes exactly.
func TestPowOverflowClasses(t *testing.T) {
	src := []float64{1e300, 1e-300, 2, 0.5, 1e10, 1e-10, 3, 7}
	for _, p := range []float64{3, -3, 100, -100} {
		dst := make([]float64, len(src))
		Pow(dst, src, p)
		for i, x := range src {
			want := math.Pow(x, p)
			switch {
			case math.IsInf(want, 1):
				if !math.IsInf(dst[i], 1) {
					t.Errorf("Pow(%v, %v) = %v, want +Inf", x, p, dst[i])
				}
			case want == 0:
				if dst[i] != 0 {
					t.Errorf("Pow(%v, %v) = %v, want 0", x, p, dst[i])
				}
			default:
				if re := relErrF64(dst[i], want); re > powRelTol64 {
					t.Errorf("Pow(%v, %v) = %v, want %v (relErr %g)", x, p, dst[i], want, re)
				}
			}
		}
	}
}

// TestLogPowParity checks the dispatched ops against the math.* reference over
// a wide finite-positive range, within the documented tolerances. On hosts
// without the SIMD kernels this compares the scalar path against itself.
func TestLogPowParity(t *testing.T) {
	const n = 257
	src := make([]float64, n)
	for i := range src {
		src[i] = math.Ldexp(1+0.5*float64(i%7), (i%200)-100)
	}

	check := func(name string, got []float64, ref func(float64) float64, tol float64) {
		for i := range src {
			want := ref(src[i])
			if re := relErrF64(got[i], want); re > tol {
				t.Errorf("%s[%d](%v) = %v, want %v (relErr %g)", name, i, src[i], got[i], want, re)
			}
		}
	}
	dst := make([]float64, n)
	Log(dst, src)
	check("Log", dst, refLn64, logRelTol64)
	Log2(dst, src)
	check("Log2", dst, refLog2_64, logRelTol64)
	Log10(dst, src)
	check("Log10", dst, refLog10_64, logRelTol64)

	for _, exp := range []float64{0.35, 0.5, 2, -1, 3.7} {
		Pow(dst, src, exp)
		for i := range src {
			want := math.Pow(src[i], exp)
			if re := relErrF64(dst[i], want); re > powRelTol64 {
				t.Errorf("Pow[%d](%v, %v) = %v, want %v (relErr %g)", i, src[i], exp, dst[i], want, re)
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
		if re := relErrF64(dst[i], want); re > powRelTol64 {
			t.Errorf("PowElem[%d] = %v, want %v (relErr %g)", i, dst[i], want, re)
		}
	}
}

// TestLogPowDispatchParity pins the per-arch dispatch wrappers against the
// pure-Go reference. On SIMD hosts this is the kernel-vs-reference gate; on
// others it compares the reference with itself.
func TestLogPowDispatchParity(t *testing.T) {
	const n = 67 // SIMD body + scalar tail
	src := make([]float64, n)
	for i := range src {
		src[i] = math.Ldexp(1+0.37*float64(i%5), (i%60)-30)
	}
	got := make([]float64, n)
	want := make([]float64, n)

	log64(got, src)
	logGo(want, src)
	for i := range got {
		if re := relErrF64(got[i], want[i]); re > logRelTol64 {
			t.Errorf("log64[%d](%v) = %v, want %v (relErr %g)", i, src[i], got[i], want[i], re)
		}
	}

	log2_64(got, src)
	log2Go(want, src)
	for i := range got {
		if re := relErrF64(got[i], want[i]); re > logRelTol64 {
			t.Errorf("log2_64[%d](%v) = %v, want %v (relErr %g)", i, src[i], got[i], want[i], re)
		}
	}

	log10_64(got, src)
	log10Go(want, src)
	for i := range got {
		if re := relErrF64(got[i], want[i]); re > logRelTol64 {
			t.Errorf("log10_64[%d](%v) = %v, want %v (relErr %g)", i, src[i], got[i], want[i], re)
		}
	}

	pow64(got, src, 0.35)
	powGo(want, src, 0.35)
	for i := range got {
		if re := relErrF64(got[i], want[i]); re > powRelTol64 {
			t.Errorf("pow64[%d](%v, 0.35) = %v, want %v (relErr %g)", i, src[i], got[i], want[i], re)
		}
	}

	expv := make([]float64, n)
	for i := range expv {
		expv[i] = -1.5 + 0.21*float64(i%17)
	}
	powElem64(got, src, expv)
	powElemGo(want, src, expv)
	for i := range got {
		if re := relErrF64(got[i], want[i]); re > powRelTol64 {
			t.Errorf("powElem64[%d](%v, %v) = %v, want %v (relErr %g)", i, src[i], expv[i], got[i], want[i], re)
		}
	}
}

func TestLogPowInPlace(t *testing.T) {
	src := []float64{1, 2, 4, 8, 16, 32, 0.5, 100}
	a := append([]float64(nil), src...)
	LogInPlace(a)
	for i := range src {
		if re := relErrF64(a[i], math.Log(src[i])); re > logRelTol64 {
			t.Errorf("LogInPlace[%d] = %v (relErr %g)", i, a[i], re)
		}
	}
	b := append([]float64(nil), src...)
	PowInPlace(b, 0.5)
	for i := range src {
		if re := relErrF64(b[i], math.Pow(src[i], 0.5)); re > powRelTol64 {
			t.Errorf("PowInPlace[%d] = %v (relErr %g)", i, b[i], re)
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
		if re := relErrF64(short[i], want); re > powRelTol64 {
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

func BenchmarkLog(b *testing.B) {
	src := make([]float64, 1024)
	for i := range src {
		src[i] = 1e-3 + float64(i)*0.7
	}
	dst := make([]float64, len(src))
	b.ReportAllocs()
	for b.Loop() {
		Log(dst, src)
	}
}

func BenchmarkPow(b *testing.B) {
	src := make([]float64, 1024)
	for i := range src {
		src[i] = 1e-3 + float64(i)*0.7
	}
	dst := make([]float64, len(src))
	b.ReportAllocs()
	for b.Loop() {
		Pow(dst, src, 0.35)
	}
}
