package f32

import (
	"math"
	"testing"
)

// Tolerances for the dispatched log/pow paths.
//
// The SIMD log core is the Cephes logf degree-8 polynomial in z = m-1 over
// [sqrt(2)/2, sqrt(2)): measured worst-case relative error is ~1.4e-7
// (about 1.2 float32 ulps) across the full range including subnormals.
// logRelTol32 leaves ~3x headroom while still failing hard on any real
// kernel bug. The pure-Go fallback computes through float64 math and passes
// trivially.
//
// Pow feeds p*Log(x) through the Exp core: the dominant error is the log's
// relative error amplified by |p*ln(x)| <= 88, ~1.4e-5 measured; powRelTol32
// mirrors the Exp tests' 1e-4.
const (
	logRelTol32 = 5e-7
	powRelTol32 = 1e-4

	// fltMinNormal32 is FLT_MIN, the smallest normal float32.
	fltMinNormal32 = 1.1754943508222875e-38
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
		{"Log(e)=1", Log, math.E, 1},
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

// TestLogEdgeCases pins the documented non-finite behavior (matching math.Log)
// on the dispatched path. The slice is long enough to hit the SIMD body, so
// the kernel's special-lane blends are exercised alongside the scalar tail.
func TestLogEdgeCases(t *testing.T) {
	src := []float32{
		0, -1, float32(math.Copysign(0, -1)), float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()),
		2, 4, 8, 16, // pad past one 8-lane block so blends and tail both run
	}
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
			if math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) {
				if !bitsEqF32(dst[i], want) {
					t.Errorf("%s(%v) = %v, want %v", fn.name, x, dst[i], want)
				}
				continue
			}
			if re := relErrF32(dst[i], want); re > logRelTol32 {
				t.Errorf("%s(%v) = %v, want %v (relErr %g)", fn.name, x, dst[i], want, re)
			}
		}
	}
}

// TestLogSubnormals checks that subnormal inputs go through the kernel's
// normalization pre-scale (or the scalar fallback) and still come out within
// tolerance.
func TestLogSubnormals(t *testing.T) {
	src := []float32{
		1.401298464324817e-45, // smallest subnormal
		1e-44, 1e-40, 3e-39,   // assorted subnormals
		fltMinNormal32,         // FLT_MIN, smallest normal
		1.1754942106924411e-38, // largest subnormal
		1e-30, 1,
	}
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
			if re := relErrF32(dst[i], want); re > logRelTol32 {
				t.Errorf("%s(%g) = %v, want %v (relErr %g)", fn.name, x, dst[i], want, re)
			}
		}
	}
}

// TestLogAccuracy sweeps the full finite-positive range, the near-1
// neighborhood where the reduction switches sign, and the mantissa-reduction
// boundaries around sqrt(2)/2 and sqrt(2).
func TestLogAccuracy(t *testing.T) {
	src := make([]float32, 0, 1330)
	for e := -126; e <= 127; e++ {
		for _, m := range []float64{1.0, 1.3, 1.4142134, 1.4142137, 1.7, 1.9999999} {
			src = append(src, float32(math.Ldexp(m, e)))
		}
	}
	// Dense near 1 (both sides), where log -> 0.
	for i := -40; i <= 40; i++ {
		src = append(src, 1+float32(i)*1e-4, 1+float32(i)*1e-7)
	}
	src = append(src, math.MaxFloat32, fltMinNormal32, math.Sqrt2, math.Sqrt2/2)

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
			if re := relErrF32(dst[i], want); re > logRelTol32 {
				t.Errorf("%s(%g) = %v, want %v (relErr %g)", fn.name, x, dst[i], want, re)
			}
		}
	}
}

// TestLogLengths runs Log over every length from 1 to 40 so the scalar
// remainder loop is exercised alongside the SIMD body (the AVX body consumes
// 8 elements at a time, the NEON body 4).
func TestLogLengths(t *testing.T) {
	for n := 1; n <= 40; n++ {
		src := make([]float32, n)
		dst := make([]float32, n)
		for i := range src {
			src[i] = float32(math.Ldexp(1+0.13*float64(i), 2*i%50-25))
		}
		Log(dst, src)
		for i, x := range src {
			want := float32(math.Log(float64(x)))
			if re := relErrF32(dst[i], want); re > logRelTol32 {
				t.Errorf("len=%d Log(%v)[%d] = %v, want %v (relErr %g)", n, x, i, dst[i], want, re)
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
		{2, 0, 1},
		{0, 0, 1},
		{0, 2, 0},
		{2, 1, 2},
		{10, 3, 1000},
	}
	for _, tt := range tests {
		dst := make([]float32, 1)
		Pow(dst, []float32{tt.base}, tt.exp)
		if d := math.Abs(float64(dst[0]-tt.want)) / (1 + math.Abs(float64(tt.want))); d > powRelTol32 {
			t.Errorf("Pow(%v, %v) = %v, want %v", tt.base, tt.exp, dst[0], tt.want)
		}
	}

	edges := []struct{ base, exp float32 }{
		{-2, 0.5},
		{0, -1},
		{float32(math.Inf(1)), 2},
		{float32(math.NaN()), 1},
		{2, float32(math.Inf(1))},
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

// TestPowEdgeLanesSIMDLength repeats the math.Pow edge cases at a length that
// engages the SIMD body, so the dispatch-level scalar fallback for slices
// containing non-positive or non-finite bases is exercised.
func TestPowEdgeLanesSIMDLength(t *testing.T) {
	src := []float32{2, -2, 0, float32(math.Inf(1)), float32(math.NaN()), 4, 0.5, 9, 16, 25}
	for _, p := range []float32{0.5, 2, -1, 0, float32(math.NaN()), float32(math.Inf(1))} {
		dst := make([]float32, len(src))
		Pow(dst, src, p)
		for i, x := range src {
			want := float32(math.Pow(float64(x), float64(p)))
			if math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) || want == 0 || x <= 0 {
				if !bitsEqF32(dst[i], want) {
					t.Errorf("Pow(%v, %v) = %v, want %v (exact class)", x, p, dst[i], want)
				}
				continue
			}
			if re := relErrF32(dst[i], want); re > powRelTol32 {
				t.Errorf("Pow(%v, %v) = %v, want %v (relErr %g)", x, p, dst[i], want, re)
			}
		}
	}

	base := []float32{2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	expv := []float32{1, 2, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), 0.5, 0, -2, 3, 1.5}
	dst := make([]float32, len(base))
	PowElem(dst, base, expv)
	for i := range base {
		want := float32(math.Pow(float64(base[i]), float64(expv[i])))
		if math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) || want == 0 {
			if !bitsEqF32(dst[i], want) {
				t.Errorf("PowElem(%v, %v) = %v, want %v (exact class)", base[i], expv[i], dst[i], want)
			}
			continue
		}
		if re := relErrF32(dst[i], want); re > powRelTol32 {
			t.Errorf("PowElem(%v, %v) = %v, want %v (relErr %g)", base[i], expv[i], dst[i], want, re)
		}
	}
}

// TestPowOverflowClasses pins the overflow/underflow behavior of the SIMD
// path: for positive finite bases, results that overflow in math.Pow must
// come out as +Inf and results that underflow must come out as 0.
func TestPowOverflowClasses(t *testing.T) {
	src := []float32{1e30, 1e-30, 2, 0.5, 1e10, 1e-10, 3, 7}
	for _, p := range []float32{3, -3, 20, -20} {
		dst := make([]float32, len(src))
		Pow(dst, src, p)
		for i, x := range src {
			want := math.Pow(float64(x), float64(p))
			switch {
			case want > math.MaxFloat32*2: // clearly past float32 overflow
				if !math.IsInf(float64(dst[i]), 1) {
					t.Errorf("Pow(%v, %v) = %v, want +Inf", x, p, dst[i])
				}
			case want != 0 && want < 1e-40: // clearly past float32 underflow
				if dst[i] != 0 {
					t.Errorf("Pow(%v, %v) = %v, want 0", x, p, dst[i])
				}
			case float32(want) == 0 || math.IsInf(float64(float32(want)), 0):
				// borderline class, skip
			default:
				if re := relErrF32(dst[i], float32(want)); re > powRelTol32 {
					t.Errorf("Pow(%v, %v) = %v, want %v (relErr %g)", x, p, dst[i], want, re)
				}
			}
		}
	}
}

// TestLogPowParity checks the dispatched ops against the float64 math
// reference over a wide finite-positive range, within the documented
// tolerances. On hosts without the SIMD kernels this exercises the scalar
// path, which matches the reference exactly.
func TestLogPowParity(t *testing.T) {
	const n = 257 // odd, exercises the vector tail
	src := make([]float32, n)
	for i := range src {
		// Logarithmically spread positives across denormal-to-large magnitudes.
		src[i] = float32(math.Ldexp(1+0.5*float64(i%7), (i%80)-40))
	}

	check := func(name string, got []float32, ref func(float64) float64, tol float64) {
		for i := range src {
			want := float32(ref(float64(src[i])))
			if re := relErrF32(got[i], want); re > tol {
				t.Errorf("%s[%d](%v) = %v, want %v (relErr %g)", name, i, src[i], got[i], want, re)
			}
		}
	}
	dst := make([]float32, n)
	Log(dst, src)
	check("Log", dst, math.Log, logRelTol32)
	Log2(dst, src)
	check("Log2", dst, math.Log2, logRelTol32)
	Log10(dst, src)
	check("Log10", dst, math.Log10, logRelTol32)

	for _, exp := range []float32{0.35, 0.5, 2, -1, 3.7} {
		Pow(dst, src, exp)
		for i := range src {
			want := float32(math.Pow(float64(src[i]), float64(exp)))
			if re := relErrF32(dst[i], want); re > powRelTol32 {
				t.Errorf("Pow[%d](%v, %v) = %v, want %v (relErr %g)", i, src[i], exp, dst[i], want, re)
			}
		}
	}

	expv := make([]float32, n)
	for i := range expv {
		expv[i] = float32(0.1 + 0.05*float64(i%13))
	}
	PowElem(dst, src, expv)
	for i := range src {
		want := float32(math.Pow(float64(src[i]), float64(expv[i])))
		if re := relErrF32(dst[i], want); re > powRelTol32 {
			t.Errorf("PowElem[%d] = %v, want %v (relErr %g)", i, dst[i], want, re)
		}
	}
}

// TestLogPowDispatchParity pins the per-arch dispatch wrappers against the
// pure-Go reference. On SIMD hosts this is the kernel-vs-reference gate; on
// others it compares the reference with itself.
func TestLogPowDispatchParity(t *testing.T) {
	const n = 67 // SIMD body + scalar tail
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(math.Ldexp(1+0.37*float64(i%5), (i%60)-30))
	}
	got := make([]float32, n)
	want := make([]float32, n)

	log32(got, src)
	logGo(want, src)
	for i := range got {
		if re := relErrF32(got[i], want[i]); re > logRelTol32 {
			t.Errorf("log32[%d](%v) = %v, want %v (relErr %g)", i, src[i], got[i], want[i], re)
		}
	}

	log2_32(got, src)
	log2Go(want, src)
	for i := range got {
		if re := relErrF32(got[i], want[i]); re > logRelTol32 {
			t.Errorf("log2_32[%d](%v) = %v, want %v (relErr %g)", i, src[i], got[i], want[i], re)
		}
	}

	log10_32(got, src)
	log10Go(want, src)
	for i := range got {
		if re := relErrF32(got[i], want[i]); re > logRelTol32 {
			t.Errorf("log10_32[%d](%v) = %v, want %v (relErr %g)", i, src[i], got[i], want[i], re)
		}
	}

	pow32(got, src, 0.35)
	powGo(want, src, 0.35)
	for i := range got {
		if re := relErrF32(got[i], want[i]); re > powRelTol32 {
			t.Errorf("pow32[%d](%v, 0.35) = %v, want %v (relErr %g)", i, src[i], got[i], want[i], re)
		}
	}

	expv := make([]float32, n)
	for i := range expv {
		expv[i] = float32(-1.5 + 0.21*float64(i%17))
	}
	powElem32(got, src, expv)
	powElemGo(want, src, expv)
	for i := range got {
		if re := relErrF32(got[i], want[i]); re > powRelTol32 {
			t.Errorf("powElem32[%d](%v, %v) = %v, want %v (relErr %g)", i, src[i], expv[i], got[i], want[i], re)
		}
	}
}

func TestLogPowInPlace(t *testing.T) {
	src := []float32{1, 2, 4, 8, 16, 32, 0.5, 100}
	a := append([]float32(nil), src...)
	LogInPlace(a)
	for i := range src {
		if re := relErrF32(a[i], float32(math.Log(float64(src[i])))); re > logRelTol32 {
			t.Errorf("LogInPlace[%d] = %v (relErr %g)", i, a[i], re)
		}
	}
	b := append([]float32(nil), src...)
	PowInPlace(b, 0.5)
	for i := range src {
		if re := relErrF32(b[i], float32(math.Pow(float64(src[i]), 0.5))); re > powRelTol32 {
			t.Errorf("PowInPlace[%d] = %v (relErr %g)", i, b[i], re)
		}
	}
}

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
		if re := relErrF32(short[i], want); re > powRelTol32 {
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

func BenchmarkLog(b *testing.B) {
	src := make([]float32, 1024)
	for i := range src {
		src[i] = 1e-3 + float32(i)*0.7
	}
	dst := make([]float32, len(src))
	b.ReportAllocs()
	for b.Loop() {
		Log(dst, src)
	}
}

func BenchmarkPow(b *testing.B) {
	src := make([]float32, 1024)
	for i := range src {
		src[i] = 1e-3 + float32(i)*0.7
	}
	dst := make([]float32, len(src))
	b.ReportAllocs()
	for b.Loop() {
		Pow(dst, src, 0.35)
	}
}
