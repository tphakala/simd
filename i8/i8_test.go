package i8

import (
	"testing"
)

// genI8 produces deterministic pseudo-random int8 data spanning the full range,
// including the -128/127 extremes, so saturation and sign-extension boundaries
// are exercised.
func genI8(n int, seed uint32) []int8 {
	s := make([]int8, n)
	x := seed*2654435761 + 1
	for i := range s {
		x = x*1664525 + 1013904223
		s[i] = int8(x >> 17) // spread across the byte range
	}
	return s
}

// lengths sweeps sub-threshold, single-block, multi-block, and ragged-tail sizes
// for both 16- and 32-byte vector kernels. 24 and 25 sit just past the reduce
// dispatch threshold (16) with n%16 == 8 and 9, exercising the 8-wide AVX2 tail
// block in Sum/DotProduct with a zero and a one-element scalar remainder (#149);
// 31/63/255 give it a full 7-element scalar remainder, 1000 a zero one.
var lengths = []int{0, 1, 2, 3, 7, 8, 15, 16, 17, 24, 25, 31, 32, 33, 63, 64, 65, 100, 255, 256, 257, 1000, 1024, 1031}

func TestAddSaturate(t *testing.T) {
	// Literal saturation cases validate the reference independently of the SIMD path.
	cases := []struct{ a, b, want int8 }{
		{0, 0, 0},
		{127, 1, 127},   // positive saturation
		{100, 100, 127}, // positive saturation
		{-128, -1, -128},
		{-100, -100, -128}, // negative saturation
		{50, -60, -10},
		{127, -128, -1},
	}
	for _, c := range cases {
		dst := make([]int8, 1)
		AddSaturate(dst, []int8{c.a}, []int8{c.b})
		if dst[0] != c.want {
			t.Errorf("AddSaturate(%d, %d) = %d, want %d", c.a, c.b, dst[0], c.want)
		}
	}
	// Parity against the reference across lengths.
	for _, n := range lengths {
		a, b := genI8(n, 1), genI8(n, 2)
		got := make([]int8, n)
		want := make([]int8, n)
		AddSaturate(got, a, b)
		addSatGo(want, a, b)
		assertI8Eq(t, "AddSaturate", n, got, want)
	}
}

func TestSubSaturate(t *testing.T) {
	cases := []struct{ a, b, want int8 }{
		{0, 0, 0},
		{127, -1, 127},   // positive saturation
		{-128, 1, -128},  // negative saturation
		{100, -100, 127}, // positive saturation
		{-100, 100, -128},
		{10, 30, -20},
		{-128, -128, 0},  // -128 - (-128) = 0
		{127, -128, 127}, // 127 - (-128) = 255 -> saturates to 127
	}
	for _, c := range cases {
		dst := make([]int8, 1)
		SubSaturate(dst, []int8{c.a}, []int8{c.b})
		if dst[0] != c.want {
			t.Errorf("SubSaturate(%d, %d) = %d, want %d", c.a, c.b, dst[0], c.want)
		}
	}
	for _, n := range lengths {
		a, b := genI8(n, 3), genI8(n, 4)
		got := make([]int8, n)
		want := make([]int8, n)
		SubSaturate(got, a, b)
		subSatGo(want, a, b)
		assertI8Eq(t, "SubSaturate", n, got, want)
	}
}

func TestToInt16(t *testing.T) {
	for _, n := range lengths {
		src := genI8(n, 5)
		got := make([]int16, n)
		ToInt16(got, src)
		for i := range src {
			if got[i] != int16(src[i]) {
				t.Fatalf("ToInt16 n=%d: got[%d]=%d, want %d", n, i, got[i], int16(src[i]))
			}
		}
	}
}

func TestToInt32(t *testing.T) {
	for _, n := range lengths {
		src := genI8(n, 6)
		got := make([]int32, n)
		ToInt32(got, src)
		for i := range src {
			if got[i] != int32(src[i]) {
				t.Fatalf("ToInt32 n=%d: got[%d]=%d, want %d", n, i, got[i], int32(src[i]))
			}
		}
	}
}

func TestSum(t *testing.T) {
	// Literal cases.
	if got := Sum(nil); got != 0 {
		t.Errorf("Sum(nil) = %d, want 0", got)
	}
	if got := Sum([]int8{127, 127, 1, -128}); got != 127 {
		t.Errorf("Sum([127,127,1,-128]) = %d, want 127", got)
	}
	// Parity across lengths.
	for _, n := range lengths {
		a := genI8(n, 7)
		if got, want := Sum(a), sumGo(a); got != want {
			t.Errorf("Sum n=%d: got %d, want %d", n, got, want)
		}
	}
	// int32 accumulation must not narrow before the total: 300 elements of 127
	// sum to 38100, which exceeds int16 but fits int32.
	big := make([]int8, 300)
	for i := range big {
		big[i] = 127
	}
	if got, want := Sum(big), int32(300*127); got != want {
		t.Errorf("Sum(300x127) = %d, want %d", got, want)
	}
}

func TestDotProduct(t *testing.T) {
	if got := DotProduct(nil, nil); got != 0 {
		t.Errorf("DotProduct(nil) = %d, want 0", got)
	}
	// (-128)*(-128) = 16384, exceeds int16 but fits int32.
	if got := DotProduct([]int8{-128, 1}, []int8{-128, 2}); got != 16384+2 {
		t.Errorf("DotProduct([-128,1],[-128,2]) = %d, want %d", got, 16384+2)
	}
	for _, n := range lengths {
		a, b := genI8(n, 8), genI8(n, 9)
		if got, want := DotProduct(a, b), dotGo(a, b); got != want {
			t.Errorf("DotProduct n=%d: got %d, want %d", n, got, want)
		}
	}
	// Mismatched lengths clamp to the shorter operand.
	if got, want := DotProduct([]int8{1, 2, 3}, []int8{4, 5}), int32(1*4+2*5); got != want {
		t.Errorf("DotProduct mismatched len = %d, want %d", got, want)
	}
	// int32 two's-complement wraparound: 140000 elements of 127*127=16129 sum to
	// 2_258_060_000, which overflows int32. The result must wrap exactly like the
	// reference (verified here against an independent int64->int32 truncation).
	const wn = 140000
	wa := make([]int8, wn)
	wb := make([]int8, wn)
	for i := range wa {
		wa[i], wb[i] = 127, 127
	}
	wantWrap := int32(int64(len(wa)) * 127 * 127) // non-constant: truncates at runtime
	if got := DotProduct(wa, wb); got != wantWrap {
		t.Errorf("DotProduct wraparound = %d, want %d", got, wantWrap)
	}
	if wantWrap >= 0 {
		t.Fatalf("test setup: expected wraparound to a negative value, got %d", wantWrap)
	}
}

func TestMinMax(t *testing.T) {
	if lo, hi := MinMax(nil); lo != 0 || hi != 0 {
		t.Errorf("MinMax(nil) = (%d,%d), want (0,0)", lo, hi)
	}
	if lo, hi := MinMax([]int8{5}); lo != 5 || hi != 5 {
		t.Errorf("MinMax([5]) = (%d,%d), want (5,5)", lo, hi)
	}
	if lo, hi := MinMax([]int8{0, -128, 127, 3, -1}); lo != -128 || hi != 127 {
		t.Errorf("MinMax = (%d,%d), want (-128,127)", lo, hi)
	}
	for _, n := range lengths {
		if n == 0 {
			continue
		}
		a := genI8(n, 10)
		gotLo, gotHi := MinMax(a)
		wantLo, wantHi := minMaxGo(a)
		if gotLo != wantLo || gotHi != wantHi {
			t.Errorf("MinMax n=%d: got (%d,%d), want (%d,%d)", n, gotLo, gotHi, wantLo, wantHi)
		}
	}
}

func TestMin(t *testing.T) {
	cases := []struct{ a, b, want int8 }{
		{0, 0, 0},
		{5, 3, 3},
		{-128, 127, -128},
		{127, -128, -128},
		{-1, -2, -2},
		{-128, -128, -128},
	}
	for _, c := range cases {
		dst := make([]int8, 1)
		Min(dst, []int8{c.a}, []int8{c.b})
		if dst[0] != c.want {
			t.Errorf("Min(%d, %d) = %d, want %d", c.a, c.b, dst[0], c.want)
		}
	}
	for _, n := range lengths {
		a, b := genI8(n, 13), genI8(n, 14)
		got := make([]int8, n)
		want := make([]int8, n)
		Min(got, a, b)
		minGo(want, a, b)
		assertI8Eq(t, "Min", n, got, want)
	}
	// Mismatched lengths clamp to the shortest operand.
	got := make([]int8, 3)
	Min(got, []int8{1, 2, 3}, []int8{0, 9})
	if got[0] != 0 || got[1] != 2 || got[2] != 0 {
		t.Errorf("Min mismatched len = %v, want [0 2 0]", got)
	}
}

func TestMax(t *testing.T) {
	cases := []struct{ a, b, want int8 }{
		{0, 0, 0},
		{5, 3, 5},
		{-128, 127, 127},
		{127, -128, 127},
		{-1, -2, -1},
		{-128, -128, -128},
	}
	for _, c := range cases {
		dst := make([]int8, 1)
		Max(dst, []int8{c.a}, []int8{c.b})
		if dst[0] != c.want {
			t.Errorf("Max(%d, %d) = %d, want %d", c.a, c.b, dst[0], c.want)
		}
	}
	for _, n := range lengths {
		a, b := genI8(n, 15), genI8(n, 16)
		got := make([]int8, n)
		want := make([]int8, n)
		Max(got, a, b)
		maxGo(want, a, b)
		assertI8Eq(t, "Max", n, got, want)
	}
}

func TestClamp(t *testing.T) {
	cases := []struct {
		v, lo, hi, want int8
	}{
		{0, -10, 10, 0},
		{-50, -10, 10, -10},
		{50, -10, 10, 10},
		{-128, -128, 127, -128},
		{127, -128, 127, 127},
		{5, 5, 5, 5},      // lo == hi
		{5, 10, -10, -10}, // lo > hi: every element maps to hi
		{-5, 10, -10, -10},
	}
	for _, c := range cases {
		dst := make([]int8, 1)
		Clamp(dst, []int8{c.v}, c.lo, c.hi)
		if dst[0] != c.want {
			t.Errorf("Clamp(%d, lo=%d, hi=%d) = %d, want %d", c.v, c.lo, c.hi, dst[0], c.want)
		}
	}
	for _, n := range lengths {
		src := genI8(n, 17)
		got := make([]int8, n)
		want := make([]int8, n)
		Clamp(got, src, -20, 20)
		clampGo(want, src, -20, 20)
		assertI8Eq(t, "Clamp", n, got, want)
	}
}

func TestAbs(t *testing.T) {
	cases := []struct{ a, want int8 }{
		{0, 0},
		{5, 5},
		{-5, 5},
		{127, 127},
		{-127, 127},
		{-128, 127}, // saturating: |−128| clamps to 127
	}
	for _, c := range cases {
		dst := make([]int8, 1)
		Abs(dst, []int8{c.a})
		if dst[0] != c.want {
			t.Errorf("Abs(%d) = %d, want %d", c.a, dst[0], c.want)
		}
	}
	for _, n := range lengths {
		a := genI8(n, 18)
		got := make([]int8, n)
		want := make([]int8, n)
		Abs(got, a)
		absGo(want, a)
		assertI8Eq(t, "Abs", n, got, want)
	}
}

func TestNeg(t *testing.T) {
	cases := []struct{ a, want int8 }{
		{0, 0},
		{5, -5},
		{-5, 5},
		{127, -127},
		{-127, 127},
		{-128, 127}, // saturating: −(−128) clamps to 127
	}
	for _, c := range cases {
		dst := make([]int8, 1)
		Neg(dst, []int8{c.a})
		if dst[0] != c.want {
			t.Errorf("Neg(%d) = %d, want %d", c.a, dst[0], c.want)
		}
	}
	for _, n := range lengths {
		a := genI8(n, 19)
		got := make([]int8, n)
		want := make([]int8, n)
		Neg(got, a)
		negGo(want, a)
		assertI8Eq(t, "Neg", n, got, want)
	}
}

func TestMaxAbs(t *testing.T) {
	if got := MaxAbs(nil); got != 0 {
		t.Errorf("MaxAbs(nil) = %d, want 0", got)
	}
	cases := []struct {
		a    []int8
		want int
	}{
		{[]int8{0}, 0},
		{[]int8{5, -3, 2}, 5},
		{[]int8{-3, 5, -2}, 5},
		{[]int8{-128}, 128}, // |−128| = 128 does not fit int8
		{[]int8{127, -128, 1}, 128},
		{[]int8{-1, -2, -127}, 127},
	}
	for _, c := range cases {
		if got := MaxAbs(c.a); got != c.want {
			t.Errorf("MaxAbs(%v) = %d, want %d", c.a, got, c.want)
		}
	}
	for _, n := range lengths {
		if n == 0 {
			continue
		}
		a := genI8(n, 20)
		if got, want := MaxAbs(a), maxAbsGo(a); got != want {
			t.Errorf("MaxAbs n=%d: got %d, want %d", n, got, want)
		}
	}
}

func TestAbsDiff(t *testing.T) {
	cases := []struct{ a, b, want int8 }{
		{0, 0, 0},
		{5, 3, 2},
		{3, 5, 2},
		{-5, 5, 10},
		{127, -128, 127}, // |255| saturates to 127
		{-128, 127, 127}, // |−255| saturates to 127
		{-128, -128, 0},
		{100, -100, 127}, // |200| saturates to 127
	}
	for _, c := range cases {
		dst := make([]int8, 1)
		AbsDiff(dst, []int8{c.a}, []int8{c.b})
		if dst[0] != c.want {
			t.Errorf("AbsDiff(%d, %d) = %d, want %d", c.a, c.b, dst[0], c.want)
		}
	}
	for _, n := range lengths {
		a, b := genI8(n, 21), genI8(n, 22)
		got := make([]int8, n)
		want := make([]int8, n)
		AbsDiff(got, a, b)
		absDiffGo(want, a, b)
		assertI8Eq(t, "AbsDiff", n, got, want)
	}
	// Mismatched lengths clamp to the shortest operand.
	got := make([]int8, 3)
	AbsDiff(got, []int8{10, 20, 30}, []int8{1, 2})
	if got[0] != 9 || got[1] != 18 || got[2] != 0 {
		t.Errorf("AbsDiff mismatched len = %v, want [9 18 0]", got)
	}
}

func TestAddScalarSaturate(t *testing.T) {
	cases := []struct{ a, s, want int8 }{
		{0, 0, 0},
		{100, 100, 127}, // positive saturation
		{-100, -100, -128},
		{50, -60, -10},
		{127, 1, 127},
		{-128, -1, -128},
	}
	for _, c := range cases {
		dst := make([]int8, 1)
		AddScalarSaturate(dst, []int8{c.a}, c.s)
		if dst[0] != c.want {
			t.Errorf("AddScalarSaturate(%d, %d) = %d, want %d", c.a, c.s, dst[0], c.want)
		}
	}
	for _, n := range lengths {
		a := genI8(n, 23)
		got := make([]int8, n)
		want := make([]int8, n)
		AddScalarSaturate(got, a, -37)
		addScalarSatGo(want, a, -37)
		assertI8Eq(t, "AddScalarSaturate", n, got, want)
	}
}

func TestSubScalarSaturate(t *testing.T) {
	cases := []struct{ a, s, want int8 }{
		{0, 0, 0},
		{100, -100, 127}, // positive saturation
		{-100, 100, -128},
		{10, 30, -20},
		{127, -128, 127}, // 127 - (-128) = 255 -> 127
		{-128, 1, -128},
	}
	for _, c := range cases {
		dst := make([]int8, 1)
		SubScalarSaturate(dst, []int8{c.a}, c.s)
		if dst[0] != c.want {
			t.Errorf("SubScalarSaturate(%d, %d) = %d, want %d", c.a, c.s, dst[0], c.want)
		}
	}
	for _, n := range lengths {
		a := genI8(n, 24)
		got := make([]int8, n)
		want := make([]int8, n)
		SubScalarSaturate(got, a, 37)
		subScalarSatGo(want, a, 37)
		assertI8Eq(t, "SubScalarSaturate", n, got, want)
	}
}

func TestSumAbs(t *testing.T) {
	if got := SumAbs(nil); got != 0 {
		t.Errorf("SumAbs(nil) = %d, want 0", got)
	}
	cases := []struct {
		a    []int8
		want int32
	}{
		{[]int8{0}, 0},
		{[]int8{5, -3, 2}, 10},
		{[]int8{-128}, 128}, // |−128| = 128
		{[]int8{-128, 127, -1}, 256},
	}
	for _, c := range cases {
		if got := SumAbs(c.a); got != c.want {
			t.Errorf("SumAbs(%v) = %d, want %d", c.a, got, c.want)
		}
	}
	for _, n := range lengths {
		a := genI8(n, 25)
		if got, want := SumAbs(a), sumAbsGo(a); got != want {
			t.Errorf("SumAbs n=%d: got %d, want %d", n, got, want)
		}
	}
	// int32 accumulation headroom: 300 elements of -128 sum to 38400 (> int16).
	big := make([]int8, 300)
	for i := range big {
		big[i] = -128
	}
	if got, want := SumAbs(big), int32(300*128); got != want {
		t.Errorf("SumAbs(300x-128) = %d, want %d", got, want)
	}
}

func TestSAD(t *testing.T) {
	if got := SAD(nil, nil); got != 0 {
		t.Errorf("SAD(nil) = %d, want 0", got)
	}
	cases := []struct {
		a, b []int8
		want int32
	}{
		{[]int8{0}, []int8{0}, 0},
		{[]int8{5, 3}, []int8{3, 5}, 4},
		{[]int8{-1}, []int8{1}, 2},       // signed: |−1−1| = 2 (not unsigned 254)
		{[]int8{127}, []int8{-128}, 255}, // true |255|, NOT saturated
		{[]int8{-128}, []int8{127}, 255},
	}
	for _, c := range cases {
		if got := SAD(c.a, c.b); got != c.want {
			t.Errorf("SAD(%v, %v) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
	for _, n := range lengths {
		a, b := genI8(n, 26), genI8(n, 27)
		if got, want := SAD(a, b), sadGo(a, b); got != want {
			t.Errorf("SAD n=%d: got %d, want %d", n, got, want)
		}
	}
	// Mismatched lengths clamp to the shorter operand.
	if got, want := SAD([]int8{10, 20, 30}, []int8{1, 2}), int32(9+18); got != want {
		t.Errorf("SAD mismatched len = %d, want %d", got, want)
	}
}

func TestZeroAllocations(t *testing.T) {
	const n = 1024
	a, b := genI8(n, 11), genI8(n, 12)
	d8 := make([]int8, n)
	d16 := make([]int16, n)
	d32 := make([]int32, n)

	checks := []struct {
		name string
		fn   func()
	}{
		{"AddSaturate", func() { AddSaturate(d8, a, b) }},
		{"SubSaturate", func() { SubSaturate(d8, a, b) }},
		{"ToInt16", func() { ToInt16(d16, a) }},
		{"ToInt32", func() { ToInt32(d32, a) }},
		{"Sum", func() { _ = Sum(a) }},
		{"DotProduct", func() { _ = DotProduct(a, b) }},
		{"MinMax", func() { _, _ = MinMax(a) }},
		{"Min", func() { Min(d8, a, b) }},
		{"Max", func() { Max(d8, a, b) }},
		{"Clamp", func() { Clamp(d8, a, -20, 20) }},
		{"Abs", func() { Abs(d8, a) }},
		{"Neg", func() { Neg(d8, a) }},
		{"MaxAbs", func() { _ = MaxAbs(a) }},
		{"AbsDiff", func() { AbsDiff(d8, a, b) }},
		{"AddScalarSaturate", func() { AddScalarSaturate(d8, a, 7) }},
		{"SubScalarSaturate", func() { SubScalarSaturate(d8, a, 7) }},
		{"SumAbs", func() { _ = SumAbs(a) }},
		{"SAD", func() { _ = SAD(a, b) }},
	}
	for _, c := range checks {
		if got := testing.AllocsPerRun(10, c.fn); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", c.name, got)
		}
	}
}

// TestTrailingCapacityUntouched verifies the binary ops and conversions write
// exactly n elements and leave trailing dst capacity alone. It runs sizes both
// below and above the SIMD dispatch thresholds (32/16/8), so the assembly
// kernels and their scalar tails are covered, not just the pure-Go fallback.
func TestTrailingCapacityUntouched(t *testing.T) {
	for _, n := range []int{3, 35} {
		a, b := genI8(n, 1), genI8(n, 2)

		dst := fillI8(n+2, 42)
		AddSaturate(dst[:n], a, b)
		if dst[n] != 42 || dst[n+1] != 42 {
			t.Errorf("AddSaturate (n=%d) clobbered trailing capacity: %v", n, dst[n:])
		}

		dst = fillI8(n+2, 42)
		SubSaturate(dst[:n], a, b)
		if dst[n] != 42 || dst[n+1] != 42 {
			t.Errorf("SubSaturate (n=%d) clobbered trailing capacity: %v", n, dst[n:])
		}

		dst16 := fillI16(n+2, 4242)
		ToInt16(dst16[:n], a)
		if dst16[n] != 4242 || dst16[n+1] != 4242 {
			t.Errorf("ToInt16 (n=%d) clobbered trailing capacity: %v", n, dst16[n:])
		}

		dst32 := fillI32(n+2, 424242)
		ToInt32(dst32[:n], a)
		if dst32[n] != 424242 || dst32[n+1] != 424242 {
			t.Errorf("ToInt32 (n=%d) clobbered trailing capacity: %v", n, dst32[n:])
		}

		dst = fillI8(n+2, 42)
		Min(dst[:n], a, b)
		if dst[n] != 42 || dst[n+1] != 42 {
			t.Errorf("Min (n=%d) clobbered trailing capacity: %v", n, dst[n:])
		}

		dst = fillI8(n+2, 42)
		Max(dst[:n], a, b)
		if dst[n] != 42 || dst[n+1] != 42 {
			t.Errorf("Max (n=%d) clobbered trailing capacity: %v", n, dst[n:])
		}

		dst = fillI8(n+2, 42)
		Clamp(dst[:n], a, -20, 20)
		if dst[n] != 42 || dst[n+1] != 42 {
			t.Errorf("Clamp (n=%d) clobbered trailing capacity: %v", n, dst[n:])
		}

		dst = fillI8(n+2, 42)
		Abs(dst[:n], a)
		if dst[n] != 42 || dst[n+1] != 42 {
			t.Errorf("Abs (n=%d) clobbered trailing capacity: %v", n, dst[n:])
		}

		dst = fillI8(n+2, 42)
		Neg(dst[:n], a)
		if dst[n] != 42 || dst[n+1] != 42 {
			t.Errorf("Neg (n=%d) clobbered trailing capacity: %v", n, dst[n:])
		}

		dst = fillI8(n+2, 42)
		AbsDiff(dst[:n], a, b)
		if dst[n] != 42 || dst[n+1] != 42 {
			t.Errorf("AbsDiff (n=%d) clobbered trailing capacity: %v", n, dst[n:])
		}

		dst = fillI8(n+2, 42)
		AddScalarSaturate(dst[:n], a, 9)
		if dst[n] != 42 || dst[n+1] != 42 {
			t.Errorf("AddScalarSaturate (n=%d) clobbered trailing capacity: %v", n, dst[n:])
		}

		dst = fillI8(n+2, 42)
		SubScalarSaturate(dst[:n], a, 9)
		if dst[n] != 42 || dst[n+1] != 42 {
			t.Errorf("SubScalarSaturate (n=%d) clobbered trailing capacity: %v", n, dst[n:])
		}
	}
}

// fillI8 mirrors fillI16/fillI32: the sentinel value is passed explicitly at
// each call site so the trailing-capacity checks read uniformly across widths.
//
//nolint:unparam // sentinel kept explicit to match the fillI16/fillI32 family
func fillI8(n int, v int8) []int8 {
	s := make([]int8, n)
	for i := range s {
		s[i] = v
	}
	return s
}
func fillI16(n int, v int16) []int16 {
	s := make([]int16, n)
	for i := range s {
		s[i] = v
	}
	return s
}
func fillI32(n int, v int32) []int32 {
	s := make([]int32, n)
	for i := range s {
		s[i] = v
	}
	return s
}

func assertI8Eq(t *testing.T, op string, n int, got, want []int8) {
	t.Helper()
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s n=%d: got[%d]=%d, want %d", op, n, i, got[i], want[i])
		}
	}
}
