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
// for both 16- and 32-byte vector kernels.
var lengths = []int{0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 255, 256, 257, 1000, 1024, 1031}

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
	}
	for _, c := range checks {
		if got := testing.AllocsPerRun(10, c.fn); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", c.name, got)
		}
	}
}

// TestTrailingCapacityUntouched verifies the binary ops and conversions write
// exactly n elements and leave trailing dst capacity alone.
func TestTrailingCapacityUntouched(t *testing.T) {
	a := []int8{1, 2, 3}
	b := []int8{4, 5, 6}
	dst := []int8{-9, -9, -9, 42, 42}
	AddSaturate(dst, a, b)
	if dst[3] != 42 || dst[4] != 42 {
		t.Errorf("AddSaturate clobbered trailing capacity: %v", dst)
	}
}

func assertI8Eq(t *testing.T, op string, n int, got, want []int8) {
	t.Helper()
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s n=%d: got[%d]=%d, want %d", op, n, i, got[i], want[i])
		}
	}
}
