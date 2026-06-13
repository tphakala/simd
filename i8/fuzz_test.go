package i8

import "testing"

// Differential fuzz targets for the i8 primitives. Every i8 kernel is bit-exact
// against its pure-Go reference by construction (saturating arithmetic is
// deterministic; the int32-accumulated reductions wrap identically regardless of
// summation order), so each target asserts exact equality. The high-value bug
// class is tail/remainder handling at arbitrary lengths around the 8/16/32-byte
// unrolls and the SIMD dispatch thresholds; the seeds bracket those boundaries
// and the fuzzer widens the length space. Seeds run under plain `go test`;
// `go test -fuzz=FuzzXxx` explores further.

// i8FromBytes reinterprets raw bytes as int8s (the full -128..127 range,
// including the saturation and sign-extension extremes).
func i8FromBytes(raw []byte) []int8 {
	out := make([]int8, len(raw))
	for i, b := range raw {
		out[i] = int8(b)
	}
	return out
}

// lenSeeds seeds raw byte buffers whose lengths cover 0 through ~70, hitting
// every remainder around the 8/16/32-byte unrolls, plus a couple larger blocks.
func lenSeeds(f *testing.F) {
	f.Helper()
	lens := []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 47, 48, 63, 64, 65, 70, 128, 257}
	for _, n := range lens {
		raw := make([]byte, n)
		for i := range raw {
			raw[i] = byte(i*37 + 11)
		}
		f.Add(raw)
	}
}

func FuzzI8Saturate(f *testing.F) {
	lenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := i8FromBytes(raw)
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		got := make([]int8, h)
		want := make([]int8, h)

		AddSaturate(got, a, b)
		addSatGo(want, a, b)
		assertI8Eq(t, "AddSaturate", h, got, want)

		SubSaturate(got, a, b)
		subSatGo(want, a, b)
		assertI8Eq(t, "SubSaturate", h, got, want)
	})
}

func FuzzI8Reduce(f *testing.F) {
	lenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := i8FromBytes(raw)
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]

		if got, want := Sum(v), sumGo(v); got != want {
			t.Fatalf("Sum: got %d want %d (len=%d)", got, want, len(v))
		}
		if got, want := DotProduct(a, b), dotGo(a, b); got != want {
			t.Fatalf("DotProduct: got %d want %d (len=%d)", got, want, h)
		}
	})
}

func FuzzI8MinMax(f *testing.F) {
	lenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := i8FromBytes(raw)
		if len(v) == 0 {
			return
		}
		gotMin, gotMax := MinMax(v)
		wantMin, wantMax := minMaxGo(v)
		if gotMin != wantMin || gotMax != wantMax {
			t.Fatalf("MinMax got (%d,%d) want (%d,%d) (len=%d)", gotMin, gotMax, wantMin, wantMax, len(v))
		}
	})
}

func FuzzI8Elementwise(f *testing.F) {
	lenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := i8FromBytes(raw)
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]

		gotMin := make([]int8, h)
		wantMin := make([]int8, h)
		Min(gotMin, a, b)
		minGo(wantMin, a, b)
		assertI8Eq(t, "Min", h, gotMin, wantMin)

		gotMax := make([]int8, h)
		wantMax := make([]int8, h)
		Max(gotMax, a, b)
		maxGo(wantMax, a, b)
		assertI8Eq(t, "Max", h, gotMax, wantMax)

		// Derive lo/hi from the first two bytes (or defaults for tiny inputs);
		// the fuzzer will explore lo > hi too.
		var lo, hi int8
		if len(v) >= 2 {
			lo, hi = v[0], v[1]
		}
		gotClamp := make([]int8, len(v))
		wantClamp := make([]int8, len(v))
		Clamp(gotClamp, v, lo, hi)
		clampGo(wantClamp, v, lo, hi)
		assertI8Eq(t, "Clamp", len(v), gotClamp, wantClamp)

		gotAbs := make([]int8, len(v))
		wantAbs := make([]int8, len(v))
		Abs(gotAbs, v)
		absGo(wantAbs, v)
		assertI8Eq(t, "Abs", len(v), gotAbs, wantAbs)

		gotNeg := make([]int8, len(v))
		wantNeg := make([]int8, len(v))
		Neg(gotNeg, v)
		negGo(wantNeg, v)
		assertI8Eq(t, "Neg", len(v), gotNeg, wantNeg)
	})
}

func FuzzI8AbsOps(f *testing.F) {
	lenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := i8FromBytes(raw)

		if got, want := MaxAbs(v), maxAbsGo(v); got != want {
			t.Fatalf("MaxAbs: got %d want %d (len=%d)", got, want, len(v))
		}

		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		got := make([]int8, h)
		want := make([]int8, h)
		AbsDiff(got, a, b)
		absDiffGo(want, a, b)
		assertI8Eq(t, "AbsDiff", h, got, want)
	})
}

func FuzzI8Convert(f *testing.F) {
	lenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := i8FromBytes(raw)
		d16 := make([]int16, len(v))
		ToInt16(d16, v)
		for i := range v {
			if d16[i] != int16(v[i]) {
				t.Fatalf("ToInt16: index %d got %d want %d (len=%d)", i, d16[i], int16(v[i]), len(v))
			}
		}
		d32 := make([]int32, len(v))
		ToInt32(d32, v)
		for i := range v {
			if d32[i] != int32(v[i]) {
				t.Fatalf("ToInt32: index %d got %d want %d (len=%d)", i, d32[i], int32(v[i]), len(v))
			}
		}
	})
}
