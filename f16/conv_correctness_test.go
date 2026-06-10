package f16

import (
	"math"
	"testing"
)

// f16IsNaN reports whether a Float16 bit pattern is NaN (exponent all ones,
// mantissa non-zero). Hardware conversions quiet signaling NaNs and may not
// preserve the exact payload the scalar reference produces, so NaN inputs are
// compared by class, not by bits.
func f16IsNaN(h Float16) bool {
	return h&0x7C00 == 0x7C00 && h&0x03FF != 0
}

// fromFloat32Corpus returns float32 inputs that stress every FromFloat32 branch:
// rounding ties, the overflow boundary, the full denormal range including the
// underflow rounding boundary at 2^-25, signed zero, and the special values.
func fromFloat32Corpus() []float32 {
	return []float32{
		0, float32(math.Copysign(0, -1)), // +0, -0
		float32(math.Inf(1)), float32(math.Inf(-1)),
		float32(math.NaN()), -float32(math.NaN()),
		1, -1, 2, -2, 0.5, -0.5, 3.14159, -100, 100,
		65504, -65504, // max finite f16
		65505, 65519, 65520, 65536, // overflow boundary (>= 65520 -> Inf under RNE)
		6.1035156e-05,                    // smallest normal f16 (2^-14)
		6.097555e-05,                     // largest f16 denormal region (just below 2^-14)
		5.9604645e-08,                    // smallest positive f16 denormal (2^-24)
		math.Float32frombits(0x33592e01), // between 2^-25 and 2^-24: rounds up to 0x0001
		2.9802322e-08,                    // 2^-25 exactly: RNE -> 0 (even)
		1.4901161e-08,                    // below 2^-25: flushes to 0
		1 + 1.0/2048,                     // 1 + 2^-11: tie at 1.0, RNE -> 1.0
		1 + 1.0/1024,                     // 1 + 2^-10: exactly representable
		1 + 3.0/2048,                     // tie up, RNE -> 1 + 2^-10
		1024.5, 2048.25,                  // large values that round
	}
}

// TestToFloat32SliceParity converts every Float16 value through the public
// dispatch (F16C on amd64, NEON on arm64, Go elsewhere) and checks bit-exact
// parity with the scalar reference, NaN by class.
func TestToFloat32SliceParity(t *testing.T) {
	const total = 1 << 16
	src := make([]Float16, total)
	for i := range src {
		src[i] = Float16(i)
	}
	got := make([]float32, total)
	ToFloat32Slice(got, src)

	for i := range src {
		want := toFloat32Go(src[i])
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got[i])) {
				t.Fatalf("h=%#04x: got %v, want NaN", src[i], got[i])
			}
			continue
		}
		if math.Float32bits(got[i]) != math.Float32bits(want) {
			t.Fatalf("h=%#04x: got bits %#08x, want %#08x", src[i], math.Float32bits(got[i]), math.Float32bits(want))
		}
	}
}

// TestFromFloat32SliceParity feeds the boundary corpus plus a large random sweep
// through the public dispatch and asserts bit-exact parity with the scalar
// reference for non-NaN inputs, class parity for NaN. On arm64 this validates the
// NEON kernel; on amd64 the F16C kernel.
func TestFromFloat32SliceParity(t *testing.T) {
	src := fromFloat32Corpus()
	seed := uint32(0x9e3779b9)
	for range 8192 {
		seed = seed*1664525 + 1013904223
		src = append(src, math.Float32frombits(seed))
	}
	got := make([]Float16, len(src))
	FromFloat32Slice(got, src)

	for i, f := range src {
		want := fromFloat32Go(f)
		if math.IsNaN(float64(f)) {
			if !f16IsNaN(got[i]) {
				t.Fatalf("src=%v (%#08x): got %#04x, want a NaN", f, math.Float32bits(f), got[i])
			}
			continue
		}
		if got[i] != want {
			t.Fatalf("src=%v (%#08x): got %#04x, want %#04x", f, math.Float32bits(f), got[i], want)
		}
	}
}

// TestConvRoundTripExhaustive round-trips every non-Inf/NaN Float16 value through
// ToFloat32Slice and back through FromFloat32Slice and asserts exact recovery.
// This covers all representable FromFloat32 outputs, including the full denormal
// range, against whichever kernel the platform dispatches to.
func TestConvRoundTripExhaustive(t *testing.T) {
	const total = 1 << 16
	src := make([]Float16, 0, total)
	for i := range total {
		h := Float16(i)
		if h&0x7C00 == 0x7C00 { // skip Inf/NaN exponent
			continue
		}
		src = append(src, h)
	}
	f32 := make([]float32, len(src))
	ToFloat32Slice(f32, src)
	back := make([]Float16, len(src))
	FromFloat32Slice(back, f32)

	for i, h := range src {
		if h&0x7FFF == 0 && back[i]&0x7FFF == 0 { // +0/-0 both fine
			continue
		}
		if back[i] != h {
			t.Fatalf("round-trip h=%#04x: got %#04x", h, back[i])
		}
	}
}

// TestConvSliceDispatchLengths exercises the kernel-prefix plus Go-remainder split
// across lengths 0..24 in both directions.
func TestConvSliceDispatchLengths(t *testing.T) {
	for n := 0; n <= 24; n++ {
		src16 := make([]Float16, n)
		for i := range src16 {
			src16[i] = Float16((i*37 + 1) & 0x7BFF) // avoid Inf/NaN exponent
		}
		f32 := make([]float32, n)
		ToFloat32Slice(f32, src16)
		for i := range f32 {
			want := toFloat32Go(src16[i])
			if math.Float32bits(f32[i]) != math.Float32bits(want) {
				t.Fatalf("ToFloat32Slice n=%d i=%d: got %#08x want %#08x", n, i, math.Float32bits(f32[i]), math.Float32bits(want))
			}
		}
		back := make([]Float16, n)
		FromFloat32Slice(back, f32)
		for i := range back {
			want := fromFloat32Go(f32[i])
			if back[i] != want {
				t.Fatalf("FromFloat32Slice n=%d i=%d: got %#04x want %#04x", n, i, back[i], want)
			}
		}
	}
}

// TestConvSliceAllocs confirms the public conversion APIs stay allocation-free.
func TestConvSliceAllocs(t *testing.T) {
	src16 := make([]Float16, 1024)
	for i := range src16 {
		src16[i] = Float16((i * 7) & 0x7BFF)
	}
	f32 := make([]float32, 1024)
	if allocs := testing.AllocsPerRun(100, func() { ToFloat32Slice(f32, src16) }); allocs != 0 {
		t.Errorf("ToFloat32Slice allocs = %v, want 0", allocs)
	}
	out16 := make([]Float16, 1024)
	if allocs := testing.AllocsPerRun(100, func() { FromFloat32Slice(out16, f32) }); allocs != 0 {
		t.Errorf("FromFloat32Slice allocs = %v, want 0", allocs)
	}
}
