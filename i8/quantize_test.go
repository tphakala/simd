package i8

import (
	"math"
	"math/rand"
	"testing"
)

// TestQuantizeRandomizedDifferential runs many random cases through the public
// path and the Go reference and asserts bit-exact agreement. It gives the active
// SIMD kernel (AVX2 or NEON) broad differential coverage beyond the deterministic
// parity sweep; on the rpi5 it is the on-device stress for the NEON kernels.
func TestQuantizeRandomizedDifferential(t *testing.T) {
	r := rand.New(rand.NewSource(0x132))
	for iter := range 4000 {
		n := r.Intn(200)

		// Quantize: random float32 bit patterns (NaN/Inf/subnormal included).
		fsrc := make([]float32, n)
		for i := range fsrc {
			fsrc[i] = math.Float32frombits(r.Uint32())
		}
		fscale := math.Float32frombits(r.Uint32())
		fzp := int8(r.Intn(256) - 128)
		gq := make([]int8, n)
		wq := make([]int8, n)
		Quantize(gq, fsrc, fscale, fzp)
		quantizeGo(wq, fsrc, fscale, fzp)
		for i := range fsrc {
			if gq[i] != wq[i] {
				t.Fatalf("Quantize[%d]=%d want %d (src=%v scale=%v zp=%d n=%d iter=%d)",
					i, gq[i], wq[i], fsrc[i], fscale, fzp, n, iter)
			}
		}

		// Dequantize.
		isrc := make([]int8, n)
		for i := range isrc {
			isrc[i] = int8(r.Intn(256) - 128)
		}
		dscale := math.Float32frombits(r.Uint32())
		dzp := int8(r.Intn(256) - 128)
		gd := make([]float32, n)
		wd := make([]float32, n)
		Dequantize(gd, isrc, dscale, dzp)
		dequantizeGo(wd, isrc, dscale, dzp)
		assertF32Bits(t, "Dequantize", n, gd, wd)

		// Requantize: random accumulators (with extremes), Q31-ish multipliers
		// and shifts spanning the in-domain and reroute ranges.
		acc := make([]int32, n)
		for i := range acc {
			switch r.Intn(8) {
			case 0:
				acc[i] = math.MaxInt32
			case 1:
				acc[i] = math.MinInt32
			default:
				acc[i] = int32(r.Uint32())
			}
		}
		mul := int32(r.Uint32())
		shift := r.Intn(80) - 40 // covers [-31,30] plus reroute on both ends
		rzp := int8(r.Intn(256) - 128)
		gr := make([]int8, n)
		wr := make([]int8, n)
		Requantize(gr, acc, mul, shift, rzp)
		requantizeGo(wr, acc, mul, shift, rzp)
		for i := range acc {
			if gr[i] != wr[i] {
				t.Fatalf("Requantize[%d]=%d want %d (acc=%d mul=%d shift=%d zp=%d n=%d iter=%d)",
					i, gr[i], wr[i], acc[i], mul, shift, rzp, n, iter)
			}
		}
	}
}

// Tests for the Part-of-#132 quantization vertical (Quantize, Dequantize,
// Requantize). The pure-Go references in i8_go.go are the source of truth; the
// SIMD kernels are validated bit-exact against them by the parity sweeps and the
// differential fuzzers here. The known-answer tables lock the semantics down so
// a reference bug cannot hide behind a self-consistent SIMD/Go pair.

// genF32 produces deterministic float32 data mixing exact half-integers (to
// exercise round-to-even ties, and the kernels' clamp-then-round vs the
// reference's round-then-clamp), fractional values that span the saturation
// bounds, and small integers.
func genF32(n int, seed uint32) []float32 {
	s := make([]float32, n)
	x := seed*2654435761 + 1
	for i := range s {
		x = x*1664525 + 1013904223
		switch i % 3 {
		case 0:
			s[i] = float32(int32(x)>>16) * 0.5 // half-integers: RNE ties
		case 1:
			s[i] = float32(int32(x)) / 700.0 // fractional, spans the clamp
		default:
			s[i] = float32(int32(x) >> 20) // small integers
		}
	}
	return s
}

// genI32 produces deterministic int32 accumulators across the full range.
func genI32(n int, seed uint32) []int32 {
	s := make([]int32, n)
	x := seed*2654435761 + 1
	for i := range s {
		x = x*1664525 + 1013904223
		s[i] = int32(x)
	}
	return s
}

// quantizeCombos and requantizeCombos are the parameter sets swept alongside the
// length sweep.
var (
	quantizeScales = []float32{1.0, 0.5, 2.0, 127.0, 0.007}
	quantizeZPs    = []int8{0, -5, 7, 127, -128}

	requantizeMuls = []int32{
		0x40000000, // 0.5 in Q31
		0x7FFFFFFF, // ~1.0 in Q31 (MaxInt32)
		0x60000000,
		0x40000001,
		0x2000_0000,
	}
	requantizeShifts = []int{-31, -8, -1, 0, 1, 8, 30}
	requantizeZPs    = []int8{0, -5, 7, 127, -128}
)

// TestQuantizeParity sweeps lengths and (scale, zeroPoint) combos, asserting the
// public path is bit-exact with the Go reference.
func TestQuantizeParity(t *testing.T) {
	for _, n := range lengths {
		src := genF32(n, uint32(n)+1)
		for _, scale := range quantizeScales {
			for _, zp := range quantizeZPs {
				got := make([]int8, n)
				want := make([]int8, n)
				Quantize(got, src, scale, zp)
				quantizeGo(want, src, scale, zp)
				assertI8Eq(t, "Quantize", n, got, want)
			}
		}
	}
}

// TestDequantizeParity sweeps lengths and (scale, zeroPoint) combos, asserting
// the public path is bit-exact (bit-for-bit float32) with the Go reference.
func TestDequantizeParity(t *testing.T) {
	for _, n := range lengths {
		src := genI8(n, uint32(n)+3)
		for _, scale := range quantizeScales {
			for _, zp := range quantizeZPs {
				got := make([]float32, n)
				want := make([]float32, n)
				Dequantize(got, src, scale, zp)
				dequantizeGo(want, src, scale, zp)
				assertF32Bits(t, "Dequantize", n, got, want)
			}
		}
	}
}

// TestRequantizeParity sweeps lengths and (multiplier, shift, zeroPoint) combos,
// asserting the public path is bit-exact with the Go reference.
func TestRequantizeParity(t *testing.T) {
	for _, n := range lengths {
		acc := genI32(n, uint32(n)+7)
		if n > 0 {
			acc[0] = math.MaxInt32
			acc[n-1] = math.MinInt32
		}
		for _, mul := range requantizeMuls {
			for _, shift := range requantizeShifts {
				for _, zp := range requantizeZPs {
					got := make([]int8, n)
					want := make([]int8, n)
					Requantize(got, acc, mul, shift, zp)
					requantizeGo(want, acc, mul, shift, zp)
					assertI8Eq(t, "Requantize", n, got, want)
				}
			}
		}
	}
}

// TestQuantizeSemantics is the known-answer table for Quantize: ties round to
// even, NaN maps to zeroPoint, +/-Inf saturate, and tiny/huge scales clamp both
// ways.
func TestQuantizeSemantics(t *testing.T) {
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	cases := []struct {
		name  string
		src   []float32
		scale float32
		zp    int8
		want  []int8
	}{
		{"ties_to_even_scale1", []float32{0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5}, 1.0, 0, []int8{0, 2, 2, 4, 0, -2, -2}},
		{"ties_to_even_scale2", []float32{3, 5, 7, -3, -5, -7}, 2.0, 0, []int8{2, 2, 4, -2, -2, -4}},
		{"nan_to_zeropoint", []float32{nan, nan, nan}, 1.0, 7, []int8{7, 7, 7}},
		{"pos_inf_saturates", []float32{inf, inf}, 1.0, 5, []int8{127, 127}},
		{"neg_inf_saturates", []float32{-inf, -inf}, 1.0, 5, []int8{-128, -128}},
		{"huge_scale_to_zero", []float32{1, -1, 1e9}, 1e30, 0, []int8{0, 0, 0}},
		{"tiny_scale_clamps", []float32{1, -1}, 1e-30, 0, []int8{127, -128}},
		{"zp_max", []float32{0, 1, -1, -255}, 1.0, 127, []int8{127, 127, 126, -128}},
		{"zp_min", []float32{0, 1, 255}, 1.0, -128, []int8{-128, -127, 127}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := make([]int8, len(c.src))
			Quantize(got, c.src, c.scale, c.zp)
			assertI8Eq(t, c.name, len(c.src), got, c.want)
			// The reference must agree with the hand-computed answer too.
			ref := make([]int8, len(c.src))
			quantizeGo(ref, c.src, c.scale, c.zp)
			assertI8Eq(t, c.name+"/ref", len(c.src), ref, c.want)
		})
	}
}

// TestDequantizeSemantics is the known-answer table for Dequantize.
func TestDequantizeSemantics(t *testing.T) {
	cases := []struct {
		name  string
		src   []int8
		scale float32
		zp    int8
		want  []float32
	}{
		{"scale2_zp3", []int8{5, -128, 127}, 2.0, 3, []float32{4, -262, 248}},
		{"scale_half_zp0", []int8{7, -1, 0}, 0.5, 0, []float32{3.5, -0.5, 0}},
		{"zp_min_extremes", []int8{127, -128}, 2.0, -128, []float32{510, 0}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := make([]float32, len(c.src))
			Dequantize(got, c.src, c.scale, c.zp)
			assertF32Bits(t, c.name, len(c.src), got, c.want)
		})
	}
}

// TestRequantizeSemantics is the known-answer table for Requantize, including
// the SRDHM saturation extremes and the round-half-up identity at multiplier =
// 1<<30, shift = 0.
func TestRequantizeSemantics(t *testing.T) {
	const half = int32(1) << 30 // 0.5 in Q31
	cases := []struct {
		name  string
		acc   []int32
		mul   int32
		shift int
		zp    int8
		want  []int8
	}{
		// multiplier 1<<30, shift 0 == round-half-up(acc/2).
		{"half_shift0", []int32{5, 4, -3, -4, 1, -1, 2, 0}, half, 0, 0, []int8{3, 2, -1, -2, 1, 0, 1, 0}},
		// ~1.0 multiplier, shift 0: value passes through, then saturates.
		{"unit_shift0", []int32{100, 127, 200, -200}, 0x7FFFFFFF, 0, 0, []int8{100, 127, 127, -128}},
		// Accumulator extremes with a ~1.0 multiplier saturate.
		{"acc_extremes", []int32{math.MaxInt32, math.MinInt32}, 0x7FFFFFFF, 0, 0, []int8{127, -128}},
		// shift 30 with a zero accumulator is exactly the zero point.
		{"shift30_zero", []int32{0, 0}, half, 30, 5, []int8{5, 5}},
		// zeroPoint offset is added after the rescale.
		{"half_shift0_zp", []int32{5, -3}, half, 0, 10, []int8{13, 9}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := make([]int8, len(c.acc))
			Requantize(got, c.acc, c.mul, c.shift, c.zp)
			assertI8Eq(t, c.name, len(c.acc), got, c.want)
			ref := make([]int8, len(c.acc))
			requantizeGo(ref, c.acc, c.mul, c.shift, c.zp)
			assertI8Eq(t, c.name+"/ref", len(c.acc), ref, c.want)
		})
	}
}

// TestRequantizeCautionA is the mandatory guard for the NEON RoundingDivideByPOT
// identity at right == 0. With shift == 0 the rounding-divide is a pass-through,
// so negative odd pre-divide accumulators must survive unchanged; a naive
// sign-fixup (pre-adding -1 to negatives) would turn -3 into -4. multiplier =
// 1<<30 makes SRDHM(acc) = floor((acc+1)/2), so acc = -6 and -7 both reach a
// pre-divide value of -3, which must stay -3.
func TestRequantizeCautionA(t *testing.T) {
	const half = int32(1) << 30
	acc := []int32{-6, -7, -3, -5, -1, -2, 3, 5}
	want := []int8{-3, -3, -1, -2, 0, -1, 2, 3}

	got := make([]int8, len(acc))
	Requantize(got, acc, half, 0, 0)
	assertI8Eq(t, "Requantize/cautionA", len(acc), got, want)

	// Also assert against the reference so the check is a differential too.
	ref := make([]int8, len(acc))
	requantizeGo(ref, acc, half, 0, 0)
	assertI8Eq(t, "Requantize/cautionA-ref", len(acc), ref, want)

	// Sweep a longer run of negatives at shift == 0 to exercise the SIMD tail as
	// well as the head block.
	for _, n := range []int{8, 15, 16, 17, 31, 33, 64} {
		big := make([]int32, n)
		for i := range big {
			big[i] = int32(-(2*i + 1)) // negative, odd
		}
		g := make([]int8, n)
		w := make([]int8, n)
		Requantize(g, big, half, 0, 3)
		requantizeGo(w, big, half, 0, 3)
		assertI8Eq(t, "Requantize/cautionA-sweep", n, g, w)
	}
}

// TestQuantizeCautionB proves the int8 pack keeps lane order: a distinct,
// in-range ascending ramp must come out in the same order. A cross-lane
// permutation bug in the AVX2 VPACKSSDW/VPERMQ/VPACKSSWB sequence would scramble
// it. The lengths straddle the 8/16-wide pack boundaries.
func TestQuantizeCautionB(t *testing.T) {
	for _, n := range []int{8, 15, 16, 17, 31, 32, 33, 64, 96} {
		base := -n / 2 // center the ramp so every value is a distinct int8
		src := make([]float32, n)
		want := make([]int8, n)
		for i := range src {
			v := base + i
			src[i] = float32(v)
			want[i] = int8(v)
		}
		got := make([]int8, n)
		Quantize(got, src, 1.0, 0)
		assertI8Eq(t, "Quantize/cautionB", n, got, want)
	}
}

// TestQuantizeZeroAllocations asserts the three new functions allocate nothing.
func TestQuantizeZeroAllocations(t *testing.T) {
	const n = 1024
	f := genF32(n, 5)
	q := genI8(n, 6)
	acc := genI32(n, 8)
	d8 := make([]int8, n)
	d32 := make([]float32, n)

	checks := []struct {
		name string
		fn   func()
	}{
		{"Quantize", func() { Quantize(d8, f, 0.5, 3) }},
		{"Dequantize", func() { Dequantize(d32, q, 0.5, 3) }},
		{"Requantize", func() { Requantize(d8, acc, 0x40000000, -2, 3) }},
	}
	for _, c := range checks {
		if got := testing.AllocsPerRun(10, c.fn); got != 0 {
			t.Errorf("%s allocated %v times per run, want 0", c.name, got)
		}
	}
}

// TestQuantizeTrailingCapacity verifies each function writes exactly n elements
// and leaves trailing dst capacity untouched, at sizes above and below the SIMD
// dispatch thresholds.
func TestQuantizeTrailingCapacity(t *testing.T) {
	for _, n := range []int{3, 20, 40} {
		f := genF32(n, 2)
		q := genI8(n, 4)
		acc := genI32(n, 9)

		d8 := fillI8(n+2, 42)
		Quantize(d8[:n], f, 0.5, 1)
		if d8[n] != 42 || d8[n+1] != 42 {
			t.Errorf("Quantize (n=%d) clobbered trailing capacity: %v", n, d8[n:])
		}

		d8 = fillI8(n+2, 42)
		Requantize(d8[:n], acc, 0x40000000, -1, 1)
		if d8[n] != 42 || d8[n+1] != 42 {
			t.Errorf("Requantize (n=%d) clobbered trailing capacity: %v", n, d8[n:])
		}

		d32 := make([]float32, n+2)
		d32[n], d32[n+1] = 42, 42
		Dequantize(d32[:n], q, 0.5, 1)
		if d32[n] != 42 || d32[n+1] != 42 {
			t.Errorf("Dequantize (n=%d) clobbered trailing capacity: %v", n, d32[n:])
		}
	}
}

// assertF32Bits compares two float32 slices bit-for-bit, treating two NaNs as
// equal regardless of payload (a hardware multiply and Go's may differ there).
func assertF32Bits(t *testing.T, op string, n int, got, want []float32) {
	t.Helper()
	for i := range n {
		g, w := got[i], want[i]
		if g != g && w != w { // both NaN
			continue
		}
		if math.Float32bits(g) != math.Float32bits(w) {
			t.Fatalf("%s[%d] = %v (0x%08x), want %v (0x%08x) (len=%d)",
				op, i, g, math.Float32bits(g), w, math.Float32bits(w), n)
		}
	}
}

// --- Differential fuzzers (public path vs Go reference) ---

// f32sFromBytes reinterprets raw bytes as float32s via Float32frombits, so NaN,
// Inf and subnormals all appear.
func f32sFromBytes(raw []byte) []float32 {
	out := make([]float32, len(raw)/4)
	for i := range out {
		b := raw[i*4:]
		bits := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}

// i32sFromBytes reinterprets raw bytes as int32s.
func i32sFromBytes(raw []byte) []int32 {
	out := make([]int32, len(raw)/4)
	for i := range out {
		b := raw[i*4:]
		out[i] = int32(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
	}
	return out
}

func lenSeeds4(f *testing.F) {
	f.Helper()
	lens := []int{0, 4, 8, 12, 28, 32, 36, 60, 64, 68, 128, 260, 512}
	for _, n := range lens {
		raw := make([]byte, n)
		for i := range raw {
			raw[i] = byte(i*37 + 11)
		}
		f.Add(raw, uint32(0x40000000), byte(3))
	}
}

func FuzzI8Quantize(f *testing.F) {
	lenSeeds4(f)
	f.Fuzz(func(t *testing.T, raw []byte, scaleBits uint32, zpByte byte) {
		src := f32sFromBytes(raw)
		scale := math.Float32frombits(scaleBits)
		zp := int8(zpByte)
		got := make([]int8, len(src))
		want := make([]int8, len(src))
		Quantize(got, src, scale, zp)
		quantizeGo(want, src, scale, zp)
		for i := range src {
			if got[i] != want[i] {
				t.Fatalf("Quantize[%d] = %d, want %d (src=%v scale=%v zp=%d len=%d)",
					i, got[i], want[i], src[i], scale, zp, len(src))
			}
		}
	})
}

func FuzzI8Dequantize(f *testing.F) {
	lenSeeds4(f)
	f.Fuzz(func(t *testing.T, raw []byte, scaleBits uint32, zpByte byte) {
		src := make([]int8, len(raw))
		for i, b := range raw {
			src[i] = int8(b)
		}
		scale := math.Float32frombits(scaleBits)
		zp := int8(zpByte)
		got := make([]float32, len(src))
		want := make([]float32, len(src))
		Dequantize(got, src, scale, zp)
		dequantizeGo(want, src, scale, zp)
		assertF32Bits(t, "Dequantize", len(src), got, want)
	})
}

func FuzzI8Requantize(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8}, int32(0x40000000), 0, byte(3))
	f.Add(make([]byte, 64), int32(0x7FFFFFFF), -8, byte(0))
	f.Add(make([]byte, 33*4), int32(0x60000000), 5, byte(127))
	f.Fuzz(func(t *testing.T, raw []byte, mul int32, shift int, zpByte byte) {
		acc := i32sFromBytes(raw)
		zp := int8(zpByte)
		got := make([]int8, len(acc))
		want := make([]int8, len(acc))
		Requantize(got, acc, mul, shift, zp)
		requantizeGo(want, acc, mul, shift, zp)
		for i := range acc {
			if got[i] != want[i] {
				t.Fatalf("Requantize[%d] = %d, want %d (acc=%d mul=%d shift=%d zp=%d len=%d)",
					i, got[i], want[i], acc[i], mul, shift, zp, len(acc))
			}
		}
	})
}
