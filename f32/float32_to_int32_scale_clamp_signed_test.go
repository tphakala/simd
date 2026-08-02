package f32

import (
	"math"
	"testing"
)

// float32ToInt32ScaleClampSignedRef is an independent oracle for
// Float32ToInt32ScaleClampSigned. The magnitude path uses float64 intermediates
// (each float32(...) cast rounds exactly once, reproducing the two-rounding
// no-FMA contract by a route the compiler cannot fuse), and the sign is applied
// with math.Copysign rather than the production bit manipulation, so a bug in
// either half is caught by a genuinely different computation.
func float32ToInt32ScaleClampSignedRef(dst []int32, mag, sign []float32, scale, offset, minV, maxV float32) {
	n := min(len(dst), len(mag), len(sign))
	for i := range n {
		p := float32(float64(mag[i]) * float64(scale)) // product, rounded to float32
		v := float32(float64(p) + float64(offset))     // + offset, rounded to float32
		if v != v {                                    // NaN magnitude -> 0 (sign not applied)
			dst[i] = 0
			continue
		}
		if v < minV { // max(v, minV)
			v = minV
		}
		if v > maxV { // min(v, maxV); inverted bounds (minV>maxV) yield maxV
			v = maxV
		}
		// copysign in the float domain, before the truncating conversion.
		v = float32(math.Copysign(float64(v), float64(sign[i])))
		dst[i] = int32(v) // truncate toward zero
	}
}

// signRamp32 returns a deterministic sign source: alternating and drifting signs,
// including exact zeros, so lanes exercise both sign bits at every residue.
func signRamp32(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i%7)-3) * 0.5 // -1.5..1.5, hits 0 at i%7==3
	}
	return s
}

func TestFloat32ToInt32ScaleClampSigned(t *testing.T) {
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	negZero := float32(math.Copysign(0, -1))
	negNaN := math.Float32frombits(math.Float32bits(nan) | (1 << 31)) // NaN with bit 31 set
	cases := []struct {
		name                      string
		mag, sign                 []float32
		scale, offset, minV, maxV float32
	}{
		{"empty", nil, nil, 2.0, 0.5, -100, 100},
		{"single_pos", []float32{1.5}, []float32{3.0}, 2.0, 0.4, -100, 100},
		{"single_neg", []float32{1.5}, []float32{-3.0}, 2.0, 0.4, -100, 100},
		{"four", []float32{1, 0.5, 2, 3}, []float32{-1, 1, -1, 1}, 3.0, 0.25, -100, 100},
		{"eight", []float32{2, 1, 0.25, 0, 0.25, 1, 2, 3.5}, []float32{-5, 5, -5, 5, -5, 5, -5, 5}, 4.0, 0.4054, -1000, 1000},
		{"nine", float32Ramp32(9), signRamp32(9), 7.0, 0.5, -5000, 5000},
		{"block16", float32Ramp32(16), signRamp32(16), 7.0, 0.5, -5000, 5000},
		{"block17", float32Ramp32(17), signRamp32(17), 7.0, 0.5, -5000, 5000},
		{"residue_31", float32Ramp32(31), signRamp32(31), 3.0, 0.1, -2000, 2000},
		// A negative magnitude source with a positive sign: copysign abs's the
		// magnitude first, so the result is positive (this is what distinguishes
		// copysign from a plain conditional negate).
		{"neg_mag_pos_sign", []float32{-4, -4, -4, -4}, []float32{1, 1, 1, 1}, 1.0, 0.0, -100, 100},
		{"neg_mag_neg_sign", []float32{-4, -4, -4, -4}, []float32{-1, -1, -1, -1}, 1.0, 0.0, -100, 100},
		// Negative-zero sign bit must produce a negative result (bit 31, not < 0).
		{"neg_zero_sign", []float32{5, 5, 5, 5}, []float32{negZero, negZero, negZero, negZero}, 1.0, 0.0, -100, 100},
		// Zero magnitude with a negative sign: copysign(0) stays 0 as int32.
		{"zero_mag_neg_sign", []float32{0, 0, 0, 0}, []float32{-1, -1, -1, -1}, 1.0, 0.0, -100, 100},
		// +Inf -> maxV, -Inf -> minV in the magnitude, then signed; NaN magnitude -> 0.
		{"inf_nan_mag", []float32{inf, -inf, nan, 5, -5, nan, inf, -inf}, []float32{-1, -1, -1, -1, 1, 1, 1, 1}, 1.0, 0.0, -100, 100},
		// NaN magnitude -> 0 regardless of sign, even when the clamp range excludes 0.
		{"nan_mag_excl_zero", []float32{nan, 50, nan, 20}, []float32{-1, -1, -1, -1}, 1.0, 0.0, 10, 100},
		// NaN sign source: only its sign bit is read (positive NaN -> positive).
		{"nan_sign", []float32{5, 5, 5, 5}, []float32{nan, nan, nan, nan}, 1.0, 0.0, -100, 100},
		// Negative NaN sign source: bit 31 is set, so the result is negative.
		{"neg_nan_sign", []float32{5, 5, 5, 5}, []float32{negNaN, negNaN, negNaN, negNaN}, 1.0, 0.0, -100, 100},
		// Inverted bounds (minV > maxV): every clamped value is maxV, then signed.
		{"inverted", []float32{-100, 0, 100, 5}, []float32{-1, 1, -1, 1}, 1.0, 0.0, 50, 10},
		// Truncation toward zero on both signs.
		{"trunc", []float32{0.7, 1.5, 2.999, 2.999, 0.999, 0.001}, []float32{-1, -1, 1, -1, 1, -1}, 1.0, 0.0, -1000, 1000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dst := make([]int32, len(tc.mag))
			Float32ToInt32ScaleClampSigned(dst, tc.mag, tc.sign, tc.scale, tc.offset, tc.minV, tc.maxV)
			want := make([]int32, len(tc.mag))
			float32ToInt32ScaleClampSignedRef(want, tc.mag, tc.sign, tc.scale, tc.offset, tc.minV, tc.maxV)
			for i := range dst {
				if dst[i] != want[i] {
					t.Errorf("[%d] = %d, want %d (mag=%g sign=%g scale=%g offset=%g minV=%g maxV=%g)",
						i, dst[i], want[i], tc.mag[i], tc.sign[i], tc.scale, tc.offset, tc.minV, tc.maxV)
				}
			}
		})
	}
}

// TestFloat32ToInt32ScaleClampSignedGo exercises the pure-Go fallback directly
// against the oracle, catching a compiler that fuses the reference's magnitude
// multiply-add into a single-rounding FMA.
func TestFloat32ToInt32ScaleClampSignedGo(t *testing.T) {
	mag := float32Ramp32(200)
	sign := signRamp32(200)
	for i := range 7 { // a few offset-boundary values with alternating signs
		mag = append(mag, float32(i)+0.4054)
		sign = append(sign, float32(1-2*(i&1)))
	}
	dst := make([]int32, len(mag))
	want := make([]int32, len(mag))
	float32ToInt32ScaleClampSignedGo(dst, mag, sign, 4097.0, 0.4054, -1e9, 1e9)
	float32ToInt32ScaleClampSignedRef(want, mag, sign, 4097.0, 0.4054, -1e9, 1e9)
	for i := range dst {
		if dst[i] != want[i] {
			t.Fatalf("Go[%d] = %d, want %d (mag=%g sign=%g)", i, dst[i], want[i], mag[i], sign[i])
		}
	}
}

// TestFloat32ToInt32ScaleClampSigned_NoFMA is an FMA detector on the magnitude
// path, identical in construction to the unsigned kernel's detector: with
// mag=scale=4097 and offset=-(2^24+2^13) the correct two-rounding magnitude is 0,
// so every lane must be 0; a fused multiply-add would yield a non-zero magnitude.
func TestFloat32ToInt32ScaleClampSigned_NoFMA(t *testing.T) {
	const offset = -16785408.0 // -(2^24 + 2^13)
	mag := make([]float32, 12) // > 8 so the AVX body runs, includes a residue
	sign := make([]float32, 12)
	for i := range mag {
		mag[i] = 4097.0
		sign[i] = -1.0 // negative sign; copysign(0) is still 0, so lanes stay 0
	}
	dst := make([]int32, len(mag))
	Float32ToInt32ScaleClampSigned(dst, mag, sign, 4097.0, offset, -1000, 1000)
	for i := range dst {
		if dst[i] != 0 {
			t.Fatalf("[%d] = %d, want 0 (non-zero means the magnitude multiply and add fused into an FMA)", i, dst[i])
		}
	}
}

// TestFloat32ToInt32ScaleClampSigned_TierParity asserts the dispatched SIMD path
// is bit-identical to the pure-Go reference across every 8-wide AVX and 4-wide
// NEON residue. The "no relaxed tier" contract: AVX == NEON == Go, exactly.
func TestFloat32ToInt32ScaleClampSigned_TierParity(t *testing.T) {
	for n := 1; n <= 40; n++ {
		mag := make([]float32, n)
		sign := make([]float32, n)
		for i := range mag {
			mag[i] = float32(i%97-48) * 0.37
			sign[i] = float32((i*13)%11 - 5) // spans both sign bits and zero
		}
		got := make([]int32, n)
		want := make([]int32, n)
		Float32ToInt32ScaleClampSigned(got, mag, sign, 3.0, 0.4054, -5000, 5000)
		float32ToInt32ScaleClampSignedGo(want, mag, sign, 3.0, 0.4054, -5000, 5000)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("n=%d [%d]: SIMD=%d Go=%d (mag=%g sign=%g)", n, i, got[i], want[i], mag[i], sign[i])
			}
		}
	}
}

// TestFloat32ToInt32ScaleClampSigned_ContractEdgeSIMD drives the extreme
// clean-conversion int32 bounds through the vector body (len >= 8) with signs.
// The signed kernel is a SEPARATE assembly function from the unsigned one, so the
// sibling's edge test does not cover its abs-then-convert path. minV/maxV are the
// largest magnitudes that stay bit-identical across tiers (|bound| <= 2147483520,
// per the doc): a floor-clamp with a positive sign abs's to +2147483520, which
// must convert to exactly that rather than overflowing to 0x80000000 / saturating.
func TestFloat32ToInt32ScaleClampSigned_ContractEdgeSIMD(t *testing.T) {
	const minV, maxV = -2147483520.0, 2147483520.0
	// Values that clamp to the floor, the ceiling, and land mid-range, spanning
	// past the 8-wide AVX block so the vector path and its overlap tail both run.
	mag := []float32{-5e9, 5e9, -5e9, 5e9, 1000, -1000, 0, 2147483520, -2147483520, 3}
	sign := []float32{1, 1, -1, -1, 1, -1, -1, 1, 1, -1}
	dst := make([]int32, len(mag))
	Float32ToInt32ScaleClampSigned(dst, mag, sign, 1.0, 0.0, minV, maxV)
	want := make([]int32, len(mag))
	float32ToInt32ScaleClampSignedRef(want, mag, sign, 1.0, 0.0, minV, maxV)
	for i := range dst {
		if dst[i] != want[i] {
			t.Fatalf("[%d] = %d, want %d (mag=%g sign=%g)", i, dst[i], want[i], mag[i], sign[i])
		}
	}
	// The exact edge case from the contract note: floor-clamp (mag=-5e9 -> minV)
	// with a positive sign abs's to +2147483520 and must NOT overflow.
	if dst[0] != 2147483520 {
		t.Fatalf("floor-clamp with positive sign = %d, want 2147483520 (overflow to 0x80000000?)", dst[0])
	}
	if dst[2] != -2147483520 { // same floor, negative sign, stays at the clean floor
		t.Fatalf("floor-clamp with negative sign = %d, want -2147483520", dst[2])
	}
}

// TestFloat32ToInt32ScaleClampSigned_MatchesUnsignedPlusSign ties the primitive
// to its documented role as the signed sibling of Float32ToInt32ScaleClamp: the
// result must equal the unsigned magnitude quantization with a copysign applied.
// Because truncation toward zero commutes with the sign, copysign(int32(clamp),
// sign) equals the abs of the unsigned int32 carrying sign[i]'s sign bit.
func TestFloat32ToInt32ScaleClampSigned_MatchesUnsignedPlusSign(t *testing.T) {
	n := 33
	mag := make([]float32, n)
	sign := make([]float32, n)
	for i := range mag {
		mag[i] = float32(i%50) * 0.5 // non-negative magnitudes (the rectified common case)
		sign[i] = float32((i%5)-2) * 3
	}
	const scale, offset, minV, maxV = 2.0, 0.4054, 0.0, 1000.0

	got := make([]int32, n)
	Float32ToInt32ScaleClampSigned(got, mag, sign, scale, offset, minV, maxV)

	unsigned := make([]int32, n)
	Float32ToInt32ScaleClamp(unsigned, mag, scale, offset, minV, maxV)
	for i := range got {
		a := unsigned[i]
		if a < 0 {
			a = -a
		}
		want := a
		if math.Signbit(float64(sign[i])) {
			want = -a
		}
		if got[i] != want {
			t.Fatalf("[%d] signed=%d, unsigned=%d sign=%g -> want %d", i, got[i], unsigned[i], sign[i], want)
		}
	}
}

// TestFloat32ToInt32ScaleClampSigned_Consumer covers the motivating pattern
// (go-aac quantize_bands): a magnitude computed from a rectified value, with the
// original sample's sign reattached. dst[i] == copysign(quantize(|x[i]|), x[i]).
func TestFloat32ToInt32ScaleClampSigned_Consumer(t *testing.T) {
	negZero := float32(math.Copysign(0, -1)) // a real -0.0 (the literal -0.0 is +0.0 in Go)
	x := []float32{-3.2, 3.2, -0.4, 0.4, -100, 100, 0, negZero, 7.7, -7.7, 1e6, -1e6}
	mag := make([]float32, len(x))
	for i, v := range x {
		mag[i] = float32(math.Abs(float64(v))) // rectified magnitude source
	}
	const scale, offset, minV, maxV = 0.5, 0.4054, 0.0, 5000.0

	dst := make([]int32, len(x))
	Float32ToInt32ScaleClampSigned(dst, mag, x, scale, offset, minV, maxV)
	for i, v := range x {
		q := float32(float64(mag[i])*scale) + offset
		if q > maxV {
			q = maxV
		}
		mI := int32(q) // magnitude is non-negative here
		want := mI
		if math.Signbit(float64(v)) {
			want = -mI
		}
		if dst[i] != want {
			t.Fatalf("[%d] x=%g: got %d want %d", i, v, dst[i], want)
		}
	}
}

func TestFloat32ToInt32ScaleClampSigned_AllocFree(t *testing.T) {
	mag := float32Ramp32(512)
	sign := signRamp32(512)
	dst := make([]int32, len(mag))
	if a := testing.AllocsPerRun(10, func() {
		Float32ToInt32ScaleClampSigned(dst, mag, sign, 3.0, 0.4054, -5000, 5000)
	}); a != 0 {
		t.Fatalf("safe: allocations = %v, want 0", a)
	}
	if a := testing.AllocsPerRun(10, func() {
		Float32ToInt32ScaleClampSignedUnsafe(dst, mag, sign, 3.0, 0.4054, -5000, 5000)
	}); a != 0 {
		t.Fatalf("unsafe: allocations = %v, want 0", a)
	}
}

// TestFloat32ToInt32ScaleClampSigned_LengthMismatch processes exactly
// min(len(dst), len(mag), len(sign)) elements and leaves the dst tail untouched.
func TestFloat32ToInt32ScaleClampSigned_LengthMismatch(t *testing.T) {
	mag := []float32{1, 2, 3, 4, 5}
	sign := []float32{-1, 1, -1, 1} // shorter than mag
	dst := make([]int32, 6)         // longer than mag
	for i := range dst {
		dst[i] = -999 // sentinel
	}
	Float32ToInt32ScaleClampSigned(dst, mag, sign, 2.0, 0.5, -1000, 1000)
	// n = min(6, 5, 4) = 4 elements written.
	want := make([]int32, 4)
	float32ToInt32ScaleClampSignedRef(want, mag[:4], sign[:4], 2.0, 0.5, -1000, 1000)
	for i := range 4 {
		if dst[i] != want[i] {
			t.Fatalf("[%d] = %d, want %d", i, dst[i], want[i])
		}
	}
	if dst[4] != -999 || dst[5] != -999 {
		t.Fatalf("tail overwritten: dst[4]=%d dst[5]=%d, want sentinel -999", dst[4], dst[5])
	}
}

func TestFloat32ToInt32ScaleClampSignedUnsafe(t *testing.T) {
	mag := float32Ramp32(37)
	sign := signRamp32(37)
	dst := make([]int32, len(mag))
	safe := make([]int32, len(mag))
	Float32ToInt32ScaleClampSignedUnsafe(dst, mag, sign, 3.0, 0.4, -1500, 1500)
	Float32ToInt32ScaleClampSigned(safe, mag, sign, 3.0, 0.4, -1500, 1500)
	for i := range dst {
		if dst[i] != safe[i] {
			t.Fatalf("[%d] unsafe=%d safe=%d", i, dst[i], safe[i])
		}
	}
}

// FuzzFloat32ToInt32ScaleClampSigned differentially fuzzes the dispatched SIMD
// path against the pure-Go reference: any divergence (magnitude path, clamp, sign
// bit, residue handling) fails.
func FuzzFloat32ToInt32ScaleClampSigned(f *testing.F) {
	f.Add(uint64(0x1234567), 17, float32(3.0), float32(0.4054), float32(-5000), float32(5000))
	f.Add(uint64(0xdeadbeef), 40, float32(1.0), float32(0.0), float32(-100), float32(100))
	f.Add(uint64(1), math.MinInt, float32(1.0), float32(0.0), float32(-100), float32(100)) // regression: n=MinInt must not panic in normalization
	f.Fuzz(func(t *testing.T, seed uint64, n int, scale, offset, minV, maxV float32) {
		// Bound n FIRST, then take the absolute value: negating before bounding
		// would overflow on n == math.MinInt (-math.MinInt is still math.MinInt),
		// leaving a negative length that panics make. After %300, |n| < 300.
		n %= 300
		if n < 0 {
			n = -n
		}
		// The signed kernel abs's the clamped magnitude before the truncating
		// conversion, so cross-tier bit-identity needs max(|minV|,|maxV|) <=
		// 2147483520 (tighter than the unsigned kernel; see the doc). NaN bounds are
		// out of contract too. Outside this window the tiers legitimately diverge
		// (x86 integer-indefinite vs ARM64 saturation), so do not compare there.
		const cleanMax = 2147483520.0
		if minV != minV || maxV != maxV ||
			math.Abs(float64(minV)) > cleanMax || math.Abs(float64(maxV)) > cleanMax {
			t.Skip("minV/maxV out of the signed clean-conversion range")
		}
		mag := make([]float32, n)
		sign := make([]float32, n)
		r := seed | 1
		next := func() float32 {
			r ^= r << 13
			r ^= r >> 7
			r ^= r << 17
			return math.Float32frombits(uint32(r))
		}
		for i := range mag {
			mag[i] = next()
			sign[i] = next()
		}
		got := make([]int32, n)
		want := make([]int32, n)
		Float32ToInt32ScaleClampSigned(got, mag, sign, scale, offset, minV, maxV)
		float32ToInt32ScaleClampSignedGo(want, mag, sign, scale, offset, minV, maxV)
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("n=%d [%d]: SIMD=%d Go=%d (mag=%g sign=%g scale=%g offset=%g minV=%g maxV=%g)",
					n, i, got[i], want[i], mag[i], sign[i], scale, offset, minV, maxV)
			}
		}
	})
}

func BenchmarkFloat32ToInt32ScaleClampSigned(b *testing.B) {
	mag := float32Ramp32(4096)
	sign := signRamp32(4096)
	dst := make([]int32, len(mag))
	b.ReportAllocs()
	b.SetBytes(int64(len(mag) * 4))
	for b.Loop() {
		Float32ToInt32ScaleClampSigned(dst, mag, sign, 3.0, 0.4054, -5000, 5000)
	}
}

// BenchmarkFloat32ToInt32ScaleClampSigned_TwoPass is the composition the fused
// kernel replaces: quantize with wide vector stores, then a scalar pass reloading
// dst with narrow int32 loads to negate. It exists to substantiate that the fused
// single-pass form avoids the store-to-load stall.
func BenchmarkFloat32ToInt32ScaleClampSigned_TwoPass(b *testing.B) {
	mag := float32Ramp32(4096)
	sign := signRamp32(4096)
	dst := make([]int32, len(mag))
	b.ReportAllocs()
	b.SetBytes(int64(len(mag) * 4))
	for b.Loop() {
		Float32ToInt32ScaleClamp(dst, mag, 3.0, 0.4054, -5000, 5000)
		for i := range dst {
			if math.Signbit(float64(sign[i])) {
				dst[i] = -dst[i]
			}
		}
	}
}
