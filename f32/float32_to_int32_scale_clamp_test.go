package f32

import (
	"math"
	"testing"
)

// float32ToInt32ScaleClampRef is an independent oracle for
// Float32ToInt32ScaleClamp. It uses float64 intermediates: a float32 operand is
// exact in float64, and each float32(...) cast rounds exactly once, so this
// computes one float32 multiply-rounding then one float32 add-rounding (the
// two-rounding, no-FMA contract) by a different route than the production Go
// reference, and the float64 form cannot be compiler-fused. The NaN/clamp/
// truncate logic matches the kernels (NaN -> 0; FMAX(minV) then FMIN(maxV); trunc
// toward zero).
func float32ToInt32ScaleClampRef(dst []int32, src []float32, scale, offset, minV, maxV float32) {
	n := min(len(dst), len(src))
	for i := range n {
		p := float32(float64(src[i]) * float64(scale)) // product, rounded to float32
		v := float32(float64(p) + float64(offset))     // + offset, rounded to float32
		if v != v {                                    // NaN -> 0
			dst[i] = 0
			continue
		}
		if v < minV { // max(v, minV)
			v = minV
		}
		if v > maxV { // min(v, maxV); inverted bounds (minV>maxV) yield maxV, as on the kernels
			v = maxV
		}
		dst[i] = int32(v) // truncate toward zero
	}
}

func float32Ramp32(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(i%2001-1000) * 0.5
	}
	return s
}

func TestFloat32ToInt32ScaleClamp(t *testing.T) {
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	cases := []struct {
		name                      string
		src                       []float32
		scale, offset, minV, maxV float32
	}{
		{"empty", nil, 2.0, 0.5, -100, 100},
		{"single", []float32{1.5}, 2.0, 0.4, -100, 100},
		{"four", []float32{-1, -0.5, 0.5, 1}, 3.0, 0.25, -100, 100},
		{"eight", []float32{-2, -1, -0.25, 0, 0.25, 1, 2, 3.5}, 4.0, 0.4054, -1000, 1000},
		{"nine", float32Ramp32(9), 7.0, 0.5, -5000, 5000},
		{"block16", float32Ramp32(16), 7.0, 0.5, -5000, 5000},
		{"block17", float32Ramp32(17), 7.0, 0.5, -5000, 5000},
		{"residue_31", float32Ramp32(31), 3.0, 0.1, -2000, 2000},
		// Clamp both ends.
		{"clamp_both", []float32{-1e6, 1e6, -50, 50, 200, -200, 0, 0.5}, 1.0, 0.0, -100, 100},
		// Negative truncation toward zero (scale 1, offset 0 so v == src).
		{"trunc", []float32{-0.7, -1.5, -2.999, 2.999, 0.999, -0.001}, 1.0, 0.0, -1000, 1000},
		// +Inf -> maxV, -Inf -> minV, NaN -> 0 (with bounds that span 0).
		{"inf_nan_span0", []float32{inf, -inf, nan, 5, -5, nan, inf, -inf}, 1.0, 0.0, -100, 100},
		// NaN -> 0 even when the clamp range EXCLUDES zero (distinguishes NaN->0
		// from NaN->clamp(0); the latter would give 10 here).
		{"nan_excl_zero", []float32{nan, 50, nan, 20}, 1.0, 0.0, 10, 100},
		// Inverted bounds: every result is int32(maxV).
		{"inverted", []float32{-100, 0, 100, 5}, 1.0, 0.0, 50, 10},
		// Contract-edge bounds with huge inputs.
		{"contract_edge", []float32{5e9, -5e9, inf, -inf, 1.5}, 1.0, 0.0, -2147483648.0, 2147483520.0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dst := make([]int32, len(tc.src))
			Float32ToInt32ScaleClamp(dst, tc.src, tc.scale, tc.offset, tc.minV, tc.maxV)
			want := make([]int32, len(tc.src))
			float32ToInt32ScaleClampRef(want, tc.src, tc.scale, tc.offset, tc.minV, tc.maxV)
			for i := range dst {
				if dst[i] != want[i] {
					t.Errorf("[%d] = %d, want %d (src=%g scale=%g offset=%g minV=%g maxV=%g)",
						i, dst[i], want[i], tc.src[i], tc.scale, tc.offset, tc.minV, tc.maxV)
				}
			}
		})
	}
}

// TestFloat32ToInt32ScaleClampGo exercises the pure-Go fallback directly against
// the oracle. This is the test that catches the Go compiler fusing the reference
// itself into an FMADDS on arm64 (which would round once instead of twice).
func TestFloat32ToInt32ScaleClampGo(t *testing.T) {
	src := float32Ramp32(200)
	for i := range 7 {
		src = append(src, float32(i)+0.4054) // a few offset-boundary values
	}
	dst := make([]int32, len(src))
	want := make([]int32, len(src))
	float32ToInt32ScaleClampGo(dst, src, 4097.0, 0.4054, -1e9, 1e9)
	float32ToInt32ScaleClampRef(want, src, 4097.0, 0.4054, -1e9, 1e9)
	for i := range dst {
		if dst[i] != want[i] {
			t.Fatalf("Go[%d] = %d, want %d (src=%g)", i, dst[i], want[i], src[i])
		}
	}
}

// TestFloat32ToInt32ScaleClamp_NoFMA is an FMA detector. With
// src=scale=4097 and offset=-(2^24+2^13), the true product 4097*4097 =
// 16785409 rounds to float32 16785408 (ties-to-even), so the correct
// two-rounding result is (16785408 + offset) = 0. A fused multiply-add keeps the
// unrounded 16785409 and yields 1. Every dispatched path must return 0.
func TestFloat32ToInt32ScaleClamp_NoFMA(t *testing.T) {
	const offset = -16785408.0 // -(2^24 + 2^13)
	src := make([]float32, 12) // > 8 so the AVX body runs, includes a residue
	for i := range src {
		src[i] = 4097.0
	}
	dst := make([]int32, len(src))
	Float32ToInt32ScaleClamp(dst, src, 4097.0, offset, -1000, 1000)
	for i := range dst {
		if dst[i] != 0 {
			t.Fatalf("[%d] = %d, want 0 (a non-zero result means the multiply and add fused into an FMA)", i, dst[i])
		}
	}
}

func TestFloat32ToInt32ScaleClamp_Large(t *testing.T) {
	// Lengths cover every 8-wide AVX residue (0..7: e.g. 10->2, 11->3, 14->6) and
	// every 4-wide NEON residue, plus multiples that skip the overlap tail.
	for _, n := range []int{4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 31, 32, 33, 63, 96, 1000, 4096} {
		src := make([]float32, n)
		for i := range src {
			// Sweep so some values clamp at each end.
			src[i] = float32(i%400-200) * 12.0
		}
		dst := make([]int32, n)
		want := make([]int32, n)
		Float32ToInt32ScaleClamp(dst, src, 3.0, 0.4054, -2000, 2000)
		float32ToInt32ScaleClampGo(want, src, 3.0, 0.4054, -2000, 2000)
		for i := range dst {
			if dst[i] != want[i] {
				t.Fatalf("n=%d [%d] = %d, want %d", n, i, dst[i], want[i])
			}
		}
	}
}

// TestFloat32ToInt32ScaleClamp_ContractEdgeSIMD drives the max/min contract
// bounds (2147483520 = 2^31-128, -2147483648) through the vector body at len >= 8
// (the table's contract_edge case is 5 elements and takes the scalar path on
// amd64), so the CVTTPS2DQ / FCVTZS of a value clamped to the extreme bound is
// exercised on the SIMD path itself.
func TestFloat32ToInt32ScaleClamp_ContractEdgeSIMD(t *testing.T) {
	const maxV = 2147483520.0
	const minV = -2147483648.0
	inf := float32(math.Inf(1))
	src := make([]float32, 40)
	for i := range src {
		switch i % 4 {
		case 0:
			src[i] = 1e30 // huge -> clamps to maxV
		case 1:
			src[i] = -1e30 // -> minV
		case 2:
			src[i] = inf // -> maxV
		default:
			src[i] = float32(i) - 20 // in-range
		}
	}
	dst := make([]int32, len(src))
	want := make([]int32, len(src))
	Float32ToInt32ScaleClamp(dst, src, 1.0, 0.0, minV, maxV)
	float32ToInt32ScaleClampRef(want, src, 1.0, 0.0, minV, maxV)
	for i := range dst {
		if dst[i] != want[i] {
			t.Fatalf("[%d] = %d, want %d (src=%g)", i, dst[i], want[i], src[i])
		}
	}
	// The clamped-to-max lanes must be exactly int32(2147483520), not the
	// integer-indefinite 0x80000000.
	if dst[0] != int32(maxV) {
		t.Errorf("clamp-to-maxV lane = %d, want %d", dst[0], int32(maxV))
	}
}

func TestFloat32ToInt32ScaleClamp_LengthMismatch(t *testing.T) {
	// dst shorter than src: only len(dst) elements processed, src not over-read.
	src := float32Ramp32(20)
	dst := make([]int32, 12)
	Float32ToInt32ScaleClamp(dst, src, 2.0, 0.5, -1000, 1000)
	want := make([]int32, 12)
	float32ToInt32ScaleClampRef(want, src[:12], 2.0, 0.5, -1000, 1000)
	for i := range dst {
		if dst[i] != want[i] {
			t.Fatalf("[%d] = %d, want %d", i, dst[i], want[i])
		}
	}
	// src shorter than dst: dst tail untouched.
	src = float32Ramp32(10)
	dst = make([]int32, 16)
	for i := range dst {
		dst[i] = -12345 // sentinel
	}
	Float32ToInt32ScaleClamp(dst, src, 2.0, 0.5, -1000, 1000)
	for i := 10; i < 16; i++ {
		if dst[i] != -12345 {
			t.Errorf("dst[%d] = %d, want untouched sentinel -12345", i, dst[i])
		}
	}
}

func TestFloat32ToInt32ScaleClampUnsafe(t *testing.T) {
	src := float32Ramp32(40)
	dst := make([]int32, len(src))
	safe := make([]int32, len(src))
	Float32ToInt32ScaleClampUnsafe(dst, src, 3.0, 0.4, -1500, 1500)
	Float32ToInt32ScaleClamp(safe, src, 3.0, 0.4, -1500, 1500)
	for i := range dst {
		if dst[i] != safe[i] {
			t.Fatalf("Unsafe[%d] = %d, want %d", i, dst[i], safe[i])
		}
	}
}

func TestFloat32ToInt32ScaleClamp_AllocFree(t *testing.T) {
	src := float32Ramp32(256)
	dst := make([]int32, len(src))
	if got := testing.AllocsPerRun(100, func() {
		Float32ToInt32ScaleClamp(dst, src, 3.0, 0.4, -2000, 2000)
	}); got != 0 {
		t.Errorf("Float32ToInt32ScaleClamp allocated %v times per run, want 0", got)
	}
	if got := testing.AllocsPerRun(100, func() {
		Float32ToInt32ScaleClampUnsafe(dst, src, 3.0, 0.4, -2000, 2000)
	}); got != 0 {
		t.Errorf("Float32ToInt32ScaleClampUnsafe allocated %v times per run, want 0", got)
	}
}

func FuzzFloat32ToInt32ScaleClamp(f *testing.F) {
	f.Add(uint32(0x3f800000), uint32(0x3e800000), int16(300), int16(-50), 20)
	f.Fuzz(func(t *testing.T, scaleBits, offsetBits uint32, hi, lo int16, n int) {
		if n < 0 {
			n = -n
		}
		n %= 200
		scale := math.Float32frombits(scaleBits)
		offset := math.Float32frombits(offsetBits)
		if scale != scale || offset != offset { // skip NaN scale/offset (out of contract)
			return
		}
		// Keep bounds in-contract and finite. lo/hi are small ints scaled up.
		minV := float32(lo) * 1000.0
		maxV := float32(hi) * 1000.0
		src := make([]float32, n)
		for i := range src {
			src[i] = math.Float32frombits(scaleBits*uint32(i+1) ^ offsetBits)
		}
		dst := make([]int32, n)
		want := make([]int32, n)
		Float32ToInt32ScaleClamp(dst, src, scale, offset, minV, maxV)
		float32ToInt32ScaleClampGo(want, src, scale, offset, minV, maxV)
		for i := range dst {
			if dst[i] != want[i] {
				t.Fatalf("n=%d [%d] = %d, want %d (src=%g scale=%g offset=%g minV=%g maxV=%g)",
					n, i, dst[i], want[i], src[i], scale, offset, minV, maxV)
			}
		}
	})
}

func BenchmarkFloat32ToInt32ScaleClamp(b *testing.B) {
	src := make([]float32, 1024)
	for i := range src {
		src[i] = float32(i%400-200) * 8.0
	}
	dst := make([]int32, len(src))
	b.SetBytes(int64(len(src)) * 8) // 4 bytes read + 4 bytes written per element
	for b.Loop() {
		Float32ToInt32ScaleClamp(dst, src, 3.0, 0.4054, -2000, 2000)
	}
}
