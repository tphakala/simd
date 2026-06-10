package f16

import (
	"encoding/binary"
	"math"
	"testing"
)

// Differential fuzz targets for the f16 conversion kernels (F16C on AMD64, NEON
// on ARM64). The dispatched public op must agree with the pure-Go reference for
// every length; the high-value bug class is tail/remainder handling at arbitrary
// lengths around the SIMD lane unrolls. Comparison policy follows the existing
// conversion parity tests: NaN matches any NaN, everything else is bit-exact.

// f16sFromBits reinterprets raw bytes as Float16 (uint16), 2 bytes per element,
// covering the whole 16-bit space including Inf, NaN, subnormals, and ±0.
func f16sFromBits(raw []byte) []Float16 {
	out := make([]Float16, len(raw)/2)
	for i := range out {
		out[i] = binary.LittleEndian.Uint16(raw[i*2:])
	}
	return out
}

// f32sFromBits reinterprets raw bytes as float32, 4 bytes per element, organically
// producing NaN, Inf, denormals, and values that round to f16 subnormals.
func f32sFromBits(raw []byte) []float32 {
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// addLenSeeds seeds raw byte buffers covering element counts 0 through ~70 (and a
// couple larger) around the SIMD unrolls; byte counts are multiples of 4 so both
// the 2-byte (Float16) and 4-byte (float32) reinterpretations hit boundaries.
func addLenSeeds(f *testing.F) {
	f.Helper()
	for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65, 70, 128, 257} {
		raw := make([]byte, n*4)
		for i := range raw {
			raw[i] = byte(i*37 + 11)
		}
		f.Add(raw)
	}
}

func FuzzF16ToFloat32(f *testing.F) {
	addLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		src := f16sFromBits(raw)
		got := make([]float32, len(src))
		want := make([]float32, len(src))
		ToFloat32Slice(got, src)
		toFloat32SliceGo(want, src)
		for i := range got {
			if math.IsNaN(float64(want[i])) {
				if !math.IsNaN(float64(got[i])) {
					t.Fatalf("ToFloat32Slice h=%#04x: got %v want NaN (len=%d)", src[i], got[i], len(src))
				}
				continue
			}
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("ToFloat32Slice h=%#04x: got bits %#08x want %#08x (len=%d)", src[i], math.Float32bits(got[i]), math.Float32bits(want[i]), len(src))
			}
		}
	})
}

func FuzzF16FromFloat32(f *testing.F) {
	addLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		src := f32sFromBits(raw)
		got := make([]Float16, len(src))
		want := make([]Float16, len(src))
		FromFloat32Slice(got, src)
		fromFloat32SliceGo(want, src)
		for i, fv := range src {
			// Differential policy keys off the reference output, not the input.
			// When the Go reference yields a NaN the SIMD path must also yield a
			// NaN, but the payload may differ (the F16C instruction and the Go
			// bit-twiddle reference quiet NaNs differently). Otherwise require a
			// bit-exact match: this also covers float32 NaNs whose set mantissa
			// bits all live in the truncated low 13 bits, which both paths map to
			// the same f16 infinity (e.g. 0xff800030 -> 0xfc00).
			if f16IsNaN(want[i]) {
				if !f16IsNaN(got[i]) {
					t.Fatalf("FromFloat32Slice src=%v (%#08x): got %#04x want a NaN (len=%d)", fv, math.Float32bits(fv), got[i], len(src))
				}
				continue
			}
			if got[i] != want[i] {
				t.Fatalf("FromFloat32Slice src=%v (%#08x): got %#04x want %#04x (len=%d)", fv, math.Float32bits(fv), got[i], want[i], len(src))
			}
		}
	})
}

func FuzzF16RoundTrip(f *testing.F) {
	addLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		// f16 -> f32 -> f16 is the identity for every non-NaN value (the upward
		// conversion is exact and the downward one rounds it back unchanged);
		// NaN maps to some NaN. This exercises both kernels back to back across
		// arbitrary tail lengths.
		src := f16sFromBits(raw)
		f32 := make([]float32, len(src))
		back := make([]Float16, len(src))
		ToFloat32Slice(f32, src)
		FromFloat32Slice(back, f32)
		for i := range src {
			if f16IsNaN(src[i]) {
				if !f16IsNaN(back[i]) {
					t.Fatalf("round-trip h=%#04x: got %#04x want a NaN (len=%d)", src[i], back[i], len(src))
				}
				continue
			}
			if back[i] != src[i] {
				t.Fatalf("round-trip h=%#04x: got %#04x (len=%d)", src[i], back[i], len(src))
			}
		}
	})
}
