package i32

import (
	"encoding/binary"
	"testing"
)

// Differential fuzz targets for the i32 primitives. Every i32 kernel is
// bit-exact against its pure-Go reference by construction (integer arithmetic
// has no accumulation-order or rounding freedom), so each target asserts exact
// equality. The high-value bug class is tail/remainder handling at arbitrary
// lengths around the 8/16-lane unrolls and the SIMD dispatch thresholds; the
// seeds bracket those boundaries and the fuzzer widens the length space. Seeds
// run under plain `go test`; `go test -fuzz=FuzzXxx` explores further.

// i32sFromBits reinterprets raw bytes as little-endian int32s, one per 4-byte
// chunk. This organically reaches the full int32 range including MinInt32/
// MaxInt32, where the sign bit and wraparound are most likely to be mishandled.
func i32sFromBits(raw []byte) []int32 {
	out := make([]int32, len(raw)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// addLenSeeds seeds raw byte buffers whose int32 element counts (bytes ÷ 4)
// cover 0 through ~70, hitting every remainder around the 8/16-lane unrolls,
// plus a couple of larger blocks.
func addLenSeeds(f *testing.F) {
	f.Helper()
	lens := []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65, 70, 128, 257}
	for _, n := range lens {
		raw := make([]byte, n*4)
		for i := range raw {
			raw[i] = byte(i*37 + 11)
		}
		f.Add(raw)
	}
}

func equalI32(t *testing.T, op string, got, want []int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length got %d want %d", op, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s: index %d got %d want %d (len=%d)", op, i, got[i], want[i], len(got))
		}
	}
}

func FuzzI32Add(f *testing.F) {
	addLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := i32sFromBits(raw)
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		got := make([]int32, h)
		want := make([]int32, h)
		Add(got, a, b)
		addGo(want, a, b)
		equalI32(t, "Add", got, want)

		Sub(got, a, b)
		subGo(want, a, b)
		equalI32(t, "Sub", got, want)
	})
}

func FuzzI32MinMax(f *testing.F) {
	addLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		res := i32sFromBits(raw)
		if len(res) == 0 {
			return
		}
		gotMin, gotMax := MinMax(res)
		wantMin, wantMax := minMaxGo(res)
		if gotMin != wantMin || gotMax != wantMax {
			t.Fatalf("MinMax got (%d,%d) want (%d,%d) (len=%d)", gotMin, gotMax, wantMin, wantMax, len(res))
		}
	})
}

func FuzzI32Interleave(f *testing.F) {
	addLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := i32sFromBits(raw)
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		dst := make([]int32, 2*h)
		dstRef := make([]int32, 2*h)
		Interleave2(dst, a, b)
		interleave2Go(dstRef, a, b)
		equalI32(t, "Interleave2", dst, dstRef)

		da := make([]int32, h)
		db := make([]int32, h)
		daRef := make([]int32, h)
		dbRef := make([]int32, h)
		Deinterleave2(da, db, dst)
		deinterleave2Go(daRef, dbRef, dstRef)
		equalI32(t, "Deinterleave2.a", da, daRef)
		equalI32(t, "Deinterleave2.b", db, dbRef)
	})
}
