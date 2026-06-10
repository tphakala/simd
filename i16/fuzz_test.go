package i16

import (
	"encoding/binary"
	"testing"
)

// Differential fuzz targets for the i16 interleave kernels. Both ops are pure
// data movement, so they are bit-exact against the pure-Go reference for every
// length; the high-value bug class is tail/remainder handling at arbitrary
// lengths around the SIMD lane unrolls. Seeds run under plain `go test`;
// `go test -fuzz=FuzzXxx` explores further.

// i16sFromBits reinterprets raw bytes as little-endian int16s, one per 2-byte
// chunk, reaching the full int16 range including MinInt16/MaxInt16.
func i16sFromBits(raw []byte) []int16 {
	out := make([]int16, len(raw)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(raw[i*2:]))
	}
	return out
}

// addLenSeeds seeds raw byte buffers whose int16 element counts (bytes ÷ 2)
// cover 0 through ~70, hitting every remainder around the SIMD unrolls, plus a
// couple of larger blocks.
func addLenSeeds(f *testing.F) {
	f.Helper()
	for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32, 33, 47, 48, 63, 64, 65, 70, 128, 257} {
		raw := make([]byte, n*2)
		for i := range raw {
			raw[i] = byte(i*37 + 11)
		}
		f.Add(raw)
	}
}

func equalI16(t *testing.T, op string, got, want []int16) {
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

func FuzzI16Interleave(f *testing.F) {
	addLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := i16sFromBits(raw)
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		dst := make([]int16, 2*h)
		dstRef := make([]int16, 2*h)
		Interleave2(dst, a, b)
		interleave2Go(dstRef, a, b)
		equalI16(t, "Interleave2", dst, dstRef)

		da := make([]int16, h)
		db := make([]int16, h)
		daRef := make([]int16, h)
		dbRef := make([]int16, h)
		Deinterleave2(da, db, dst)
		deinterleave2Go(daRef, dbRef, dstRef)
		equalI16(t, "Deinterleave2.a", da, daRef)
		equalI16(t, "Deinterleave2.b", db, dbRef)
	})
}
