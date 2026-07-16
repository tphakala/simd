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

// addDotLenSeeds seeds raw buffers for FuzzI16Dot, which splits the decoded
// int16s into two operands. The per-operand length is therefore raw/4 bytes,
// not raw/2, so addLenSeeds cannot be reused: halving its buffers a second time
// lands the operands on a sparse set that misses 9 and 17, the one-past-the-
// block tails for the 8-wide and 16-wide bodies, and leaves most seeds below
// every dispatch threshold where the assertion degrades to dotGo against
// itself. These sizes make the operand length itself sweep 0..40.
func addDotLenSeeds(f *testing.F) {
	f.Helper()
	sizes := make([]int, 0, 46)
	for n := range 41 {
		sizes = append(sizes, n)
	}
	sizes = append(sizes, 47, 48, 63, 64, 65, 128)
	for _, n := range sizes {
		raw := make([]byte, n*4)
		for i := range raw {
			raw[i] = byte(i*37 + 11)
		}
		f.Add(raw)
	}
}

// FuzzI16Dot differentially fuzzes the widening dot product against the pure-Go
// reference. Unlike the interleave targets, the interesting bug class here is
// arithmetic as much as tail handling: the raw bytes reach MinInt16 freely, so
// the fuzzer readily builds inputs whose int32 accumulator overflows, which is
// exactly where a saturating or mis-reassociated kernel would diverge. It also
// checks dotOracle, so a fault in the reference itself cannot hide by agreeing
// with the kernels.
func FuzzI16Dot(f *testing.F) {
	addDotLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := i16sFromBits(raw)
		h := len(v) / 2
		a, b := v[:h], v[h:2*h]
		got := DotProduct(a, b)
		if want := dotGo(a, b); got != want {
			t.Fatalf("DotProduct(len=%d) = %d, want %d (reference)", h, got, want)
		}
		if want := dotOracle(a, b); got != want {
			t.Fatalf("DotProduct(len=%d) = %d, want %d (int64 oracle)", h, got, want)
		}
		// Mismatched lengths must clamp, not read out of bounds. Both orders:
		// the kernels take min(len(a), len(b)) from two separate slice headers,
		// so shortening a and shortening b are distinct paths.
		if h > 1 {
			if got, want := DotProduct(a, b[:h-1]), dotOracle(a, b[:h-1]); got != want {
				t.Fatalf("DotProduct(len=%d, len=%d) = %d, want %d", h, h-1, got, want)
			}
			if got, want := DotProduct(a[:h-1], b), dotOracle(a[:h-1], b); got != want {
				t.Fatalf("DotProduct(len=%d, len=%d) = %d, want %d", h-1, h, got, want)
			}
		}
	})
}

// FuzzI16XCorr differentially fuzzes multi-lag correlation. The high-value bug
// class is the lag blocking: the kernel evaluates 4 lags per call and the
// dispatcher finishes the remainder with the dot kernel, so arbitrary lag
// counts exercise that seam. It asserts the defining property directly, that
// every lag equals the dot product at that offset, which also catches a kernel
// that drifted from DotProduct even if xcorrGo drifted with it.
func FuzzI16XCorr(f *testing.F) {
	for _, xn := range []int{1, 4, 8, 9, 15, 16, 17, 32, 33} {
		for _, lags := range []int{1, 3, 4, 5, 8, 9} {
			raw := make([]byte, (xn+xn+lags)*2)
			for i := range raw {
				raw[i] = byte(i*29 + 7)
			}
			f.Add(raw, uint8(xn), uint8(lags))
		}
	}
	f.Fuzz(func(t *testing.T, raw []byte, xnRaw, lagsRaw uint8) {
		v := i16sFromBits(raw)
		xn := int(xnRaw) % 64
		lags := int(lagsRaw)%32 + 1
		if xn == 0 || len(v) < xn {
			return
		}
		x := v[:xn]
		y := v[xn:]
		dst := make([]int32, lags)
		ref := make([]int32, lags)
		XCorr(dst, x, y)
		xcorrGo(ref, x, y)
		for k := range xcorrLags(dst, x, y) {
			if dst[k] != ref[k] {
				t.Fatalf("XCorr(xn=%d, lags=%d, len(y)=%d): dst[%d] = %d, want %d", xn, lags, len(y), k, dst[k], ref[k])
			}
			// The defining property: every lag is the dot product at that offset.
			if want := DotProduct(x, y[k:k+xn]); dst[k] != want {
				t.Fatalf("XCorr(xn=%d, lags=%d): dst[%d] = %d, want DotProduct = %d", xn, lags, k, dst[k], want)
			}
		}
	})
}
