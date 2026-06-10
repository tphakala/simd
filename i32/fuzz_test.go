package i32

import (
	"encoding/binary"
	"testing"
)

// Differential fuzz targets for the i32 codec primitives. Every i32 kernel is
// bit-exact against its pure-Go reference by construction (integer arithmetic
// has no accumulation-order or rounding freedom), so each target asserts exact
// equality. The high-value bug class is tail/remainder handling at arbitrary
// lengths around the 8/16-lane unrolls and the SIMD dispatch thresholds; the
// seeds bracket those boundaries and the fuzzer widens the length space. Seeds
// run under plain `go test`; `go test -fuzz=FuzzXxx` explores further.

// i32sFromBits reinterprets raw bytes as little-endian int32s, one per 4-byte
// chunk. This organically reaches the full int32 range including MinInt32/
// MaxInt32, which is where the zigzag fold and the order-4 difference overflow.
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

func FuzzI32MidSide(f *testing.F) {
	addLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := i32sFromBits(raw)
		h := len(v) / 2
		left, right := v[:h], v[h:2*h]
		mid := make([]int32, h)
		side := make([]int32, h)
		midRef := make([]int32, h)
		sideRef := make([]int32, h)
		MidSideEncode(mid, side, left, right)
		midSideEncodeGo(midRef, sideRef, left, right)
		equalI32(t, "MidSideEncode.mid", mid, midRef)
		equalI32(t, "MidSideEncode.side", side, sideRef)

		// Round-trip: decode the SIMD-encoded mid/side and compare to the
		// reference decode of the same channels.
		l2 := make([]int32, h)
		r2 := make([]int32, h)
		lRef := make([]int32, h)
		rRef := make([]int32, h)
		MidSideDecode(l2, r2, mid, side)
		midSideDecodeGo(lRef, rRef, midRef, sideRef)
		equalI32(t, "MidSideDecode.left", l2, lRef)
		equalI32(t, "MidSideDecode.right", r2, rRef)
	})
}

func FuzzI32Diff(f *testing.F) {
	addLenSeeds(f)
	type diffPair struct {
		name string
		simd func(dst, src []int32)
		ref  func(dst, src []int32)
	}
	pairs := []diffPair{
		{"Diff1", Diff1, diff1Go}, {"Diff2", Diff2, diff2Go},
		{"Diff3", Diff3, diff3Go}, {"Diff4", Diff4, diff4Go},
	}
	restorePairs := []diffPair{
		{"Restore1", Restore1, restore1Go}, {"Restore2", Restore2, restore2Go},
		{"Restore3", Restore3, restore3Go}, {"Restore4", Restore4, restore4Go},
	}
	f.Fuzz(func(t *testing.T, raw []byte) {
		src := i32sFromBits(raw)
		got := make([]int32, len(src))
		want := make([]int32, len(src))
		for _, p := range pairs {
			p.simd(got, src)
			p.ref(want, src)
			equalI32(t, p.name, got, want)
		}
		for _, p := range restorePairs {
			p.simd(got, src)
			p.ref(want, src)
			equalI32(t, p.name, got, want)
		}
	})
}

func FuzzI32LPC(f *testing.F) {
	addLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := i32sFromBits(raw)
		n := len(v)
		if n < 3 {
			return
		}
		// order in [1, min(maxLPCRestoreOrder, n-1)] so the public guard
		// (order == 0 || order >= n) is not triggered and the kernel runs.
		maxOrder := min(n-1, maxLPCRestoreOrder)
		order := 1 + int(raw[0])%maxOrder
		shift := uint(raw[len(raw)-1]) % (maxLPCShift + 1)
		// Use the leading `order` samples as coefficients (full int32 range);
		// both the SIMD and Go paths accumulate in int64 with identical wrapping.
		coeffs := append([]int32(nil), v[:order]...)
		samples := v

		res := make([]int32, n)
		resRef := make([]int32, n)
		LPCResidualEncode(res, samples, coeffs, shift)
		lpcResidualEncodeGo(resRef, samples, coeffs, shift)
		equalI32(t, "LPCResidualEncode", res, resRef)

		out := make([]int32, n)
		outRef := make([]int32, n)
		LPCRestore(out, res, coeffs, shift)
		lpcRestoreGo(outRef, resRef, coeffs, shift)
		equalI32(t, "LPCRestore", out, outRef)
	})
}

func FuzzI32Rice(f *testing.F) {
	addLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		res := i32sFromBits(raw)

		// RiceSums across both the 15-wide and 31-wide (5-bit) kernels.
		for _, m := range []int{riceParamCount, riceMaxParam5 + 1} {
			sums := make([]uint64, m)
			sumsRef := make([]uint64, m)
			RiceSums(sums, res)
			riceSumsGo(sumsRef, res)
			for i := range sums {
				if sums[i] != sumsRef[i] {
					t.Fatalf("RiceSums[m=%d] col %d got %d want %d (len=%d)", m, i, sums[i], sumsRef[i], len(res))
				}
			}
		}

		if got, want := ZigzagSum(res), zigzagSumGo(res); got != want {
			t.Fatalf("ZigzagSum got %d want %d (len=%d)", got, want, len(res))
		}
	})
}

func FuzzI32FixedAbsSums(f *testing.F) {
	addLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		src := i32sFromBits(raw)
		var got, want [5]uint64
		FixedAbsSums(src, &got)
		fixedAbsSumsGo(src, &want)
		if got != want {
			t.Fatalf("FixedAbsSums got %v want %v (len=%d)", got, want, len(src))
		}
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
