package i8

import "testing"

// residueLengths isolate the 16-wide and 8-wide AVX2 pre-blocks (#149) that the
// shared `lengths` sweep does not pin down on its own. Each is past the
// blockSat32 (32) dispatch threshold so the SIMD path runs, and the residue
// n%32 selects which pre-blocks fire:
//
//	40  -> residue  8  (0b01000): 8-wide block only, empty scalar tail
//	48  -> residue 16  (0b10000): 16-wide block only, empty scalar tail
//	55  -> residue 23  (0b10111): 16-wide block only (bit3=0), 7-element scalar tail
//	63  -> residue 31  (0b11111): 16-wide + 8-wide blocks + 7-element scalar tail
//
// n=48 in particular is the only case that runs the 16-wide block with the
// 8-wide block skipped, which the shared table's residue-8/9 entries never hit.
var residueLengths = []int{40, 48, 55, 63}

// TestElementwiseResiduePreBlocks checks every element-wise op against its Go
// reference at the residue lengths that isolate the new SIMD pre-blocks. A bug in
// a pre-block (wrong op, operand order, or pointer advance) shows up here as a
// mismatch at exactly the bytes the block wrote.
func TestElementwiseResiduePreBlocks(t *testing.T) {
	for _, n := range residueLengths {
		a, b := genI8(n, 1), genI8(n, 2)
		got := make([]int8, n)
		want := make([]int8, n)

		// dst, a: single input ops.
		Abs(got, a)
		absGo(want, a)
		assertI8Eq(t, "Abs", n, got, want)

		Neg(got, a)
		negGo(want, a)
		assertI8Eq(t, "Neg", n, got, want)

		// dst, a, b: two-input ops.
		AddSaturate(got, a, b)
		addSatGo(want, a, b)
		assertI8Eq(t, "AddSaturate", n, got, want)

		SubSaturate(got, a, b)
		subSatGo(want, a, b)
		assertI8Eq(t, "SubSaturate", n, got, want)

		Min(got, a, b)
		minGo(want, a, b)
		assertI8Eq(t, "Min", n, got, want)

		Max(got, a, b)
		maxGo(want, a, b)
		assertI8Eq(t, "Max", n, got, want)

		AbsDiff(got, a, b)
		absDiffGo(want, a, b)
		assertI8Eq(t, "AbsDiff", n, got, want)

		// dst, src, lo, hi: clamp, both lo<hi and lo>hi (max-then-min ordering).
		for _, lohi := range []struct{ lo, hi int8 }{{-40, 90}, {50, -50}} {
			Clamp(got, a, lohi.lo, lohi.hi)
			clampGo(want, a, lohi.lo, lohi.hi)
			assertI8Eq(t, "Clamp", n, got, want)
		}

		// dst, a, s: scalar-broadcast ops. Exercise a positive and a negative s so
		// both saturation directions run through the reused sVec X-half.
		for _, s := range []int8{37, -100} {
			AddScalarSaturate(got, a, s)
			addScalarSatGo(want, a, s)
			assertI8Eq(t, "AddScalarSaturate", n, got, want)

			SubScalarSaturate(got, a, s)
			subScalarSatGo(want, a, s)
			assertI8Eq(t, "SubScalarSaturate", n, got, want)
		}
	}
}

// TestElementwiseInPlaceForwardBlocks guards the forward-block in-place safety:
// the pre-blocks read each input byte then write its output before advancing, so
// dst==a aliasing must produce the same result as the out-of-place call even for
// the non-idempotent ops. An overlapping/backward block would double-transform
// bytes it re-read from dst; a forward block does not. n=55 (residue 23) runs the
// 16-wide block in place; n=63 (residue 31) runs both the 16-wide and 8-wide
// blocks (plus a 7-element scalar tail) in place, so the two lengths together
// cover both forward pre-blocks with aliased dst.
func TestElementwiseInPlaceForwardBlocks(t *testing.T) {
	for _, n := range []int{55, 63} {
		a := genI8(n, 1)
		b := genI8(n, 2)

		// Abs in place: dst == a.
		{
			want := make([]int8, n)
			Abs(want, a) // out-of-place reference
			inplace := append([]int8(nil), a...)
			Abs(inplace, inplace)
			assertI8Eq(t, "Abs in-place", n, inplace, want)
		}

		// AddSaturate in place: dst == a (first input aliases output).
		{
			want := make([]int8, n)
			AddSaturate(want, a, b)
			inplace := append([]int8(nil), a...)
			AddSaturate(inplace, inplace, b)
			assertI8Eq(t, "AddSaturate in-place", n, inplace, want)
		}
	}
}
