//go:build arm64

package f32

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// The minIdxOfSumRows4NEON kernel scores four rows (one per NEON lane) against
// the candidate vector a. For a block the dispatcher passes:
//
//	rev == 0 (slide +1): k sliced at row r's window start; lane l is row r+l.
//	rev == 1 (slide -1): k sliced at row r+3's window start; lane l is row r+3-l,
//	                     so the kernel reverses the two result vectors at store.
//
// Both signs load k[i:i+4] ascending (windows slide by one element per
// candidate). With a union slice of length n+3 the reference reproduces the
// kernel exactly: rev 0 == minIdxOfSumRowsGo(a, kUnion, base 0, slide +1) over
// four rows, and rev 1 == the same with base 3, slide -1 (row m reads
// kUnion[3-m : 3-m+n], which reverses to lane m).
func refBaseSlideForRev(rev int) (base, slide int) {
	if rev == 1 {
		return 3, -1
	}
	return 0, 1
}

// TestMinIdxOfSumRows4NEON_ParityWithGo drives the kernel directly for both rev
// values across n in 1..80 with quantized-tie data plus injected NaN/+Inf in
// BOTH operands, including NaN at a[0] and a[n-1] so the broadcast operand's
// NaN-stickiness is exercised (no prior kernel-path test put NaN in a). Every
// lane must be bit-identical to the Go reference.
func TestMinIdxOfSumRows4NEON_ParityWithGo(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON not available")
	}
	set := quant025Set()
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	rng := rand.New(rand.NewSource(0xC0FFEE))

	for n := 1; n <= 80; n++ {
		for _, rev := range []int{0, 1} {
			for trial := range 24 {
				a := make([]float32, n)
				kUnion := make([]float32, n+3)
				fillQuant025(rng, a, set)
				fillQuant025(rng, kUnion, set)
				injectSpecials32(rng, kUnion, nan, pinf)
				// NaN/+Inf into the broadcast operand a too, and force NaN at
				// the ends on a rotating subset of trials.
				injectSpecials32(rng, a, nan, pinf)
				if trial%5 == 0 {
					a[0] = nan
				}
				if trial%7 == 0 && n > 1 {
					a[n-1] = nan
				}

				vals := make([]float32, 4)
				idxs := make([]int32, 4)
				minIdxOfSumRows4NEON(vals, idxs, a, kUnion, rev)

				refVals := make([]float32, 4)
				refIdxs := make([]int32, 4)
				base, slide := refBaseSlideForRev(rev)
				minIdxOfSumRowsGo(refVals, refIdxs, a, kUnion, base, slide)

				for l := range 4 {
					if idxs[l] != refIdxs[l] || !bitsEqF32(vals[l], refVals[l]) {
						t.Fatalf("n=%d rev=%d trial=%d lane=%d: kernel (%d,%#x) ref (%d,%#x)\n a=%v\n k=%v",
							n, rev, trial, l, idxs[l], math.Float32bits(vals[l]),
							refIdxs[l], math.Float32bits(refVals[l]), a, kUnion)
					}
				}
			}
		}
	}
}

// TestMinIdxOfSumRows4NEON_LengthExact allocates every buffer with len == cap ==
// exactly the span the kernel may touch (vals/idxs are 4; k is n+3, the union of
// the four rows' windows), exercising the kernel at the tight window with no
// trailing slack. This checks correctness at the exact span; a deliberate tail
// over-read is caught by TestMinIdxOfSumRows4NEON_OverReadPoison (a small heap
// over-read alone neither faults nor trips -race on hand-written asm loads, so
// exact length is a parity check, not an over-read detector). Both rev values.
func TestMinIdxOfSumRows4NEON_LengthExact(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON not available")
	}
	set := quant025Set()
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	rng := rand.New(rand.NewSource(0x1E0FE))

	for n := 1; n <= 40; n++ {
		for _, rev := range []int{0, 1} {
			// Two-arg make gives cap == len, so each buffer spans exactly the
			// window the kernel may touch (k is n+3, the union of four rows).
			a := make([]float32, n)
			kUnion := make([]float32, n+3)
			fillQuant025(rng, a, set)
			fillQuant025(rng, kUnion, set)
			injectSpecials32(rng, kUnion, nan, pinf)

			vals := make([]float32, 4)
			idxs := make([]int32, 4)
			minIdxOfSumRows4NEON(vals, idxs, a, kUnion, rev)

			refVals := make([]float32, 4)
			refIdxs := make([]int32, 4)
			base, slide := refBaseSlideForRev(rev)
			minIdxOfSumRowsGo(refVals, refIdxs, a, kUnion, base, slide)

			for l := range 4 {
				if idxs[l] != refIdxs[l] || !bitsEqF32(vals[l], refVals[l]) {
					t.Fatalf("n=%d rev=%d lane=%d: kernel (%d,%#x) ref (%d,%#x)",
						n, rev, l, idxs[l], math.Float32bits(vals[l]),
						refIdxs[l], math.Float32bits(refVals[l]))
				}
			}
		}
	}
}

// TestMinIdxOfSumRows4NEON_OverReadPoison plants a min-winning poison (-Inf) in a
// full lane-width block immediately past the kernel's valid k window (both rev
// values read k[0:n+3] ascending) and asserts the result is unchanged. a and the
// valid window are finite, so every real candidate is finite and a[i] + (-Inf) =
// -Inf strictly wins the less-than min: if a future edit over-reads the k tail by
// even one element, the poison flips that lane's (idx, val) and the test fails
// loudly. This is the detector the +Inf-padded parity and LengthExact tests
// cannot be: +Inf is the min-reduction identity (an over-read into it never wins,
// so it stays invisible), and a small heap over-read does not fault.
func TestMinIdxOfSumRows4NEON_OverReadPoison(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON not available")
	}
	const width = 4
	neginf := float32(math.Inf(-1))

	for n := 1; n <= 40; n++ {
		for _, rev := range []int{0, 1} {
			valid := n + width - 1 // the exact k span the kernel reads
			k := make([]float32, valid+width)
			for i := range valid {
				k[i] = float32((i*7)%23)*0.5 - 3.0 // finite, controlled
			}
			for i := valid; i < len(k); i++ {
				k[i] = neginf // a full width-block of min-winning poison
			}
			// One finite guard element behind a: a buggy kernel's extra
			// candidate also broadcasts a[n], and 0 + (-Inf) = -Inf still
			// wins, so detection never depends on uncontrolled heap bits.
			aBack := make([]float32, n+1)
			for i := range n {
				aBack[i] = float32((i*5)%17)*0.25 + 0.5 // finite
			}
			a := aBack[:n]

			vals := make([]float32, width)
			idxs := make([]int32, width)
			minIdxOfSumRows4NEON(vals, idxs, a, k, rev)

			// Reference over ONLY the valid window (poison sliced off) is what a
			// non-over-reading kernel must reproduce.
			refVals := make([]float32, width)
			refIdxs := make([]int32, width)
			base, slide := refBaseSlideForRev(rev)
			minIdxOfSumRowsGo(refVals, refIdxs, a, k[:valid], base, slide)

			for l := range width {
				if idxs[l] != refIdxs[l] || !bitsEqF32(vals[l], refVals[l]) {
					t.Fatalf("n=%d rev=%d lane=%d: kernel (%d,%#x) ref (%d,%#x): over-read into -Inf poison?",
						n, rev, l, idxs[l], math.Float32bits(vals[l]),
						refIdxs[l], math.Float32bits(refVals[l]))
				}
			}
		}
	}
}

// TestMinIdxOfSumRowsDispatch_ReachesNEON drives the public API (which routes
// through the arm64 minIdxOfSumRows32 dispatcher) with a 9-row call (two NEON
// blocks plus one Go remainder row) and a 3-row call (all Go remainder), for
// slide +1, slide -1, and a non-unit slide that must fall through to Go. Every
// result must match the batched Go reference bit-exactly.
func TestMinIdxOfSumRowsDispatch_ReachesNEON(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON not available")
	}
	set := quant025Set()
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	rng := rand.New(rand.NewSource(0xD15A))

	for _, rows := range []int{9, 3} {
		for _, slide := range []int{1, -1, 2} {
			const n = 7
			base, klen := windowLayout(rows, n, slide)

			a := make([]float32, n)
			k := make([]float32, klen)
			fillQuant025(rng, a, set)
			fillQuant025(rng, k, set)
			injectSpecials32(rng, k, nan, pinf)

			vals := make([]float32, rows)
			idxs := make([]int32, rows)
			MinIdxOfSumRows(vals, idxs, a, k, base, slide)

			refVals := make([]float32, rows)
			refIdxs := make([]int32, rows)
			minIdxOfSumRowsGo(refVals, refIdxs, a, k, base, slide)

			for r := range rows {
				if idxs[r] != refIdxs[r] || !bitsEqF32(vals[r], refVals[r]) {
					t.Fatalf("rows=%d slide=%d row=%d: got (%d,%#x) ref (%d,%#x)",
						rows, slide, r, idxs[r], math.Float32bits(vals[r]),
						refIdxs[r], math.Float32bits(refVals[r]))
				}
			}
		}
	}
}

// TestMinIdxOfSumRows_NEONSignedZero drives the signed-zero incumbent case from
// Task B through the public API on this arch (a full four-row block, so the NEON
// kernel handles it) and asserts the -0.0 incumbent keeps its exact bits and
// index against a later +0.0 candidate, plus bit-exactness of the whole block.
func TestMinIdxOfSumRows_NEONSignedZero(t *testing.T) {
	if !hasNEON {
		t.Skip("NEON not available")
	}
	negZero := math.Float32frombits(0x80000000)
	const negZeroBits = uint32(0x80000000)

	// n=2, a = {-0.0, 1.0}. Row 0 window {-0.0, -1.0}:
	//   -0.0 + -0.0 = -0.0 (idx 0)
	//    1.0 + -1.0 = +0.0 (idx 1); +0.0 < -0.0 is false, so -0.0 keeps its bits.
	const n, rows, slide = 2, 4, 1
	base, klen := windowLayout(rows, n, slide)
	a := []float32{negZero, 1.0}
	k := make([]float32, klen)
	// Deterministic finite fill, then plant the signed-zero row at window 0.
	for i := range k {
		k[i] = float32(i)*0.25 - 1.0
	}
	k[0] = negZero
	k[1] = -1.0

	vals := make([]float32, rows)
	idxs := make([]int32, rows)
	MinIdxOfSumRows(vals, idxs, a, k, base, slide)

	refVals := make([]float32, rows)
	refIdxs := make([]int32, rows)
	minIdxOfSumRowsGo(refVals, refIdxs, a, k, base, slide)

	for r := range rows {
		if idxs[r] != refIdxs[r] || !bitsEqF32(vals[r], refVals[r]) {
			t.Fatalf("row=%d: got (%d,%#x) ref (%d,%#x)",
				r, idxs[r], math.Float32bits(vals[r]), refIdxs[r], math.Float32bits(refVals[r]))
		}
	}
	if idxs[0] != 0 || math.Float32bits(vals[0]) != negZeroBits {
		t.Errorf("signed-zero row 0 = (%d, %#x), want (0, %#x)",
			idxs[0], math.Float32bits(vals[0]), negZeroBits)
	}
}

// BenchmarkMinIdxOfSumRows4NEON_N drives the minIdxOfSumRows4NEON kernel
// directly at the dispatcher's exact slicing for a slide +1, four-row block
// (rev 0, base 0): k is the n+3 union of the four rows' windows, matching
// what minIdxOfSumRows32 in f32_arm64.go passes for r == 0.
func BenchmarkMinIdxOfSumRows4NEON_N(b *testing.B) {
	if !hasNEON {
		b.Skip("NEON not available")
	}
	for _, n := range []int{11, 14, 17} {
		a := make([]float32, n)
		kUnion := make([]float32, n+3)
		for i := range a {
			a[i] = float32(i%100) + 0.5
		}
		for i := range kUnion {
			kUnion[i] = float32((i+37)%100) + 0.5
		}
		vals := make([]float32, 4)
		idxs := make([]int32, 4)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				minIdxOfSumRows4NEON(vals, idxs, a, kUnion, 0)
			}
		})
	}
}
