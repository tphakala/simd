//go:build amd64

package f32

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// The AVX2 minIdxOfSumRows kernels score W rows (one per SIMD lane) against the
// candidate vector a. For a block the dispatcher passes:
//
//	rev == 0 (slide +1): k sliced at row r's window start; lane l is row r+l.
//	rev == 1 (slide -1): k sliced at row r+(W-1)'s window start; lane l is row
//	                     r+(W-1)-l, so the kernel reverses the result vectors at
//	                     store.
//
// Both signs load k[i:i+W] ascending (windows slide by one element per
// candidate). With a union slice of length n+(W-1) the Go reference reproduces
// the kernel exactly: rev 0 == minIdxOfSumRowsGo(a, kUnion, base 0, slide +1)
// over W rows, and rev 1 == the same with base W-1, slide -1 (row m reads
// kUnion[(W-1)-m : (W-1)-m+n], which reverses to lane m).
func refBaseSlideForRevW(rev, width int) (base, slide int) {
	if rev == 1 {
		return width - 1, -1
	}
	return 0, 1
}

// runRowsAVX2Parity drives an AVX2 row kernel of the given width directly for
// both rev values across n in 1..80 with quantized-tie data plus injected
// NaN/+Inf in BOTH operands, including NaN at a[0] and a[n-1] so the broadcast
// operand's NaN-stickiness is exercised (no prior kernel-path test put NaN in
// a). Every lane must be bit-identical to the Go reference.
func runRowsAVX2Parity(t *testing.T, width int, kernel func([]float32, []int32, []float32, []float32, int)) {
	t.Helper()
	set := quant025Set()
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	rng := rand.New(rand.NewSource(0xC0FFEE))

	for n := 1; n <= 80; n++ {
		for _, rev := range []int{0, 1} {
			for trial := range 24 {
				a := make([]float32, n)
				kUnion := make([]float32, n+width-1)
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

				vals := make([]float32, width)
				idxs := make([]int32, width)
				kernel(vals, idxs, a, kUnion, rev)

				refVals := make([]float32, width)
				refIdxs := make([]int32, width)
				base, slide := refBaseSlideForRevW(rev, width)
				minIdxOfSumRowsGo(refVals, refIdxs, a, kUnion, base, slide)

				for l := range width {
					if idxs[l] != refIdxs[l] || !bitsEqF32(vals[l], refVals[l]) {
						t.Fatalf("width=%d n=%d rev=%d trial=%d lane=%d: kernel (%d,%#x) ref (%d,%#x)\n a=%v\n k=%v",
							width, n, rev, trial, l, idxs[l], math.Float32bits(vals[l]),
							refIdxs[l], math.Float32bits(refVals[l]), a, kUnion)
					}
				}
			}
		}
	}
}

// runRowsAVX2LengthExact allocates every buffer with len == cap == exactly the
// span the kernel may touch (vals/idxs are width; k is n+(width-1), the union of
// the block rows' windows), exercising the kernel at the tight window with no
// trailing slack. This checks correctness at the exact span; a deliberate tail
// over-read is caught by runRowsAVX2OverReadPoison (a small heap over-read alone
// neither faults nor trips -race on hand-written asm loads, so exact length is a
// parity check, not an over-read detector). Both rev values.
func runRowsAVX2LengthExact(t *testing.T, width int, kernel func([]float32, []int32, []float32, []float32, int)) {
	t.Helper()
	set := quant025Set()
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	rng := rand.New(rand.NewSource(0x1E0FE))

	for n := 1; n <= 40; n++ {
		for _, rev := range []int{0, 1} {
			// Two-arg make gives cap == len, so each buffer spans exactly the
			// window the kernel may touch (k is n+width-1, the union of the
			// block rows' windows).
			a := make([]float32, n)
			kUnion := make([]float32, n+width-1)
			fillQuant025(rng, a, set)
			fillQuant025(rng, kUnion, set)
			injectSpecials32(rng, kUnion, nan, pinf)

			vals := make([]float32, width)
			idxs := make([]int32, width)
			kernel(vals, idxs, a, kUnion, rev)

			refVals := make([]float32, width)
			refIdxs := make([]int32, width)
			base, slide := refBaseSlideForRevW(rev, width)
			minIdxOfSumRowsGo(refVals, refIdxs, a, kUnion, base, slide)

			for l := range width {
				if idxs[l] != refIdxs[l] || !bitsEqF32(vals[l], refVals[l]) {
					t.Fatalf("width=%d n=%d rev=%d lane=%d: kernel (%d,%#x) ref (%d,%#x)",
						width, n, rev, l, idxs[l], math.Float32bits(vals[l]),
						refIdxs[l], math.Float32bits(refVals[l]))
				}
			}
		}
	}
}

// runRowsAVX2OverReadPoison plants a min-winning poison (-Inf) in a full
// lane-width block immediately past the kernel's valid k window (both rev values
// read k[0:n+width-1] ascending) and asserts the result is unchanged. a and the
// valid window are finite, so every real candidate is finite and a[i] + (-Inf) =
// -Inf strictly wins the less-than min: if a future edit over-reads the k tail by
// even one element, the poison flips that lane's (idx, val) and the test fails
// loudly. This is the detector the +Inf-padded parity and LengthExact tests
// cannot be: +Inf is the min-reduction identity (an over-read into it never wins,
// so it stays invisible), and a small heap over-read does not fault.
func runRowsAVX2OverReadPoison(t *testing.T, width int, kernel func([]float32, []int32, []float32, []float32, int)) {
	t.Helper()
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
			kernel(vals, idxs, a, k, rev)

			// Reference over ONLY the valid window (poison sliced off) is what a
			// non-over-reading kernel must reproduce.
			refVals := make([]float32, width)
			refIdxs := make([]int32, width)
			base, slide := refBaseSlideForRevW(rev, width)
			minIdxOfSumRowsGo(refVals, refIdxs, a, k[:valid], base, slide)

			for l := range width {
				if idxs[l] != refIdxs[l] || !bitsEqF32(vals[l], refVals[l]) {
					t.Fatalf("width=%d n=%d rev=%d lane=%d: kernel (%d,%#x) ref (%d,%#x): over-read into -Inf poison?",
						width, n, rev, l, idxs[l], math.Float32bits(vals[l]),
						refIdxs[l], math.Float32bits(refVals[l]))
				}
			}
		}
	}
}

func TestMinIdxOfSumRows8AVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available; minIdxOfSumRows8AVX2 requires AVX2")
	}
	runRowsAVX2Parity(t, 8, minIdxOfSumRows8AVX2)
}

func TestMinIdxOfSumRows8AVX2_OverReadPoison(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available; minIdxOfSumRows8AVX2 requires AVX2")
	}
	runRowsAVX2OverReadPoison(t, 8, minIdxOfSumRows8AVX2)
}

func TestMinIdxOfSumRows4AVX2_OverReadPoison(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available; minIdxOfSumRows4AVX2 requires AVX2")
	}
	runRowsAVX2OverReadPoison(t, 4, minIdxOfSumRows4AVX2)
}

func TestMinIdxOfSumRows4AVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available; minIdxOfSumRows4AVX2 requires AVX2")
	}
	runRowsAVX2Parity(t, 4, minIdxOfSumRows4AVX2)
}

func TestMinIdxOfSumRows8AVX2_LengthExact(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available; minIdxOfSumRows8AVX2 requires AVX2")
	}
	runRowsAVX2LengthExact(t, 8, minIdxOfSumRows8AVX2)
}

func TestMinIdxOfSumRows4AVX2_LengthExact(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available; minIdxOfSumRows4AVX2 requires AVX2")
	}
	runRowsAVX2LengthExact(t, 4, minIdxOfSumRows4AVX2)
}

// TestMinIdxOfSumRowsDispatch_ReachesAVX2 drives the public API (which routes
// through the amd64 minIdxOfSumRows32 dispatcher) with a 13-row call (one 8-wide
// block, one 4-wide block, one Go remainder row) and a 21-row call (two 8-wide
// blocks, one 4-wide block, one Go remainder row), for slide +1, slide -1, and a
// non-unit slide that must fall through to Go. Every result must match the
// batched Go reference bit-exactly.
func TestMinIdxOfSumRowsDispatch_ReachesAVX2(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available; minIdxOfSumRows32 AVX2 path requires AVX2")
	}
	set := quant025Set()
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	rng := rand.New(rand.NewSource(0xD15A))

	for _, rows := range []int{13, 21} {
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

// runMinIdxOfSumRowsKernelBench drives an AVX2 row kernel of the given width
// directly at the dispatcher's exact slicing for a slide +1 block (rev 0,
// base 0): k is the n+(width-1) union of the block's row windows, matching
// what minIdxOfSumRows32 in f32_amd64.go passes for r == 0.
func runMinIdxOfSumRowsKernelBench(b *testing.B, width int, kernel func([]float32, []int32, []float32, []float32, int)) {
	b.Helper()
	for _, n := range []int{11, 14, 17} {
		a := make([]float32, n)
		kUnion := make([]float32, n+width-1)
		for i := range a {
			a[i] = float32(i%100) + 0.5
		}
		for i := range kUnion {
			kUnion[i] = float32((i+37)%100) + 0.5
		}
		vals := make([]float32, width)
		idxs := make([]int32, width)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				kernel(vals, idxs, a, kUnion, 0)
			}
		})
	}
}

func BenchmarkMinIdxOfSumRows8AVX2_N(b *testing.B) {
	if !cpu.X86.AVX2 {
		b.Skip("AVX2 not available; minIdxOfSumRows8AVX2 requires AVX2")
	}
	runMinIdxOfSumRowsKernelBench(b, 8, minIdxOfSumRows8AVX2)
}

func BenchmarkMinIdxOfSumRows4AVX2_N(b *testing.B) {
	if !cpu.X86.AVX2 {
		b.Skip("AVX2 not available; minIdxOfSumRows4AVX2 requires AVX2")
	}
	runMinIdxOfSumRowsKernelBench(b, 4, minIdxOfSumRows4AVX2)
}
