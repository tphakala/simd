//go:build arm64

package i8

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestMinMaxNEON_ParityWithGo guards the overlapping-final-block tail and the
// 2-block-unrolled body of minMaxNEON against the scalar reference. It plants a
// unique extreme in a mid-block lane and another in the last (tail) lane, so a
// kernel that drops a vector lane, mis-seeds an accumulator pair, or skips the
// overlap block on the single-block fast path is caught. The swap variant covers
// both reduces. The sizes concentrate on the single-block-plus-ragged-tail range
// n in [16,31] where the #285 dropped-tail bug class lives (the single-block CBZ
// must reach the overlap check, skipping the no-op foldpairs, so the tail still
// runs), plus larger ragged
// sizes that exercise the odd-block and pairs paths.
//
// minMaxNEON is called directly (not via the public MinMax) so the kernel runs on
// exactly these lengths instead of being routed to the Go reference below the
// dispatch floor. The n >= minNEON16 gate is a correctness floor once the overlap
// block reloads a[n-16], so lengths below it are skipped.
func TestMinMaxNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	sizes := []int{16, 17, 18, 19, 20, 23, 24, 25, 30, 31, 32, 33, 47, 48, 49, 63, 64, 65, 95, 96, 127, 128, 255, 256, 257}
	for _, n := range sizes {
		if n < minNEON16 {
			continue // minMaxNEON is called directly here and needs a full 16-elem block
		}
		for _, swap := range []bool{false, true} {
			a := make([]int8, n)
			// A tame in-range body keeps the two planted extremes the unique
			// min and max: int8(i%7)-3 stays within [-3,3].
			for i := range a {
				a[i] = int8(i%7) - 3
			}
			mid, tail := int8(math.MinInt8), int8(math.MaxInt8)
			if swap {
				mid, tail = tail, mid
			}
			a[n/2] = mid
			a[n-1] = tail
			gotMin, gotMax := minMaxNEON(a)
			wantMin, wantMax := minMaxGo(a)
			if gotMin != wantMin || gotMax != wantMax {
				t.Fatalf("n=%d swap=%v: minMaxNEON = (%d, %d), want (%d, %d) (Go)",
					n, swap, gotMin, gotMax, wantMin, wantMax)
			}
		}
	}
}
