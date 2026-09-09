//go:build arm64

package i32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestMinMaxNEON_SmallRaggedOverlap concentrates on the single-block-plus-ragged
// range n in [4,7] that the general paritySizes sweep misses (it has 7 but not
// 5 or 6). Once minMaxNEON re-folds the last res[n-4:n] block instead of a scalar
// tail, the single-block fast path must reach the overlap check (skipping the
// no-op foldpairs) so the overlap still runs; a kernel that jumps straight to the
// reduce would drop the last (n mod 4) elements. A unique planted extreme in the tail lane catches exactly
// that. minMaxNEON is called directly so the kernel runs on these lengths rather
// than being routed to the Go reference below the dispatch floor.
func TestMinMaxNEON_SmallRaggedOverlap(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range []int{4, 5, 6, 7, 8, 9, 10, 11, 13, 15} {
		if n < minNEONMinMax {
			continue
		}
		for _, swap := range []bool{false, true} {
			res := make([]int32, n)
			for i := range res {
				res[i] = int32(i%5) - 2 // tame body in [-2,2]
			}
			mid, tail := int32(math.MinInt32), int32(math.MaxInt32)
			if swap {
				mid, tail = tail, mid
			}
			res[n/2] = mid
			res[n-1] = tail
			gotMin, gotMax := minMaxNEON(res)
			wantMin, wantMax := minMaxGo(res)
			if gotMin != wantMin || gotMax != wantMax {
				t.Fatalf("n=%d swap=%v: minMaxNEON = (%d, %d), want (%d, %d) (Go)",
					n, swap, gotMin, gotMax, wantMin, wantMax)
			}
		}
	}
}
