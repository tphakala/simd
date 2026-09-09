package i32

import (
	"fmt"
	"testing"
)

// BenchmarkMinMax_N measures the overlapping-final-block tail that the i32
// minMaxNEON kernel uses (issue #286). The residue of 1..3 int32 elements (n mod 4)
// that a scalar CSEL tail
// used to serve is now re-folded as one overlapping .4S block before the
// horizontal reduce. The fixed BenchmarkMinMax_1000 (n%4==0) is residue-free and
// never runs the overlap block, so ragged residue lengths must be measured
// explicitly. 4 and 4096 are aligned sentinels (overlap skipped, the control);
// 5/7/15/31/63/255/1023 are ragged so the overlap block runs. The i32 tail is at
// most 3 elements, so this is the smaller-payoff sibling of the i8 overlap and is
// adopted only if the A76 A/B shows a repeatable win.
func BenchmarkMinMax_N(b *testing.B) {
	for _, n := range []int{4, 5, 7, 15, 31, 63, 255, 1023, 4096} {
		res := genI32(n, 1)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 4)
			for b.Loop() {
				_, _ = MinMax(res)
			}
		})
	}
}

// BenchmarkMaxAbs_N is the MaxAbs sibling of BenchmarkMinMax_N (issue #289): it
// guards the overlapping-final-block tail on the MaxAbs reduction at ragged
// residue sizes. Instead of serving the (n mod 4) residue with a serial
// compare/cmov scalar chain, one overlapping final .4S block re-folds the last
// full block (idempotent signed min/max makes the double-count harmless). The
// fixed-size MaxAbs benchmarks are residue-free (1000) or large (1003), so the
// small ragged residues must be measured explicitly. n=8 is the aligned sentinel
// (overlap skipped); 5/9 are residue 1 (the worst case, where a whole .4S fold
// replaces a single scalar iteration), 7/15 are residue 3, and the larger ragged
// sizes show the tail shrinking to a negligible fraction of the work.
func BenchmarkMaxAbs_N(b *testing.B) {
	for _, n := range []int{5, 7, 8, 9, 15, 25, 63, 255, 1023} {
		a := genI32(n, 1)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n) * 4)
			for b.Loop() {
				_ = MaxAbs(a)
			}
		})
	}
}
