package i32

import (
	"fmt"
	"testing"
)

// BenchmarkMinMax_N measures the overlapping-final-block tail added to minMaxNEON
// in #286. The residue of 1..3 int32 elements (n mod 4) that a scalar CSEL tail
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
