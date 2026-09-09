package i8

import (
	"fmt"
	"testing"
)

// BenchmarkMaxAbs_N and BenchmarkMinMax_N guard the overlapping-final-block tail
// on the MaxAbs/MinMax reductions: instead of serving the (n mod width) residue
// with a serial compare/cmov scalar chain, one overlapping final vector block
// re-folds the last full block. amd64 got this for both MaxAbs and MinMax in #149;
// arm64 NEON now has it for both MinMax (issue #286) and MaxAbs (issue #289).
// The fixed 4096-byte benchmarks are residue-free and never run the overlap block,
// so ragged residue lengths must be measured explicitly. 32 is an aligned sentinel
// (overlap skipped on the 16- and 32-wide kernels alike). 17 and 25 are the
// single-block-plus-ragged-tail range: on the arm64 16-wide kernels they are one
// block (the 2-block unroll never runs, only the odd-block fold and the overlap),
// the most tail-dominated NEON case; on amd64 they fall below the 32-wide dispatch
// threshold and measure the sub-threshold path there. At the arm64 16-wide width
// 40/63/95/248 are ragged and run the overlap while 48 is aligned there
// (48 mod 16 == 0); at the amd64 32-wide width all of 40/48/63/95/248 are ragged,
// with 63 and 95 at the worst-case residue 31.
func BenchmarkMaxAbs_N(b *testing.B) {
	for _, n := range []int{17, 25, 32, 40, 48, 63, 95, 248} {
		a := genI8(n, 1)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				_ = MaxAbs(a)
			}
		})
	}
}

func BenchmarkMinMax_N(b *testing.B) {
	for _, n := range []int{17, 25, 32, 40, 48, 63, 95, 248} {
		a := genI8(n, 1)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				_, _ = MinMax(a)
			}
		})
	}
}
