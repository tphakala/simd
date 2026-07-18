package i8

import (
	"fmt"
	"testing"
)

// BenchmarkMaxAbs_N and BenchmarkMinMax_N guard the overlapping-final-block tail
// (#149) on the 32-wide MaxAbs/MinMax reductions: a residue of 1..31 bytes was
// served by a serial compare/cmov scalar chain (up to 31 dependent steps), now
// absorbed by one overlapping 32-wide block. The fixed 4096-byte benchmarks are
// all n%32==0 and never run the overlap block, so residue lengths must be
// measured explicitly. 32 is the aligned sentinel (overlap skipped, the untouched
// control); 40/48/63/95/248 are ragged so the overlap block runs, with 63 and 95
// at the worst-case residue 31.
func BenchmarkMaxAbs_N(b *testing.B) {
	for _, n := range []int{32, 40, 48, 63, 95, 248} {
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
	for _, n := range []int{32, 40, 48, 63, 95, 248} {
		a := genI8(n, 1)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			b.SetBytes(int64(n))
			for b.Loop() {
				_, _ = MinMax(a)
			}
		})
	}
}
