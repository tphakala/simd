//go:build amd64

package c128

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestDotProductKernelsAMD64 exercises the SSE and AVX kernels directly against
// the scalar references, so both tiers are covered regardless of which one the
// active CPU selects.
func TestDotProductKernelsAMD64(t *testing.T) {
	for n := 0; n <= 20; n++ {
		a := dpSeq(n, 1)
		b := dpSeq(n, 4)
		want := dotProductGo(a, b)
		wantConj := dotProductConjGo(a, b)

		if cpu.X86.SSE2 {
			if got := dotProductSSE2(a, b); !dpClose(got, want) {
				t.Errorf("dotProductSSE2 n=%d = %v, want %v", n, got, want)
			}
			if got := dotProductConjSSE2(a, b); !dpClose(got, wantConj) {
				t.Errorf("dotProductConjSSE2 n=%d = %v, want %v", n, got, wantConj)
			}
		}
		if cpu.X86.AVX && cpu.X86.FMA {
			if got := dotProductAVX(a, b); !dpClose(got, want) {
				t.Errorf("dotProductAVX n=%d = %v, want %v", n, got, want)
			}
			if got := dotProductConjAVX(a, b); !dpClose(got, wantConj) {
				t.Errorf("dotProductConjAVX n=%d = %v, want %v", n, got, wantConj)
			}
		}
	}
}
