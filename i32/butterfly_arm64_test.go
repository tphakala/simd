//go:build arm64

package i32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestButterflyNEON_ParityWithGo drives the kernel directly across the full
// tier-3 sweep, over lengths the dispatcher would never route to it, so a
// threshold change cannot quietly reduce this to a test of the Go reference
// against itself. Index 0 carries MinInt32/1 (the difference underflows) and the
// last index MaxInt32/1 (the sum overflows), so both wraps run at every length,
// and the lane order is checked against the oracle at every position via the
// value-matrix test in butterfly_test.go.
func TestButterflyNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range tier3Lengths {
		lo := genI32(n, 41)
		hi := genI32(n, 42)
		if n > 0 {
			lo[0], hi[0] = math.MinInt32, 1
			lo[n-1], hi[n-1] = math.MaxInt32, 1
		}
		origLo := append([]int32(nil), lo...)
		origHi := append([]int32(nil), hi...)

		gotLo := append([]int32(nil), lo...)
		gotHi := append([]int32(nil), hi...)
		wantLo := append([]int32(nil), lo...)
		wantHi := append([]int32(nil), hi...)
		butterflyNEON(gotLo, gotHi)
		butterflyGo(wantLo, wantHi)
		for i := range gotLo {
			if gotLo[i] != wantLo[i] || gotHi[i] != wantHi[i] {
				t.Fatalf("butterflyNEON n=%d: at %d got (%d,%d), want (%d,%d)", n, i, gotLo[i], gotHi[i], wantLo[i], wantHi[i])
			}
			if ws, wd := butterflyOracle(origLo[i], origHi[i]); gotLo[i] != ws || gotHi[i] != wd {
				t.Fatalf("butterflyNEON n=%d: at %d got (%d,%d), oracle (%d,%d)", n, i, gotLo[i], gotHi[i], ws, wd)
			}
		}
	}
}

// TestButterflyNEON_NoOverwrite guards the scalar tail: the kernel may not write
// past n when n is not a multiple of the 4-lane block, in either slice.
func TestButterflyNEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 11
	lo := genI32(n+8, 43)
	hi := genI32(n+8, 44)
	for i := n; i < len(lo); i++ {
		lo[i] = math.MaxInt32
		hi[i] = math.MinInt32
	}
	butterflyNEON(lo[:n], hi[:n])
	for i := n; i < len(lo); i++ {
		if lo[i] != math.MaxInt32 {
			t.Errorf("butterflyNEON wrote past end of lo at %d = %d", i, lo[i])
		}
		if hi[i] != math.MinInt32 {
			t.Errorf("butterflyNEON wrote past end of hi at %d = %d", i, hi[i])
		}
	}
}

// TestButterflyNEON_AllocFree asserts the kernel runs allocation-free, the repo's
// zero-allocation contract enforced at the kernel boundary.
func TestButterflyNEON_AllocFree(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 1024
	lo := make([]int32, n)
	hi := make([]int32, n)
	if got := testing.AllocsPerRun(100, func() { butterflyNEON(lo, hi) }); got != 0 {
		t.Errorf("butterflyNEON allocated %v times per run, want 0", got)
	}
}

// TestButterflyDispatch_ReachesNEON pins the dispatch state Butterfly depends on.
// It is a white-box check: the NEON kernel is bit-identical to the Go reference by
// design, so a dispatcher that silently routed every call to Go would pass every
// parity test. It must not call t.Parallel(): it reads package-level dispatch
// state.
func TestButterflyDispatch_ReachesNEON(t *testing.T) {
	if hasNEON != cpu.ARM64.NEON {
		t.Fatalf("hasNEON = %v but cpu.ARM64.NEON = %v: dispatch flag is not wired to CPU detection", hasNEON, cpu.ARM64.NEON)
	}
	if minNEONButterfly > 8 {
		t.Fatalf("minNEONButterfly = %d exceeds two vector blocks: Butterfly would not vectorize at the lengths it was written for", minNEONButterfly)
	}
}
