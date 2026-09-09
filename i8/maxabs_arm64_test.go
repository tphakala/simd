//go:build arm64

package i8

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestMaxAbsNEON_ParityWithGo drives the kernel directly across lengths that force
// the 2-block-unrolled 16-wide body plus every overlap-tail remainder (block
// boundaries and ragged residues in 16..128), over lengths the dispatcher would
// route the same way, so a threshold change cannot quietly reduce this to a test of
// the Go reference against itself. -128 rides index 0 and the last index: its
// magnitude 128 is the unique peak, so the overlap block must fold the tail lane in
// or the result drops below 128.
func TestMaxAbsNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	lens := []int{16, 17, 19, 23, 31, 32, 33, 40, 47, 48, 63, 64, 65, 79, 80, 95, 127, 128}
	for _, n := range lens {
		a := genI8(n, 71)
		a[0] = -128
		a[n-1] = -128
		if got, want := maxAbsNEON(a), maxAbsGo(a); got != want {
			t.Fatalf("maxAbsNEON n=%d = %d, want %d (reference)", n, got, want)
		}
	}
}

// TestMaxAbsNEON_PlantedExtreme plants the result-driving extreme (-128, magnitude
// 128) at every position of a two-block-plus-tail length, so a kernel that drops a
// vector lane, mis-seeds a UMAX accumulator, or skips the overlap tail misses the
// extreme where it lives and is caught. n=33 is two 16-wide blocks plus a residue-1
// tail (exercises the 2-block unroll, the odd-block fold, the chain combine, and the
// overlap block); n=17 is the single-block fast path plus a residue-1 tail.
func TestMaxAbsNEON_PlantedExtreme(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range []int{17, 33} {
		for pos := range n {
			a := make([]int8, n)
			for i := range a {
				a[i] = int8(i%50 - 25) // tame body, magnitude <= 25
			}
			a[pos] = -128
			if got, want := maxAbsNEON(a), maxAbsGo(a); got != want {
				t.Fatalf("maxAbsNEON n=%d pos=%d = %d, want %d", n, pos, got, want)
			}
		}
	}
}

// TestMaxAbsNEON_OverRead catches a kernel that reads past len(a). The in-range body
// is tame (magnitude <= 25, so the true peak is 25), while the slack past n is
// poisoned with -128 (magnitude 128), which would dominate the UMAX-of-ABS reduction
// if read. a is backing[:n] over a backing of length n+16 (one full 16-wide block of
// slack), so a kernel that reads a stray block or a scalar tail past n lands in the
// poisoned (still allocated) memory and its result jumps to 128; a correct kernel
// stops at n and stays at the tame peak.
func TestMaxAbsNEON_OverRead(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range []int{16, 17, 23, 31, 33, 47, 63} {
		backing := make([]int8, n+16)
		for i := range backing {
			backing[i] = int8(i%50 - 25) // tame body: magnitude <= 25
		}
		for i := n; i < len(backing); i++ {
			backing[i] = -128 // poison the slack with the peak magnitude
		}
		a := backing[:n]
		if got, want := maxAbsNEON(a), maxAbsGo(a); got != want {
			t.Fatalf("maxAbsNEON n=%d = %d, want %d: kernel read past n into poisoned slack", n, got, want)
		}
	}
}

// TestMaxAbsNEON_AllocFree asserts the kernel runs allocation-free, the repo's
// zero-allocation contract enforced at the kernel boundary.
func TestMaxAbsNEON_AllocFree(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	a := make([]int8, 1024)
	for i := range a {
		a[i] = int8(i*7 - 500)
	}
	if got := testing.AllocsPerRun(100, func() { _ = maxAbsNEON(a) }); got != 0 {
		t.Errorf("maxAbsNEON allocated %v times per run, want 0", got)
	}
}

// TestMaxAbsDispatch_ReachesNEON pins the dispatch state MaxAbs depends on. It is a
// white-box check: the NEON kernel is bit-identical to the Go reference by design, so
// a dispatcher that silently routed every call to Go would pass every parity test.
// It must not call t.Parallel(): it reads package-level dispatch state.
func TestMaxAbsDispatch_ReachesNEON(t *testing.T) {
	if hasNEON != cpu.ARM64.NEON {
		t.Fatalf("hasNEON = %v but cpu.ARM64.NEON = %v: dispatch flag is not wired to CPU detection", hasNEON, cpu.ARM64.NEON)
	}
	if minNEON16 > 32 {
		t.Fatalf("minNEON16 = %d exceeds two vector blocks: MaxAbs would not vectorize at the lengths it was written for", minNEON16)
	}
	// Lower bound: the overlap tail reloads a[n-16], so the kernel must never be
	// dispatched below one 16-byte block or it would read out of bounds.
	if minNEON16 < 16 {
		t.Fatalf("minNEON16 = %d is below maxAbsNEON's 16-byte block: the overlap reload of a[n-16] would read out of bounds", minNEON16)
	}
}
