//go:build arm64

package i32

import (
	"fmt"
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestSumSqShiftedQ31NEON_ParityWithGo drives the kernel directly across the full
// tier-3 sweep and the shift matrix, over lengths the dispatcher would never route
// to it, so a threshold change cannot quietly reduce this to a test of the Go
// reference against itself. The kernel's VEOR/CBZ prologue makes n below one block
// (and n=0) valid to call directly. MinInt32 rides index 0 and MaxInt32 the last
// index so the SHL32/square/accumulate wraps are exercised at every length.
func TestSumSqShiftedQ31NEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range tier3Lengths {
		a := genI32(n, 81)
		if n > 0 {
			a[0] = math.MinInt32
			a[n-1] = math.MaxInt32
		}
		for _, s := range sumSqShifts {
			got := sumSqShiftedQ31NEON(a, s)
			if want := sumSqShiftedQ31Go(a, s); got != want {
				t.Fatalf("sumSqShiftedQ31NEON n=%d shift=%d = %d, want %d (reference)", n, s, got, want)
			}
			if want := sumSqShiftedQ31Oracle(a, s); got != want {
				t.Fatalf("sumSqShiftedQ31NEON n=%d shift=%d = %d, want %d (oracle)", n, s, got, want)
			}
		}
	}
}

// TestSumSqShiftedQ31NEON_PlantedTerm plants a distinctive nonzero term at every
// position of a block-plus-tail length over an all-zero body, so a kernel that
// drops a vector lane or skips the scalar tail loses the planted term and is caught.
func TestSumSqShiftedQ31NEON_PlantedTerm(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 11 // two 4-wide blocks + 3 tail
	const planted = int32(0x2BADF00D)
	for _, s := range []int{0, 1, 7} {
		want := sumSqShiftedQ31Oracle([]int32{planted}, s)
		for pos := range n {
			a := make([]int32, n)
			a[pos] = planted
			if got := sumSqShiftedQ31NEON(a, s); got != want {
				t.Fatalf("sumSqShiftedQ31NEON planted pos=%d shift=%d = %d, want %d", pos, s, got, want)
			}
		}
	}
}

// TestSumSqShiftedQ31NEON_OverRead catches a kernel that reads past len(a). The
// in-range body is all zero (term 0), while the slack past n is poisoned with a
// value whose per-term contribution is ODD, hence of full additive order 2^32: for
// an ADDITIVE reduction the poison must not cancel at any block count (a zero or an
// even-order value like MinInt32 would be an invisible identity, the opposite of a
// min/max reduction). a is backing[:n] over a backing of n+4 (one full 4-wide block
// of slack), so a stray block or a tail past n lands in the poison and flips the
// result away from the oracle over a[:n]. The test asserts the poison term is odd,
// so the poison choice cannot silently become an identity.
func TestSumSqShiftedQ31NEON_OverRead(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const poison = int32(0x2BADF00D)
	for _, s := range []int{0, 1} {
		if term := sumSqShiftedQ31Go([]int32{poison}, s); term%2 == 0 {
			t.Fatalf("poison term for shift=%d is even (%d): an even-order poison cannot expose an over-read", s, term)
		}
		for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 17, 23} {
			backing := make([]int32, n+4)
			for i := n; i < len(backing); i++ {
				backing[i] = poison
			}
			a := backing[:n] // a[:n] is all zero
			if got, want := sumSqShiftedQ31NEON(a, s), sumSqShiftedQ31Oracle(a, s); got != want {
				t.Fatalf("sumSqShiftedQ31NEON n=%d shift=%d = %d, want %d: kernel read past n into poisoned slack", n, s, got, want)
			}
		}
	}
}

// TestSumSqShiftedQ31NEON_AllocFree asserts the kernel runs allocation-free, the
// repo's zero-allocation contract enforced at the kernel boundary.
func TestSumSqShiftedQ31NEON_AllocFree(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	a := make([]int32, 1024)
	for i := range a {
		a[i] = int32(i*7 - 3000)
	}
	if got := testing.AllocsPerRun(100, func() { _ = sumSqShiftedQ31NEON(a, 5) }); got != 0 {
		t.Errorf("sumSqShiftedQ31NEON allocated %v times per run, want 0", got)
	}
}

// TestSumSqShiftedQ31Dispatch_ReachesSIMD checks the two dispatch preconditions the
// NEON kernel needs to be reached: hasNEON is wired to CPU detection, and the
// threshold stays low enough that realistic lengths clear it. Because the kernel is
// bit-identical to the Go reference by design, the parity tests cannot tell whether
// the kernel or the Go path ran, so these checks are what keep a mis-wired flag or a
// regressed-huge threshold from silently sending every call to Go; they do NOT prove
// the kernel dispatch branch still exists (that would need a seam this package does
// not have). It must not call t.Parallel(): it reads package-level dispatch state.
func TestSumSqShiftedQ31Dispatch_ReachesSIMD(t *testing.T) {
	if hasNEON != cpu.ARM64.NEON {
		t.Fatalf("hasNEON = %v but cpu.ARM64.NEON = %v: dispatch flag is not wired to CPU detection", hasNEON, cpu.ARM64.NEON)
	}
	if minNEONSumSqShiftedQ31 > 16 {
		t.Fatalf("minNEONSumSqShiftedQ31 = %d exceeds two vector blocks: it would not vectorize at the lengths it was written for", minNEONSumSqShiftedQ31)
	}
}

// BenchmarkSumSqShiftedQ31CrossoverNEON sweeps the NEON kernel directly against the
// Go reference so minNEONSumSqShiftedQ31 can be confirmed on the target hardware. It
// benchmarks the kernel directly rather than the dispatched SumSqShiftedQ31, so the
// threshold under test does not gate the measurement.
func BenchmarkSumSqShiftedQ31CrossoverNEON(b *testing.B) {
	if !cpu.ARM64.NEON {
		b.Skip("NEON not available")
	}
	for _, n := range []int{4, 8, 16, 32, 64, 128, 256} {
		b.Run(fmt.Sprintf("NEON_n%d", n), func(b *testing.B) { benchmarkSumSqShiftedQ31(b, n, sumSqShiftedQ31NEON) })
		b.Run(fmt.Sprintf("Go_n%d", n), func(b *testing.B) { benchmarkSumSqShiftedQ31(b, n, sumSqShiftedQ31Go) })
	}
}
