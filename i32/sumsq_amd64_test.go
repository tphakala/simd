//go:build amd64

package i32

import (
	"fmt"
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestSumSqShiftedQ31AVX2_ParityWithGo drives the kernel directly across the full
// tier-3 sweep and the shift matrix, over lengths the dispatcher would never route
// to it, so a threshold change cannot quietly reduce this to a test of the Go
// reference against itself. The kernel's VPXOR/JZ prologue makes n below one block
// (and n=0) valid to call directly. MinInt32 rides index 0 and MaxInt32 the last
// index so the SHL32/square/accumulate wraps are exercised at every length.
func TestSumSqShiftedQ31AVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range tier3Lengths {
		a := genI32(n, 81)
		if n > 0 {
			a[0] = math.MinInt32
			a[n-1] = math.MaxInt32
		}
		for _, s := range sumSqShifts {
			got := sumSqShiftedQ31AVX2(a, s)
			if want := sumSqShiftedQ31Go(a, s); got != want {
				t.Fatalf("sumSqShiftedQ31AVX2 n=%d shift=%d = %d, want %d (reference)", n, s, got, want)
			}
			if want := sumSqShiftedQ31Oracle(a, s); got != want {
				t.Fatalf("sumSqShiftedQ31AVX2 n=%d shift=%d = %d, want %d (oracle)", n, s, got, want)
			}
		}
	}
}

// TestSumSqShiftedQ31AVX2_PlantedTerm plants a distinctive nonzero term at every
// position of a block-plus-tail length over an all-zero body, so a kernel that
// drops a vector lane or skips the scalar tail loses the planted term and is caught.
func TestSumSqShiftedQ31AVX2_PlantedTerm(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 11 // one 8-wide block + 3 tail
	const planted = int32(0x2BADF00D)
	for _, s := range []int{0, 1, 7} {
		want := sumSqShiftedQ31Oracle([]int32{planted}, s)
		for pos := range n {
			a := make([]int32, n)
			a[pos] = planted
			if got := sumSqShiftedQ31AVX2(a, s); got != want {
				t.Fatalf("sumSqShiftedQ31AVX2 planted pos=%d shift=%d = %d, want %d", pos, s, got, want)
			}
		}
	}
}

// TestSumSqShiftedQ31AVX2_OverRead catches a kernel that reads past len(a). The
// in-range body is all zero (term 0), while the slack past n is poisoned with a
// value whose per-term contribution is ODD, hence of full additive order 2^32: for
// an ADDITIVE reduction the poison must not cancel at any block count (a zero or an
// even-order value like MinInt32 would be an invisible identity, the opposite of a
// min/max reduction). a is backing[:n] over a backing of n+8 (one full 8-wide block
// of slack), so a stray block or a tail past n lands in the poison and flips the
// result away from the oracle over a[:n]. The test asserts the poison term is odd,
// so the poison choice cannot silently become an identity.
func TestSumSqShiftedQ31AVX2_OverRead(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const poison = int32(0x2BADF00D)
	for _, s := range []int{0, 1} {
		if term := sumSqShiftedQ31Go([]int32{poison}, s); term%2 == 0 {
			t.Fatalf("poison term for shift=%d is even (%d): an even-order poison cannot expose an over-read", s, term)
		}
		for _, n := range []int{1, 3, 4, 5, 8, 9, 11, 13, 16, 17, 23, 31} {
			backing := make([]int32, n+8)
			for i := n; i < len(backing); i++ {
				backing[i] = poison
			}
			a := backing[:n] // a[:n] is all zero
			if got, want := sumSqShiftedQ31AVX2(a, s), sumSqShiftedQ31Oracle(a, s); got != want {
				t.Fatalf("sumSqShiftedQ31AVX2 n=%d shift=%d = %d, want %d: kernel read past n into poisoned slack", n, s, got, want)
			}
		}
	}
}

// TestSumSqShiftedQ31AVX2_AllocFree asserts the kernel runs allocation-free, the
// repo's zero-allocation contract enforced at the kernel boundary.
func TestSumSqShiftedQ31AVX2_AllocFree(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	a := make([]int32, 1024)
	for i := range a {
		a[i] = int32(i*7 - 3000)
	}
	if got := testing.AllocsPerRun(100, func() { _ = sumSqShiftedQ31AVX2(a, 5) }); got != 0 {
		t.Errorf("sumSqShiftedQ31AVX2 allocated %v times per run, want 0", got)
	}
}

// TestSumSqShiftedQ31Dispatch_ReachesSIMD checks the two dispatch preconditions the
// AVX2 kernel needs to be reached: hasAVX2 is wired to CPU detection, and the
// threshold stays low enough that realistic lengths clear it. Because the kernel is
// bit-identical to the Go reference by design, the parity tests cannot tell whether
// the kernel or the Go path ran, so these checks are what keep a mis-wired flag or a
// regressed-huge threshold from silently sending every call to Go; they do NOT prove
// the kernel dispatch branch still exists (that would need a seam this package does
// not have). It must not call t.Parallel(): it reads package-level dispatch state.
func TestSumSqShiftedQ31Dispatch_ReachesSIMD(t *testing.T) {
	if hasAVX2 != cpu.X86.AVX2 {
		t.Fatalf("hasAVX2 = %v but cpu.X86.AVX2 = %v: dispatch flag is not wired to CPU detection", hasAVX2, cpu.X86.AVX2)
	}
	// The threshold stays within two 8-wide blocks (measured: this kernel wins from
	// the first block), so a regression to a large threshold that left realistic short
	// bands on the slower Go path is caught here.
	if minAVX2SumSqShiftedQ31 > 16 {
		t.Fatalf("minAVX2SumSqShiftedQ31 = %d exceeds two vector blocks: it would not vectorize at the lengths it was written for", minAVX2SumSqShiftedQ31)
	}
}

// BenchmarkSumSqShiftedQ31CrossoverAVX2 sweeps the AVX2 kernel directly against the
// Go reference across the SIMD/scalar crossover region so minAVX2SumSqShiftedQ31 can
// be tuned on the target hardware. It benchmarks the kernel directly rather than the
// dispatched SumSqShiftedQ31, so the threshold under test does not gate the
// measurement.
func BenchmarkSumSqShiftedQ31CrossoverAVX2(b *testing.B) {
	if !cpu.X86.AVX2 {
		b.Skip("AVX2 not available")
	}
	for _, n := range []int{8, 16, 32, 64, 128, 160, 256, 512} {
		b.Run(fmt.Sprintf("AVX2_n%d", n), func(b *testing.B) { benchmarkSumSqShiftedQ31(b, n, sumSqShiftedQ31AVX2) })
		b.Run(fmt.Sprintf("Go_n%d", n), func(b *testing.B) { benchmarkSumSqShiftedQ31(b, n, sumSqShiftedQ31Go) })
	}
}
