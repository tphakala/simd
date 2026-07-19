//go:build amd64

package i32

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// TestMaxAbsAVX2_ParityWithGo drives the kernel directly across lengths that force
// the 8-wide vector body plus every scalar-tail remainder (block boundaries and
// primes in 8..40), over lengths the dispatcher would route the same way, so a
// threshold change cannot quietly reduce this to a test of the Go reference against
// itself. MinInt32 rides index 0 so the wrapping -minVal combine is exercised, and
// MaxInt32 rides the last index so the scalar tail must be folded in.
func TestMaxAbsAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	lens := []int{8, 9, 11, 13, 16, 17, 19, 23, 24, 29, 31, 32, 37, 40}
	for _, n := range lens {
		a := genI32(n, 71)
		a[0] = math.MinInt32
		a[n-1] = math.MaxInt32
		if got, want := maxAbsAVX2(a), maxAbsGo(a); got != want {
			t.Fatalf("maxAbsAVX2 n=%d = %d, want %d (reference)", n, got, want)
		}
		if got, want := maxAbsAVX2(a), maxAbsOracle(a); got != want {
			t.Fatalf("maxAbsAVX2 n=%d = %d, want %d (oracle)", n, got, want)
		}
	}
}

// TestMaxAbsAVX2_PlantedExtreme plants the result-driving extreme at every position
// of a block-plus-tail length, so a kernel that drops a vector lane or skips the
// scalar tail misses the extreme where it lives and is caught. MaxInt32 drives the
// answer through the max accumulator; MinInt32+1 (a large-magnitude negative)
// drives it through the min accumulator, so both lanes are load-bearing.
func TestMaxAbsAVX2_PlantedExtreme(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 11 // one 8-wide block + 3 tail
	for _, ext := range []int32{math.MaxInt32, math.MinInt32 + 1} {
		for pos := range n {
			a := make([]int32, n)
			for i := range a {
				a[i] = int32(i%3 - 1) // tame body: -1, 0, 1
			}
			a[pos] = ext
			if got, want := maxAbsAVX2(a), maxAbsOracle(a); got != want {
				t.Fatalf("maxAbsAVX2 ext=%d pos=%d = %d, want %d", ext, pos, got, want)
			}
		}
	}
}

// TestMaxAbsAVX2_OverRead catches a kernel that reads past len(a). The in-range
// body is tame (values in -1..1, so the true peak magnitude is 1), while the slack
// past n is poisoned with the absolute extremes MinInt32 and MaxInt32, which would
// dominate the signed min/max reduction if read. a is backing[:n] over a backing of
// length n+8 (one full 8-wide block of slack), so a kernel that reads a stray block
// or a scalar tail past n lands in the poisoned (still allocated) memory and its
// result flips away from the oracle over a[:n]; a correct kernel stops at n and
// stays equal. For a min/max reduction the poison must be MORE extreme than any
// in-range element, the opposite of an additive kernel where a zero or an
// even-count MinInt32 would be an invisible identity.
func TestMaxAbsAVX2_OverRead(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range []int{8, 9, 11, 13, 16, 17, 23, 31} {
		backing := make([]int32, n+8)
		for i := range backing {
			backing[i] = int32(i%3 - 1) // tame body: -1, 0, 1
		}
		for i := n; i < len(backing); i++ {
			if i%2 == 0 {
				backing[i] = math.MinInt32
			} else {
				backing[i] = math.MaxInt32
			}
		}
		a := backing[:n]
		if got, want := maxAbsAVX2(a), maxAbsOracle(a); got != want {
			t.Fatalf("maxAbsAVX2 n=%d = %d, want %d: kernel read past n into poisoned slack", n, got, want)
		}
	}
}

// TestMaxAbsAVX2_AllocFree asserts the kernel runs allocation-free, the repo's
// zero-allocation contract enforced at the kernel boundary.
func TestMaxAbsAVX2_AllocFree(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	a := make([]int32, 1024)
	for i := range a {
		a[i] = int32(i*7 - 3000)
	}
	if got := testing.AllocsPerRun(100, func() { _ = maxAbsAVX2(a) }); got != 0 {
		t.Errorf("maxAbsAVX2 allocated %v times per run, want 0", got)
	}
}

// TestMaxAbsDispatch_ReachesSIMD pins the dispatch state MaxAbs depends on. It is a
// white-box check: the AVX2 kernel is bit-identical to the Go reference by design,
// so a dispatcher that silently routed every call to Go would pass every parity
// test. It must not call t.Parallel(): it reads package-level dispatch state.
func TestMaxAbsDispatch_ReachesSIMD(t *testing.T) {
	if hasAVX2 != cpu.X86.AVX2 {
		t.Fatalf("hasAVX2 = %v but cpu.X86.AVX2 = %v: dispatch flag is not wired to CPU detection", hasAVX2, cpu.X86.AVX2)
	}
	if minAVX2MaxAbs > 16 {
		t.Fatalf("minAVX2MaxAbs = %d exceeds two vector blocks: MaxAbs would not vectorize at the lengths it was written for", minAVX2MaxAbs)
	}
}
