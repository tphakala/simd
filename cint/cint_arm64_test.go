//go:build arm64

package cint

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// Kernel-direct tests for the NEON path. They call the assembly kernels straight,
// over lengths the dispatcher would never route to them, so a threshold change
// cannot quietly reduce these to a test of the Go reference against itself. The
// kernels are bit-identical to the Go reference by design, so the dispatch-pin test
// is what proves the SIMD path is actually reached.

// kernelEvenLengthsNEON drives the kernels at whole-pair lengths spanning zero,
// sub-block, single-block, block+tail and multi-block sizes on the 4-complex
// (8 int32) Mul blocks and the 4-int32 Add/Sub/MulByScalar blocks.
var kernelEvenLengthsNEON = []int{2, 4, 6, 8, 10, 12, 14, 16, 18, 22, 24, 30, 32, 34, 64, 66, 128, 130}

func TestMulNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range kernelEvenLengthsNEON {
		a := genI32(n, 101)
		tw := genI16(n, 102)
		plantExtremes(a, tw)
		got := make([]int32, n)
		mulNEON(got, a, tw)
		checkMul(t, "mulNEON", false, got, a, tw)
	}
}

func TestMulConjNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range kernelEvenLengthsNEON {
		a := genI32(n, 103)
		tw := genI16(n, 104)
		plantExtremes(a, tw)
		got := make([]int32, n)
		mulConjNEON(got, a, tw)
		checkMul(t, "mulConjNEON", true, got, a, tw)
	}
}

func TestMulByScalarNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	ss := []int16{math.MinInt16, math.MaxInt16, 1, -1, 0x4000}
	for _, n := range kernelEvenLengthsNEON {
		for _, s := range ss {
			a := genI32(n, 105)
			if n >= 2 {
				a[0] = math.MinInt32
				a[n-1] = math.MaxInt32
			}
			orig := append([]int32(nil), a...)
			mulByScalarNEON(a, s)
			for i := range a {
				if want := sMulBig(orig[i], s); a[i] != want {
					t.Fatalf("mulByScalarNEON n=%d s=%d at %d: got %d want %d", n, s, i, a[i], want)
				}
			}
		}
	}
}

func TestAddSubNEON_ParityWithGo(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	for _, n := range kernelEvenLengthsNEON {
		a := genI32(n, 106)
		b := genI32(n, 107)
		if n >= 2 {
			a[0], b[0] = math.MinInt32, -1
			a[n-1], b[n-1] = math.MaxInt32, 1
		}
		got := make([]int32, n)
		ref := make([]int32, n)
		addNEON(got, a, b)
		addGo(ref, a, b)
		for i := range got {
			if got[i] != ref[i] {
				t.Fatalf("addNEON n=%d at %d: got %d want %d", n, i, got[i], ref[i])
			}
		}
		subNEON(got, a, b)
		subGo(ref, a, b)
		for i := range got {
			if got[i] != ref[i] {
				t.Fatalf("subNEON n=%d at %d: got %d want %d", n, i, got[i], ref[i])
			}
		}
	}
}

// TestMulNEON_OverRead catches a kernel that reads past n or writes past n. The
// in-range body is tame (values in -2..2), the slack past n in a and tw is poisoned
// with the absolute extremes, and the slack in dst carries a sentinel. Backings are
// n + one full 8-int32 (8-int16) block, so a stray read lands in the poison and
// flips the in-range results, and a stray write overwrites the dst sentinel.
func TestMulNEON_OverRead(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	testMulOverReadNEON(t, "mulNEON", false, mulNEON)
	testMulOverReadNEON(t, "mulConjNEON", true, mulConjNEON)
}

func testMulOverReadNEON(t *testing.T, name string, conj bool, kernel func(dst, a []int32, tw []int16)) {
	t.Helper()
	const slack = 8
	const sentinel = int32(0x5EED1234)
	for _, n := range []int{8, 10, 12, 14, 18, 22, 30} {
		ab := make([]int32, n+slack)
		twb := make([]int16, n+slack)
		dstb := make([]int32, n+slack)
		for i := range ab {
			ab[i] = int32(i%5 - 2) // tame body: -2..2
			twb[i] = int16(i%5 - 2)
			dstb[i] = sentinel
		}
		for i := n; i < n+slack; i++ {
			if i%2 == 0 {
				ab[i], twb[i] = math.MinInt32, math.MinInt16
			} else {
				ab[i], twb[i] = math.MaxInt32, math.MaxInt16
			}
		}
		a, tw, dst := ab[:n], twb[:n], dstb[:n]
		kernel(dst, a, tw)
		checkMul(t, name, conj, dst, a, tw)
		for i := n; i < n+slack; i++ {
			if dstb[i] != sentinel {
				t.Fatalf("%s n=%d wrote past n at dst[%d] = %d (want sentinel)", name, n, i, dstb[i])
			}
		}
	}
}

// TestMulByScalarNEON_NoOverwrite guards the in-place scalar tail: it may not write
// past n when n is not a multiple of the 4-lane block.
func TestMulByScalarNEON_NoOverwrite(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 10
	a := make([]int32, n+8)
	for i := range a {
		a[i] = math.MaxInt32
	}
	mulByScalarNEON(a[:n], 0x1234)
	for i := n; i < len(a); i++ {
		if a[i] != math.MaxInt32 {
			t.Errorf("mulByScalarNEON wrote past end at a[%d] = %d", i, a[i])
		}
	}
}

// TestMulNEON_AllocFree asserts the kernels run allocation-free at the kernel
// boundary.
func TestMulNEON_AllocFree(t *testing.T) {
	if !cpu.ARM64.NEON {
		t.Skip("NEON not available")
	}
	const n = 1024
	a := make([]int32, n)
	b := make([]int32, n)
	tw := make([]int16, n)
	dst := make([]int32, n)
	if got := testing.AllocsPerRun(100, func() { mulNEON(dst, a, tw) }); got != 0 {
		t.Errorf("mulNEON allocated %v times per run, want 0", got)
	}
	if got := testing.AllocsPerRun(100, func() { mulConjNEON(dst, a, tw) }); got != 0 {
		t.Errorf("mulConjNEON allocated %v times per run, want 0", got)
	}
	if got := testing.AllocsPerRun(100, func() { mulByScalarNEON(a, 0x1234) }); got != 0 {
		t.Errorf("mulByScalarNEON allocated %v times per run, want 0", got)
	}
	if got := testing.AllocsPerRun(100, func() { addNEON(dst, a, b) }); got != 0 {
		t.Errorf("addNEON allocated %v times per run, want 0", got)
	}
	if got := testing.AllocsPerRun(100, func() { subNEON(dst, a, b) }); got != 0 {
		t.Errorf("subNEON allocated %v times per run, want 0", got)
	}
}

// TestDispatchReachesSIMD_NEON pins the dispatch state the public API depends on. It
// is a white-box check: the NEON kernels are bit-identical to the Go reference by
// design, so a dispatcher that silently routed every call to Go would pass every
// parity test. It must not call t.Parallel(): it reads package-level dispatch state.
func TestDispatchReachesSIMD_NEON(t *testing.T) {
	if hasNEON != cpu.ARM64.NEON {
		t.Fatalf("hasNEON = %v but cpu.ARM64.NEON = %v: dispatch flag is not wired to CPU detection", hasNEON, cpu.ARM64.NEON)
	}
	for name, th := range map[string]int{
		"minNEONAdd": minNEONAdd, "minNEONSub": minNEONSub,
		"minNEONMulByScalar": minNEONMulByScalar, "minNEONMul": minNEONMul,
		"minNEONMulConj": minNEONMulConj,
	} {
		if th > 16 {
			t.Fatalf("%s = %d exceeds two vector blocks: the op would not vectorize at the lengths it was written for", name, th)
		}
	}
}
