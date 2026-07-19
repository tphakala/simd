//go:build amd64

package cint

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// Kernel-direct tests for the AVX2 path. They call the assembly kernels straight,
// over lengths the dispatcher would never route to them (below the threshold), so a
// threshold change cannot quietly reduce these to a test of the Go reference
// against itself. The kernels are bit-identical to the Go reference by design, so
// the dispatch-pin test is what proves the SIMD path is actually reached.

// kernelEvenLengths drives the kernels at whole-pair lengths spanning zero, sub-
// block, single-block, block+tail and multi-block sizes on the 4-complex (8 int32)
// AVX2 blocks.
var kernelEvenLengths = []int{2, 4, 6, 8, 10, 12, 14, 16, 18, 22, 24, 30, 32, 34, 64, 66, 128, 130}

func TestMulAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range kernelEvenLengths {
		a := genI32(n, 101)
		tw := genI16(n, 102)
		plantExtremes(a, tw)
		got := make([]int32, n)
		mulAVX2(got, a, tw)
		checkMul(t, "mulAVX2", false, got, a, tw)
	}
}

func TestMulConjAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range kernelEvenLengths {
		a := genI32(n, 103)
		tw := genI16(n, 104)
		plantExtremes(a, tw)
		got := make([]int32, n)
		mulConjAVX2(got, a, tw)
		checkMul(t, "mulConjAVX2", true, got, a, tw)
	}
}

func TestMulByScalarAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	ss := []int16{math.MinInt16, math.MaxInt16, 1, -1, 0x4000}
	for _, n := range kernelEvenLengths {
		for _, s := range ss {
			a := genI32(n, 105)
			if n >= 2 {
				a[0] = math.MinInt32
				a[n-1] = math.MaxInt32
			}
			orig := append([]int32(nil), a...)
			mulByScalarAVX2(a, s)
			for i := range a {
				if want := sMulBig(orig[i], s); a[i] != want {
					t.Fatalf("mulByScalarAVX2 n=%d s=%d at %d: got %d want %d", n, s, i, a[i], want)
				}
			}
		}
	}
}

func TestAddSubAVX2_ParityWithGo(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	for _, n := range kernelEvenLengths {
		a := genI32(n, 106)
		b := genI32(n, 107)
		if n >= 2 {
			a[0], b[0] = math.MinInt32, -1
			a[n-1], b[n-1] = math.MaxInt32, 1
		}
		got := make([]int32, n)
		ref := make([]int32, n)
		addAVX2(got, a, b)
		addGo(ref, a, b)
		for i := range got {
			if got[i] != ref[i] {
				t.Fatalf("addAVX2 n=%d at %d: got %d want %d", n, i, got[i], ref[i])
			}
		}
		subAVX2(got, a, b)
		subGo(ref, a, b)
		for i := range got {
			if got[i] != ref[i] {
				t.Fatalf("subAVX2 n=%d at %d: got %d want %d", n, i, got[i], ref[i])
			}
		}
	}
}

// TestMulAVX2_OverRead catches a kernel that reads past n or writes past n. The
// in-range body is tame (values in -2..2, so every product is small), while the
// slack past n in a and tw is poisoned with the absolute extremes, and the slack in
// dst carries a sentinel. Backings are n + one full 8-int32 (8-int16) block, so a
// kernel that reads a stray block lands in the poison and its in-range results flip
// away from the tame oracle, and a kernel that writes a stray block overwrites the
// dst sentinel. A correct kernel stops exactly at n on both sides.
func TestMulAVX2_OverRead(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	testMulOverRead(t, "mulAVX2", false, mulAVX2)
	testMulOverRead(t, "mulConjAVX2", true, mulConjAVX2)
}

func testMulOverRead(t *testing.T, name string, conj bool, kernel func(dst, a []int32, tw []int16)) {
	t.Helper()
	const slack = 8 // one full 4-complex block in int32 (and in int16) units
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
		checkMul(t, name, conj, dst, a, tw) // in-range lanes must be tame
		for i := n; i < n+slack; i++ {
			if dstb[i] != sentinel {
				t.Fatalf("%s n=%d wrote past n at dst[%d] = %d (want sentinel)", name, n, i, dstb[i])
			}
		}
	}
}

// TestMulByScalarAVX2_NoOverwrite guards the in-place scalar tail: it may not write
// past n when n is not a multiple of the 8-lane block.
func TestMulByScalarAVX2_NoOverwrite(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 10
	a := make([]int32, n+8)
	for i := range a {
		a[i] = math.MaxInt32
	}
	mulByScalarAVX2(a[:n], 0x1234)
	for i := n; i < len(a); i++ {
		if a[i] != math.MaxInt32 {
			t.Errorf("mulByScalarAVX2 wrote past end at a[%d] = %d", i, a[i])
		}
	}
}

// TestMulAVX2_AllocFree asserts the kernels run allocation-free at the kernel
// boundary.
func TestMulAVX2_AllocFree(t *testing.T) {
	if !cpu.X86.AVX2 {
		t.Skip("AVX2 not available")
	}
	const n = 1024
	a := make([]int32, n)
	b := make([]int32, n)
	tw := make([]int16, n)
	dst := make([]int32, n)
	if got := testing.AllocsPerRun(100, func() { mulAVX2(dst, a, tw) }); got != 0 {
		t.Errorf("mulAVX2 allocated %v times per run, want 0", got)
	}
	if got := testing.AllocsPerRun(100, func() { mulConjAVX2(dst, a, tw) }); got != 0 {
		t.Errorf("mulConjAVX2 allocated %v times per run, want 0", got)
	}
	if got := testing.AllocsPerRun(100, func() { mulByScalarAVX2(a, 0x1234) }); got != 0 {
		t.Errorf("mulByScalarAVX2 allocated %v times per run, want 0", got)
	}
	if got := testing.AllocsPerRun(100, func() { addAVX2(dst, a, b) }); got != 0 {
		t.Errorf("addAVX2 allocated %v times per run, want 0", got)
	}
	if got := testing.AllocsPerRun(100, func() { subAVX2(dst, a, b) }); got != 0 {
		t.Errorf("subAVX2 allocated %v times per run, want 0", got)
	}
}

// TestDispatchReachesSIMD_AVX2 pins the dispatch state the public API depends on. It
// is a white-box check: the AVX2 kernels are bit-identical to the Go reference by
// design, so a dispatcher that silently routed every call to Go would pass every
// parity test. It must not call t.Parallel(): it reads package-level dispatch state.
func TestDispatchReachesSIMD_AVX2(t *testing.T) {
	if hasAVX2 != cpu.X86.AVX2 {
		t.Fatalf("hasAVX2 = %v but cpu.X86.AVX2 = %v: dispatch flag is not wired to CPU detection", hasAVX2, cpu.X86.AVX2)
	}
	for name, th := range map[string]int{
		"minAVX2Add": minAVX2Add, "minAVX2Sub": minAVX2Sub,
		"minAVX2MulByScalar": minAVX2MulByScalar, "minAVX2Mul": minAVX2Mul,
		"minAVX2MulConj": minAVX2MulConj,
	} {
		if th > 16 {
			t.Fatalf("%s = %d exceeds two vector blocks: the op would not vectorize at the lengths it was written for", name, th)
		}
	}
}
