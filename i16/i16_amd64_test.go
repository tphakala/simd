//go:build amd64

package i16

import (
	"math"
	"testing"

	"github.com/tphakala/simd/cpu"
)

// interleaveKernel/deinterleaveKernel describe one SIMD tier so the parity and
// no-overwrite checks run identically against AVX2 and SSE2.
type interleaveKernel struct {
	name      string
	available bool
	fn        func(dst, a, b []int16)
}

type deinterleaveKernel struct {
	name      string
	available bool
	fn        func(a, b, src []int16)
}

func interleaveKernels() []interleaveKernel {
	return []interleaveKernel{
		{"AVX2", cpu.X86.AVX2, interleave2AVX2},
		{"SSE2", cpu.X86.SSE2, interleave2SSE2},
	}
}

func deinterleaveKernels() []deinterleaveKernel {
	return []deinterleaveKernel{
		{"AVX2", cpu.X86.AVX2, deinterleave2AVX2},
		{"SSE2", cpu.X86.SSE2, deinterleave2SSE2},
	}
}

func TestInterleave2_ParityWithGo(t *testing.T) {
	for _, k := range interleaveKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, n := range paritySizes {
				a := make([]int16, n)
				b := make([]int16, n)
				fillPattern(a, b)

				gotSIMD := make([]int16, n*2)
				gotGo := make([]int16, n*2)
				k.fn(gotSIMD, a, b)
				interleave2Go(gotGo, a, b)

				for i := range gotGo {
					if gotSIMD[i] != gotGo[i] {
						t.Fatalf("n=%d: interleave2%s[%d] = %d, want %d (Go)", n, k.name, i, gotSIMD[i], gotGo[i])
					}
				}
			}
		})
	}
}

func TestDeinterleave2_ParityWithGo(t *testing.T) {
	for _, k := range deinterleaveKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, n := range paritySizes {
				src := make([]int16, n*2)
				for i := range src {
					src[i] = int16(i) ^ math.MinInt16
				}

				aSIMD := make([]int16, n)
				bSIMD := make([]int16, n)
				aGo := make([]int16, n)
				bGo := make([]int16, n)
				k.fn(aSIMD, bSIMD, src)
				deinterleave2Go(aGo, bGo, src)

				for i := range aGo {
					if aSIMD[i] != aGo[i] {
						t.Fatalf("n=%d: deinterleave2%s a[%d] = %d, want %d (Go)", n, k.name, i, aSIMD[i], aGo[i])
					}
					if bSIMD[i] != bGo[i] {
						t.Fatalf("n=%d: deinterleave2%s b[%d] = %d, want %d (Go)", n, k.name, i, bSIMD[i], bGo[i])
					}
				}
			}
		})
	}
}

// TestInterleave2_NoOverwrite guards the scalar tail: the kernel must not write
// past n*2 output elements even when n is not a multiple of the block.
func TestInterleave2_NoOverwrite(t *testing.T) {
	const n = 23 // one AVX2 block (16) + 7 tail; one SSE2 block (8) leaves a 7 tail too
	for _, k := range interleaveKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			a := make([]int16, n)
			b := make([]int16, n)
			fillPattern(a, b)
			dst := make([]int16, n*2+8)
			for i := range dst {
				dst[i] = math.MaxInt16 // sentinel
			}
			k.fn(dst[:n*2], a, b)
			for i := n * 2; i < len(dst); i++ {
				if dst[i] != math.MaxInt16 {
					t.Errorf("interleave2%s wrote past end at dst[%d] = %d", k.name, i, dst[i])
				}
			}
		})
	}
}

// TestDeinterleave2_NoOverwrite guards both output buffers' scalar tails.
func TestDeinterleave2_NoOverwrite(t *testing.T) {
	const n = 23
	for _, k := range deinterleaveKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			src := make([]int16, n*2)
			for i := range src {
				src[i] = int16(i) ^ math.MinInt16
			}
			a := make([]int16, n+8)
			b := make([]int16, n+8)
			for i := range a {
				a[i] = math.MaxInt16
				b[i] = math.MaxInt16
			}
			k.fn(a[:n], b[:n], src)
			for i := n; i < len(a); i++ {
				if a[i] != math.MaxInt16 {
					t.Errorf("deinterleave2%s wrote past end of a at [%d] = %d", k.name, i, a[i])
				}
				if b[i] != math.MaxInt16 {
					t.Errorf("deinterleave2%s wrote past end of b at [%d] = %d", k.name, i, b[i])
				}
			}
		})
	}
}

// dotKernel describes one amd64 SIMD tier so the dot parity checks run
// identically against AVX2 and SSE2.
type dotKernel struct {
	name      string
	available bool
	fn        func(a, b []int16) int32
}

func dotKernels() []dotKernel {
	return []dotKernel{
		{"AVX2", cpu.X86.AVX2, dotAVX2},
		{"SSE2", cpu.X86.SSE2, dotSSE2},
	}
}

// TestDotAMD64_ParityWithGo exercises each kernel directly rather than through
// DotProduct, so a dispatch threshold change can never quietly turn this into a
// test of the Go reference against itself.
func TestDotAMD64_ParityWithGo(t *testing.T) {
	for _, k := range dotKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, n := range dotLengths {
				a, b := genI16(n, 61), genI16(n, 62)
				if got, want := k.fn(a, b), dotGo(a, b); got != want {
					t.Errorf("dot%s n=%d: got %d, want %d", k.name, n, got, want)
				}
			}
		})
	}
}

// TestDotAMD64_MinInt16 pins the PMADDWD overflow case: a pair of
// (-32768 * -32768) sums to 2^31 inside one instruction and must wrap to
// MinInt32, matching the scalar reference, rather than saturate.
func TestDotAMD64_MinInt16(t *testing.T) {
	for _, k := range dotKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for n := 1; n <= 64; n++ {
				a := make([]int16, n)
				b := make([]int16, n)
				for i := range a {
					a[i], b[i] = math.MinInt16, math.MinInt16
				}
				if got, want := k.fn(a, b), dotOracle(a, b); got != want {
					t.Errorf("dot%s all-MinInt16 n=%d: got %d, want %d", k.name, n, got, want)
				}
			}
		})
	}
}

// TestDotAMD64_Clamp verifies the in-assembly min(len(a), len(b)): the kernel
// must not read the longer operand past the shorter one's length.
func TestDotAMD64_Clamp(t *testing.T) {
	for _, k := range dotKernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, n := range dotLengths {
				if n == 0 {
					continue
				}
				long, short := genI16(n+37, 63), genI16(n, 64)
				if got, want := k.fn(long, short), dotOracle(long, short); got != want {
					t.Errorf("dot%s clamp n=%d: got %d, want %d", k.name, n, got, want)
				}
				if got, want := k.fn(short, long), dotOracle(short, long); got != want {
					t.Errorf("dot%s clamp (swapped) n=%d: got %d, want %d", k.name, n, got, want)
				}
			}
		})
	}
}

// TestDotDispatch_ReachesSIMD asserts that DotProduct actually routes to a SIMD
// kernel rather than silently falling back to the Go reference. See the arm64
// counterpart for why this must be a white-box check: the kernels are
// bit-identical to dotGo, so a dead dispatcher passes every parity test.
//
// It must not call t.Parallel(): it reads package-level dispatch state.
func TestDotDispatch_ReachesSIMD(t *testing.T) {
	if hasSSE2 != cpu.X86.SSE2 {
		t.Fatalf("hasSSE2 = %v but cpu.X86.SSE2 = %v: dispatch flag is not wired to CPU detection", hasSSE2, cpu.X86.SSE2)
	}
	if hasAVX2 != cpu.X86.AVX2 {
		t.Fatalf("hasAVX2 = %v but cpu.X86.AVX2 = %v: dispatch flag is not wired to CPU detection", hasAVX2, cpu.X86.AVX2)
	}
	// SSE2 is in the GOAMD64=v1 baseline, so on amd64 the dot product must
	// always have a SIMD path: a false hasSSE2 means every call runs scalar.
	if !hasSSE2 {
		t.Fatal("hasSSE2 is false on amd64: PMADDWD is SSE2 baseline, so DotProduct should never fall back to Go here")
	}
	if minSSE2Elements > 2*8 || minAVX2Elements > 2*16 {
		t.Fatalf("dispatch thresholds too high (SSE2 %d, AVX2 %d): DotProduct would not vectorize at the short lengths it was written for",
			minSSE2Elements, minAVX2Elements)
	}
}
