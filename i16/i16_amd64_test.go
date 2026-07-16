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
	if minSSE2Dot > 2*8 || minAVX2Dot > 2*16 {
		t.Fatalf("dispatch thresholds too high (SSE2 %d, AVX2 %d): DotProduct would not vectorize at the short lengths it was written for",
			minSSE2Dot, minAVX2Dot)
	}
}

// xcorr4Kernels mirrors dotKernels for the 4-lag cross-correlation tiers.
type xcorr4Kernel struct {
	name      string
	available bool
	fn        func(dst []int32, x, y []int16)
}

func xcorr4Kernels() []xcorr4Kernel {
	return []xcorr4Kernel{
		{"AVX2", cpu.X86.AVX2, xcorr4AVX2},
		{"SSE2", cpu.X86.SSE2, xcorr4SSE2},
	}
}

// TestXCorr4AMD64_ParityWithGo drives each 4-lag kernel directly, over lengths
// the dispatcher would never route to it, so a threshold change cannot quietly
// reduce this to a test of the Go reference against itself.
func TestXCorr4AMD64_ParityWithGo(t *testing.T) {
	for _, k := range xcorr4Kernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, xn := range []int{1, 2, 7, 8, 9, 15, 16, 17, 23, 31, 32, 33, 64, 240} {
				x := genI16(xn, 121)
				y := genI16(xn+3, 122) // exactly the window the dispatcher passes
				dst := make([]int32, xcorrLagBlock)
				k.fn(dst, x, y)
				for lag := range xcorrLagBlock {
					if got, want := dst[lag], dotOracle(x, y[lag:]); got != want {
						t.Errorf("xcorr4%s xn=%d: dst[%d] = %d, want %d", k.name, xn, lag, got, want)
					}
				}
			}
		})
	}
}

// TestXCorr4AMD64_ShortWindowIsBounded feeds each kernel a y window shorter
// than its contract and asserts it clamps rather than reading past the end.
func TestXCorr4AMD64_ShortWindowIsBounded(t *testing.T) {
	for _, k := range xcorr4Kernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			x := genI16(32, 123)
			for _, yn := range []int{0, 1, 2, 3, 4, 8, 16, 31, 34} {
				y := genI16(yn, 124)
				dst := make([]int32, xcorrLagBlock)
				k.fn(dst, x, y) // must not fault or read past y
				n := max(min(len(x), yn-3), 0)
				for lag := range xcorrLagBlock {
					if got, want := dst[lag], dotOracle(x[:n], y[min(lag, yn):]); got != want {
						t.Errorf("xcorr4%s short window yn=%d: dst[%d] = %d, want %d", k.name, yn, lag, got, want)
					}
				}
			}
		})
	}
}

// TestXCorrDispatch_ReachesSIMD pins the dispatch STATE that XCorr's SIMD path
// depends on. See the arm64 counterpart for what it does NOT prove (nothing
// here establishes that xcorrI16 calls a kernel, because the kernel is
// bit-identical to the Go reference and a dead dispatcher passes every test)
// and for the build-tagged hook that would close that gap.
func TestXCorrDispatch_ReachesSIMD(t *testing.T) {
	if hasSSE2 != cpu.X86.SSE2 {
		t.Fatalf("hasSSE2 = %v but cpu.X86.SSE2 = %v: dispatch flag is not wired to CPU detection", hasSSE2, cpu.X86.SSE2)
	}
	if hasAVX2 != cpu.X86.AVX2 {
		t.Fatalf("hasAVX2 = %v but cpu.X86.AVX2 = %v: dispatch flag is not wired to CPU detection", hasAVX2, cpu.X86.AVX2)
	}
	if !hasSSE2 {
		t.Fatal("hasSSE2 is false on amd64: PMADDWD is SSE2 baseline, so XCorr should never fall back to Go here")
	}
	// One vector block each, matching the kernel bodies. A bound of 2x the
	// block would be a tautology against the real values (8 and 16).
	if minSSE2XCorr > 8 || minAVX2XCorr > 16 {
		t.Fatalf("XCorr thresholds too high (SSE2 %d, AVX2 %d): would not vectorize at the x lengths it was written for",
			minSSE2XCorr, minAVX2XCorr)
	}
}

// TestXCorr4AMD64_LongWindowIsClamped covers the other half of the kernel's
// n = min(len(x), len(y)-3) clamp. ShortWindowIsBounded only probes
// len(y)-3 < len(x); ParityWithGo passes len(y) == len(x)+3, where both
// operands of the min are equal and a mutant that dropped the min entirely
// would still agree.
func TestXCorr4AMD64_LongWindowIsClamped(t *testing.T) {
	for _, k := range xcorr4Kernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, xn := range []int{1, 8, 9, 16, 17, 32} {
				// x MUST be a prefix of a longer allocation: the mutant this
				// test exists for reads past the end of x, and past a
				// standalone slice that is zeroed memory, which multiplies to
				// 0 and leaves the answer correct. Non-zero bytes past x are
				// what make the over-read observable rather than a coin flip
				// on heap layout. Without this, the test detects nothing on
				// amd64 at any xn.
				backing := genI16(xn+200, 209)
				x := backing[:xn]
				y := genI16(xn+40, 210) // len(y)-3 far exceeds len(x)
				dst := make([]int32, xcorrLagBlock)
				k.fn(dst, x, y)
				for lag := range xcorrLagBlock {
					if got, want := dst[lag], dotOracle(x, y[lag:]); got != want {
						t.Errorf("xcorr4%s long window xn=%d: dst[%d] = %d, want %d", k.name, xn, lag, got, want)
					}
				}
			}
		})
	}
}
