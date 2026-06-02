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
