//go:build amd64

package i8

import (
	"testing"

	"github.com/tphakala/simd/cpu"
)

// dotOracle is an independent scalar reference for a single int32-accumulated
// int8 dot product with two's-complement wraparound, over min(len(a), len(b))
// elements. It shares no code with dotGo or the kernels.
func dotOracle(a, b []int8) int32 {
	var s int32
	for i := range min(len(a), len(b)) {
		s += int32(a[i]) * int32(b[i])
	}
	return s
}

// dp4Kernel is one direct 4-row DotProductBatch kernel plus whether the running
// CPU supports it. Driving the kernels directly (rather than only through the
// dispatcher) pins each one at lengths the dispatch thresholds would never route
// to it, so a threshold change cannot quietly reduce these to a test of the Go
// reference against itself.
type dp4Kernel struct {
	name      string
	available bool
	fn        func(res []int32, r0, r1, r2, r3, vec []int8)
}

func dotProduct4Kernels() []dp4Kernel {
	return []dp4Kernel{
		{"AVXVNNI", cpu.X86.AVXVNNI, dotProduct4AVXVNNI},
		{"AVX2", cpu.X86.AVX2, dotProduct4AVX2},
	}
}

// dotProduct4Lengths sweeps every length 1..80 (so every transition between the
// 32-wide loop, the 16- and 8-wide XMM peels, and the scalar tail is pinned at
// several block counts) plus a few larger aligned and ragged sizes.
var dotProduct4Lengths = func() []int {
	lengths := make([]int, 0, 88)
	for n := 1; n <= 80; n++ {
		lengths = append(lengths, n)
	}
	return append(lengths, 96, 127, 128, 255, 256, 257)
}()

// TestDotProduct4AMD64_ParityWithGo drives each 4-row kernel directly across the
// length sweep and checks all four row results against the independent oracle.
func TestDotProduct4AMD64_ParityWithGo(t *testing.T) {
	for _, k := range dotProduct4Kernels() {
		t.Run(k.name, func(t *testing.T) {
			if !k.available {
				t.Skipf("%s not available", k.name)
			}
			for _, n := range dotProduct4Lengths {
				r0 := genI8(n, uint32(n)*4+1)
				r1 := genI8(n, uint32(n)*4+2)
				r2 := genI8(n, uint32(n)*4+3)
				r3 := genI8(n, uint32(n)*4+4)
				vec := genI8(n, uint32(n)*4+5)
				res := make([]int32, 4)
				k.fn(res, r0, r1, r2, r3, vec)
				for i, row := range [][]int8{r0, r1, r2, r3} {
					if got, want := res[i], dotOracle(row, vec); got != want {
						t.Fatalf("dotProduct4%s n=%d: res[%d] = %d, want %d", k.name, n, i, got, want)
					}
				}
			}
		})
	}
}

// TestDotProduct4AMD64_Extremes pins the bias+correction math (the AVX-VNNI
// kernel biases each row byte by XOR 0x80 to feed VPDPBUSD, then subtracts
// 128*sum(vec)) at the int8 extremes and across forced int32 wraparound. A
// saturating kernel, or a wrong correction term, diverges here.
func TestDotProduct4AMD64_Extremes(t *testing.T) {
	// 200031 = 16 + 8 + 6250*32 + 7, so a single call exercises the 16- and
	// 8-wide XMM peels, the 32-wide YMM loop, and the 7-element scalar tail, all
	// while 200031*127*127 (and *(-128)*(-128)) overflow int32 many times over, so
	// the bias correction is stressed through every path under wraparound.
	const nBig = 200031
	fill := func(n int, v int8) []int8 {
		s := make([]int8, n)
		for i := range s {
			s[i] = v
		}
		return s
	}
	alt := func(n int, a, b int8) []int8 { // alternating extremes
		s := make([]int8, n)
		for i := range s {
			if i%2 == 0 {
				s[i] = a
			} else {
				s[i] = b
			}
		}
		return s
	}
	cases := []struct {
		name           string
		r0, r1, r2, r3 []int8
		vec            []int8
	}{
		{"all_max", fill(nBig, 127), fill(nBig, 127), fill(nBig, 127), fill(nBig, 127), fill(nBig, 127)},
		{"all_min", fill(nBig, -128), fill(nBig, -128), fill(nBig, -128), fill(nBig, -128), fill(nBig, -128)},
		{"min_vec", fill(nBig, 127), fill(nBig, -128), fill(nBig, 1), fill(nBig, -1), fill(nBig, -128)},
		{"alt", alt(nBig, 127, -128), alt(nBig, -128, 127), alt(nBig, 1, -1), alt(nBig, -1, 1), alt(nBig, 127, -128)},
	}
	for _, k := range dotProduct4Kernels() {
		if !k.available {
			continue
		}
		t.Run(k.name, func(t *testing.T) {
			for _, c := range cases {
				res := make([]int32, 4)
				k.fn(res, c.r0, c.r1, c.r2, c.r3, c.vec)
				rows := [][]int8{c.r0, c.r1, c.r2, c.r3}
				for i, row := range rows {
					if got, want := res[i], dotOracle(row, c.vec); got != want {
						t.Fatalf("dotProduct4%s %s: res[%d] = %d, want %d", k.name, c.name, i, got, want)
					}
				}
			}
		})
	}
}

// TestDotProduct4AMD64_AllocFree asserts the direct kernel path allocates
// nothing. Length 63 exercises both the width-aligned body and the scalar tail.
func TestDotProduct4AMD64_AllocFree(t *testing.T) {
	for _, k := range dotProduct4Kernels() {
		if !k.available {
			continue
		}
		t.Run(k.name, func(t *testing.T) {
			r0 := genI8(63, 1)
			r1 := genI8(63, 2)
			r2 := genI8(63, 3)
			r3 := genI8(63, 4)
			vec := genI8(63, 5)
			res := make([]int32, 4)
			if got := testing.AllocsPerRun(10, func() { k.fn(res, r0, r1, r2, r3, vec) }); got != 0 {
				t.Fatalf("dotProduct4%s allocated %v times, want 0", k.name, got)
			}
		})
	}
}
