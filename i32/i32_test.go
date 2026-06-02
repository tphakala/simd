package i32

import (
	"math"
	"testing"
)

// Tests for Interleave2 and Deinterleave2.
//
// Unlike the float packages, the interesting cases for int32 are negative
// values and the extremes of the type, because the SIMD kernels move 32-bit
// lanes by bit pattern. If a kernel ever corrupted the sign or high bits those
// values would expose it.

func TestInterleave2(t *testing.T) {
	tests := []struct {
		name string
		a    []int32
		b    []int32
		want []int32
	}{
		{"single", []int32{1}, []int32{2}, []int32{1, 2}},
		{"two", []int32{1, 3}, []int32{2, 4}, []int32{1, 2, 3, 4}},
		{"four", []int32{1, 3, 5, 7}, []int32{2, 4, 6, 8}, []int32{1, 2, 3, 4, 5, 6, 7, 8}},
		{"eight", []int32{1, 3, 5, 7, 9, 11, 13, 15}, []int32{2, 4, 6, 8, 10, 12, 14, 16},
			[]int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
		{"negatives_and_extremes",
			[]int32{math.MinInt32, -1, 0, math.MaxInt32},
			[]int32{math.MaxInt32, 1, -2, math.MinInt32},
			[]int32{math.MinInt32, math.MaxInt32, -1, 1, 0, -2, math.MaxInt32, math.MinInt32}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]int32, len(tt.a)*2)
			Interleave2(dst, tt.a, tt.b)
			for i, want := range tt.want {
				if dst[i] != want {
					t.Errorf("Interleave2()[%d] = %d, want %d", i, dst[i], want)
				}
			}
		})
	}
}

func TestInterleave2_Large(t *testing.T) {
	// Sizes chosen to straddle the SIMD block size: exact multiples plus
	// off-by-one tails on both the 4-lane (NEON) and 8-lane (AVX) widths.
	sizes := []int{8, 16, 17, 100, 1024, 1025}
	for _, n := range sizes {
		a := make([]int32, n)
		b := make([]int32, n)
		for i := range a {
			a[i] = int32(i*2) ^ math.MinInt32 // flip the sign bit to exercise high bits
			b[i] = int32(i*2 + 1)
		}

		dst := make([]int32, n*2)
		Interleave2(dst, a, b)

		for i := range n {
			if dst[i*2] != a[i] {
				t.Errorf("size=%d: Interleave2()[%d] = %d, want %d", n, i*2, dst[i*2], a[i])
			}
			if dst[i*2+1] != b[i] {
				t.Errorf("size=%d: Interleave2()[%d] = %d, want %d", n, i*2+1, dst[i*2+1], b[i])
			}
		}
	}
}

func TestDeinterleave2(t *testing.T) {
	tests := []struct {
		name  string
		src   []int32
		wantA []int32
		wantB []int32
	}{
		{"single", []int32{1, 2}, []int32{1}, []int32{2}},
		{"two", []int32{1, 2, 3, 4}, []int32{1, 3}, []int32{2, 4}},
		{"four", []int32{1, 2, 3, 4, 5, 6, 7, 8}, []int32{1, 3, 5, 7}, []int32{2, 4, 6, 8}},
		{"extremes",
			[]int32{math.MinInt32, math.MaxInt32, -1, 1, 0, -2, 42, -42},
			[]int32{math.MinInt32, -1, 0, 42},
			[]int32{math.MaxInt32, 1, -2, -42}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := len(tt.src) / 2
			a := make([]int32, n)
			b := make([]int32, n)
			Deinterleave2(a, b, tt.src)
			for i, want := range tt.wantA {
				if a[i] != want {
					t.Errorf("Deinterleave2() a[%d] = %d, want %d", i, a[i], want)
				}
			}
			for i, want := range tt.wantB {
				if b[i] != want {
					t.Errorf("Deinterleave2() b[%d] = %d, want %d", i, b[i], want)
				}
			}
		})
	}
}

func TestDeinterleave2_Large(t *testing.T) {
	sizes := []int{8, 16, 17, 100, 1024, 1025}
	for _, n := range sizes {
		src := make([]int32, n*2)
		for i := range src {
			src[i] = int32(i) ^ math.MinInt32
		}

		a := make([]int32, n)
		b := make([]int32, n)
		Deinterleave2(a, b, src)

		for i := range n {
			wantA := int32(i*2) ^ math.MinInt32
			wantB := int32(i*2+1) ^ math.MinInt32
			if a[i] != wantA {
				t.Errorf("size=%d: Deinterleave2() a[%d] = %d, want %d", n, i, a[i], wantA)
			}
			if b[i] != wantB {
				t.Errorf("size=%d: Deinterleave2() b[%d] = %d, want %d", n, i, b[i], wantB)
			}
		}
	}
}

func TestInterleaveDeinterleaveRoundTrip(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 100}
	for _, n := range sizes {
		a := make([]int32, n)
		b := make([]int32, n)
		for i := range a {
			a[i] = int32(i) ^ math.MinInt32
			b[i] = int32(-i - 1)
		}

		dst := make([]int32, n*2)
		Interleave2(dst, a, b)

		gotA := make([]int32, n)
		gotB := make([]int32, n)
		Deinterleave2(gotA, gotB, dst)

		for i := range n {
			if gotA[i] != a[i] {
				t.Errorf("size=%d: round-trip a[%d] = %d, want %d", n, i, gotA[i], a[i])
			}
			if gotB[i] != b[i] {
				t.Errorf("size=%d: round-trip b[%d] = %d, want %d", n, i, gotB[i], b[i])
			}
		}
	}
}

// TestInterleave2_Clamp verifies the pair count is clamped to the shortest of
// dst/2, a and b, leaving the rest of every buffer untouched.
func TestInterleave2_Clamp(t *testing.T) {
	a := []int32{1, 2, 3, 4}
	b := []int32{10, 20}      // shorter than a
	dst := make([]int32, 100) // longer than needed

	Interleave2(dst, a, b)

	want := []int32{1, 10, 2, 20}
	for i, w := range want {
		if dst[i] != w {
			t.Errorf("dst[%d] = %d, want %d", i, dst[i], w)
		}
	}
	for i := len(want); i < len(dst); i++ {
		if dst[i] != 0 {
			t.Errorf("dst[%d] = %d, want untouched 0", i, dst[i])
		}
	}
}

func TestDeinterleave2_Clamp(t *testing.T) {
	src := []int32{1, 2, 3, 4, 5, 6}
	a := make([]int32, 2) // only room for 2 pairs
	b := make([]int32, 10)

	Deinterleave2(a, b, src)

	wantA := []int32{1, 3}
	wantB := []int32{2, 4}
	for i, w := range wantA {
		if a[i] != w {
			t.Errorf("a[%d] = %d, want %d", i, a[i], w)
		}
	}
	for i, w := range wantB {
		if b[i] != w {
			t.Errorf("b[%d] = %d, want %d", i, b[i], w)
		}
	}
	for i := len(wantB); i < len(b); i++ {
		if b[i] != 0 {
			t.Errorf("b[%d] = %d, want untouched 0", i, b[i])
		}
	}
}

func TestInterleave2_Empty(t *testing.T) {
	// No panics and no writes on empty/zero-length inputs.
	Interleave2(nil, nil, nil)
	Interleave2([]int32{}, []int32{}, []int32{})
	dst := []int32{99, 99}
	Interleave2(dst, nil, []int32{1})
	if dst[0] != 99 || dst[1] != 99 {
		t.Errorf("empty a wrote into dst: %v", dst)
	}
}

func TestDeinterleave2_Empty(t *testing.T) {
	Deinterleave2(nil, nil, nil)
	Deinterleave2([]int32{}, []int32{}, []int32{})
	a := []int32{99}
	Deinterleave2(a, []int32{1}, nil)
	if a[0] != 99 {
		t.Errorf("empty src wrote into a: %v", a)
	}
}

func TestInterleave2_AllocFree(t *testing.T) {
	a := make([]int32, 1024)
	b := make([]int32, 1024)
	dst := make([]int32, 2048)
	if got := testing.AllocsPerRun(100, func() { Interleave2(dst, a, b) }); got != 0 {
		t.Errorf("Interleave2 allocated %v times per run, want 0", got)
	}
}

func TestDeinterleave2_AllocFree(t *testing.T) {
	src := make([]int32, 2048)
	a := make([]int32, 1024)
	b := make([]int32, 1024)
	if got := testing.AllocsPerRun(100, func() { Deinterleave2(a, b, src) }); got != 0 {
		t.Errorf("Deinterleave2 allocated %v times per run, want 0", got)
	}
}
