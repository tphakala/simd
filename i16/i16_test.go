package i16

import (
	"math"
	"testing"
)

// Tests for Interleave2 and Deinterleave2.
//
// As with i32, the interesting cases for int16 are negative values and the
// extremes of the type, because the SIMD kernels move 16-bit lanes by bit
// pattern. If a kernel ever corrupted the sign or high bits those values would
// expose it.

func TestInterleave2(t *testing.T) {
	tests := []struct {
		name string
		a    []int16
		b    []int16
		want []int16
	}{
		{"single", []int16{1}, []int16{2}, []int16{1, 2}},
		{"two", []int16{1, 3}, []int16{2, 4}, []int16{1, 2, 3, 4}},
		{"four", []int16{1, 3, 5, 7}, []int16{2, 4, 6, 8}, []int16{1, 2, 3, 4, 5, 6, 7, 8}},
		{"eight", []int16{1, 3, 5, 7, 9, 11, 13, 15}, []int16{2, 4, 6, 8, 10, 12, 14, 16},
			[]int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}},
		{"negatives_and_extremes",
			[]int16{math.MinInt16, -1, 0, math.MaxInt16},
			[]int16{math.MaxInt16, 1, -2, math.MinInt16},
			[]int16{math.MinInt16, math.MaxInt16, -1, 1, 0, -2, math.MaxInt16, math.MinInt16}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]int16, len(tt.a)*2)
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
	// Sizes chosen to straddle the SIMD block sizes: exact multiples plus
	// off-by-one tails on the 8-lane (NEON/SSE2) and 16-lane (AVX2) widths, with
	// 8..15 routing through SSE2 on an AVX2 host.
	sizes := []int{8, 12, 15, 16, 17, 100, 1024, 1025}
	for _, n := range sizes {
		a := make([]int16, n)
		b := make([]int16, n)
		for i := range a {
			a[i] = int16(i*2) ^ math.MinInt16 // flip the sign bit to exercise high bits
			b[i] = int16(i*2 + 1)
		}

		dst := make([]int16, n*2)
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
		src   []int16
		wantA []int16
		wantB []int16
	}{
		{"single", []int16{1, 2}, []int16{1}, []int16{2}},
		{"two", []int16{1, 2, 3, 4}, []int16{1, 3}, []int16{2, 4}},
		{"four", []int16{1, 2, 3, 4, 5, 6, 7, 8}, []int16{1, 3, 5, 7}, []int16{2, 4, 6, 8}},
		{"extremes",
			[]int16{math.MinInt16, math.MaxInt16, -1, 1, 0, -2, 42, -42},
			[]int16{math.MinInt16, -1, 0, 42},
			[]int16{math.MaxInt16, 1, -2, -42}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := len(tt.src) / 2
			a := make([]int16, n)
			b := make([]int16, n)
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
	sizes := []int{8, 12, 15, 16, 17, 100, 1024, 1025}
	for _, n := range sizes {
		src := make([]int16, n*2)
		for i := range src {
			src[i] = int16(i) ^ math.MinInt16
		}

		a := make([]int16, n)
		b := make([]int16, n)
		Deinterleave2(a, b, src)

		for i := range n {
			wantA := int16(i*2) ^ math.MinInt16
			wantB := int16(i*2+1) ^ math.MinInt16
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
	sizes := []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 100}
	for _, n := range sizes {
		a := make([]int16, n)
		b := make([]int16, n)
		for i := range a {
			a[i] = int16(i) ^ math.MinInt16
			b[i] = int16(-i - 1)
		}

		dst := make([]int16, n*2)
		Interleave2(dst, a, b)

		gotA := make([]int16, n)
		gotB := make([]int16, n)
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
	a := []int16{1, 2, 3, 4}
	b := []int16{10, 20}      // shorter than a
	dst := make([]int16, 100) // longer than needed

	Interleave2(dst, a, b)

	want := []int16{1, 10, 2, 20}
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
	src := []int16{1, 2, 3, 4, 5, 6}
	a := make([]int16, 2) // only room for 2 pairs
	b := make([]int16, 10)

	Deinterleave2(a, b, src)

	wantA := []int16{1, 3}
	wantB := []int16{2, 4}
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

// TestInterleave2_OddDstTail verifies that when dst is the limiting factor and
// has an odd length, the pair count rounds down and the trailing element is
// left untouched (the "ragged tail of dst" guarantee).
func TestInterleave2_OddDstTail(t *testing.T) {
	a := []int16{1, 2, 3, 4, 5}
	b := []int16{10, 20, 30, 40, 50}
	dst := []int16{0, 0, 0, 0, 99} // odd length: only 2 pairs fit, dst[4] is the tail

	Interleave2(dst, a, b)

	want := []int16{1, 10, 2, 20}
	for i, w := range want {
		if dst[i] != w {
			t.Errorf("dst[%d] = %d, want %d", i, dst[i], w)
		}
	}
	if dst[4] != 99 {
		t.Errorf("odd dst tail dst[4] = %d, want untouched 99", dst[4])
	}
}

// TestDeinterleave2_OddSrcTail verifies that an odd-length src does not panic
// and leaves its trailing element unread (the "ragged tail of src" guarantee).
func TestDeinterleave2_OddSrcTail(t *testing.T) {
	src := []int16{1, 2, 3, 4, 5} // odd length: 2 full pairs, src[4] is the tail
	a := make([]int16, 5)
	b := make([]int16, 5)
	for i := range a {
		a[i] = 99
		b[i] = 99
	}

	Deinterleave2(a, b, src)

	wantA := []int16{1, 3}
	wantB := []int16{2, 4}
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
	// Only 2 pairs were processed; the rest of a and b stay untouched.
	for i := len(wantA); i < len(a); i++ {
		if a[i] != 99 {
			t.Errorf("a[%d] = %d, want untouched 99", i, a[i])
		}
		if b[i] != 99 {
			t.Errorf("b[%d] = %d, want untouched 99", i, b[i])
		}
	}
}

func TestInterleave2_Empty(t *testing.T) {
	// No panics and no writes on empty/zero-length inputs.
	Interleave2(nil, nil, nil)
	Interleave2([]int16{}, []int16{}, []int16{})
	dst := []int16{99, 99}
	Interleave2(dst, nil, []int16{1})
	if dst[0] != 99 || dst[1] != 99 {
		t.Errorf("empty a wrote into dst: %v", dst)
	}
}

func TestDeinterleave2_Empty(t *testing.T) {
	Deinterleave2(nil, nil, nil)
	Deinterleave2([]int16{}, []int16{}, []int16{})
	a := []int16{99}
	Deinterleave2(a, []int16{1}, nil)
	if a[0] != 99 {
		t.Errorf("empty src wrote into a: %v", a)
	}
}

func TestInterleave2_AllocFree(t *testing.T) {
	a := make([]int16, 1024)
	b := make([]int16, 1024)
	dst := make([]int16, 2048)
	if got := testing.AllocsPerRun(100, func() { Interleave2(dst, a, b) }); got != 0 {
		t.Errorf("Interleave2 allocated %v times per run, want 0", got)
	}
}

func TestDeinterleave2_AllocFree(t *testing.T) {
	src := make([]int16, 2048)
	a := make([]int16, 1024)
	b := make([]int16, 1024)
	if got := testing.AllocsPerRun(100, func() { Deinterleave2(a, b, src) }); got != 0 {
		t.Errorf("Deinterleave2 allocated %v times per run, want 0", got)
	}
}
