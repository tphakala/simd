package i32

import (
	"math"
	"testing"
)

// Tests for the element-wise integer primitives (Add, Sub) and the FLAC
// mid/side stereo decorrelation (MidSideEncode, MidSideDecode).
//
// As with interleave, the interesting int32 cases are negatives and the type
// extremes: the SIMD kernels operate on 32-bit lanes, so a kernel that
// mishandled the sign bit or wrapped differently than Go would be exposed by
// MinInt32/MaxInt32 inputs.

func TestAdd(t *testing.T) {
	tests := []struct {
		name string
		a    []int32
		b    []int32
		want []int32
	}{
		{"basic", []int32{1, 2, 3}, []int32{10, 20, 30}, []int32{11, 22, 33}},
		{"negatives", []int32{-5, 7, -1}, []int32{5, -7, 1}, []int32{0, 0, 0}},
		{"wraps", []int32{math.MaxInt32, math.MinInt32}, []int32{1, -1},
			[]int32{math.MinInt32, math.MaxInt32}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]int32, len(tt.a))
			Add(dst, tt.a, tt.b)
			for i, want := range tt.want {
				if dst[i] != want {
					t.Errorf("Add()[%d] = %d, want %d", i, dst[i], want)
				}
			}
		})
	}
}

func TestSub(t *testing.T) {
	tests := []struct {
		name string
		a    []int32
		b    []int32
		want []int32
	}{
		{"basic", []int32{11, 22, 33}, []int32{10, 20, 30}, []int32{1, 2, 3}},
		{"negatives", []int32{0, 0}, []int32{5, -7}, []int32{-5, 7}},
		{"wraps", []int32{math.MinInt32, math.MaxInt32}, []int32{1, -1},
			[]int32{math.MaxInt32, math.MinInt32}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]int32, len(tt.a))
			Sub(dst, tt.a, tt.b)
			for i, want := range tt.want {
				if dst[i] != want {
					t.Errorf("Sub()[%d] = %d, want %d", i, dst[i], want)
				}
			}
		})
	}
}

// midSideEncodeScalar / midSideDecodeScalar are the by-the-spec definitions used
// to pin MidSideEncode/MidSideDecode independently of the package's own Go
// reference, so a mistake duplicated into the reference cannot hide.
func midSideEncodeScalar(l, r int32) (mid, side int32) {
	return (l + r) >> 1, l - r
}

func midSideDecodeScalar(mid, side int32) (l, r int32) {
	sum := (mid << 1) | (side & 1)
	return (sum + side) >> 1, (sum - side) >> 1
}

func TestMidSideEncode(t *testing.T) {
	left := []int32{5, 2, -3, 5, math.MaxInt32, math.MinInt32}
	right := []int32{5, -3, 2, 4, 1, -1}
	mid := make([]int32, len(left))
	side := make([]int32, len(left))

	MidSideEncode(mid, side, left, right)

	for i := range left {
		wantMid, wantSide := midSideEncodeScalar(left[i], right[i])
		if mid[i] != wantMid {
			t.Errorf("MidSideEncode mid[%d] = %d, want %d", i, mid[i], wantMid)
		}
		if side[i] != wantSide {
			t.Errorf("MidSideEncode side[%d] = %d, want %d", i, side[i], wantSide)
		}
	}
}

func TestMidSideDecode(t *testing.T) {
	mid := []int32{5, -1, -1, 4, 0}
	side := []int32{0, -5, 5, 1, 7}
	left := make([]int32, len(mid))
	right := make([]int32, len(mid))

	MidSideDecode(left, right, mid, side)

	for i := range mid {
		wantL, wantR := midSideDecodeScalar(mid[i], side[i])
		if left[i] != wantL {
			t.Errorf("MidSideDecode left[%d] = %d, want %d", i, left[i], wantL)
		}
		if right[i] != wantR {
			t.Errorf("MidSideDecode right[%d] = %d, want %d", i, right[i], wantR)
		}
	}
}

// TestMidSideRoundTrip is the property that matters to a FLAC codec: for inputs
// within the codec's effective bit depth (no l+r overflow), decode(encode(l,r))
// reconstructs the original samples exactly, including odd parity.
func TestMidSideRoundTrip(t *testing.T) {
	sizes := []int{1, 2, 3, 7, 8, 9, 16, 17, 100, 1000}
	for _, n := range sizes {
		left := make([]int32, n)
		right := make([]int32, n)
		for i := range left {
			// 24-bit-ranged values with mixed parity and sign so the parity
			// reconstruction is exercised; small enough that l+r cannot overflow.
			left[i] = int32((i*37+11)%(1<<23)) - (1 << 22)
			right[i] = int32((i*53+7)%(1<<23)) - (1 << 22)
		}

		mid := make([]int32, n)
		side := make([]int32, n)
		MidSideEncode(mid, side, left, right)

		gotL := make([]int32, n)
		gotR := make([]int32, n)
		MidSideDecode(gotL, gotR, mid, side)

		for i := range left {
			if gotL[i] != left[i] || gotR[i] != right[i] {
				t.Fatalf("n=%d round-trip[%d] = (%d,%d), want (%d,%d)",
					n, i, gotL[i], gotR[i], left[i], right[i])
			}
		}
	}
}

func TestAddSub_Clamp(t *testing.T) {
	a := []int32{1, 2, 3, 4}
	b := []int32{10, 20}        // shorter than a
	dst := make([]int32, 100)   // longer than needed
	Add(dst, a, b)
	want := []int32{11, 22}
	for i, w := range want {
		if dst[i] != w {
			t.Errorf("Add clamp dst[%d] = %d, want %d", i, dst[i], w)
		}
	}
	for i := len(want); i < len(dst); i++ {
		if dst[i] != 0 {
			t.Errorf("Add wrote past clamp at dst[%d] = %d, want untouched 0", i, dst[i])
		}
	}
}

func TestMidSide_Clamp(t *testing.T) {
	left := []int32{8, 12, 4}
	right := []int32{2, 4, 2}
	mid := make([]int32, 2)  // only room for 2
	side := make([]int32, 10)
	MidSideEncode(mid, side, left, right)
	if mid[0] != 5 || mid[1] != 8 {
		t.Errorf("MidSideEncode clamp mid = %v, want [5 8 ...]", mid[:2])
	}
	for i := 2; i < len(side); i++ {
		if side[i] != 0 {
			t.Errorf("MidSideEncode wrote past clamp at side[%d] = %d", i, side[i])
		}
	}
}

func TestDecorrelate_Empty(t *testing.T) {
	// No panics and no writes on empty/nil inputs.
	Add(nil, nil, nil)
	Sub([]int32{}, []int32{}, []int32{})
	MidSideEncode(nil, nil, nil, nil)
	MidSideDecode(nil, nil, nil, nil)
	dst := []int32{99}
	Add(dst, nil, []int32{1})
	if dst[0] != 99 {
		t.Errorf("Add wrote on empty input: %v", dst)
	}
}

func TestAdd_AllocFree(t *testing.T) {
	a := make([]int32, 1024)
	b := make([]int32, 1024)
	dst := make([]int32, 1024)
	if got := testing.AllocsPerRun(100, func() { Add(dst, a, b) }); got != 0 {
		t.Errorf("Add allocated %v times per run, want 0", got)
	}
}

func TestSub_AllocFree(t *testing.T) {
	a := make([]int32, 1024)
	b := make([]int32, 1024)
	dst := make([]int32, 1024)
	if got := testing.AllocsPerRun(100, func() { Sub(dst, a, b) }); got != 0 {
		t.Errorf("Sub allocated %v times per run, want 0", got)
	}
}

func TestMidSideEncode_AllocFree(t *testing.T) {
	left := make([]int32, 1024)
	right := make([]int32, 1024)
	mid := make([]int32, 1024)
	side := make([]int32, 1024)
	if got := testing.AllocsPerRun(100, func() { MidSideEncode(mid, side, left, right) }); got != 0 {
		t.Errorf("MidSideEncode allocated %v times per run, want 0", got)
	}
}

func TestMidSideDecode_AllocFree(t *testing.T) {
	mid := make([]int32, 1024)
	side := make([]int32, 1024)
	left := make([]int32, 1024)
	right := make([]int32, 1024)
	if got := testing.AllocsPerRun(100, func() { MidSideDecode(left, right, mid, side) }); got != 0 {
		t.Errorf("MidSideDecode allocated %v times per run, want 0", got)
	}
}
