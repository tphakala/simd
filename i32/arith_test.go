package i32

import (
	"math"
	"testing"
)

// Tests for the element-wise integer primitives Add and Sub.
//
// The interesting int32 cases are negatives and the type extremes: the SIMD
// kernels operate on 32-bit lanes, so a kernel that mishandled the sign bit or
// wrapped differently than Go would be exposed by MinInt32/MaxInt32 inputs.

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

func TestAddSub_Clamp(t *testing.T) {
	a := []int32{1, 2, 3, 4}
	b := []int32{10, 20}      // shorter than a
	dst := make([]int32, 100) // longer than needed
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

func TestArith_Empty(t *testing.T) {
	// No panics and no writes on empty/nil inputs.
	Add(nil, nil, nil)
	Sub([]int32{}, []int32{}, []int32{})
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
