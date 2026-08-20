package c64

import "testing"

// mulRealRef is the independent scalar reference for MulReal.
func mulRealRef(a []complex64, s []float32) []complex64 {
	out := make([]complex64, len(a))
	for i := range a {
		out[i] = complex(real(a[i])*s[i], imag(a[i])*s[i])
	}
	return out
}

func TestMulReal(t *testing.T) {
	tests := []struct {
		name string
		a    []complex64
		s    []float32
	}{
		{"single", []complex64{1 + 2i}, []float32{3}},
		{"pair", []complex64{1 + 2i, 3 + 4i}, []float32{2, -1}},
		{"zeros", []complex64{1 + 1i, 2 + 2i}, []float32{0, 0}},
		{"negative", []complex64{-1 - 2i, 3 - 4i}, []float32{-2, 0.5}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]complex64, len(tt.a))
			MulReal(dst, tt.a, tt.s)
			want := mulRealRef(tt.a, tt.s)
			for i := range dst {
				if !complexClose(dst[i], want[i]) {
					t.Errorf("MulReal()[%d] = %v, want %v", i, dst[i], want[i])
				}
			}
		})
	}
}

func TestMulReal_Large(t *testing.T) {
	const n = 100
	a := make([]complex64, n)
	s := make([]float32, n)
	for i := range n {
		a[i] = complex(float32(i+1), float32(-i))
		s[i] = float32(i) * 0.25
	}
	dst := make([]complex64, n)
	MulReal(dst, a, s)
	want := mulRealRef(a, s)
	for i := range n {
		if !complexClose(dst[i], want[i]) {
			t.Errorf("MulReal_Large()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

// TestMulReal_Alias verifies the documented dst-may-alias-a contract: an in-place
// MulReal(a, a, s) equals the same computation into a separate buffer.
func TestMulReal_Alias(t *testing.T) {
	a := []complex64{1 + 2i, 3 + 4i, 5 + 6i, 7 + 8i}
	s := []float32{0.5, -1, 2, 0.25}
	want := mulRealRef(a, s)
	inPlace := make([]complex64, len(a))
	copy(inPlace, a)
	MulReal(inPlace, inPlace, s)
	for i := range inPlace {
		if !complexClose(inPlace[i], want[i]) {
			t.Errorf("MulReal alias[%d] = %v, want %v", i, inPlace[i], want[i])
		}
	}
}

// TestMulReal_Clamp checks the min-of-three length clamp: only min(len) elements
// are written and the tail of a longer dst is left untouched.
func TestMulReal_Clamp(t *testing.T) {
	a := []complex64{1 + 1i, 2 + 2i, 3 + 3i, 4 + 4i}
	s := []float32{2, 3, 4}     // shortest operand: 3
	dst := make([]complex64, 5) // longest: 5
	sentinel := complex64(9 + 9i)
	for i := range dst {
		dst[i] = sentinel
	}
	MulReal(dst, a, s)
	want := mulRealRef(a[:3], s)
	for i := range 3 {
		if !complexClose(dst[i], want[i]) {
			t.Errorf("MulReal_Clamp[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
	for i := 3; i < len(dst); i++ {
		if dst[i] != sentinel {
			t.Errorf("MulReal_Clamp wrote past clamp at [%d] = %v, want sentinel %v", i, dst[i], sentinel)
		}
	}
}

func TestMulRealGo(t *testing.T) {
	a := []complex64{1 + 2i, 3 + 4i}
	s := []float32{2, -1}
	dst := make([]complex64, 2)
	mulRealGo(dst, a, s)
	want := mulRealRef(a, s)
	for i := range dst {
		if !complexClose(dst[i], want[i]) {
			t.Errorf("mulRealGo()[%d] = %v, want %v", i, dst[i], want[i])
		}
	}
}

func TestMulReal_Empty(_ *testing.T) {
	var a, dst []complex64
	var s []float32
	MulReal(dst, a, s)
}

func TestMulReal_Alloc(t *testing.T) {
	const n = 1024
	a := make([]complex64, n)
	s := make([]float32, n)
	for i := range n {
		a[i] = complex(float32(i), float32(-i))
		s[i] = float32(i) * 0.001
	}
	dst := make([]complex64, n)
	if allocs := testing.AllocsPerRun(100, func() { MulReal(dst, a, s) }); allocs != 0 {
		t.Errorf("MulReal allocated %v times per run, want 0", allocs)
	}
}
