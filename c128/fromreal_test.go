package c128

import "testing"

func TestFromReal(t *testing.T) {
	tests := []struct {
		name string
		src  []float64
		want []complex128
	}{
		{"empty", nil, nil},
		{"single", []float64{1}, []complex128{1 + 0i}},
		{"pair", []float64{1, 2}, []complex128{1 + 0i, 2 + 0i}},
		{"negative", []float64{-1, -2}, []complex128{-1 + 0i, -2 + 0i}},
		{"zero", []float64{0, 0}, []complex128{0 + 0i, 0 + 0i}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.src) == 0 {
				dst := make([]complex128, 0)
				FromReal(dst, tt.src)
				return
			}
			dst := make([]complex128, len(tt.src))
			FromReal(dst, tt.src)
			for i := range dst {
				if !complexClose(dst[i], tt.want[i]) {
					t.Errorf("FromReal()[%d] = %v, want %v", i, dst[i], tt.want[i])
				}
			}
		})
	}
}

func TestFromReal_Large(t *testing.T) {
	n := 100
	src := make([]float64, n)
	for i := range n {
		src[i] = float64(i + 1)
	}

	dst := make([]complex128, n)
	FromReal(dst, src)

	for i := range n {
		expected := complex(src[i], 0)
		if !complexClose(dst[i], expected) {
			t.Errorf("FromReal_Large()[%d] = %v, want %v", i, dst[i], expected)
		}
	}
}

// TestFromReal_AllocFree pins the zero-allocation contract: FromReal writes into
// the caller-provided destination and must not allocate.
func TestFromReal_AllocFree(t *testing.T) {
	src := make([]float64, 1024)
	dst := make([]complex128, 1024)
	for i := range src {
		src[i] = float64(i%2400 - 1200)
	}
	if a := testing.AllocsPerRun(50, func() { FromReal(dst, src) }); a != 0 {
		t.Errorf("FromReal allocated %v times per run, want 0", a)
	}
}
