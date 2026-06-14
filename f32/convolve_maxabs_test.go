package f32

import (
	"math"
	"testing"
)

// refConvolveValidMaxAbs is the reference: a full valid convolution into a
// materialized output, then a scalar abs-max over it.
func refConvolveValidMaxAbs(signal, kernel []float32) float32 {
	if len(kernel) == 0 || len(signal) < len(kernel) {
		return 0
	}
	out := make([]float32, len(signal)-len(kernel)+1)
	ConvolveValid(out, signal, kernel)
	return scalarMaxAbs(out)
}

func TestConvolveValidMaxAbs(t *testing.T) {
	for _, sl := range []int{1, 4, 7, 16, 33, 128} {
		for _, kl := range []int{1, 2, 3, 5, 8} {
			if sl < kl {
				continue
			}
			signal := make([]float32, sl)
			kernel := make([]float32, kl)
			for i := range signal {
				signal[i] = float32(math.Sin(float64(i)*0.3)) - 0.4*float32(i%3)
			}
			for i := range kernel {
				kernel[i] = 0.5 - float32(i)*0.1
			}
			got := ConvolveValidMaxAbs(signal, kernel)
			want := refConvolveValidMaxAbs(signal, kernel)
			// Both paths run the same dispatched dotProduct, so they match exactly.
			if got != want {
				t.Errorf("sl=%d kl=%d: got %v want %v", sl, kl, got, want)
			}
		}
	}
}

func TestConvolveValidMaxAbsEdge(t *testing.T) {
	if got := ConvolveValidMaxAbs(nil, nil); got != 0 {
		t.Errorf("nil/nil = %v want 0", got)
	}
	if got := ConvolveValidMaxAbs([]float32{1, 2}, []float32{1, 2, 3}); got != 0 {
		t.Errorf("short signal = %v want 0", got)
	}
	if got := ConvolveValidMaxAbs([]float32{1, 2, 3}, nil); got != 0 {
		t.Errorf("empty kernel = %v want 0", got)
	}
}

func TestConvolveValidMaxAbsMulti(t *testing.T) {
	signal := make([]float32, 64)
	for i := range signal {
		signal[i] = float32(math.Cos(float64(i) * 0.21))
	}
	kernels := [][]float32{
		{0.2, -0.1, 0.05, 0.3},
		{-0.4, 0.4, -0.2, 0.1},
		{0.9, 0.0, -0.9, 0.5},
		{0.1, 0.1, 0.1, 0.1},
	}
	got := ConvolveValidMaxAbsMulti(signal, kernels)
	var want float32
	for _, k := range kernels {
		if v := refConvolveValidMaxAbs(signal, k); v > want {
			want = v
		}
	}
	if got != want {
		t.Errorf("Multi got %v want %v", got, want)
	}
}

func TestConvolveValidMaxAbsMultiEdge(t *testing.T) {
	signal := []float32{1, 2, 3, 4}
	if got := ConvolveValidMaxAbsMulti(signal, nil); got != 0 {
		t.Errorf("no kernels = %v want 0", got)
	}
	if got := ConvolveValidMaxAbsMulti(signal, [][]float32{{}}); got != 0 {
		t.Errorf("empty first kernel = %v want 0", got)
	}
	if got := ConvolveValidMaxAbsMulti([]float32{1}, [][]float32{{1, 2, 3}}); got != 0 {
		t.Errorf("signal shorter than kernel = %v want 0", got)
	}
}

func TestConvolveValidMaxAbsMultiPanic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic on mismatched kernel lengths")
		}
	}()
	ConvolveValidMaxAbsMulti([]float32{1, 2, 3, 4}, [][]float32{{1, 2}, {1, 2, 3}})
}

func TestConvolveValidMaxAbsNoAlloc(t *testing.T) {
	signal := make([]float32, 256)
	for i := range signal {
		signal[i] = float32(i%9) - 4
	}
	kernel := []float32{0.1, -0.2, 0.3, -0.4}
	if n := testing.AllocsPerRun(100, func() { _ = ConvolveValidMaxAbs(signal, kernel) }); n != 0 {
		t.Errorf("ConvolveValidMaxAbs allocated %v, want 0", n)
	}
	kernels := [][]float32{{0.1, -0.2, 0.3, -0.4}, {0.2, 0.2, 0.2, 0.2}}
	if n := testing.AllocsPerRun(100, func() { _ = ConvolveValidMaxAbsMulti(signal, kernels) }); n != 0 {
		t.Errorf("ConvolveValidMaxAbsMulti allocated %v, want 0", n)
	}
}
