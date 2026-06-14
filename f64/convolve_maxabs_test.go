package f64

import (
	"math"
	"testing"
)

// refConvolveValidMaxAbs is the reference: a full valid convolution into a
// materialized output, then a scalar abs-max over it.
func refConvolveValidMaxAbs(signal, kernel []float64) float64 {
	if len(kernel) == 0 || len(signal) < len(kernel) {
		return 0
	}
	out := make([]float64, len(signal)-len(kernel)+1)
	ConvolveValid(out, signal, kernel)
	return scalarMaxAbs(out)
}

func TestConvolveValidMaxAbs(t *testing.T) {
	for _, sl := range []int{1, 4, 7, 16, 33, 128} {
		for _, kl := range []int{1, 2, 3, 5, 8} {
			if sl < kl {
				continue
			}
			signal := make([]float64, sl)
			kernel := make([]float64, kl)
			for i := range signal {
				signal[i] = math.Sin(float64(i)*0.3) - 0.4*float64(i%3)
			}
			for i := range kernel {
				kernel[i] = 0.5 - float64(i)*0.1
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
	if got := ConvolveValidMaxAbs([]float64{1, 2}, []float64{1, 2, 3}); got != 0 {
		t.Errorf("short signal = %v want 0", got)
	}
	if got := ConvolveValidMaxAbs([]float64{1, 2, 3}, nil); got != 0 {
		t.Errorf("empty kernel = %v want 0", got)
	}
}

func TestConvolveValidMaxAbsMulti(t *testing.T) {
	signal := make([]float64, 64)
	for i := range signal {
		signal[i] = math.Cos(float64(i) * 0.21)
	}
	kernels := [][]float64{
		{0.2, -0.1, 0.05, 0.3},
		{-0.4, 0.4, -0.2, 0.1},
		{0.9, 0.0, -0.9, 0.5},
		{0.1, 0.1, 0.1, 0.1},
	}
	got := ConvolveValidMaxAbsMulti(signal, kernels)
	want := 0.0
	for _, k := range kernels {
		if v := refConvolveValidMaxAbs(signal, k); v > want {
			want = v
		}
	}
	// Both paths run the same dispatched dotProduct, so they match exactly.
	if got != want {
		t.Errorf("Multi got %v want %v", got, want)
	}
}

func TestConvolveValidMaxAbsMultiEdge(t *testing.T) {
	signal := []float64{1, 2, 3, 4}
	if got := ConvolveValidMaxAbsMulti(signal, nil); got != 0 {
		t.Errorf("no kernels = %v want 0", got)
	}
	if got := ConvolveValidMaxAbsMulti(signal, [][]float64{{}}); got != 0 {
		t.Errorf("empty first kernel = %v want 0", got)
	}
	if got := ConvolveValidMaxAbsMulti([]float64{1}, [][]float64{{1, 2, 3}}); got != 0 {
		t.Errorf("signal shorter than kernel = %v want 0", got)
	}
}

func TestConvolveValidMaxAbsMultiPanic(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic on mismatched kernel lengths")
		}
	}()
	ConvolveValidMaxAbsMulti([]float64{1, 2, 3, 4}, [][]float64{{1, 2}, {1, 2, 3}})
}

func TestConvolveValidMaxAbsNoAlloc(t *testing.T) {
	signal := make([]float64, 256)
	for i := range signal {
		signal[i] = float64(i%9) - 4
	}
	kernel := []float64{0.1, -0.2, 0.3, -0.4}
	if n := testing.AllocsPerRun(100, func() { _ = ConvolveValidMaxAbs(signal, kernel) }); n != 0 {
		t.Errorf("ConvolveValidMaxAbs allocated %v, want 0", n)
	}
	kernels := [][]float64{{0.1, -0.2, 0.3, -0.4}, {0.2, 0.2, 0.2, 0.2}}
	if n := testing.AllocsPerRun(100, func() { _ = ConvolveValidMaxAbsMulti(signal, kernels) }); n != 0 {
		t.Errorf("ConvolveValidMaxAbsMulti allocated %v, want 0", n)
	}
}

// TestConvolveValidMaxAbsScalarOracle checks the dispatched (SIMD) path against an
// independent pure-Go math.FMA oracle rather than ConvolveValid, so a bug shared
// by the fused convolution and the dot-product path is still caught. Tolerance,
// since the SIMD lane-parallel summation order differs from the sequential oracle.
func TestConvolveValidMaxAbsScalarOracle(t *testing.T) {
	for _, kl := range []int{1, 2, 3, 4, 5, 8, 13, 16, 17, 31, 32, 33, 64} {
		for _, sl := range []int{kl, kl + 1, 65, 128, 257} {
			if sl < kl {
				continue
			}
			signal := make([]float64, sl)
			kernel := make([]float64, kl)
			for i := range signal {
				signal[i] = math.Sin(float64(i)*0.17) - 0.25*float64(i%5)
			}
			for i := range kernel {
				kernel[i] = math.Cos(float64(i)*0.23) * 0.6
			}
			got := ConvolveValidMaxAbs(signal, kernel)
			var want float64
			for i := range sl - kl + 1 {
				var s float64
				for j := range kl {
					s = math.FMA(signal[i+j], kernel[j], s)
				}
				if a := math.Abs(s); a > want {
					want = a
				}
			}
			if d := math.Abs(got - want); d > 1e-9*(1+want) {
				t.Errorf("sl=%d kl=%d: got %v want(scalar) %v diff=%g", sl, kl, got, want, d)
			}
		}
	}
}
