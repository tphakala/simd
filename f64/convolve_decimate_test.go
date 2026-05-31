package f64

import (
	"math"
	"math/rand"
	"testing"
)

// convolveDecimateRef is the scalar reference (correctness oracle) for
// ConvolveDecimate. It computes the same strided valid-convolution outputs in
// plain Go so the SIMD paths can be checked against it.
func convolveDecimateRef(signal, kernel []float64, factor, phase int) []float64 {
	kLen := len(kernel)
	if kLen == 0 || factor < 1 || phase < 0 {
		return nil
	}
	var out []float64
	for pos := phase; pos+kLen <= len(signal); pos += factor {
		var sum float64
		for i := range kLen {
			sum += signal[pos+i] * kernel[i]
		}
		out = append(out, sum)
	}
	return out
}

func TestConvolveDecimate_Basic(t *testing.T) {
	tests := []struct {
		name   string
		signal []float64
		kernel []float64
		factor int
		phase  int
		want   []float64
	}{
		{
			name:   "factor1 equals ConvolveValid",
			signal: []float64{1, 2, 3, 4, 5},
			kernel: []float64{1, 1, 1},
			factor: 1,
			phase:  0,
			want:   []float64{6, 9, 12},
		},
		{
			name:   "factor2 phase0",
			signal: []float64{1, 2, 3, 4, 5, 6},
			kernel: []float64{1, 1, 1},
			factor: 2,
			phase:  0,
			want:   []float64{6, 12},
		},
		{
			name:   "factor2 phase1",
			signal: []float64{1, 2, 3, 4, 5, 6},
			kernel: []float64{1, 1, 1},
			factor: 2,
			phase:  1,
			want:   []float64{9, 15},
		},
		{
			name:   "factor3 identity kernel",
			signal: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
			kernel: []float64{1},
			factor: 3,
			phase:  0,
			want:   []float64{1, 4, 7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float64, len(tt.want))
			ConvolveDecimate(dst, tt.signal, tt.kernel, tt.factor, tt.phase)
			for i, want := range tt.want {
				if dst[i] != want {
					t.Errorf("ConvolveDecimate()[%d] = %v, want %v", i, dst[i], want)
				}
			}
		})
	}
}

func TestConvolveDecimate_Edge(t *testing.T) {
	signal := []float64{1, 2, 3, 4, 5}

	dst := []float64{99, 99}
	ConvolveDecimate(dst, signal, nil, 1, 0)
	if dst[0] != 99 || dst[1] != 99 {
		t.Errorf("empty kernel wrote to dst: %v", dst)
	}

	dst = []float64{99, 99}
	ConvolveDecimate(dst, signal, []float64{1}, 0, 0)
	if dst[0] != 99 || dst[1] != 99 {
		t.Errorf("factor<1 wrote to dst: %v", dst)
	}

	dst = []float64{99, 99}
	ConvolveDecimate(dst, signal, []float64{1}, 1, -1)
	if dst[0] != 99 || dst[1] != 99 {
		t.Errorf("negative phase wrote to dst: %v", dst)
	}

	dst = []float64{99}
	ConvolveDecimate(dst, []float64{1, 2}, []float64{1, 2, 3}, 1, 0)
	if dst[0] != 99 {
		t.Errorf("kernel longer than signal wrote to dst: %v", dst)
	}

	dst = []float64{99}
	ConvolveDecimate(dst, signal, []float64{1, 1}, 1, 10)
	if dst[0] != 99 {
		t.Errorf("phase past end wrote to dst: %v", dst)
	}

	dst = []float64{0}
	ConvolveDecimate(dst, signal, []float64{1}, 1, 0)
	if dst[0] != 1 {
		t.Errorf("dst[0] = %v, want 1", dst[0])
	}
}

// TestConvolveDecimate_Parity checks the active SIMD path against the scalar
// reference across the factor/phase/tap/length grid, including ragged tails.
func TestConvolveDecimate_Parity(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5117D))
	randSlice := func(n int) []float64 {
		s := make([]float64, n)
		for i := range s {
			s[i] = rng.Float64()*2 - 1
		}
		return s
	}

	taps := []int{1, 3, 5, 8, 16, 20, 32, 64, 128, 241}
	factors := []int{1, 2, 3, 4}
	sigLens := []int{257, 480, 999, 4096, 8191}

	for _, kLen := range taps {
		for _, factor := range factors {
			for phase := range factor {
				for _, sigLen := range sigLens {
					if sigLen < kLen {
						continue
					}
					kernel := randSlice(kLen)
					signal := randSlice(sigLen)
					want := convolveDecimateRef(signal, kernel, factor, phase)
					dst := make([]float64, len(want))
					ConvolveDecimate(dst, signal, kernel, factor, phase)
					for i := range want {
						// FMA (single rounding) vs scalar mul+add diverge near
						// cancellations in relative terms; accept small relative
						// or absolute error.
						absErr := math.Abs(dst[i] - want[i])
						relErr := absErr
						if math.Abs(want[i]) > 1e-12 {
							relErr = absErr / math.Abs(want[i])
						}
						if relErr > 1e-12 && absErr > 1e-12 {
							t.Fatalf("taps=%d factor=%d phase=%d sigLen=%d: out[%d]=%v want %v (absErr %g)",
								kLen, factor, phase, sigLen, i, dst[i], want[i], absErr)
						}
					}
				}
			}
		}
	}
}

// TestConvolveDecimate_MatchesDotProductUnsafe is the rigorous oracle: the fused
// kernel must produce bit-identical results to a loop calling DotProductUnsafe at
// the same strided windows on every backend.
func TestConvolveDecimate_MatchesDotProductUnsafe(t *testing.T) {
	checkConvolveDecimateExact(t)
}

func checkConvolveDecimateExact(t *testing.T) {
	t.Helper()
	rng := rand.New(rand.NewSource(0xC0FFEE))
	randSlice := func(n int) []float64 {
		s := make([]float64, n)
		for i := range s {
			s[i] = rng.Float64()*2 - 1
		}
		return s
	}

	taps := []int{1, 3, 5, 8, 16, 20, 32, 64, 128, 241}
	factors := []int{1, 2, 3, 4}
	sigLens := []int{257, 480, 999, 4096, 8191}

	for _, kLen := range taps {
		for _, factor := range factors {
			for phase := range factor {
				for _, sigLen := range sigLens {
					if sigLen < kLen {
						continue
					}
					kernel := randSlice(kLen)
					signal := randSlice(sigLen)

					var want []float64
					for pos := phase; pos+kLen <= sigLen; pos += factor {
						want = append(want, DotProductUnsafe(signal[pos:pos+kLen], kernel))
					}
					dst := make([]float64, len(want))
					ConvolveDecimate(dst, signal, kernel, factor, phase)
					for i := range want {
						if dst[i] != want[i] {
							t.Fatalf("taps=%d factor=%d phase=%d sigLen=%d: out[%d]=%v, DotProductUnsafe=%v (not bit-identical)",
								kLen, factor, phase, sigLen, i, dst[i], want[i])
						}
					}
				}
			}
		}
	}
}

func TestConvolveDecimate_AllocFree(t *testing.T) {
	signal := make([]float64, 4096)
	kernel := make([]float64, 241)
	for i := range signal {
		signal[i] = float64(i%17) * 0.1
	}
	for i := range kernel {
		kernel[i] = 1.0 / 241.0
	}
	dst := make([]float64, (len(signal)-len(kernel))/2+1)

	allocs := testing.AllocsPerRun(50, func() {
		ConvolveDecimate(dst, signal, kernel, 2, 0)
	})
	if allocs != 0 {
		t.Errorf("ConvolveDecimate allocated %v times, want 0", allocs)
	}
}

// BenchmarkConvolveDecimate compares the fused primitive against a Go loop
// calling DotProductUnsafe at each strided window (the status-quo baseline).
func BenchmarkConvolveDecimate(b *testing.B) {
	cases := []struct {
		name         string
		kLen, factor int
	}{
		{"taps20_2x", 20, 2},
		{"taps32_2x", 32, 2},
		{"taps64_2x", 64, 2},
		{"taps241_2x", 241, 2},
		{"taps241_4x", 241, 4},
	}
	const sigLen = 4096
	for _, c := range cases {
		signal := make([]float64, sigLen)
		for i := range signal {
			signal[i] = float64(i%19) * 0.0517
		}
		kernel := make([]float64, c.kLen)
		for i := range kernel {
			kernel[i] = 1.0 / float64(c.kLen)
		}
		n := (sigLen-c.kLen)/c.factor + 1
		dst := make([]float64, n)
		bytes := int64(n * c.kLen * 8 * 2)

		b.Run(c.name+"/Fused", func(b *testing.B) {
			b.SetBytes(bytes)
			for b.Loop() {
				ConvolveDecimate(dst, signal, kernel, c.factor, 0)
			}
		})
		b.Run(c.name+"/DotLoop", func(b *testing.B) {
			b.SetBytes(bytes)
			for b.Loop() {
				pos := 0
				for k := range dst {
					dst[k] = DotProductUnsafe(signal[pos:pos+c.kLen], kernel)
					pos += c.factor
				}
			}
		})
	}
}
