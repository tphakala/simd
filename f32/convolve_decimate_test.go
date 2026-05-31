package f32

import (
	"math"
	"math/rand"
	"testing"
)

// convolveDecimateRef is the scalar reference (correctness oracle) for
// ConvolveDecimate. It computes the same strided valid-convolution outputs in
// plain Go so the SIMD paths can be checked against it.
func convolveDecimateRef(signal, kernel []float32, factor, phase int) []float32 {
	kLen := len(kernel)
	if kLen == 0 || factor < 1 || phase < 0 || len(signal)-kLen-phase < 0 {
		return nil
	}
	out := make([]float32, 0, (len(signal)-kLen-phase)/factor+1)
	for pos := phase; pos+kLen <= len(signal); pos += factor {
		var sum float32
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
		signal []float32
		kernel []float32
		factor int
		phase  int
		want   []float32
	}{
		{
			name:   "factor1 equals ConvolveValid",
			signal: []float32{1, 2, 3, 4, 5},
			kernel: []float32{1, 1, 1},
			factor: 1,
			phase:  0,
			want:   []float32{6, 9, 12}, // 1+2+3, 2+3+4, 3+4+5
		},
		{
			name:   "factor2 phase0",
			signal: []float32{1, 2, 3, 4, 5, 6},
			kernel: []float32{1, 1, 1},
			factor: 2,
			phase:  0,
			want:   []float32{6, 12}, // pos 0 (1+2+3), pos 2 (3+4+5); pos 4 window 4,5,6 -> 15? check below
		},
		{
			name:   "factor2 phase1",
			signal: []float32{1, 2, 3, 4, 5, 6},
			kernel: []float32{1, 1, 1},
			factor: 2,
			phase:  1,
			want:   []float32{9, 15}, // pos 1 (2+3+4), pos 3 (4+5+6)
		},
		{
			name:   "factor3 identity kernel",
			signal: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9},
			kernel: []float32{1},
			factor: 3,
			phase:  0,
			want:   []float32{1, 4, 7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]float32, len(tt.want))
			ConvolveDecimate(dst, tt.signal, tt.kernel, tt.factor, tt.phase)
			for i, want := range tt.want {
				if dst[i] != want {
					t.Errorf("ConvolveDecimate()[%d] = %v, want %v", i, dst[i], want)
				}
			}
		})
	}
}

// TestConvolveDecimate_factor2Window verifies the "factor2 phase0" comment math
// above is what we expect: with signal len 6, kernel len 3, factor 2, phase 0,
// the valid strided positions are 0 and 2 (pos 4 would need indices 4,5,6 but 6
// is out of range), so two outputs.
func TestConvolveDecimate_OutputCount(t *testing.T) {
	signal := make([]float32, 6)
	kernel := []float32{1, 1, 1}
	got := convolveDecimateRef(signal, kernel, 2, 0)
	if len(got) != 2 {
		t.Fatalf("reference output count = %d, want 2", len(got))
	}
}

func TestConvolveDecimate_Edge(t *testing.T) {
	signal := []float32{1, 2, 3, 4, 5}

	// Empty kernel: nothing written.
	dst := []float32{99, 99}
	ConvolveDecimate(dst, signal, nil, 1, 0)
	if dst[0] != 99 || dst[1] != 99 {
		t.Errorf("empty kernel wrote to dst: %v", dst)
	}

	// factor < 1: treated as no-op (precondition factor >= 1).
	dst = []float32{99, 99}
	ConvolveDecimate(dst, signal, []float32{1}, 0, 0)
	if dst[0] != 99 || dst[1] != 99 {
		t.Errorf("factor<1 wrote to dst: %v", dst)
	}

	// negative phase: no-op (precondition phase >= 0).
	dst = []float32{99, 99}
	ConvolveDecimate(dst, signal, []float32{1}, 1, -1)
	if dst[0] != 99 || dst[1] != 99 {
		t.Errorf("negative phase wrote to dst: %v", dst)
	}

	// kernel longer than signal: no valid output.
	dst = []float32{99}
	ConvolveDecimate(dst, []float32{1, 2}, []float32{1, 2, 3}, 1, 0)
	if dst[0] != 99 {
		t.Errorf("kernel longer than signal wrote to dst: %v", dst)
	}

	// phase past end: no valid output.
	dst = []float32{99}
	ConvolveDecimate(dst, signal, []float32{1, 1}, 1, 10)
	if dst[0] != 99 {
		t.Errorf("phase past end wrote to dst: %v", dst)
	}

	// dst shorter than available outputs: only len(dst) written.
	dst = []float32{0}
	ConvolveDecimate(dst, signal, []float32{1}, 1, 0) // 5 outputs available
	if dst[0] != 1 {
		t.Errorf("dst[0] = %v, want 1", dst[0])
	}
}

// TestConvolveDecimate_Parity checks the active SIMD path against the scalar
// reference across the factor/phase/tap/length grid the issue calls for,
// including ragged tails (lengths that are not multiples of the SIMD width).
func TestConvolveDecimate_Parity(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5117D))
	randSlice := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1 // [-1, 1)
		}
		return s
	}

	taps := []int{1, 4, 7, 16, 20, 32, 64, 128, 241}
	factors := []int{1, 2, 3, 4}
	// Signal lengths chosen to exercise full-vector, partial-vector and scalar
	// remainder paths plus ragged tails.
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
					dst := make([]float32, len(want))
					ConvolveDecimate(dst, signal, kernel, factor, phase)
					for i := range want {
						// The SIMD path may use FMA (single rounding) while the
						// scalar reference uses separate mul+add, so near
						// cancellations diverge in relative terms. Accept either a
						// small relative error or a small absolute error; for
						// signal/kernel in [-1,1] and <=241 taps the absolute
						// accumulation error is bounded well under 1e-4.
						absErr := math.Abs(float64(dst[i] - want[i]))
						if relErrF32(dst[i], want[i]) > 1e-4 && absErr > 1e-4 {
							t.Fatalf("taps=%d factor=%d phase=%d sigLen=%d: out[%d]=%v want %v (absErr %g)",
								kLen, factor, phase, sigLen, i, dst[i], want[i], absErr)
						}
					}
				}
			}
		}
	}
}

// TestConvolveDecimate_MatchesDotProductUnsafe is the rigorous correctness
// oracle. The fused kernel must produce bit-identical results to a loop calling
// DotProductUnsafe at the same strided windows, because its inner dot replicates
// the per-window dot exactly. This holds on every backend (asm vs asm, Go vs Go)
// and proves the fusion changes nothing numerically versus the code it replaces.
func TestConvolveDecimate_MatchesDotProductUnsafe(t *testing.T) {
	checkConvolveDecimateExact(t)
}

// checkConvolveDecimateExact runs the bit-exact oracle against the currently
// selected backend. It is shared with the amd64 per-backend test, which swaps
// the dispatch pointers to exercise SSE / Go / AVX-512 on one machine.
func checkConvolveDecimateExact(t *testing.T) {
	t.Helper()
	rng := rand.New(rand.NewSource(0xC0FFEE))
	randSlice := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = rng.Float32()*2 - 1
		}
		return s
	}

	taps := []int{1, 4, 7, 16, 20, 32, 64, 128, 241}
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

					want := make([]float32, 0, (sigLen-kLen-phase)/factor+1)
					for pos := phase; pos+kLen <= sigLen; pos += factor {
						want = append(want, DotProductUnsafe(signal[pos:pos+kLen], kernel))
					}
					dst := make([]float32, len(want))
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

// BenchmarkConvolveDecimate compares the fused primitive against the status-quo
// baseline a consumer writes today: a Go loop calling DotProductUnsafe at each
// strided window. Both compute identical results; the fused path removes the
// per-output call/dispatch overhead and keeps the kernel pointer resident.
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
		signal := make([]float32, sigLen)
		for i := range signal {
			signal[i] = float32(i%19) * 0.0517
		}
		kernel := make([]float32, c.kLen)
		for i := range kernel {
			kernel[i] = 1.0 / float32(c.kLen)
		}
		n := (sigLen-c.kLen)/c.factor + 1
		dst := make([]float32, n)
		bytes := int64(n * c.kLen * 4 * 2)

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

// TestConvolveDecimate_AllocFree confirms the primitive allocates nothing when
// the caller provides dst (the zero-alloc streaming contract).
func TestConvolveDecimate_AllocFree(t *testing.T) {
	signal := make([]float32, 4096)
	kernel := make([]float32, 241)
	for i := range signal {
		signal[i] = float32(i%17) * 0.1
	}
	for i := range kernel {
		kernel[i] = 1.0 / 241.0
	}
	dst := make([]float32, (len(signal)-len(kernel))/2+1)

	allocs := testing.AllocsPerRun(50, func() {
		ConvolveDecimate(dst, signal, kernel, 2, 0)
	})
	if allocs != 0 {
		t.Errorf("ConvolveDecimate allocated %v times, want 0", allocs)
	}
}
