package f64

import (
	"math"
	"math/rand"
	"testing"
)

// autocorrNaive is an independent, maximally simple reference: the textbook
// double loop. autocorrelateGo must equal it, and Autocorrelate (AVX2/NEON on
// capable hosts) must equal autocorrelateGo bit-for-bit.
func autocorrNaive(x []float64, maxLag int) []float64 {
	out := make([]float64, maxLag+1)
	n := len(x)
	for lag := 0; lag <= maxLag; lag++ {
		var s float64
		for i := lag; i < n; i++ {
			s += x[i] * x[i-lag]
		}
		out[lag] = s
	}
	return out
}

func randWindowed(rng *rand.Rand, n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		// Windowed integer PCM: a sample magnitude scaled by a [0,1] weight.
		amp := float64(rng.Intn(1<<21) - (1 << 20))
		x[i] = amp * rng.Float64()
	}
	return x
}

func bitsDiff(a, b []float64) (int, bool) {
	if len(a) != len(b) {
		return -1, false
	}
	for i := range a {
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			return i, false
		}
	}
	return -1, true
}

// TestAutocorrelateBitIdentical asserts the dispatched implementation (AVX2 on
// this host) reproduces the scalar reference bit-for-bit across block sizes and
// lag counts, including the partial-final-group and short-block fallback paths.
func TestAutocorrelateBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA17C0))
	blockSizes := []int{1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 35, 36, 63, 64, 256, 1024, 4096}
	maxLags := []int{0, 1, 2, 3, 4, 5, 7, 8, 11, 12, 15, 16, 31, 32}
	cases := 0
	for range 60 {
		for _, n := range blockSizes {
			x := randWindowed(rng, n)
			for _, maxLag := range maxLags {
				if maxLag >= n {
					continue
				}
				want := autocorrNaive(x, maxLag)

				gotGo := make([]float64, maxLag+1)
				autocorrelateGo(gotGo, x, maxLag)
				if i, ok := bitsDiff(want, gotGo); !ok {
					t.Fatalf("autocorrelateGo != naive: n=%d maxLag=%d lag=%d", n, maxLag, i)
				}

				gotDisp := make([]float64, maxLag+1)
				Autocorrelate(gotDisp, x, maxLag)
				if i, ok := bitsDiff(gotGo, gotDisp); !ok {
					t.Fatalf("Autocorrelate != reference: n=%d maxLag=%d lag=%d\n want=%v (%#x)\n got =%v (%#x)",
						n, maxLag, i, gotGo[i], math.Float64bits(gotGo[i]),
						gotDisp[i], math.Float64bits(gotDisp[i]))
				}
				cases++
			}
		}
	}
	t.Logf("bit-identical across %d (n, maxLag) cases", cases)
}

// TestAutocorrelateGuards covers the public-API precondition clamps.
func TestAutocorrelateGuards(t *testing.T) {
	autoc := make([]float64, 4)
	x := []float64{1, 2, 3, 4, 5}

	// Negative maxLag and empty x are no-ops.
	Autocorrelate(autoc, x, -1)
	Autocorrelate(autoc, nil, 3)

	// maxLag clamped to len(autoc)-1; lag 0 still computed.
	for i := range autoc {
		autoc[i] = -123
	}
	Autocorrelate(autoc, x, 99)
	want := autocorrNaive(x, len(autoc)-1)
	if i, ok := bitsDiff(want, autoc); !ok {
		t.Fatalf("clamped maxLag mismatch at lag %d: got %v want %v", i, autoc, want)
	}
}

// TestAutocorrelateZeroInput keeps the all-zero block (silence) well-defined:
// every lag is exactly 0.
func TestAutocorrelateZeroInput(t *testing.T) {
	x := make([]float64, 512)
	autoc := make([]float64, 13)
	Autocorrelate(autoc, x, 12)
	for lag, v := range autoc {
		if v != 0 {
			t.Fatalf("lag %d = %v, want 0", lag, v)
		}
	}
}
