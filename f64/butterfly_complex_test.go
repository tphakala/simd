package f64

import (
	"fmt"
	"math"
	"testing"
)

// =============================================================================
// BUTTERFLY COMPLEX TESTS (float64)
// =============================================================================

// butterflyComplexRef is a reference implementation of the fused butterfly
// operation using separate multiply and add (no FMA), so it is the numerically
// distinct scalar baseline the SIMD FMA path is compared against.
func butterflyComplexRef(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm []float64) {
	for i := range upperRe {
		// Complex multiply: temp = lower * twiddle
		lr, li := lowerRe[i], lowerIm[i]
		tr, ti := twRe[i], twIm[i]
		tempRe := lr*tr - li*ti
		tempIm := lr*ti + li*tr

		// Butterfly: upper' = upper + temp, lower' = upper - temp
		ur, ui := upperRe[i], upperIm[i]
		upperRe[i] = ur + tempRe
		upperIm[i] = ui + tempIm
		lowerRe[i] = ur - tempRe
		lowerIm[i] = ui - tempIm
	}
}

func TestButterflyComplex(t *testing.T) {
	sizes := []int{0, 1, 2, 3, 4, 8, 9, 16, 17, 31, 32, 33, 63, 64, 65, 128, 256, 512, 1000}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			if n == 0 {
				ButterflyComplex(nil, nil, nil, nil, nil, nil)
				return
			}

			upperRe := make([]float64, n)
			upperIm := make([]float64, n)
			lowerRe := make([]float64, n)
			lowerIm := make([]float64, n)
			twRe := make([]float64, n)
			twIm := make([]float64, n)

			// Reference copies
			upperReRef := make([]float64, n)
			upperImRef := make([]float64, n)
			lowerReRef := make([]float64, n)
			lowerImRef := make([]float64, n)

			// Initialize with test data
			for i := range n {
				angle := 2 * math.Pi * float64(i) / float64(n)
				upperRe[i] = float64(i+1) * 0.1
				upperIm[i] = float64(i+2) * 0.2
				lowerRe[i] = float64(i+3) * 0.3
				lowerIm[i] = float64(i+4) * 0.4
				twRe[i] = math.Cos(angle)
				twIm[i] = math.Sin(angle)
			}

			// Copy for reference
			copy(upperReRef, upperRe)
			copy(upperImRef, upperIm)
			copy(lowerReRef, lowerRe)
			copy(lowerImRef, lowerIm)

			// Apply SIMD version
			ButterflyComplex(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)

			// Apply reference
			butterflyComplexRef(upperReRef, upperImRef, lowerReRef, lowerImRef, twRe, twIm)

			// Compare results - relative tolerance for FMA vs separate mul+add
			// rounding differences (float64 FMA divergence is ~1 ULP).
			relTol := func(got, want float64) bool {
				diff := math.Abs(got - want)
				return diff <= 1e-9 || diff <= math.Abs(want)*1e-11
			}
			for i := range n {
				if !relTol(upperRe[i], upperReRef[i]) {
					t.Errorf("upperRe[%d] = %v, want %v", i, upperRe[i], upperReRef[i])
				}
				if !relTol(upperIm[i], upperImRef[i]) {
					t.Errorf("upperIm[%d] = %v, want %v", i, upperIm[i], upperImRef[i])
				}
				if !relTol(lowerRe[i], lowerReRef[i]) {
					t.Errorf("lowerRe[%d] = %v, want %v", i, lowerRe[i], lowerReRef[i])
				}
				if !relTol(lowerIm[i], lowerImRef[i]) {
					t.Errorf("lowerIm[%d] = %v, want %v", i, lowerIm[i], lowerImRef[i])
				}
			}
		})
	}
}

func TestButterflyComplex_KnownValues(t *testing.T) {
	// upper = 1+2i, lower = 3+4i, twiddle = cos(π/4)+i*sin(π/4)
	// temp = lower * twiddle
	// upper' = upper + temp, lower' = upper - temp
	cos45 := math.Cos(math.Pi / 4)
	sin45 := math.Sin(math.Pi / 4)

	upperRe := []float64{1}
	upperIm := []float64{2}
	lowerRe := []float64{3}
	lowerIm := []float64{4}
	twRe := []float64{cos45}
	twIm := []float64{sin45}

	ButterflyComplex(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)

	// Expected values computed manually
	tempRe := 3*cos45 - 4*sin45
	tempIm := 3*sin45 + 4*cos45
	wantUpperRe := 1 + tempRe
	wantUpperIm := 2 + tempIm
	wantLowerRe := 1 - tempRe
	wantLowerIm := 2 - tempIm

	if math.Abs(upperRe[0]-wantUpperRe) > 1e-12 {
		t.Errorf("upperRe = %v, want %v", upperRe[0], wantUpperRe)
	}
	if math.Abs(upperIm[0]-wantUpperIm) > 1e-12 {
		t.Errorf("upperIm = %v, want %v", upperIm[0], wantUpperIm)
	}
	if math.Abs(lowerRe[0]-wantLowerRe) > 1e-12 {
		t.Errorf("lowerRe = %v, want %v", lowerRe[0], wantLowerRe)
	}
	if math.Abs(lowerIm[0]-wantLowerIm) > 1e-12 {
		t.Errorf("lowerIm = %v, want %v", lowerIm[0], wantLowerIm)
	}
}

// TestButterflyComplex_KnownValuesSIMD pins the SIMD kernels to hand-derived
// absolute values, not just a differential comparison against the Go path. n=9
// enters the AVX 4-wide and NEON 2-wide main loops AND their scalar tails; every
// element uses the same inputs, so every output (main-loop and tail) must equal
// the same hand-computed result.
func TestButterflyComplex_KnownValuesSIMD(t *testing.T) {
	const n = 9 // >= 4: exercises the SIMD main loop plus the scalar remainder
	cos45 := math.Cos(math.Pi / 4)
	sin45 := math.Sin(math.Pi / 4)

	upperRe := make([]float64, n)
	upperIm := make([]float64, n)
	lowerRe := make([]float64, n)
	lowerIm := make([]float64, n)
	twRe := make([]float64, n)
	twIm := make([]float64, n)
	for i := range n {
		upperRe[i], upperIm[i] = 1, 2
		lowerRe[i], lowerIm[i] = 3, 4
		twRe[i], twIm[i] = cos45, sin45
	}

	ButterflyComplex(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)

	tempRe := 3*cos45 - 4*sin45
	tempIm := 3*sin45 + 4*cos45
	wantUpperRe, wantUpperIm := 1+tempRe, 2+tempIm
	wantLowerRe, wantLowerIm := 1-tempRe, 2-tempIm

	for i := range n {
		if math.Abs(upperRe[i]-wantUpperRe) > 1e-12 {
			t.Errorf("upperRe[%d] = %v, want %v", i, upperRe[i], wantUpperRe)
		}
		if math.Abs(upperIm[i]-wantUpperIm) > 1e-12 {
			t.Errorf("upperIm[%d] = %v, want %v", i, upperIm[i], wantUpperIm)
		}
		if math.Abs(lowerRe[i]-wantLowerRe) > 1e-12 {
			t.Errorf("lowerRe[%d] = %v, want %v", i, lowerRe[i], wantLowerRe)
		}
		if math.Abs(lowerIm[i]-wantLowerIm) > 1e-12 {
			t.Errorf("lowerIm[%d] = %v, want %v", i, lowerIm[i], wantLowerIm)
		}
	}
}

// TestButterflyComplex_LengthMismatch verifies the minLen6 clamp: with unequal
// slice lengths, only the first min(len) elements are processed and the tails of
// the longer, written slices are left untouched.
func TestButterflyComplex_LengthMismatch(t *testing.T) {
	const long, short = 16, 6 // twIm (length short) drives n = 6

	upperRe := make([]float64, long)
	upperIm := make([]float64, long)
	lowerRe := make([]float64, long)
	lowerIm := make([]float64, long)
	twRe := make([]float64, long)
	twIm := make([]float64, short) // shortest slice clamps n

	for i := range long {
		angle := 2 * math.Pi * float64(i) / float64(long)
		upperRe[i] = float64(i+1) * 0.1
		upperIm[i] = float64(i+2) * 0.2
		lowerRe[i] = float64(i+3) * 0.3
		lowerIm[i] = float64(i+4) * 0.4
		twRe[i] = math.Cos(angle)
	}
	for i := range short {
		twIm[i] = math.Sin(2 * math.Pi * float64(i) / float64(short))
	}

	// Snapshot the tails of the four written slices (indices [short, long)).
	upReTail := append([]float64(nil), upperRe[short:]...)
	upImTail := append([]float64(nil), upperIm[short:]...)
	loReTail := append([]float64(nil), lowerRe[short:]...)
	loImTail := append([]float64(nil), lowerIm[short:]...)

	// Independent reference over the first `short` elements only.
	wantUpRe := append([]float64(nil), upperRe[:short]...)
	wantUpIm := append([]float64(nil), upperIm[:short]...)
	wantLoRe := append([]float64(nil), lowerRe[:short]...)
	wantLoIm := append([]float64(nil), lowerIm[:short]...)
	butterflyComplexRef(wantUpRe, wantUpIm, wantLoRe, wantLoIm, twRe[:short], twIm[:short])

	ButterflyComplex(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)

	// The first `short` elements must match the reference.
	for i := range short {
		if math.Abs(upperRe[i]-wantUpRe[i]) > 1e-12 {
			t.Errorf("upperRe[%d] = %v, want %v", i, upperRe[i], wantUpRe[i])
		}
		if math.Abs(upperIm[i]-wantUpIm[i]) > 1e-12 {
			t.Errorf("upperIm[%d] = %v, want %v", i, upperIm[i], wantUpIm[i])
		}
		if math.Abs(lowerRe[i]-wantLoRe[i]) > 1e-12 {
			t.Errorf("lowerRe[%d] = %v, want %v", i, lowerRe[i], wantLoRe[i])
		}
		if math.Abs(lowerIm[i]-wantLoIm[i]) > 1e-12 {
			t.Errorf("lowerIm[%d] = %v, want %v", i, lowerIm[i], wantLoIm[i])
		}
	}
	// Elements past n must be untouched in all four written slices.
	for i := range upReTail {
		k := short + i
		if upperRe[k] != upReTail[i] {
			t.Errorf("upperRe[%d] modified past clamp: got %v, want %v", k, upperRe[k], upReTail[i])
		}
		if upperIm[k] != upImTail[i] {
			t.Errorf("upperIm[%d] modified past clamp: got %v, want %v", k, upperIm[k], upImTail[i])
		}
		if lowerRe[k] != loReTail[i] {
			t.Errorf("lowerRe[%d] modified past clamp: got %v, want %v", k, lowerRe[k], loReTail[i])
		}
		if lowerIm[k] != loImTail[i] {
			t.Errorf("lowerIm[%d] modified past clamp: got %v, want %v", k, lowerIm[k], loImTail[i])
		}
	}
}

func TestButterflyComplex_SIMDvsGo(t *testing.T) {
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128, 256, 512, 1024}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			upperRe := make([]float64, n)
			upperIm := make([]float64, n)
			lowerRe := make([]float64, n)
			lowerIm := make([]float64, n)
			twRe := make([]float64, n)
			twIm := make([]float64, n)

			// Go copies
			upperReGo := make([]float64, n)
			upperImGo := make([]float64, n)
			lowerReGo := make([]float64, n)
			lowerImGo := make([]float64, n)

			// Initialize with varied data to catch bugs
			for i := range n {
				angle := 2 * math.Pi * float64(i) / float64(n)
				upperRe[i] = math.Sin(float64(i) * 0.123)
				upperIm[i] = math.Cos(float64(i) * 0.456)
				lowerRe[i] = math.Sin(float64(i) * 0.789)
				lowerIm[i] = math.Cos(float64(i) * 0.012)
				twRe[i] = math.Cos(angle)
				twIm[i] = math.Sin(angle)
			}

			copy(upperReGo, upperRe)
			copy(upperImGo, upperIm)
			copy(lowerReGo, lowerRe)
			copy(lowerImGo, lowerIm)

			// SIMD version
			ButterflyComplex(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)

			// Pure Go version
			butterflyComplex64Go(upperReGo, upperImGo, lowerReGo, lowerImGo, twRe, twIm)

			// Compare (only difference is FMA vs separate mul+add, ~1 ULP)
			for i := range n {
				if math.Abs(upperRe[i]-upperReGo[i]) > 1e-12 {
					t.Errorf("upperRe[%d]: SIMD=%v, Go=%v", i, upperRe[i], upperReGo[i])
				}
				if math.Abs(upperIm[i]-upperImGo[i]) > 1e-12 {
					t.Errorf("upperIm[%d]: SIMD=%v, Go=%v", i, upperIm[i], upperImGo[i])
				}
				if math.Abs(lowerRe[i]-lowerReGo[i]) > 1e-12 {
					t.Errorf("lowerRe[%d]: SIMD=%v, Go=%v", i, lowerRe[i], lowerReGo[i])
				}
				if math.Abs(lowerIm[i]-lowerImGo[i]) > 1e-12 {
					t.Errorf("lowerIm[%d]: SIMD=%v, Go=%v", i, lowerIm[i], lowerImGo[i])
				}
			}
		})
	}
}

func TestButterflyComplex_FFTPattern(t *testing.T) {
	// Test the exact FFT butterfly pattern at different stages.
	fftSizes := []int{8, 16, 32, 64, 128, 256, 512, 1024}

	for _, fftSize := range fftSizes {
		t.Run(fmt.Sprintf("fft=%d", fftSize), func(t *testing.T) {
			// Precompute twiddle factors
			twRe := make([]float64, fftSize/2)
			twIm := make([]float64, fftSize/2)
			for k := range fftSize / 2 {
				angle := -2 * math.Pi * float64(k) / float64(fftSize)
				twRe[k] = math.Cos(angle)
				twIm[k] = math.Sin(angle)
			}

			// Test at different stages
			for stage := 0; (1 << stage) < fftSize; stage++ {
				halfSize := 1 << stage
				fullSize := halfSize * 2
				twStep := fftSize / fullSize

				t.Run(fmt.Sprintf("stage=%d", stage), func(t *testing.T) {
					// Create test data
					upperRe := make([]float64, halfSize)
					upperIm := make([]float64, halfSize)
					lowerRe := make([]float64, halfSize)
					lowerIm := make([]float64, halfSize)
					gatheredTwRe := make([]float64, halfSize)
					gatheredTwIm := make([]float64, halfSize)

					// Copy data for reference
					upperReRef := make([]float64, halfSize)
					upperImRef := make([]float64, halfSize)
					lowerReRef := make([]float64, halfSize)
					lowerImRef := make([]float64, halfSize)

					for k := range halfSize {
						angle := 2 * math.Pi * float64(k) / float64(halfSize)
						upperRe[k] = math.Cos(angle)
						upperIm[k] = math.Sin(angle)
						lowerRe[k] = math.Sin(angle * 2)
						lowerIm[k] = math.Cos(angle * 2)
						gatheredTwRe[k] = twRe[k*twStep]
						gatheredTwIm[k] = twIm[k*twStep]
					}

					copy(upperReRef, upperRe)
					copy(upperImRef, upperIm)
					copy(lowerReRef, lowerRe)
					copy(lowerImRef, lowerIm)

					// Apply SIMD butterfly
					ButterflyComplex(upperRe, upperIm, lowerRe, lowerIm, gatheredTwRe, gatheredTwIm)

					// Apply reference butterfly
					butterflyComplex64Go(upperReRef, upperImRef, lowerReRef, lowerImRef, gatheredTwRe, gatheredTwIm)

					// Compare
					for k := range halfSize {
						if math.Abs(upperRe[k]-upperReRef[k]) > 1e-12 {
							t.Errorf("stage=%d k=%d upperRe: SIMD=%v, ref=%v", stage, k, upperRe[k], upperReRef[k])
						}
						if math.Abs(upperIm[k]-upperImRef[k]) > 1e-12 {
							t.Errorf("stage=%d k=%d upperIm: SIMD=%v, ref=%v", stage, k, upperIm[k], upperImRef[k])
						}
						if math.Abs(lowerRe[k]-lowerReRef[k]) > 1e-12 {
							t.Errorf("stage=%d k=%d lowerRe: SIMD=%v, ref=%v", stage, k, lowerRe[k], lowerReRef[k])
						}
						if math.Abs(lowerIm[k]-lowerImRef[k]) > 1e-12 {
							t.Errorf("stage=%d k=%d lowerIm: SIMD=%v, ref=%v", stage, k, lowerIm[k], lowerImRef[k])
						}
					}
				})
			}
		})
	}
}

// TestButterflyComplex_AllocFree pins the zero-allocation guarantee. The direct
// cpu-flag dispatch keeps //go:noescape effective; routing through an init-time
// function-pointer table would reintroduce heap allocations here.
func TestButterflyComplex_AllocFree(t *testing.T) {
	const n = 256
	upperRe := make([]float64, n)
	upperIm := make([]float64, n)
	lowerRe := make([]float64, n)
	lowerIm := make([]float64, n)
	twRe := make([]float64, n)
	twIm := make([]float64, n)
	for i := range n {
		angle := 2 * math.Pi * float64(i) / float64(n)
		upperRe[i] = float64(i) * 0.1
		lowerRe[i] = float64(i) * 0.3
		twRe[i] = math.Cos(angle)
		twIm[i] = math.Sin(angle)
	}
	fn := func() { ButterflyComplex(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm) }
	if a := testing.AllocsPerRun(50, fn); a != 0 {
		t.Errorf("ButterflyComplex allocated %v times per run, want 0", a)
	}
}

func BenchmarkButterflyComplex(b *testing.B) {
	sizes := []int{64, 128, 256, 512, 1024, 4096}

	benchFn := func(b *testing.B, n int, fn func(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm []float64)) {
		b.Helper()
		upperRe := make([]float64, n)
		upperIm := make([]float64, n)
		lowerRe := make([]float64, n)
		lowerIm := make([]float64, n)
		twRe := make([]float64, n)
		twIm := make([]float64, n)

		for i := range n {
			angle := 2 * math.Pi * float64(i) / float64(n)
			upperRe[i] = float64(i) * 0.1
			upperIm[i] = float64(i) * 0.2
			lowerRe[i] = float64(i) * 0.3
			lowerIm[i] = float64(i) * 0.4
			twRe[i] = math.Cos(angle)
			twIm[i] = math.Sin(angle)
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8 * 6)) // 6 slices of float64

		for i := 0; i < b.N; i++ {
			fn(upperRe, upperIm, lowerRe, lowerIm, twRe, twIm)
		}
	}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("SIMD_%d", n), func(b *testing.B) {
			benchFn(b, n, ButterflyComplex)
		})
		b.Run(fmt.Sprintf("Go_%d", n), func(b *testing.B) {
			benchFn(b, n, butterflyComplex64Go)
		})
	}
}
