package f64

// DSP Scenario Tests - Aggressive testing with realistic DSP workloads
// These tests compare SIMD implementations against pure Go reference implementations
// using typical audio/signal processing patterns and sizes.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

// =============================================================================
// Test Data Generators for DSP Scenarios
// =============================================================================

// generateSineWave creates a sine wave signal at the given frequency
func generateSineWave64(samples int, frequency, sampleRate float64) []float64 {
	signal := make([]float64, samples)
	for i := range signal {
		t := float64(i) / sampleRate
		signal[i] = math.Sin(2 * math.Pi * frequency * t)
	}
	return signal
}

// generateWhiteNoise creates random noise in range [-1, 1]
func generateWhiteNoise64(samples int, seed int64) []float64 {
	rng := rand.New(rand.NewSource(seed))
	noise := make([]float64, samples)
	for i := range noise {
		noise[i] = rng.Float64()*2 - 1
	}
	return noise
}

// generateImpulse creates an impulse response
func generateImpulse64(length, position int) []float64 {
	impulse := make([]float64, length)
	if position < length {
		impulse[position] = 1.0
	}
	return impulse
}

// generateLinearRamp creates a linear ramp from 0 to 1
func generateLinearRamp64(samples int) []float64 {
	ramp := make([]float64, samples)
	for i := range ramp {
		ramp[i] = float64(i) / float64(samples-1)
	}
	return ramp
}

// generateLowPassFIR creates a simple low-pass FIR filter kernel
func generateLowPassFIR64(taps int, cutoff float64) []float64 {
	kernel := make([]float64, taps)
	middle := taps / 2
	sum := 0.0
	for i := range kernel {
		n := float64(i - middle)
		if n == 0 {
			kernel[i] = 2 * cutoff
		} else {
			kernel[i] = math.Sin(2*math.Pi*cutoff*n) / (math.Pi * n)
		}
		// Hamming window
		kernel[i] *= 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(taps-1))
		sum += kernel[i]
	}
	// Normalize
	for i := range kernel {
		kernel[i] /= sum
	}
	return kernel
}

// =============================================================================
// Pure Go Reference Implementations for DSP Operations
// =============================================================================

func goConvolveValid64(signal, kernel []float64) []float64 {
	if len(kernel) == 0 || len(signal) < len(kernel) {
		return nil
	}
	validLen := len(signal) - len(kernel) + 1
	dst := make([]float64, validLen)
	for i := range dst {
		var sum float64
		for j := range kernel {
			sum += signal[i+j] * kernel[j]
		}
		dst[i] = sum
	}
	return dst
}

func goEnergyRMS64(signal []float64) float64 {
	if len(signal) == 0 {
		return 0
	}
	var sumSq float64
	for _, v := range signal {
		sumSq += v * v
	}
	return math.Sqrt(sumSq / float64(len(signal)))
}

func goNormalize64(signal []float64) []float64 {
	dst := make([]float64, len(signal))
	var magnitude float64
	for _, v := range signal {
		magnitude += v * v
	}
	magnitude = math.Sqrt(magnitude)
	if magnitude < 1e-10 {
		copy(dst, signal)
		return dst
	}
	invMag := 1.0 / magnitude
	for i, v := range signal {
		dst[i] = v * invMag
	}
	return dst
}

func goMixSignals64(a, b []float64, mixRatio float64) []float64 {
	n := min(len(b), len(a))
	dst := make([]float64, n)
	for i := range dst {
		dst[i] = a[i]*(1-mixRatio) + b[i]*mixRatio
	}
	return dst
}

func goApplyGain64(signal []float64, gain float64) []float64 {
	dst := make([]float64, len(signal))
	for i, v := range signal {
		dst[i] = v * gain
	}
	return dst
}

func goCrossCorrelation64(a, b []float64) float64 {
	n := min(len(b), len(a))
	var sum float64
	for i := range n {
		sum += a[i] * b[i]
	}
	return sum
}

func goEuclideanDistance64(a, b []float64) float64 {
	n := min(len(b), len(a))
	var sum float64
	for i := range n {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// =============================================================================
// DSP Scenario Tests
// =============================================================================

// TestDSP_FIRFilter tests FIR filtering with various kernel sizes
func TestDSP_FIRFilter(t *testing.T) {
	// Typical FIR filter sizes in audio processing
	filterSizes := []int{21, 64, 127, 241, 512}
	signalSizes := []int{1024, 4096, 8192}

	for _, filterSize := range filterSizes {
		for _, signalSize := range signalSizes {
			t.Run(generateTestName("FIR", signalSize, filterSize), func(t *testing.T) {
				signal := generateSineWave64(signalSize, 440.0, 44100.0)
				kernel := generateLowPassFIR64(filterSize, 0.25)

				// Reference implementation
				expected := goConvolveValid64(signal, kernel)

				// SIMD implementation
				validLen := signalSize - filterSize + 1
				got := make([]float64, validLen)
				ConvolveValid(got, signal, kernel)

				assert.Len(t, got, len(expected), "output lengths should match")
				for i := range expected {
					assert.InDelta(t, expected[i], got[i], 1e-10,
						"ConvolveValid mismatch at index %d for filter=%d signal=%d", i, filterSize, signalSize)
				}
			})
		}
	}
}

// TestDSP_DotProductAccumulation tests dot product with FIR-like accumulation patterns
func TestDSP_DotProductAccumulation(t *testing.T) {
	// Simulate polyphase filter bank processing
	numPhases := []int{2, 4, 8}
	tapSizes := []int{32, 64, 128, 241}

	for _, phases := range numPhases {
		for _, taps := range tapSizes {
			t.Run(generateTestName("Polyphase", phases, taps), func(t *testing.T) {
				// Create polyphase coefficients
				coeffs := make([][]float64, phases)
				for p := range phases {
					coeffs[p] = make([]float64, taps)
					for i := range taps {
						coeffs[p][i] = math.Sin(float64(p*taps+i) * 0.1)
					}
				}

				// Input signal
				input := generateSineWave64(taps, 1000.0, 48000.0)

				// Reference: compute each phase's dot product
				expectedResults := make([]float64, phases)
				for p := range phases {
					expectedResults[p] = goCrossCorrelation64(coeffs[p], input)
				}

				// SIMD batch processing
				gotResults := make([]float64, phases)
				DotProductBatch(gotResults, coeffs, input)

				for p := range phases {
					assert.InDelta(t, expectedResults[p], gotResults[p], 1e-10,
						"DotProductBatch phase %d mismatch", p)
				}
			})
		}
	}
}

// TestDSP_AudioMixing tests audio signal mixing operations
func TestDSP_AudioMixing(t *testing.T) {
	// Common audio buffer sizes
	bufferSizes := []int{64, 128, 256, 512, 1024, 2048, 4096}
	mixRatios := []float64{0.0, 0.25, 0.5, 0.75, 1.0}

	for _, size := range bufferSizes {
		for _, ratio := range mixRatios {
			t.Run(generateTestName("Mix", size, int(ratio*100)), func(t *testing.T) {
				signalA := generateSineWave64(size, 440.0, 44100.0)
				signalB := generateSineWave64(size, 880.0, 44100.0)

				// Reference
				expected := goMixSignals64(signalA, signalB, ratio)

				// SIMD implementation using Scale and Add
				got := make([]float64, size)
				temp := make([]float64, size)

				// got = signalA * (1-ratio) + signalB * ratio
				Scale(got, signalA, 1-ratio)
				Scale(temp, signalB, ratio)
				Add(got, got, temp)

				for i := range expected {
					assert.InDelta(t, expected[i], got[i], 1e-14,
						"Audio mix mismatch at index %d", i)
				}
			})
		}
	}
}

// TestDSP_GainStaging tests gain application typical in audio processing
func TestDSP_GainStaging(t *testing.T) {
	sizes := []int{256, 512, 1024, 4096}
	gains := []float64{0.0, 0.5, 1.0, 2.0, 0.707, -1.0} // -6dB, unity, +6dB, -3dB, invert

	for _, size := range sizes {
		for _, gain := range gains {
			t.Run(generateTestName("Gain", size, int(gain*100)), func(t *testing.T) {
				signal := generateSineWave64(size, 440.0, 44100.0)

				// Reference
				expected := goApplyGain64(signal, gain)

				// SIMD
				got := make([]float64, size)
				Scale(got, signal, gain)

				for i := range expected {
					assert.InDelta(t, expected[i], got[i], 1e-14,
						"Gain mismatch at index %d for gain=%f", i, gain)
				}
			})
		}
	}
}

// TestDSP_RMSEnergy tests RMS energy calculation
func TestDSP_RMSEnergy(t *testing.T) {
	// Use sizes that complete full sine wave cycles for accurate RMS
	// 441 samples at 100Hz/44100Hz = 1 full cycle
	sizes := []int{441, 882, 4410}

	for _, size := range sizes {
		t.Run(generateTestName("RMS", size, 0), func(t *testing.T) {
			// Test with sine wave (RMS should be 1/sqrt(2) for unit amplitude)
			sine := generateSineWave64(size, 100.0, 44100.0)
			expectedRMS := goEnergyRMS64(sine)

			// Using SIMD: sqrt(sum(x^2)/n)
			squared := make([]float64, size)
			Mul(squared, sine, sine)
			sumSq := Sum(squared)
			gotRMS := math.Sqrt(sumSq / float64(size))

			assert.InDelta(t, expectedRMS, gotRMS, 1e-10,
				"RMS mismatch for size=%d", size)

			// Verify sine wave RMS is approximately 0.707 (1/sqrt(2))
			// Use larger tolerance for first size (only 1 cycle)
			tolerance := 0.01
			if size < 1000 {
				tolerance = 0.02
			}
			assert.InDelta(t, 1.0/math.Sqrt(2), gotRMS, tolerance,
				"Sine wave RMS should be ~0.707")
		})
	}
}

// TestDSP_Normalization tests signal normalization (unit vector)
func TestDSP_Normalization(t *testing.T) {
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		t.Run(generateTestName("Normalize", size, 0), func(t *testing.T) {
			signal := generateWhiteNoise64(size, 42)

			// Reference
			expected := goNormalize64(signal)

			// SIMD
			got := make([]float64, size)
			Normalize(got, signal)

			// Check values match
			for i := range expected {
				assert.InDelta(t, expected[i], got[i], 1e-10,
					"Normalize mismatch at index %d", i)
			}

			// Verify unit length
			var magnitude float64
			for _, v := range got {
				magnitude += v * v
			}
			magnitude = math.Sqrt(magnitude)
			assert.InDelta(t, 1.0, magnitude, 1e-10,
				"Normalized vector should have unit magnitude")
		})
	}
}

// TestDSP_EuclideanDistance tests distance calculations (common in audio fingerprinting)
func TestDSP_EuclideanDistance(t *testing.T) {
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		t.Run(generateTestName("Distance", size, 0), func(t *testing.T) {
			a := generateSineWave64(size, 440.0, 44100.0)
			b := generateSineWave64(size, 441.0, 44100.0) // Slightly different frequency

			// Reference
			expected := goEuclideanDistance64(a, b)

			// SIMD
			got := EuclideanDistance(a, b)

			assert.InDelta(t, expected, got, 1e-10,
				"EuclideanDistance mismatch for size=%d", size)

			// Test identical signals should give 0
			gotSame := EuclideanDistance(a, a)
			assert.InDelta(t, 0.0, gotSame, 1e-10, "Distance to self should be 0")
		})
	}
}

// TestDSP_StereoInterleaving tests stereo audio interleave/deinterleave
func TestDSP_StereoInterleaving(t *testing.T) {
	// Common audio buffer sizes (per channel)
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		t.Run(generateTestName("Stereo", size, 0), func(t *testing.T) {
			// Generate stereo audio (left and right channels)
			left := generateSineWave64(size, 440.0, 44100.0)
			right := generateSineWave64(size, 880.0, 44100.0)

			// Interleave
			interleaved := make([]float64, size*2)
			Interleave2(interleaved, left, right)

			// Verify interleaving manually
			for i := range size {
				assert.InDelta(t, left[i], interleaved[i*2], 1e-10,
					"Left channel mismatch at sample %d", i)
				assert.InDelta(t, right[i], interleaved[i*2+1], 1e-10,
					"Right channel mismatch at sample %d", i)
			}

			// Deinterleave back
			leftOut := make([]float64, size)
			rightOut := make([]float64, size)
			Deinterleave2(leftOut, rightOut, interleaved)

			// Verify round-trip
			for i := range size {
				assert.InDelta(t, left[i], leftOut[i], 1e-10,
					"Left channel round-trip mismatch at sample %d", i)
				assert.InDelta(t, right[i], rightOut[i], 1e-10,
					"Right channel round-trip mismatch at sample %d", i)
			}
		})
	}
}

// TestDSP_MultiKernelConvolution tests polyphase-style multi-kernel convolution
func TestDSP_MultiKernelConvolution(t *testing.T) {
	// Typical polyphase resampler configurations
	configs := []struct {
		numKernels int
		kernelLen  int
		signalLen  int
	}{
		{2, 64, 1024},  // 2x resampler
		{4, 64, 1024},  // 4x resampler
		{8, 32, 512},   // 8x resampler, shorter kernels
		{2, 241, 4096}, // High-quality resampler
	}

	for _, cfg := range configs {
		t.Run(generateTestName("MultiConv", cfg.numKernels, cfg.kernelLen), func(t *testing.T) {
			signal := generateSineWave64(cfg.signalLen, 1000.0, 48000.0)

			// Create kernels
			kernels := make([][]float64, cfg.numKernels)
			for k := range cfg.numKernels {
				kernels[k] = generateLowPassFIR64(cfg.kernelLen, 0.25+float64(k)*0.05)
			}

			validLen := cfg.signalLen - cfg.kernelLen + 1
			dsts := make([][]float64, cfg.numKernels)
			for k := range cfg.numKernels {
				dsts[k] = make([]float64, validLen)
			}

			// SIMD multi-kernel
			ConvolveValidMulti(dsts, signal, kernels)

			// Compare with single ConvolveValid calls
			for k := range cfg.numKernels {
				expected := goConvolveValid64(signal, kernels[k])
				for i := range expected {
					assert.InDelta(t, expected[i], dsts[k][i], 1e-10,
						"MultiConv kernel %d mismatch at index %d", k, i)
				}
			}
		})
	}
}

// TestDSP_StatisticalFunctions tests Mean, Variance, StdDev with DSP-typical data
func TestDSP_StatisticalFunctions(t *testing.T) {
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		t.Run(generateTestName("Stats", size, 0), func(t *testing.T) {
			signal := generateWhiteNoise64(size, 123)

			// Test Mean
			var expectedMean float64
			for _, v := range signal {
				expectedMean += v
			}
			expectedMean /= float64(size)

			gotMean := Mean(signal)
			assert.InDelta(t, expectedMean, gotMean, 1e-10,
				"Mean mismatch for size=%d", size)

			// White noise should have mean close to 0
			assert.InDelta(t, 0.0, gotMean, 0.1,
				"White noise mean should be close to 0")

			// Test Variance
			var expectedVariance float64
			for _, v := range signal {
				diff := v - expectedMean
				expectedVariance += diff * diff
			}
			expectedVariance /= float64(size)

			gotVariance := Variance(signal)
			assert.InDelta(t, expectedVariance, gotVariance, 1e-6,
				"Variance mismatch for size=%d", size)

			// Test StdDev
			expectedStdDev := math.Sqrt(expectedVariance)
			gotStdDev := StdDev(signal)
			assert.InDelta(t, expectedStdDev, gotStdDev, 1e-6,
				"StdDev mismatch for size=%d", size)

			// White noise in [-1, 1] should have stddev around 0.58 (sqrt(1/3))
			assert.InDelta(t, 1.0/math.Sqrt(3), gotStdDev, 0.1,
				"White noise stddev should be close to 0.58")
		})
	}
}

// TestDSP_AccumulativeOperations tests overlap-add patterns
func TestDSP_AccumulativeOperations(t *testing.T) {
	// Simulate overlap-add synthesis
	frameSize := 512
	hopSize := 128
	numFrames := 10
	outputLen := frameSize + (numFrames-1)*hopSize

	t.Run("OverlapAdd", func(t *testing.T) {
		output := make([]float64, outputLen)

		// Create windowed frames and accumulate
		for f := range numFrames {
			frame := generateSineWave64(frameSize, 440.0, 44100.0)
			// Apply Hann window
			for i := range frame {
				frame[i] *= 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(frameSize-1)))
			}

			offset := f * hopSize
			AccumulateAdd(output, frame, offset)
		}

		// Verify the output is not all zeros
		var sum float64
		for _, v := range output {
			sum += v * v
		}
		assert.Greater(t, sum, 0.0, "Accumulated output should not be zero")

		// The overlapping Hann windows should sum to approximately 1 in the middle
		// (with 75% overlap, COLA holds)
		middleStart := frameSize
		middleEnd := outputLen - frameSize
		for i := middleStart; i < middleEnd; i += hopSize {
			// Due to overlap-add, the sum should be relatively constant
			assert.NotZero(t, output[i], "Output at position %d should not be zero", i)
		}
	})
}

// TestDSP_Clamping tests signal limiting/clipping operations
func TestDSP_Clamping(t *testing.T) {
	sizes := []int{256, 1024, 4096}
	limits := []struct {
		min, max float64
	}{
		{-1.0, 1.0},   // Standard audio normalization
		{-0.5, 0.5},   // 50% headroom
		{0.0, 1.0},    // Unipolar
		{-0.99, 0.99}, // Avoid digital maximum (common in mastering)
	}

	for _, size := range sizes {
		for _, limit := range limits {
			t.Run(generateTestName("Clamp", size, int(limit.max*100)), func(t *testing.T) {
				// Generate signal that exceeds limits
				signal := make([]float64, size)
				for i := range signal {
					signal[i] = math.Sin(float64(i)*0.1) * 2.0 // Amplitude of 2, exceeds [-1,1]
				}

				// Reference
				expected := make([]float64, size)
				for i, v := range signal {
					switch {
					case v < limit.min:
						expected[i] = limit.min
					case v > limit.max:
						expected[i] = limit.max
					default:
						expected[i] = v
					}
				}

				// SIMD
				got := make([]float64, size)
				Clamp(got, signal, limit.min, limit.max)

				for i := range expected {
					assert.InDelta(t, expected[i], got[i], 1e-10,
						"Clamp mismatch at index %d for limits [%f, %f]", i, limit.min, limit.max)
				}

				// Verify all values are within limits
				for i, v := range got {
					assert.GreaterOrEqual(t, v, limit.min,
						"Value at %d below minimum", i)
					assert.LessOrEqual(t, v, limit.max,
						"Value at %d above maximum", i)
				}
			})
		}
	}
}

// TestDSP_FMAOperations tests fused multiply-add patterns
func TestDSP_FMAOperations(t *testing.T) {
	sizes := []int{64, 256, 1024}

	for _, size := range sizes {
		t.Run(generateTestName("FMA", size, 0), func(t *testing.T) {
			a := generateSineWave64(size, 440.0, 44100.0)
			b := generateSineWave64(size, 880.0, 44100.0)
			c := generateLinearRamp64(size)

			// Reference: dst = a*b + c
			expected := make([]float64, size)
			for i := range size {
				expected[i] = a[i]*b[i] + c[i]
			}

			// SIMD
			got := make([]float64, size)
			FMA(got, a, b, c)

			for i := range expected {
				assert.InDelta(t, expected[i], got[i], 1e-14,
					"FMA mismatch at index %d", i)
			}
		})
	}
}

// TestDSP_CumulativeSum tests running sum (integration) common in envelopes
func TestDSP_CumulativeSum(t *testing.T) {
	sizes := []int{64, 256, 1024}

	for _, size := range sizes {
		t.Run(generateTestName("CumSum", size, 0), func(t *testing.T) {
			// Create an impulse train
			signal := make([]float64, size)
			for i := 0; i < size; i += size / 10 {
				signal[i] = 1.0
			}

			// Reference
			expected := make([]float64, size)
			var sum float64
			for i, v := range signal {
				sum += v
				expected[i] = sum
			}

			// SIMD
			got := make([]float64, size)
			CumulativeSum(got, signal)

			for i := range expected {
				assert.InDelta(t, expected[i], got[i], 1e-10,
					"CumulativeSum mismatch at index %d", i)
			}

			// Verify monotonic increase for non-negative input
			for i := 1; i < size; i++ {
				assert.GreaterOrEqual(t, got[i], got[i-1],
					"CumulativeSum should be monotonic for non-negative input")
			}
		})
	}
}

// =============================================================================
// Stress Tests with Large Data
// =============================================================================

func TestDSP_LargeDataStress(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}

	// Test with very large buffers (simulating long audio files)
	largeSize := 1 << 20 // 1M samples (~23 seconds at 44.1kHz)

	t.Run("LargeDotProduct", func(t *testing.T) {
		a := generateWhiteNoise64(largeSize, 1)
		b := generateWhiteNoise64(largeSize, 2)

		// Reference
		expected := goCrossCorrelation64(a, b)

		// SIMD
		got := DotProduct(a, b)

		// For large sums, allow slightly larger tolerance
		assert.InDelta(t, expected, got, math.Abs(expected)*1e-10+1e-10,
			"Large DotProduct mismatch")
	})

	t.Run("LargeSum", func(t *testing.T) {
		signal := generateWhiteNoise64(largeSize, 42)

		// Reference
		var expected float64
		for _, v := range signal {
			expected += v
		}

		// SIMD
		got := Sum(signal)

		assert.InDelta(t, expected, got, math.Abs(expected)*1e-10+1e-10,
			"Large Sum mismatch")
	})

	t.Run("LargeConvolve", func(t *testing.T) {
		signal := generateSineWave64(largeSize, 440.0, 44100.0)
		kernel := generateLowPassFIR64(127, 0.25)

		validLen := largeSize - len(kernel) + 1
		got := make([]float64, validLen)
		ConvolveValid(got, signal, kernel)

		// Just verify it doesn't panic and produces reasonable output
		assert.Len(t, got, validLen)

		// Check a few sample points against reference
		expected := goConvolveValid64(signal[:1000], kernel)
		for i := range expected {
			assert.InDelta(t, expected[i], got[i], 1e-10,
				"Large ConvolveValid mismatch at index %d", i)
		}
	})
}

// =============================================================================
// Helper Functions
// =============================================================================

func generateTestName(op string, param1, param2 int) string {
	if param2 == 0 {
		return op + "_" + itoa(param1)
	}
	return op + "_" + itoa(param1) + "x" + itoa(param2)
}

func itoa(i int) string {
	return string('0' + byte(i%10))
}
