package f32

// DSP Scenario Tests - Aggressive testing with realistic DSP workloads
// These tests compare SIMD implementations against pure Go reference implementations
// using typical audio/signal processing patterns and sizes.

import (
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
)

// =============================================================================
// Test Data Generators for DSP Scenarios
// =============================================================================

// generateSineWave creates a sine wave signal at the given frequency
func generateSineWave32(samples int, frequency, sampleRate float64) []float32 {
	signal := make([]float32, samples)
	for i := range signal {
		t := float64(i) / sampleRate
		signal[i] = float32(math.Sin(2 * math.Pi * frequency * t))
	}
	return signal
}

// generateWhiteNoise creates random noise in range [-1, 1]
func generateWhiteNoise32(samples int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	noise := make([]float32, samples)
	for i := range noise {
		noise[i] = float32(rng.Float64()*2 - 1)
	}
	return noise
}

// generateImpulse creates an impulse response
func generateImpulse32(length, position int) []float32 {
	impulse := make([]float32, length)
	if position < length {
		impulse[position] = 1.0
	}
	return impulse
}

// generateLinearRamp creates a linear ramp from 0 to 1
func generateLinearRamp32(samples int) []float32 {
	ramp := make([]float32, samples)
	for i := range ramp {
		ramp[i] = float32(float64(i) / float64(samples-1))
	}
	return ramp
}

// generateLowPassFIR creates a simple low-pass FIR filter kernel
func generateLowPassFIR32(taps int, cutoff float64) []float32 {
	kernel := make([]float32, taps)
	middle := taps / 2
	sum := float64(0)
	for i := range kernel {
		n := float64(i - middle)
		var val float64
		if n == 0 {
			val = 2 * cutoff
		} else {
			val = math.Sin(2*math.Pi*cutoff*n) / (math.Pi * n)
		}
		// Hamming window
		val *= 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(taps-1))
		kernel[i] = float32(val)
		sum += val
	}
	// Normalize
	for i := range kernel {
		kernel[i] = float32(float64(kernel[i]) / sum)
	}
	return kernel
}

// =============================================================================
// Pure Go Reference Implementations for DSP Operations
// =============================================================================

func goConvolveValid32(signal, kernel []float32) []float32 {
	if len(kernel) == 0 || len(signal) < len(kernel) {
		return nil
	}
	validLen := len(signal) - len(kernel) + 1
	dst := make([]float32, validLen)
	for i := range dst {
		var sum float32
		for j := range kernel {
			sum += signal[i+j] * kernel[j]
		}
		dst[i] = sum
	}
	return dst
}

func goEnergyRMS32(signal []float32) float32 {
	if len(signal) == 0 {
		return 0
	}
	var sumSq float64 // Use float64 for accumulation accuracy
	for _, v := range signal {
		sumSq += float64(v) * float64(v)
	}
	return float32(math.Sqrt(sumSq / float64(len(signal))))
}

func goNormalize32(signal []float32) []float32 {
	dst := make([]float32, len(signal))
	var magnitude float64
	for _, v := range signal {
		magnitude += float64(v) * float64(v)
	}
	magnitude = math.Sqrt(magnitude)
	if magnitude < 1e-10 {
		copy(dst, signal)
		return dst
	}
	invMag := float32(1.0 / magnitude)
	for i, v := range signal {
		dst[i] = v * invMag
	}
	return dst
}

func goMixSignals32(a, b []float32, mixRatio float32) []float32 {
	n := min(len(b), len(a))
	dst := make([]float32, n)
	for i := range dst {
		dst[i] = a[i]*(1-mixRatio) + b[i]*mixRatio
	}
	return dst
}

func goApplyGain32(signal []float32, gain float32) []float32 {
	dst := make([]float32, len(signal))
	for i, v := range signal {
		dst[i] = v * gain
	}
	return dst
}

func goCrossCorrelation32(a, b []float32) float32 {
	n := min(len(b), len(a))
	var sum float32
	for i := range n {
		sum += a[i] * b[i]
	}
	return sum
}

func goEuclideanDistance32(a, b []float32) float32 {
	n := min(len(b), len(a))
	var sum float64 // Use float64 for accuracy
	for i := range n {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return float32(math.Sqrt(sum))
}

// =============================================================================
// DSP Scenario Tests
// =============================================================================

// TestDSP_FIRFilter tests FIR filtering with various kernel sizes
func TestDSP_FIRFilter(t *testing.T) {
	// Typical FIR filter sizes in audio processing
	filterSizes := []int{21, 64, 127, 241}
	signalSizes := []int{1024, 4096}

	for _, filterSize := range filterSizes {
		for _, signalSize := range signalSizes {
			t.Run(generateTestName32("FIR", signalSize, filterSize), func(t *testing.T) {
				signal := generateSineWave32(signalSize, 440.0, 44100.0)
				kernel := generateLowPassFIR32(filterSize, 0.25)

				// Reference implementation
				expected := goConvolveValid32(signal, kernel)

				// SIMD implementation
				validLen := signalSize - filterSize + 1
				got := make([]float32, validLen)
				ConvolveValid(got, signal, kernel)

				assert.Len(t, got, len(expected), "output lengths should match")
				for i := range expected {
					// float32 has less precision, use 1e-5 tolerance
					assert.InDelta(t, float64(expected[i]), float64(got[i]), 1e-5,
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
	tapSizes := []int{32, 64, 128}

	for _, phases := range numPhases {
		for _, taps := range tapSizes {
			t.Run(generateTestName32("Polyphase", phases, taps), func(t *testing.T) {
				// Create polyphase coefficients
				coeffs := make([][]float32, phases)
				for p := range phases {
					coeffs[p] = make([]float32, taps)
					for i := range taps {
						coeffs[p][i] = float32(math.Sin(float64(p*taps+i) * 0.1))
					}
				}

				// Input signal
				input := generateSineWave32(taps, 1000.0, 48000.0)

				// Reference: compute each phase's dot product
				expectedResults := make([]float32, phases)
				for p := range phases {
					expectedResults[p] = goCrossCorrelation32(coeffs[p], input)
				}

				// SIMD batch processing
				gotResults := make([]float32, phases)
				DotProductBatch(gotResults, coeffs, input)

				for p := range phases {
					assert.InDelta(t, float64(expectedResults[p]), float64(gotResults[p]), 1e-4,
						"DotProductBatch phase %d mismatch", p)
				}
			})
		}
	}
}

// TestDSP_AudioMixing tests audio signal mixing operations
func TestDSP_AudioMixing(t *testing.T) {
	// Common audio buffer sizes
	bufferSizes := []int{64, 128, 256, 512, 1024}
	mixRatios := []float32{0.0, 0.25, 0.5, 0.75, 1.0}

	for _, size := range bufferSizes {
		for _, ratio := range mixRatios {
			t.Run(generateTestName32("Mix", size, int(ratio*100)), func(t *testing.T) {
				signalA := generateSineWave32(size, 440.0, 44100.0)
				signalB := generateSineWave32(size, 880.0, 44100.0)

				// Reference
				expected := goMixSignals32(signalA, signalB, ratio)

				// SIMD implementation using Scale and Add
				got := make([]float32, size)
				temp := make([]float32, size)

				// got = signalA * (1-ratio) + signalB * ratio
				Scale(got, signalA, 1-ratio)
				Scale(temp, signalB, ratio)
				Add(got, got, temp)

				for i := range expected {
					assert.InDelta(t, float64(expected[i]), float64(got[i]), 1e-6,
						"Audio mix mismatch at index %d", i)
				}
			})
		}
	}
}

// TestDSP_GainStaging tests gain application typical in audio processing
func TestDSP_GainStaging(t *testing.T) {
	sizes := []int{256, 512, 1024}
	gains := []float32{0.0, 0.5, 1.0, 2.0, 0.707, -1.0}

	for _, size := range sizes {
		for _, gain := range gains {
			t.Run(generateTestName32("Gain", size, int(gain*100)), func(t *testing.T) {
				signal := generateSineWave32(size, 440.0, 44100.0)

				// Reference
				expected := goApplyGain32(signal, gain)

				// SIMD
				got := make([]float32, size)
				Scale(got, signal, gain)

				for i := range expected {
					assert.InDelta(t, float64(expected[i]), float64(got[i]), 1e-6,
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
		t.Run(generateTestName32("RMS", size, 0), func(t *testing.T) {
			// Test with sine wave (RMS should be 1/sqrt(2) for unit amplitude)
			sine := generateSineWave32(size, 100.0, 44100.0)
			expectedRMS := goEnergyRMS32(sine)

			// Using SIMD: sqrt(sum(x^2)/n)
			squared := make([]float32, size)
			Mul(squared, sine, sine)
			sumSq := Sum(squared)
			gotRMS := float32(math.Sqrt(float64(sumSq) / float64(size)))

			assert.InDelta(t, float64(expectedRMS), float64(gotRMS), 1e-5,
				"RMS mismatch for size=%d", size)

			// Verify sine wave RMS is approximately 0.707 (1/sqrt(2))
			// Use larger tolerance due to float32 precision
			assert.InDelta(t, 1.0/math.Sqrt(2), float64(gotRMS), 0.02,
				"Sine wave RMS should be ~0.707")
		})
	}
}

// TestDSP_Normalization tests signal normalization (unit vector)
func TestDSP_Normalization(t *testing.T) {
	sizes := []int{64, 256, 1024}

	for _, size := range sizes {
		t.Run(generateTestName32("Normalize", size, 0), func(t *testing.T) {
			signal := generateWhiteNoise32(size, 42)

			// Reference
			expected := goNormalize32(signal)

			// SIMD
			got := make([]float32, size)
			Normalize(got, signal)

			// Check values match
			for i := range expected {
				assert.InDelta(t, float64(expected[i]), float64(got[i]), 1e-5,
					"Normalize mismatch at index %d", i)
			}

			// Verify unit length
			var magnitude float64
			for _, v := range got {
				magnitude += float64(v) * float64(v)
			}
			magnitude = math.Sqrt(magnitude)
			assert.InDelta(t, 1.0, magnitude, 1e-5,
				"Normalized vector should have unit magnitude")
		})
	}
}

// TestDSP_EuclideanDistance tests distance calculations (common in audio fingerprinting)
func TestDSP_EuclideanDistance(t *testing.T) {
	sizes := []int{64, 256, 1024}

	for _, size := range sizes {
		t.Run(generateTestName32("Distance", size, 0), func(t *testing.T) {
			a := generateSineWave32(size, 440.0, 44100.0)
			b := generateSineWave32(size, 441.0, 44100.0) // Slightly different frequency

			// Reference
			expected := goEuclideanDistance32(a, b)

			// SIMD
			got := EuclideanDistance(a, b)

			assert.InDelta(t, float64(expected), float64(got), 1e-4,
				"EuclideanDistance mismatch for size=%d", size)

			// Test identical signals should give 0
			gotSame := EuclideanDistance(a, a)
			assert.InDelta(t, 0.0, float64(gotSame), 1e-10, "Distance to self should be 0")
		})
	}
}

// TestDSP_StereoInterleaving tests stereo audio interleave/deinterleave
func TestDSP_StereoInterleaving(t *testing.T) {
	// Common audio buffer sizes (per channel)
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		t.Run(generateTestName32("Stereo", size, 0), func(t *testing.T) {
			// Generate stereo audio (left and right channels)
			left := generateSineWave32(size, 440.0, 44100.0)
			right := generateSineWave32(size, 880.0, 44100.0)

			// Interleave
			interleaved := make([]float32, size*2)
			Interleave2(interleaved, left, right)

			// Verify interleaving manually
			for i := range size {
				assert.InDelta(t, float64(left[i]), float64(interleaved[i*2]), 1e-10,
					"Left channel mismatch at sample %d", i)
				assert.InDelta(t, float64(right[i]), float64(interleaved[i*2+1]), 1e-10,
					"Right channel mismatch at sample %d", i)
			}

			// Deinterleave back
			leftOut := make([]float32, size)
			rightOut := make([]float32, size)
			Deinterleave2(leftOut, rightOut, interleaved)

			// Verify round-trip
			for i := range size {
				assert.InDelta(t, float64(left[i]), float64(leftOut[i]), 1e-10,
					"Left channel round-trip mismatch at sample %d", i)
				assert.InDelta(t, float64(right[i]), float64(rightOut[i]), 1e-10,
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
		{2, 64, 1024},
		{4, 64, 1024},
		{8, 32, 512},
	}

	for _, cfg := range configs {
		t.Run(generateTestName32("MultiConv", cfg.numKernels, cfg.kernelLen), func(t *testing.T) {
			signal := generateSineWave32(cfg.signalLen, 1000.0, 48000.0)

			// Create kernels
			kernels := make([][]float32, cfg.numKernels)
			for k := range cfg.numKernels {
				kernels[k] = generateLowPassFIR32(cfg.kernelLen, 0.25+float64(k)*0.05)
			}

			validLen := cfg.signalLen - cfg.kernelLen + 1
			dsts := make([][]float32, cfg.numKernels)
			for k := range cfg.numKernels {
				dsts[k] = make([]float32, validLen)
			}

			// SIMD multi-kernel
			ConvolveValidMulti(dsts, signal, kernels)

			// Compare with single ConvolveValid calls
			for k := range cfg.numKernels {
				expected := goConvolveValid32(signal, kernels[k])
				for i := range expected {
					assert.InDelta(t, float64(expected[i]), float64(dsts[k][i]), 1e-4,
						"MultiConv kernel %d mismatch at index %d", k, i)
				}
			}
		})
	}
}

// TestDSP_StatisticalFunctions tests Mean, Variance, StdDev with DSP-typical data
func TestDSP_StatisticalFunctions(t *testing.T) {
	sizes := []int{64, 256, 1024}

	for _, size := range sizes {
		t.Run(generateTestName32("Stats", size, 0), func(t *testing.T) {
			signal := generateWhiteNoise32(size, 123)

			// Test Mean
			var expectedMean float64
			for _, v := range signal {
				expectedMean += float64(v)
			}
			expectedMean /= float64(size)

			gotMean := Mean(signal)
			assert.InDelta(t, expectedMean, float64(gotMean), 1e-5,
				"Mean mismatch for size=%d", size)

			// White noise should have mean close to 0
			assert.InDelta(t, 0.0, float64(gotMean), 0.15,
				"White noise mean should be close to 0")

			// Test Variance
			var expectedVariance float64
			for _, v := range signal {
				diff := float64(v) - expectedMean
				expectedVariance += diff * diff
			}
			expectedVariance /= float64(size)

			gotVariance := Variance(signal)
			assert.InDelta(t, expectedVariance, float64(gotVariance), 1e-4,
				"Variance mismatch for size=%d", size)

			// Test StdDev
			expectedStdDev := math.Sqrt(expectedVariance)
			gotStdDev := StdDev(signal)
			assert.InDelta(t, expectedStdDev, float64(gotStdDev), 1e-4,
				"StdDev mismatch for size=%d", size)
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
		output := make([]float32, outputLen)

		// Create windowed frames and accumulate
		for f := range numFrames {
			frame := generateSineWave32(frameSize, 440.0, 44100.0)
			// Apply Hann window
			for i := range frame {
				frame[i] *= float32(0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(frameSize-1))))
			}

			offset := f * hopSize
			AccumulateAdd(output, frame, offset)
		}

		// Verify the output is not all zeros
		var sum float64
		for _, v := range output {
			sum += float64(v) * float64(v)
		}
		assert.Greater(t, sum, 0.0, "Accumulated output should not be zero")

		// The overlapping Hann windows should sum to approximately 1 in the middle
		middleStart := frameSize
		middleEnd := outputLen - frameSize
		for i := middleStart; i < middleEnd; i += hopSize {
			assert.NotZero(t, output[i], "Output at position %d should not be zero", i)
		}
	})
}

// TestDSP_Clamping tests signal limiting/clipping operations
func TestDSP_Clamping(t *testing.T) {
	sizes := []int{256, 1024}
	limits := []struct {
		min, max float32
	}{
		{-1.0, 1.0},
		{-0.5, 0.5},
		{0.0, 1.0},
	}

	for _, size := range sizes {
		for _, limit := range limits {
			t.Run(generateTestName32("Clamp", size, int(limit.max*100)), func(t *testing.T) {
				// Generate signal that exceeds limits
				signal := make([]float32, size)
				for i := range signal {
					signal[i] = float32(math.Sin(float64(i)*0.1) * 2.0)
				}

				// Reference
				expected := make([]float32, size)
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
				got := make([]float32, size)
				Clamp(got, signal, limit.min, limit.max)

				for i := range expected {
					assert.InDelta(t, float64(expected[i]), float64(got[i]), 1e-10,
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
		t.Run(generateTestName32("FMA", size, 0), func(t *testing.T) {
			a := generateSineWave32(size, 440.0, 44100.0)
			b := generateSineWave32(size, 880.0, 44100.0)
			c := generateLinearRamp32(size)

			// Reference: dst = a*b + c
			expected := make([]float32, size)
			for i := range size {
				expected[i] = a[i]*b[i] + c[i]
			}

			// SIMD
			got := make([]float32, size)
			FMA(got, a, b, c)

			for i := range expected {
				assert.InDelta(t, float64(expected[i]), float64(got[i]), 1e-6,
					"FMA mismatch at index %d", i)
			}
		})
	}
}

// TestDSP_CumulativeSum tests running sum (integration) common in envelopes
func TestDSP_CumulativeSum(t *testing.T) {
	sizes := []int{64, 256, 1024}

	for _, size := range sizes {
		t.Run(generateTestName32("CumSum", size, 0), func(t *testing.T) {
			// Create an impulse train
			signal := make([]float32, size)
			for i := 0; i < size; i += size / 10 {
				signal[i] = 1.0
			}

			// Reference
			expected := make([]float32, size)
			var sum float32
			for i, v := range signal {
				sum += v
				expected[i] = sum
			}

			// SIMD
			got := make([]float32, size)
			CumulativeSum(got, signal)

			for i := range expected {
				assert.InDelta(t, float64(expected[i]), float64(got[i]), 1e-5,
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

// TestDSP_SqrtReciprocal tests math operations common in audio normalization
func TestDSP_SqrtReciprocal(t *testing.T) {
	sizes := []int{64, 256, 1024}

	for _, size := range sizes {
		t.Run(generateTestName32("SqrtRecip", size, 0), func(t *testing.T) {
			// Generate positive values (avoid sqrt of negative)
			input := make([]float32, size)
			for i := range input {
				input[i] = float32(i+1) * 0.01 // Values 0.01 to 10.24
			}

			// Test Sqrt
			expectedSqrt := make([]float32, size)
			for i, v := range input {
				expectedSqrt[i] = float32(math.Sqrt(float64(v)))
			}

			gotSqrt := make([]float32, size)
			Sqrt(gotSqrt, input)

			for i := range expectedSqrt {
				assert.InDelta(t, float64(expectedSqrt[i]), float64(gotSqrt[i]), 1e-5,
					"Sqrt mismatch at index %d", i)
			}

			// Test Reciprocal
			expectedRecip := make([]float32, size)
			for i, v := range input {
				expectedRecip[i] = 1.0 / v
			}

			gotRecip := make([]float32, size)
			Reciprocal(gotRecip, input)

			for i := range expectedRecip {
				assert.InDelta(t, float64(expectedRecip[i]), float64(gotRecip[i]), 1e-4,
					"Reciprocal mismatch at index %d", i)
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

	// Test with large buffers (simulating audio files)
	largeSize := 1 << 18 // 256K samples (~6 seconds at 44.1kHz)

	t.Run("LargeDotProduct", func(t *testing.T) {
		a := generateWhiteNoise32(largeSize, 1)
		b := generateWhiteNoise32(largeSize, 2)

		// Reference (use float64 for accumulation)
		var expected float64
		for i := range a {
			expected += float64(a[i]) * float64(b[i])
		}

		// SIMD
		got := DotProduct(a, b)

		// float32 accumulation has less precision for large sums
		assert.InDelta(t, expected, float64(got), math.Abs(expected)*1e-4+1,
			"Large DotProduct mismatch")
	})

	t.Run("LargeSum", func(t *testing.T) {
		signal := generateWhiteNoise32(largeSize, 42)

		// Reference
		var expected float64
		for _, v := range signal {
			expected += float64(v)
		}

		// SIMD
		got := Sum(signal)

		assert.InDelta(t, expected, float64(got), math.Abs(expected)*1e-4+1,
			"Large Sum mismatch")
	})

	t.Run("LargeConvolve", func(t *testing.T) {
		signal := generateSineWave32(largeSize, 440.0, 44100.0)
		kernel := generateLowPassFIR32(127, 0.25)

		validLen := largeSize - len(kernel) + 1
		got := make([]float32, validLen)
		ConvolveValid(got, signal, kernel)

		// Just verify it doesn't panic and produces reasonable output
		assert.Len(t, got, validLen)

		// Check a few sample points against reference
		expected := goConvolveValid32(signal[:1000], kernel)
		for i := range expected {
			assert.InDelta(t, float64(expected[i]), float64(got[i]), 1e-4,
				"Large ConvolveValid mismatch at index %d", i)
		}
	})
}

// =============================================================================
// Helper Functions
// =============================================================================

func generateTestName32(op string, param1, param2 int) string {
	if param2 == 0 {
		return op + "_" + strconv.Itoa(param1)
	}
	return op + "_" + strconv.Itoa(param1) + "x" + strconv.Itoa(param2)
}
