package f32

import (
	"fmt"
	"testing"
)

// Benchmark sizes: Small, Medium, Large
var benchSizes = []int{128, 512, 1024, 4096, 16384, 65536}

// =============================================================================
// IMPORTANT: Sink Variables for Accurate Benchmarking
// =============================================================================
//
// These package-level sink variables MUST be used for functions that return values.
// Without them, the Go compiler can apply dead code elimination (DCE) to optimize
// away function calls when their results are discarded.
//
// THE BUG (DO NOT REPEAT):
//
//	for i := 0; i < b.N; i++ {
//	    _ = sumGo(a)  // WRONG: compiler may optimize this away entirely!
//	}
//
// When benchmarking Go functions in the same package, the compiler can see their
// implementation. If the result is discarded with "_ = func()", the compiler may
// determine the computation has no observable side effects and eliminate it.
// This causes the Go implementation to appear artificially faster than SIMD.
//
// Assembly functions (SIMD) are opaque to the compiler and cannot be optimized
// away, which creates a false comparison where Go appears faster.
//
// THE FIX (ALWAYS USE THIS PATTERN):
//
//	for i := 0; i < b.N; i++ {
//	    sink32 = sumGo(a)  // CORRECT: result stored in package-level variable
//	}
//
// Package-level variables cannot be optimized away because the compiler cannot
// prove they won't be read by other code after the benchmark completes.
//
// ALTERNATIVE SAFE PATTERN (also acceptable):
//
//	var result float32
//	for i := 0; i < b.N; i++ {
//	    result = Sum(a)
//	}
//	_ = result  // Use result after loop to prevent DCE
var (
	sink32   float32
	sinkBool bool
	sinkInt  int
)

// =============================================================================
// Helper functions
// =============================================================================

func makeBenchData32(n int) (a, b, c, dst []float32) {
	a = make([]float32, n)
	b = make([]float32, n)
	c = make([]float32, n)
	dst = make([]float32, n)
	for i := range n {
		a[i] = float32(i%100) + 0.5
		b[i] = float32((i+50)%100) + 0.5
		c[i] = float32((i+25)%100) + 0.5
	}
	return
}

// reportThroughput reports throughput in GB/s for the benchmark
// n is the number of float32 elements processed (4 bytes each)
func reportThroughput32(b *testing.B, n int) {
	b.Helper()
	const bytesPerElement = 4
	b.SetBytes(int64(n * bytesPerElement))
	// Also report as custom metric
	b.ReportMetric(float64(n*bytesPerElement*b.N)/(1e9*b.Elapsed().Seconds()), "GB/s")
}

// =============================================================================
// DotProduct Benchmarks
// =============================================================================

func BenchmarkDotProduct(b *testing.B) {
	for _, size := range benchSizes {
		a, bb, _, _ := makeBenchData32(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = DotProduct(a, bb)
			}
			reportThroughput32(b, size*2) // 2 input vectors * 4 bytes
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = dotProductGo(a, bb)
			}
			reportThroughput32(b, size*2)
		})
	}
}

// =============================================================================
// Element-wise Binary Operations
// =============================================================================

func BenchmarkAdd(b *testing.B) {
	for _, size := range benchSizes {
		a, bb, _, dst := makeBenchData32(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Add(dst, a, bb)
			}
			reportThroughput32(b, size*3) // 2 input + 1 output
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				addGo(dst, a, bb)
			}
			reportThroughput32(b, size*3)
		})
	}
}

func BenchmarkSub(b *testing.B) {
	for _, size := range benchSizes {
		a, bb, _, dst := makeBenchData32(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Sub(dst, a, bb)
			}
			reportThroughput32(b, size*3)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				subGo(dst, a, bb)
			}
			reportThroughput32(b, size*3)
		})
	}
}

func BenchmarkMul(b *testing.B) {
	for _, size := range benchSizes {
		a, bb, _, dst := makeBenchData32(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Mul(dst, a, bb)
			}
			reportThroughput32(b, size*3)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				mulGo(dst, a, bb)
			}
			reportThroughput32(b, size*3)
		})
	}
}

func BenchmarkDiv(b *testing.B) {
	for _, size := range benchSizes {
		a, bb, _, dst := makeBenchData32(size)
		// Avoid division by zero
		for i := range bb {
			if bb[i] == 0 {
				bb[i] = 1
			}
		}
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Div(dst, a, bb)
			}
			reportThroughput32(b, size*3)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				divGo(dst, a, bb)
			}
			reportThroughput32(b, size*3)
		})
	}
}

// =============================================================================
// Scalar Operations
// =============================================================================

func BenchmarkScale(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData32(size)
		scalar := float32(2.5)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Scale(dst, a, scalar)
			}
			reportThroughput32(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				scaleGo(dst, a, scalar)
			}
			reportThroughput32(b, size*2)
		})
	}
}

func BenchmarkAddScalar(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData32(size)
		scalar := float32(10.5)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				AddScalar(dst, a, scalar)
			}
			reportThroughput32(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				addScalarGo(dst, a, scalar)
			}
			reportThroughput32(b, size*2)
		})
	}
}

// =============================================================================
// Reduction Operations
// =============================================================================

func BenchmarkSum(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, _ := makeBenchData32(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = Sum(a)
			}
			reportThroughput32(b, size)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = sumGo(a)
			}
			reportThroughput32(b, size)
		})
	}
}

func BenchmarkMin(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, _ := makeBenchData32(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = Min(a)
			}
			reportThroughput32(b, size)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = minGo(a)
			}
			reportThroughput32(b, size)
		})
	}
}

func BenchmarkMax(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, _ := makeBenchData32(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = Max(a)
			}
			reportThroughput32(b, size)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = maxGo(a)
			}
			reportThroughput32(b, size)
		})
	}
}

// =============================================================================
// Unary Operations
// =============================================================================

func BenchmarkAbs(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData32(size)
		// Add some negative values
		for i := 0; i < len(a); i += 2 {
			a[i] = -a[i]
		}
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Abs(dst, a)
			}
			reportThroughput32(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				absGo(dst, a)
			}
			reportThroughput32(b, size*2)
		})
	}
}

func BenchmarkAbsPow34(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData32(size)
		// Include negatives; AbsPow34 takes the magnitude before the power.
		for i := 0; i < len(a); i += 2 {
			a[i] = -a[i]
		}
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				AbsPow34(dst, a)
			}
			reportThroughput32(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				absPow34Go(dst, a)
			}
			reportThroughput32(b, size*2)
		})
	}
}

func BenchmarkNeg(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData32(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Neg(dst, a)
			}
			reportThroughput32(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				negGo(dst, a)
			}
			reportThroughput32(b, size*2)
		})
	}
}

func BenchmarkSigmoid(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData32(size)
		// Use values in reasonable range for sigmoid
		for i := range a {
			a[i] = (a[i] - 50) / 10 // Range roughly -5 to +5
		}
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Sigmoid(dst, a)
			}
			reportThroughput32(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sigmoid32Go(dst, a)
			}
			reportThroughput32(b, size*2)
		})
	}
}

func BenchmarkExp(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData32(size)
		// Use values in a reasonable range for exp
		for i := range a {
			a[i] = (a[i] - 50) / 10 // Range roughly -5 to +5
		}
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Exp(dst, a)
			}
			reportThroughput32(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				exp32Go(dst, a)
			}
			reportThroughput32(b, size*2)
		})
	}
}

func BenchmarkReLU(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData32(size)
		// Use values in range that includes negative values
		for i := range a {
			a[i] = (a[i] - 50) / 10 // Range roughly -5 to +5
		}
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ReLU(dst, a)
			}
			reportThroughput32(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				relu32Go(dst, a)
			}
			reportThroughput32(b, size*2)
		})
	}
}

func BenchmarkTanh(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData32(size)
		// Use values in reasonable range for tanh
		for i := range a {
			a[i] = (a[i] - 50) / 10 // Range roughly -5 to +5
		}
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Tanh(dst, a)
			}
			reportThroughput32(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				tanh32Go(dst, a)
			}
			reportThroughput32(b, size*2)
		})
	}
}

// =============================================================================
// FMA and Clamp
// =============================================================================

func BenchmarkFMA(b *testing.B) {
	for _, size := range benchSizes {
		a, bb, c, dst := makeBenchData32(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				FMA(dst, a, bb, c)
			}
			reportThroughput32(b, size*4) // 3 input + 1 output
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fmaGo(dst, a, bb, c)
			}
			reportThroughput32(b, size*4)
		})
	}
}

func BenchmarkClamp(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData32(size)
		minVal, maxVal := float32(25.0), float32(75.0)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Clamp(dst, a, minVal, maxVal)
			}
			reportThroughput32(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				clampGo(dst, a, minVal, maxVal)
			}
			reportThroughput32(b, size*2)
		})
	}
}

// =============================================================================
// Batch Operations
// =============================================================================

func BenchmarkDotProductBatch(b *testing.B) {
	// Test with realistic embedding sizes
	vecSizes := []int{128, 256, 512}
	numRows := []int{10, 100, 1000}

	for _, vecSize := range vecSizes {
		for _, nRows := range numRows {
			vec := make([]float32, vecSize)
			rows := make([][]float32, nRows)
			results := make([]float32, nRows)

			for i := range vec {
				vec[i] = float32(i%100) + 0.5
			}
			for i := range rows {
				rows[i] = make([]float32, vecSize)
				for j := range rows[i] {
					rows[i][j] = float32((i+j)%100) + 0.5
				}
			}

			name := fmt.Sprintf("vec%d_rows%d", vecSize, nRows)
			b.Run(fmt.Sprintf("SIMD_%s", name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					DotProductBatch(results, rows, vec)
				}
				// Total bytes: nRows * vecSize * 4 (rows) + vecSize * 4 (vec)
				totalBytes := int64(nRows*vecSize*4 + vecSize*4)
				b.SetBytes(totalBytes)
			})
			b.Run(fmt.Sprintf("Go_%s", name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					dotProductBatch32Go(results, rows, vec)
				}
				totalBytes := int64(nRows*vecSize*4 + vecSize*4)
				b.SetBytes(totalBytes)
			})
		}
	}
}

func BenchmarkConvolveValid(b *testing.B) {
	// Signal and kernel size combinations
	signalSizes := []int{1024, 4096, 16384}
	kernelSizes := []int{16, 64, 256}

	for _, sigSize := range signalSizes {
		for _, kerSize := range kernelSizes {
			if kerSize >= sigSize {
				continue
			}
			signal := make([]float32, sigSize)
			kernel := make([]float32, kerSize)
			dstSize := sigSize - kerSize + 1
			dst := make([]float32, dstSize)

			for i := range signal {
				signal[i] = float32(i%100) + 0.5
			}
			for i := range kernel {
				kernel[i] = float32(i%10) + 0.1
			}

			name := fmt.Sprintf("sig%d_ker%d", sigSize, kerSize)
			b.Run(fmt.Sprintf("SIMD_%s", name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					ConvolveValid(dst, signal, kernel)
				}
				// Operations: dstSize * kerSize multiply-adds
				b.SetBytes(int64(dstSize * kerSize * 4))
			})
			b.Run(fmt.Sprintf("Go_%s", name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					convolveValid32Go(dst, signal, kernel)
				}
				b.SetBytes(int64(dstSize * kerSize * 4))
			})
		}
	}
}

// BenchmarkConvolveValidMaxAbs compares the fused kernel against the Go-level
// fusion it replaces (rc.1 behavior: a loop over the dispatched dotProduct). The
// fused kernel keeps the kernel resident and folds the abs-max in, removing the
// per-output dispatch overhead; the win is largest for short kernels.
func BenchmarkConvolveValidMaxAbs(b *testing.B) {
	signalSizes := []int{4096, 48000}
	kernelSizes := []int{16, 32, 64}

	for _, sigSize := range signalSizes {
		for _, kerSize := range kernelSizes {
			signal := make([]float32, sigSize)
			kernel := make([]float32, kerSize)
			for i := range signal {
				signal[i] = float32(i%100) - 50
			}
			for i := range kernel {
				kernel[i] = float32(i%10) - 5
			}

			name := fmt.Sprintf("sig%d_ker%d", sigSize, kerSize)
			b.Run(fmt.Sprintf("Fused_%s", name), func(b *testing.B) {
				var sink float32
				for i := 0; i < b.N; i++ {
					sink = ConvolveValidMaxAbs(signal, kernel)
				}
				_ = sink
			})
			b.Run(fmt.Sprintf("GoFusion_%s", name), func(b *testing.B) {
				var sink float32
				for i := 0; i < b.N; i++ {
					sink = convolveValidMaxAbsGo(signal, kernel)
				}
				_ = sink
			})
		}
	}
}

// =============================================================================
// Quick Summary Benchmark (run with -bench=Summary)
// =============================================================================

func BenchmarkSummary(b *testing.B) {
	// Single size for quick comparison
	size := 4096
	a, bb, c, dst := makeBenchData32(size)

	ops := []struct {
		name string
		simd func()
		goFn func()
	}{
		{"DotProduct", func() { sink32 = DotProduct(a, bb) }, func() { sink32 = dotProductGo(a, bb) }},
		{"Add", func() { Add(dst, a, bb) }, func() { addGo(dst, a, bb) }},
		{"Mul", func() { Mul(dst, a, bb) }, func() { mulGo(dst, a, bb) }},
		{"Scale", func() { Scale(dst, a, 2.5) }, func() { scaleGo(dst, a, 2.5) }},
		{"Sum", func() { sink32 = Sum(a) }, func() { sink32 = sumGo(a) }},
		{"FMA", func() { FMA(dst, a, bb, c) }, func() { fmaGo(dst, a, bb, c) }},
	}

	for _, op := range ops {
		b.Run("SIMD_"+op.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				op.simd()
			}
		})
		b.Run("Go_"+op.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				op.goFn()
			}
		})
	}
}

// =============================================================================
// Int32ToFloat32Scale Benchmarks
// =============================================================================

func BenchmarkInt32ToFloat32Scale(b *testing.B) {
	for _, size := range benchSizes {
		src := make([]int32, size)
		dst := make([]float32, size)
		for i := range src {
			src[i] = int32((i % 65536) - 32768)
		}
		scale := float32(1.0 / 32768.0)

		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Int32ToFloat32Scale(dst, src, scale)
			}
			// Read int32 (4 bytes) + write float32 (4 bytes) = 8 bytes per element
			b.SetBytes(int64(size * 8))
		})

		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				int32ToFloat32ScaleGo(dst, src, scale)
			}
			b.SetBytes(int64(size * 8))
		})
	}
}

// =============================================================================
// Int16ToFloat32Scale Benchmarks
// =============================================================================

// benchScalePair runs the SIMD and Go sub-benchmarks for one PCM conversion
// primitive at a single size. bytesPerElem is the combined read+write width per
// element, used for the throughput (MB/s) report.
func benchScalePair(b *testing.B, size, bytesPerElem int, simd, gofb func()) {
	b.Helper()
	b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			simd()
		}
		b.SetBytes(int64(size * bytesPerElem))
	})
	b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			gofb()
		}
		b.SetBytes(int64(size * bytesPerElem))
	})
}

func BenchmarkInt16ToFloat32Scale(b *testing.B) {
	for _, size := range benchSizes {
		src := make([]int16, size)
		dst := make([]float32, size)
		for i := range src {
			src[i] = int16((i % 65536) - 32768)
		}
		scale := float32(1.0 / 32768.0)
		// Read int16 (2 bytes) + write float32 (4 bytes) = 6 bytes per element.
		benchScalePair(b, size, 6,
			func() { Int16ToFloat32Scale(dst, src, scale) },
			func() { int16ToFloat32ScaleGo(dst, src, scale) })
	}
}

// =============================================================================
// Float32ToInt16Scale Benchmarks
// =============================================================================

func BenchmarkFloat32ToInt16Scale(b *testing.B) {
	for _, size := range benchSizes {
		src := make([]float32, size)
		dst := make([]int16, size)
		for i := range src {
			src[i] = float32((i%65536)-32768) / 32768.0
		}
		scale := float32(32767.0)
		// Read float32 (4 bytes) + write int16 (2 bytes) = 6 bytes per element.
		benchScalePair(b, size, 6,
			func() { Float32ToInt16Scale(dst, src, scale) },
			func() { float32ToInt16ScaleGo(dst, src, scale) })
	}
}

// =============================================================================
// MinIdxOfSum / MinIdxOfSumRows Benchmarks
// =============================================================================

// minIdxOfSumRowsBenchShapes lists the (rows, n) shapes benchmarked for
// MinIdxOfSumRows. 11x11 through 17x17 track the motivating workload, where
// rows and n both run 11 to 17 with slide -1; 8x14 straddles the NEON 4-row /
// AVX2 8-row block seam, and 32x32, 96x17, 96x96 exercise larger row counts.
var minIdxOfSumRowsBenchShapes = [][2]int{
	{8, 14},
	{11, 11},
	{11, 14},
	{14, 14},
	{14, 17},
	{17, 17},
	{32, 32},
	{96, 17},
	{96, 96},
}

// makeMinIdxOfSumRowsBenchData builds the a and k inputs for a rows x n
// MinIdxOfSumRows benchmark at slide -1, with base = rows-1 so the lowest
// row offset (row rows-1) lands at k[0], and k sized so every row's window
// fits (see windowLayout in minidxofsum_test.go).
func makeMinIdxOfSumRowsBenchData(rows, n int) (a, k []float32, base int) {
	base, klen := windowLayout(rows, n, -1)
	a = make([]float32, n)
	k = make([]float32, klen)
	for i := range a {
		a[i] = float32(i%100) + 0.5
	}
	for i := range k {
		k[i] = float32((i+37)%100) + 0.5
	}
	return a, k, base
}

func BenchmarkMinIdxOfSumRows_RxN(b *testing.B) {
	for _, shape := range minIdxOfSumRowsBenchShapes {
		rows, n := shape[0], shape[1]
		a, k, base := makeMinIdxOfSumRowsBenchData(rows, n)
		vals := make([]float32, rows)
		idxs := make([]int32, rows)
		b.Run(fmt.Sprintf("%dx%d", rows, n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MinIdxOfSumRows(vals, idxs, a, k, base, -1)
			}
			reportThroughput32(b, rows*n)
		})
	}
}

func BenchmarkMinIdxOfSumRowsGo_RxN(b *testing.B) {
	for _, shape := range minIdxOfSumRowsBenchShapes {
		rows, n := shape[0], shape[1]
		a, k, base := makeMinIdxOfSumRowsBenchData(rows, n)
		vals := make([]float32, rows)
		idxs := make([]int32, rows)
		b.Run(fmt.Sprintf("%dx%d", rows, n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				minIdxOfSumRowsGo(vals, idxs, a, k, base, -1)
			}
			reportThroughput32(b, rows*n)
		})
	}
}

// BenchmarkMinIdxOfSumRows_Slide compares slide +1 against slide -1 at a
// single 14x14 shape, to confirm the store-time-reversal design (see
// minIdxOfSumRows32 in f32_arm64.go / f32_amd64.go) keeps the two directions
// equal in cost.
func BenchmarkMinIdxOfSumRows_Slide(b *testing.B) {
	const rows, n = 14, 14
	vals := make([]float32, rows)
	idxs := make([]int32, rows)

	b.Run("plus1", func(b *testing.B) {
		base, klen := windowLayout(rows, n, 1)
		a := make([]float32, n)
		k := make([]float32, klen)
		for i := range a {
			a[i] = float32(i%100) + 0.5
		}
		for i := range k {
			k[i] = float32((i+37)%100) + 0.5
		}
		for i := 0; i < b.N; i++ {
			MinIdxOfSumRows(vals, idxs, a, k, base, 1)
		}
		reportThroughput32(b, rows*n)
	})
	b.Run("minus1", func(b *testing.B) {
		a, k, base := makeMinIdxOfSumRowsBenchData(rows, n)
		for i := 0; i < b.N; i++ {
			MinIdxOfSumRows(vals, idxs, a, k, base, -1)
		}
		reportThroughput32(b, rows*n)
	})
}

// BenchmarkMinIdxOfSum_N benchmarks the pairwise scalar MinIdxOfSum at the
// same n range as the batched MinIdxOfSumRows shapes, plus a larger n, to
// document its per-call cost next to the batched form.
func BenchmarkMinIdxOfSum_N(b *testing.B) {
	for _, n := range []int{11, 14, 17, 96} {
		a, bb, _, _ := makeBenchData32(n)
		b.Run(fmt.Sprintf("%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sinkInt, sink32 = MinIdxOfSum(a, bb)
			}
			reportThroughput32(b, n*2)
		})
	}
}
