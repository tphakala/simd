package f64

import (
	"fmt"
	"testing"
)

// Benchmark sizes: Small, Medium, Large
var benchSizes = []int{128, 512, 1024, 4096, 16384, 65536}

// =============================================================================
// Helper functions
// =============================================================================

func makeBenchData64(n int) (a, b, c, dst []float64) {
	a = make([]float64, n)
	b = make([]float64, n)
	c = make([]float64, n)
	dst = make([]float64, n)
	for i := range n {
		a[i] = float64(i%100) + 0.5
		b[i] = float64((i+50)%100) + 0.5
		c[i] = float64((i+25)%100) + 0.5
	}
	return
}

// reportThroughput reports throughput in GB/s for the benchmark
// n is the number of float64 elements processed (8 bytes each)
func reportThroughput64(b *testing.B, n int) {
	b.Helper()
	const bytesPerElement = 8
	b.SetBytes(int64(n * bytesPerElement))
	b.ReportMetric(float64(n*bytesPerElement*b.N)/(1e9*b.Elapsed().Seconds()), "GB/s")
}

// =============================================================================
// DotProduct Benchmarks
// =============================================================================

func BenchmarkDotProduct(b *testing.B) {
	for _, size := range benchSizes {
		a, bb, _, _ := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = DotProduct(a, bb)
			}
			reportThroughput64(b, size*2) // 2 input vectors * 8 bytes
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = dotProductGo(a, bb)
			}
			reportThroughput64(b, size*2)
		})
	}
}

// =============================================================================
// Element-wise Binary Operations
// =============================================================================

func BenchmarkAdd(b *testing.B) {
	for _, size := range benchSizes {
		a, bb, _, dst := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Add(dst, a, bb)
			}
			reportThroughput64(b, size*3)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				addGo(dst, a, bb)
			}
			reportThroughput64(b, size*3)
		})
	}
}

func BenchmarkSub(b *testing.B) {
	for _, size := range benchSizes {
		a, bb, _, dst := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Sub(dst, a, bb)
			}
			reportThroughput64(b, size*3)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				subGo(dst, a, bb)
			}
			reportThroughput64(b, size*3)
		})
	}
}

func BenchmarkMul(b *testing.B) {
	for _, size := range benchSizes {
		a, bb, _, dst := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Mul(dst, a, bb)
			}
			reportThroughput64(b, size*3)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				mulGo(dst, a, bb)
			}
			reportThroughput64(b, size*3)
		})
	}
}

func BenchmarkDiv(b *testing.B) {
	for _, size := range benchSizes {
		a, bb, _, dst := makeBenchData64(size)
		for i := range bb {
			if bb[i] == 0 {
				bb[i] = 1
			}
		}
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Div(dst, a, bb)
			}
			reportThroughput64(b, size*3)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				divGo(dst, a, bb)
			}
			reportThroughput64(b, size*3)
		})
	}
}

// =============================================================================
// Scalar Operations
// =============================================================================

func BenchmarkScale(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData64(size)
		scalar := 2.5
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Scale(dst, a, scalar)
			}
			reportThroughput64(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				scaleGo(dst, a, scalar)
			}
			reportThroughput64(b, size*2)
		})
	}
}

func BenchmarkAddScalar(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData64(size)
		scalar := 10.5
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				AddScalar(dst, a, scalar)
			}
			reportThroughput64(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				addScalarGo(dst, a, scalar)
			}
			reportThroughput64(b, size*2)
		})
	}
}

// =============================================================================
// Reduction Operations
// =============================================================================

func BenchmarkSum(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, _ := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = Sum(a)
			}
			reportThroughput64(b, size)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = sumGo(a)
			}
			reportThroughput64(b, size)
		})
	}
}

func BenchmarkMin(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, _ := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = Min(a)
			}
			reportThroughput64(b, size)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = minGo(a)
			}
			reportThroughput64(b, size)
		})
	}
}

func BenchmarkMax(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, _ := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = Max(a)
			}
			reportThroughput64(b, size)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = maxGo(a)
			}
			reportThroughput64(b, size)
		})
	}
}

// =============================================================================
// Unary Operations
// =============================================================================

func BenchmarkAbs(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData64(size)
		for i := 0; i < len(a); i += 2 {
			a[i] = -a[i]
		}
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Abs(dst, a)
			}
			reportThroughput64(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				absGo(dst, a)
			}
			reportThroughput64(b, size*2)
		})
	}
}

func BenchmarkNeg(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Neg(dst, a)
			}
			reportThroughput64(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				negGo(dst, a)
			}
			reportThroughput64(b, size*2)
		})
	}
}

func BenchmarkSqrt(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData64(size)
		// Ensure positive values
		for i := range a {
			if a[i] < 0 {
				a[i] = -a[i]
			}
		}
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Sqrt(dst, a)
			}
			reportThroughput64(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sqrt64Go(dst, a)
			}
			reportThroughput64(b, size*2)
		})
	}
}

func BenchmarkReciprocal(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData64(size)
		for i := range a {
			if a[i] == 0 {
				a[i] = 1
			}
		}
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Reciprocal(dst, a)
			}
			reportThroughput64(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				reciprocal64Go(dst, a)
			}
			reportThroughput64(b, size*2)
		})
	}
}

// =============================================================================
// FMA and Clamp
// =============================================================================

func BenchmarkFMA(b *testing.B) {
	for _, size := range benchSizes {
		a, bb, c, dst := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				FMA(dst, a, bb, c)
			}
			reportThroughput64(b, size*4)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fmaGo(dst, a, bb, c)
			}
			reportThroughput64(b, size*4)
		})
	}
}

func BenchmarkClamp(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData64(size)
		minVal, maxVal := 25.0, 75.0
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Clamp(dst, a, minVal, maxVal)
			}
			reportThroughput64(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				clampGo(dst, a, minVal, maxVal)
			}
			reportThroughput64(b, size*2)
		})
	}
}

// =============================================================================
// Statistical Operations
// =============================================================================

func BenchmarkMean(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, _ := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = Mean(a)
			}
			reportThroughput64(b, size)
		})
	}
}

func BenchmarkVariance(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, _ := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = Variance(a)
			}
			// Two passes: mean + variance, so 2x data read
			reportThroughput64(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = varianceFullGo(a)
			}
			// Two passes: mean + variance, so 2x data read
			reportThroughput64(b, size*2)
		})
	}
}

// BenchmarkVarianceKernel tests just the variance kernel with pre-computed mean
func BenchmarkVarianceKernel(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, _ := makeBenchData64(size)
		mean := Mean(a)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = variance64(a, mean)
			}
			reportThroughput64(b, size)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = variance64Go(a, mean)
			}
			reportThroughput64(b, size)
		})
	}
}

func BenchmarkEuclideanDistance(b *testing.B) {
	for _, size := range benchSizes {
		a, bb, _, _ := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = EuclideanDistance(a, bb)
			}
			reportThroughput64(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_ = euclideanDistance64Go(a, bb)
			}
			reportThroughput64(b, size*2)
		})
	}
}

func BenchmarkNormalize(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Normalize(dst, a)
			}
			reportThroughput64(b, size*2)
		})
	}
}

func BenchmarkCumulativeSum(b *testing.B) {
	for _, size := range benchSizes {
		a, _, _, dst := makeBenchData64(size)
		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				CumulativeSum(dst, a)
			}
			reportThroughput64(b, size*2)
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				cumulativeSum64Go(dst, a)
			}
			reportThroughput64(b, size*2)
		})
	}
}

// =============================================================================
// Batch Operations
// =============================================================================

func BenchmarkDotProductBatch(b *testing.B) {
	vecSizes := []int{128, 256, 512}
	numRows := []int{10, 100, 1000}

	for _, vecSize := range vecSizes {
		for _, nRows := range numRows {
			vec := make([]float64, vecSize)
			rows := make([][]float64, nRows)
			results := make([]float64, nRows)

			for i := range vec {
				vec[i] = float64(i%100) + 0.5
			}
			for i := range rows {
				rows[i] = make([]float64, vecSize)
				for j := range rows[i] {
					rows[i][j] = float64((i+j)%100) + 0.5
				}
			}

			name := fmt.Sprintf("vec%d_rows%d", vecSize, nRows)
			b.Run(fmt.Sprintf("SIMD_%s", name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					DotProductBatch(results, rows, vec)
				}
				totalBytes := int64(nRows*vecSize*8 + vecSize*8)
				b.SetBytes(totalBytes)
			})
			b.Run(fmt.Sprintf("Go_%s", name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					dotProductBatch64Go(results, rows, vec)
				}
				totalBytes := int64(nRows*vecSize*8 + vecSize*8)
				b.SetBytes(totalBytes)
			})
		}
	}
}

func BenchmarkConvolveValid(b *testing.B) {
	signalSizes := []int{1024, 4096, 16384}
	kernelSizes := []int{16, 64, 256}

	for _, sigSize := range signalSizes {
		for _, kerSize := range kernelSizes {
			if kerSize >= sigSize {
				continue
			}
			signal := make([]float64, sigSize)
			kernel := make([]float64, kerSize)
			dstSize := sigSize - kerSize + 1
			dst := make([]float64, dstSize)

			for i := range signal {
				signal[i] = float64(i%100) + 0.5
			}
			for i := range kernel {
				kernel[i] = float64(i%10) + 0.1
			}

			name := fmt.Sprintf("sig%d_ker%d", sigSize, kerSize)
			b.Run(fmt.Sprintf("SIMD_%s", name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					ConvolveValid(dst, signal, kernel)
				}
				b.SetBytes(int64(dstSize * kerSize * 8))
			})
			b.Run(fmt.Sprintf("Go_%s", name), func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					convolveValid64Go(dst, signal, kernel)
				}
				b.SetBytes(int64(dstSize * kerSize * 8))
			})
		}
	}
}

// =============================================================================
// Quick Summary Benchmark
// =============================================================================

func BenchmarkSummary(b *testing.B) {
	size := 4096
	a, bb, c, dst := makeBenchData64(size)

	ops := []struct {
		name string
		simd func()
		goFn func()
	}{
		{"DotProduct", func() { _ = DotProduct(a, bb) }, func() { _ = dotProductGo(a, bb) }},
		{"Add", func() { Add(dst, a, bb) }, func() { addGo(dst, a, bb) }},
		{"Mul", func() { Mul(dst, a, bb) }, func() { mulGo(dst, a, bb) }},
		{"Scale", func() { Scale(dst, a, 2.5) }, func() { scaleGo(dst, a, 2.5) }},
		{"Sum", func() { _ = Sum(a) }, func() { _ = sumGo(a) }},
		{"FMA", func() { FMA(dst, a, bb, c) }, func() { fmaGo(dst, a, bb, c) }},
		{"Sqrt", func() { Sqrt(dst, a) }, func() { sqrt64Go(dst, a) }},
		{"EuclideanDist", func() { _ = EuclideanDistance(a, bb) }, func() { _ = euclideanDistance64Go(a, bb) }},
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
