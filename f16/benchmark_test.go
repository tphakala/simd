package f16

import (
	"fmt"
	"testing"

	"github.com/tphakala/simd/f32"
)

// Benchmark sizes matching f32 package
var benchSizes = []int{128, 512, 1024, 4096, 16384, 65536}

// Sink variables to prevent dead code elimination
var (
	sink32  float32
	sink16  Float16
	sinkF32 []float32
)

// =============================================================================
// Helper functions
// =============================================================================

func makeBenchDataF16(n int) (a, b, c, dst []Float16) {
	a = make([]Float16, n)
	b = make([]Float16, n)
	c = make([]Float16, n)
	dst = make([]Float16, n)
	for i := range n {
		a[i] = FromFloat32(float32(i%100) + 0.5)
		b[i] = FromFloat32(float32((i+50)%100) + 0.5)
		c[i] = FromFloat32(float32((i+25)%100) + 0.5)
	}
	return
}

func makeBenchDataF32(n int) (a, b, c, dst []float32) {
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

// reportThroughput16 reports throughput in GB/s for Float16 (2 bytes each)
func reportThroughput16(b *testing.B, n int) {
	b.Helper()
	const bytesPerElement = 2
	b.SetBytes(int64(n * bytesPerElement))
	b.ReportMetric(float64(n*bytesPerElement*b.N)/(1e9*b.Elapsed().Seconds()), "GB/s")
}

// reportThroughput32 reports throughput in GB/s for float32 (4 bytes each)
func reportThroughput32(b *testing.B, n int) {
	b.Helper()
	const bytesPerElement = 4
	b.SetBytes(int64(n * bytesPerElement))
	b.ReportMetric(float64(n*bytesPerElement*b.N)/(1e9*b.Elapsed().Seconds()), "GB/s")
}

// =============================================================================
// DotProduct Benchmarks - f16 SIMD vs Go, and f16 vs f32
// =============================================================================

func BenchmarkDotProduct(b *testing.B) {
	for _, size := range benchSizes {
		a16, b16, _, _ := makeBenchDataF16(size)
		a32, b32, _, _ := makeBenchDataF32(size)

		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = DotProduct(a16, b16)
			}
			reportThroughput16(b, size*2)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = dotProductGo(a16, b16)
			}
			reportThroughput16(b, size*2)
		})
		b.Run(fmt.Sprintf("F32_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = f32.DotProduct(a32, b32)
			}
			reportThroughput32(b, size*2)
		})
	}
}

// BenchmarkDotProductF32 measures the FP32-widened DotProduct variant
// against the native (potentially-saturating) DotProduct and the pure-Go
// reference. On ARM64 with asimdhp, Native uses dotProductNEON (FP16 FMUL)
// while F32Wide uses dotProductWideNEON (FP32 FMLA after widening). On
// AMD64 both F32Wide and Native call dotProductGo, so they should report
// near-identical times.
func BenchmarkDotProductF32(b *testing.B) {
	for _, size := range benchSizes {
		a16, b16, _, _ := makeBenchDataF16(size)

		b.Run(fmt.Sprintf("F32Wide_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = DotProductF32(a16, b16)
			}
			reportThroughput16(b, size*2)
		})
		b.Run(fmt.Sprintf("Native_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = DotProduct(a16, b16)
			}
			reportThroughput16(b, size*2)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = dotProductGo(a16, b16)
			}
			reportThroughput16(b, size*2)
		})
	}
}

// =============================================================================
// Element-wise Binary Operations
// =============================================================================

//nolint:dupl // Benchmark structure is intentionally similar across operations
func BenchmarkAdd(b *testing.B) {
	for _, size := range benchSizes {
		a16, b16, _, dst16 := makeBenchDataF16(size)
		a32, b32, _, dst32 := makeBenchDataF32(size)

		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Add(dst16, a16, b16)
			}
			reportThroughput16(b, size*3)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				addGo(dst16, a16, b16)
			}
			reportThroughput16(b, size*3)
		})
		b.Run(fmt.Sprintf("F32_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				f32.Add(dst32, a32, b32)
			}
			reportThroughput32(b, size*3)
		})
	}
}

//nolint:dupl // Benchmark structure is intentionally similar across operations
func BenchmarkMul(b *testing.B) {
	for _, size := range benchSizes {
		a16, b16, _, dst16 := makeBenchDataF16(size)
		a32, b32, _, dst32 := makeBenchDataF32(size)

		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Mul(dst16, a16, b16)
			}
			reportThroughput16(b, size*3)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				mulGo(dst16, a16, b16)
			}
			reportThroughput16(b, size*3)
		})
		b.Run(fmt.Sprintf("F32_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				f32.Mul(dst32, a32, b32)
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
		a16, _, _, dst16 := makeBenchDataF16(size)
		a32, _, _, dst32 := makeBenchDataF32(size)
		scalar16 := FromFloat32(2.5)
		scalar32 := float32(2.5)

		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Scale(dst16, a16, scalar16)
			}
			reportThroughput16(b, size*2)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				scaleGo(dst16, a16, scalar16)
			}
			reportThroughput16(b, size*2)
		})
		b.Run(fmt.Sprintf("F32_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				f32.Scale(dst32, a32, scalar32)
			}
			reportThroughput32(b, size*2)
		})
	}
}

// =============================================================================
// Reduction Operations
// =============================================================================

//nolint:dupl // Benchmark structure is intentionally similar across operations
func BenchmarkSum(b *testing.B) {
	for _, size := range benchSizes {
		a16, _, _, _ := makeBenchDataF16(size)
		a32, _, _, _ := makeBenchDataF32(size)

		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = Sum(a16)
			}
			reportThroughput16(b, size)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = sumGo(a16)
			}
			reportThroughput16(b, size)
		})
		b.Run(fmt.Sprintf("F32_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = f32.Sum(a32)
			}
			reportThroughput32(b, size)
		})
	}
}

//nolint:dupl // Benchmark structure is intentionally similar across operations
func BenchmarkMin(b *testing.B) {
	for _, size := range benchSizes {
		a16, _, _, _ := makeBenchDataF16(size)
		a32, _, _, _ := makeBenchDataF32(size)

		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink16 = Min(a16)
			}
			reportThroughput16(b, size)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink16 = minGo(a16)
			}
			reportThroughput16(b, size)
		})
		b.Run(fmt.Sprintf("F32_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = f32.Min(a32)
			}
			reportThroughput32(b, size)
		})
	}
}

//nolint:dupl // Benchmark structure is intentionally similar across operations
func BenchmarkMax(b *testing.B) {
	for _, size := range benchSizes {
		a16, _, _, _ := makeBenchDataF16(size)
		a32, _, _, _ := makeBenchDataF32(size)

		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink16 = Max(a16)
			}
			reportThroughput16(b, size)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink16 = maxGo(a16)
			}
			reportThroughput16(b, size)
		})
		b.Run(fmt.Sprintf("F32_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = f32.Max(a32)
			}
			reportThroughput32(b, size)
		})
	}
}

// =============================================================================
// FMA (Fused Multiply-Add)
// =============================================================================

func BenchmarkFMA(b *testing.B) {
	for _, size := range benchSizes {
		a16, b16, c16, dst16 := makeBenchDataF16(size)
		a32, b32, c32, dst32 := makeBenchDataF32(size)

		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				FMA(dst16, a16, b16, c16)
			}
			reportThroughput16(b, size*4)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fmaGo(dst16, a16, b16, c16)
			}
			reportThroughput16(b, size*4)
		})
		b.Run(fmt.Sprintf("F32_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				f32.FMA(dst32, a32, b32, c32)
			}
			reportThroughput32(b, size*4)
		})
	}
}

// =============================================================================
// Activation Functions
// =============================================================================

func BenchmarkReLU(b *testing.B) {
	for _, size := range benchSizes {
		a16, _, _, dst16 := makeBenchDataF16(size)
		a32, _, _, dst32 := makeBenchDataF32(size)

		// Include negative values
		for i := 0; i < len(a16); i += 2 {
			a16[i] = FromFloat32(-ToFloat32(a16[i]))
			a32[i] = -a32[i]
		}

		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ReLU(dst16, a16)
			}
			reportThroughput16(b, size*2)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				reluGo(dst16, a16)
			}
			reportThroughput16(b, size*2)
		})
		b.Run(fmt.Sprintf("F32_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				f32.ReLU(dst32, a32)
			}
			reportThroughput32(b, size*2)
		})
	}
}

// =============================================================================
// Conversion Benchmarks
// =============================================================================

func BenchmarkToFloat32Slice(b *testing.B) {
	for _, size := range benchSizes {
		src := make([]Float16, size)
		dst := make([]float32, size)
		for i := range src {
			src[i] = FromFloat32(float32(i%100) + 0.5)
		}

		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ToFloat32Slice(dst, src)
			}
			// Read 2 bytes, write 4 bytes = 6 bytes per element
			b.SetBytes(int64(size * 6))
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				toFloat32SliceGo(dst, src)
			}
			b.SetBytes(int64(size * 6))
		})
	}
}

func BenchmarkFromFloat32Slice(b *testing.B) {
	for _, size := range benchSizes {
		src := make([]float32, size)
		dst := make([]Float16, size)
		for i := range src {
			src[i] = float32(i%100) + 0.5
		}

		b.Run(fmt.Sprintf("SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				FromFloat32Slice(dst, src)
			}
			// Read 4 bytes, write 2 bytes = 6 bytes per element
			b.SetBytes(int64(size * 6))
		})
		b.Run(fmt.Sprintf("Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				fromFloat32SliceGo(dst, src)
			}
			b.SetBytes(int64(size * 6))
		})
	}
}

// =============================================================================
// Summary Benchmark - Quick comparison at 4096 elements
// =============================================================================

func BenchmarkSummary(b *testing.B) {
	size := 4096
	a16, b16, c16, dst16 := makeBenchDataF16(size)
	a32, b32, c32, dst32 := makeBenchDataF32(size)
	scalar16 := FromFloat32(2.5)

	ops := []struct {
		name   string
		f16    func()
		f32    func()
	}{
		{"DotProduct", func() { sink32 = DotProduct(a16, b16) }, func() { sink32 = f32.DotProduct(a32, b32) }},
		{"Add", func() { Add(dst16, a16, b16) }, func() { f32.Add(dst32, a32, b32) }},
		{"Mul", func() { Mul(dst16, a16, b16) }, func() { f32.Mul(dst32, a32, b32) }},
		{"Scale", func() { Scale(dst16, a16, scalar16) }, func() { f32.Scale(dst32, a32, 2.5) }},
		{"Sum", func() { sink32 = Sum(a16) }, func() { sink32 = f32.Sum(a32) }},
		{"FMA", func() { FMA(dst16, a16, b16, c16) }, func() { f32.FMA(dst32, a32, b32, c32) }},
	}

	for _, op := range ops {
		b.Run("F16_"+op.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				op.f16()
			}
		})
		b.Run("F32_"+op.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				op.f32()
			}
		})
	}
}

// =============================================================================
// Memory Bandwidth Test - Key metric for F16 advantage
// =============================================================================

func BenchmarkMemoryBandwidth(b *testing.B) {
	// Test large arrays to highlight memory bandwidth benefits
	sizes := []int{65536, 262144, 1048576} // 64K, 256K, 1M elements

	for _, size := range sizes {
		a16, b16, _, dst16 := makeBenchDataF16(size)
		a32, b32, _, dst32 := makeBenchDataF32(size)

		// F16: processes 2*size bytes of input + size bytes of output
		b.Run(fmt.Sprintf("F16_Add_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Add(dst16, a16, b16)
			}
			b.SetBytes(int64(size * 6)) // 3 arrays * 2 bytes
		})

		// F32: processes 4*size bytes of input + 2*size bytes of output
		b.Run(fmt.Sprintf("F32_Add_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				f32.Add(dst32, a32, b32)
			}
			b.SetBytes(int64(size * 12)) // 3 arrays * 4 bytes
		})
	}
}

// =============================================================================
// EuclideanDistance / Variance / StdDev / Interleave / ClampScale (new SIMD)
// =============================================================================

func BenchmarkEuclideanDistance(b *testing.B) {
	for _, size := range benchSizes {
		a16, b16, _, _ := makeBenchDataF16(size)
		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = EuclideanDistance(a16, b16)
			}
			reportThroughput16(b, size)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = euclideanDistanceGo(a16, b16)
			}
			reportThroughput16(b, size)
		})
	}
}

func BenchmarkVariance(b *testing.B) {
	for _, size := range benchSizes {
		a16, _, _, _ := makeBenchDataF16(size)
		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = Variance(a16)
			}
			reportThroughput16(b, size)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			mean := Mean(a16)
			for i := 0; i < b.N; i++ {
				sink32 = varianceGo(a16, mean)
			}
			reportThroughput16(b, size)
		})
	}
}

func BenchmarkInterleave2(b *testing.B) {
	for _, size := range benchSizes {
		a16, b16, _, _ := makeBenchDataF16(size)
		dst := make([]Float16, size*2)
		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Interleave2(dst, a16, b16)
			}
			reportThroughput16(b, size)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				interleave2Go(dst, a16, b16)
			}
			reportThroughput16(b, size)
		})
	}
}

func BenchmarkDeinterleave2(b *testing.B) {
	for _, size := range benchSizes {
		src := make([]Float16, size*2)
		for i := range src {
			src[i] = FromFloat32(float32(i%100) + 0.5)
		}
		a16 := make([]Float16, size)
		b16 := make([]Float16, size)
		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				Deinterleave2(a16, b16, src)
			}
			reportThroughput16(b, size)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				deinterleave2Go(a16, b16, src)
			}
			reportThroughput16(b, size)
		})
	}
}

func BenchmarkClampScale(b *testing.B) {
	minV := FromFloat32(10)
	maxV := FromFloat32(90)
	sc := FromFloat32(0.0125)
	for _, size := range benchSizes {
		a16, _, _, dst := makeBenchDataF16(size)
		b.Run(fmt.Sprintf("F16_SIMD_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ClampScale(dst, a16, minV, maxV, sc)
			}
			reportThroughput16(b, size)
		})
		b.Run(fmt.Sprintf("F16_Go_%d", size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				clampScaleGo(dst, a16, minV, maxV, sc)
			}
			reportThroughput16(b, size)
		})
	}
}
