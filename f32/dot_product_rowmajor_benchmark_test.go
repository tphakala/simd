package f32

import (
	"fmt"
	"testing"
)

var (
	sinkRowMajorResults []float32
	sinkRowMajorUsed    bool
)

var rowMajorBenchDims = []int{64, 128, 768, 2048}
var rowMajorBenchRows = []int{1, 2, 4, 8, 13, 16, 32, 64, 256}

func BenchmarkDotProductIndexedRowMajor(b *testing.B) {
	for _, pattern := range []string{"contiguous", "scattered"} {
		for _, dims := range rowMajorBenchDims {
			for _, rows := range rowMajorBenchRows {
				baseRows := rows + 257
				if pattern == "contiguous" {
					baseRows = rows
				}
				base := deterministicF32Vector(1000+dims+rows, baseRows*dims)
				query := deterministicF32Vector(2000+dims, dims)
				rowIDs := makeBenchRowIDs(rows, baseRows, pattern)
				rowSlices := makeIndexedRowSlices(base, rowIDs, dims)

				b.Run(fmt.Sprintf("pattern=%s/dim=%d/rows=%d/api=simd", pattern, dims, rows), func(b *testing.B) {
					benchIndexedSIMD(b, base, query, rowIDs, dims)
				})
				b.Run(fmt.Sprintf("pattern=%s/dim=%d/rows=%d/api=loop", pattern, dims, rows), func(b *testing.B) {
					benchIndexedLoop(b, base, query, rowIDs, dims)
				})
				b.Run(fmt.Sprintf("pattern=%s/dim=%d/rows=%d/api=scalar", pattern, dims, rows), func(b *testing.B) {
					benchIndexedScalar(b, base, query, rowIDs, dims)
				})
				b.Run(fmt.Sprintf("pattern=%s/dim=%d/rows=%d/api=batch", pattern, dims, rows), func(b *testing.B) {
					benchBatchRows(b, rowSlices, query)
				})
			}
		}
	}
}

func BenchmarkDotProductStridedRowMajor(b *testing.B) {
	for _, pattern := range []string{"contiguous", "fixed_stride"} {
		for _, dims := range rowMajorBenchDims {
			stride := dims
			if pattern == "fixed_stride" {
				stride = dims + 16
			}
			for _, rows := range rowMajorBenchRows {
				base := deterministicF32Vector(3000+dims+rows+stride, rows*stride)
				query := deterministicF32Vector(4000+dims, dims)
				rowSlices := makeStridedRowSlices(base, rows, dims, stride)

				b.Run(fmt.Sprintf("pattern=%s/dim=%d/rows=%d/api=simd", pattern, dims, rows), func(b *testing.B) {
					benchStridedSIMD(b, base, query, rows, dims, stride)
				})
				b.Run(fmt.Sprintf("pattern=%s/dim=%d/rows=%d/api=loop", pattern, dims, rows), func(b *testing.B) {
					benchStridedLoop(b, base, query, rows, dims, stride)
				})
				b.Run(fmt.Sprintf("pattern=%s/dim=%d/rows=%d/api=scalar", pattern, dims, rows), func(b *testing.B) {
					benchStridedScalar(b, base, query, rows, dims, stride)
				})
				b.Run(fmt.Sprintf("pattern=%s/dim=%d/rows=%d/api=batch", pattern, dims, rows), func(b *testing.B) {
					benchBatchRows(b, rowSlices, query)
				})
			}
		}
	}
}

func benchIndexedSIMD(b *testing.B, base, query []float32, rowIDs []uint32, dims int) {
	b.Helper()
	dst := make([]float32, len(rowIDs))
	reportRowMajorBench(b, len(rowIDs), dims)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkRowMajorUsed = DotProductIndexed(dst, base, query, rowIDs, dims)
	}
	reportOptimizedMetric(b, sinkRowMajorUsed)
	reportRowMajorRate(b, len(rowIDs))
	sinkRowMajorResults = dst
}

func benchIndexedLoop(b *testing.B, base, query []float32, rowIDs []uint32, dims int) {
	b.Helper()
	dst := make([]float32, len(rowIDs))
	reportRowMajorBench(b, len(rowIDs), dims)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j, rowID := range rowIDs {
			off := int(rowID) * dims
			dst[j] = DotProduct(base[off:off+dims], query[:dims])
		}
	}
	reportRowMajorRate(b, len(rowIDs))
	sinkRowMajorResults = dst
}

func benchIndexedScalar(b *testing.B, base, query []float32, rowIDs []uint32, dims int) {
	b.Helper()
	dst := make([]float32, len(rowIDs))
	reportRowMajorBench(b, len(rowIDs), dims)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dotProductIndexedGo(dst, base, query, rowIDs, dims)
	}
	reportRowMajorRate(b, len(rowIDs))
	sinkRowMajorResults = dst
}

func benchStridedSIMD(b *testing.B, base, query []float32, rows, dims, stride int) {
	b.Helper()
	dst := make([]float32, rows)
	reportRowMajorBench(b, rows, dims)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkRowMajorUsed = DotProductStrided(dst, base, query, rows, dims, stride)
	}
	reportOptimizedMetric(b, sinkRowMajorUsed)
	reportRowMajorRate(b, rows)
	sinkRowMajorResults = dst
}

func benchStridedLoop(b *testing.B, base, query []float32, rows, dims, stride int) {
	b.Helper()
	dst := make([]float32, rows)
	reportRowMajorBench(b, rows, dims)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for row := range rows {
			off := row * stride
			dst[row] = DotProduct(base[off:off+dims], query[:dims])
		}
	}
	reportRowMajorRate(b, rows)
	sinkRowMajorResults = dst
}

func benchStridedScalar(b *testing.B, base, query []float32, rows, dims, stride int) {
	b.Helper()
	dst := make([]float32, rows)
	reportRowMajorBench(b, rows, dims)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dotProductStridedGo(dst, base, query, rows, dims, stride)
	}
	reportRowMajorRate(b, rows)
	sinkRowMajorResults = dst
}

func benchBatchRows(b *testing.B, rows [][]float32, query []float32) {
	b.Helper()
	dst := make([]float32, len(rows))
	dims := len(query)
	reportRowMajorBench(b, len(rows), dims)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotProductBatch(dst, rows, query)
	}
	reportRowMajorRate(b, len(rows))
	sinkRowMajorResults = dst
}

func reportRowMajorBench(b *testing.B, rows, dims int) {
	b.Helper()
	b.ReportAllocs()
	b.ReportMetric(float64(rows), "dots/op")
	b.SetBytes(int64(rows * dims * 2 * 4))
}

func reportRowMajorRate(b *testing.B, rows int) {
	b.Helper()
	elapsed := b.Elapsed().Seconds()
	if elapsed > 0 {
		b.ReportMetric(float64(rows)*float64(b.N)/elapsed, "dots/s")
	}
}

func reportOptimizedMetric(b *testing.B, optimized bool) {
	b.Helper()
	if optimized {
		b.ReportMetric(1, "optimized")
		return
	}
	b.ReportMetric(0, "optimized")
}

func makeBenchRowIDs(rows, baseRows int, pattern string) []uint32 {
	ids := make([]uint32, rows)
	for i := range ids {
		switch pattern {
		case "scattered":
			ids[i] = uint32((i*131 + 17) % baseRows)
		default:
			ids[i] = uint32(i)
		}
	}
	return ids
}

func makeIndexedRowSlices(base []float32, rowIDs []uint32, dims int) [][]float32 {
	rows := make([][]float32, len(rowIDs))
	for i, rowID := range rowIDs {
		off := int(rowID) * dims
		rows[i] = base[off : off+dims]
	}
	return rows
}

func makeStridedRowSlices(base []float32, rows, dims, stride int) [][]float32 {
	out := make([][]float32, rows)
	for i := range out {
		off := i * stride
		out[i] = base[off : off+dims]
	}
	return out
}
