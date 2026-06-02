package f64

import (
	"fmt"
	"math"
	"testing"
)

var sinkBatch4Results64 []float64

func TestDotProductBatchMatchesDotProductWideShapes(t *testing.T) {
	dims := []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 127, 128, 255, 256, 768}
	rowCounts := []int{0, 1, 2, 3, 4, 5, 7, 8, 16, 128}
	for _, dim := range dims {
		for _, rowCount := range rowCounts {
			t.Run("shape", func(t *testing.T) {
				vec := deterministicF64Vector(11, dim)
				rows := make([][]float64, rowCount)
				for i := range rows {
					rows[i] = deterministicF64Vector(100+i, dim)
				}
				got := make([]float64, rowCount)
				DotProductBatch(got, rows, vec)
				for i := range rows {
					// Anchor the oracle to the pure-scalar dotProductGo rather than the
					// dispatched DotProduct, so on amd64 this checks the batched SIMD
					// kernel against the scalar contract instead of SIMD-vs-SIMD.
					want := dotProductGo(rows[i], vec)
					if !closeFloat64(got[i], want) {
						t.Fatalf("dim=%d rows=%d row=%d got=%g want=%g", dim, rowCount, i, got[i], want)
					}
				}
			})
		}
	}
}

func TestDotProductBatchVariedRowLengths(t *testing.T) {
	vec := deterministicF64Vector(7, 768)
	rows := [][]float64{
		deterministicF64Vector(1, 0),
		deterministicF64Vector(2, 1),
		deterministicF64Vector(3, 15),
		deterministicF64Vector(4, 16),
		deterministicF64Vector(5, 255),
		deterministicF64Vector(6, 768),
		deterministicF64Vector(7, 769),
	}
	got := make([]float64, len(rows))
	DotProductBatch(got, rows, vec)
	for i := range rows {
		want := DotProduct(rows[i], vec)
		if !closeFloat64(got[i], want) {
			t.Fatalf("row=%d len=%d got=%g want=%g", i, len(rows[i]), got[i], want)
		}
	}
}

func TestDotProductBatchAllocs(t *testing.T) {
	for _, rowCount := range []int{4, 8, 16, 128} {
		vec := deterministicF64Vector(21, 768)
		rows := make([][]float64, rowCount)
		for i := range rows {
			rows[i] = deterministicF64Vector(2000+i, 768)
		}
		results := make([]float64, rowCount)
		allocs := testing.AllocsPerRun(100, func() {
			DotProductBatch(results, rows, vec)
		})
		if allocs != 0 {
			t.Fatalf("DotProductBatch rows=%d allocations = %v, want 0", rowCount, allocs)
		}
	}
}

func BenchmarkDotProductBatch768xRows(b *testing.B) {
	for _, rowsN := range []int{4, 8, 16, 128} {
		b.Run(fmt.Sprintf("batch_rows_%d", rowsN), func(b *testing.B) {
			benchmarkDotProductBatch768Rows(b, rowsN, true)
		})
		b.Run(fmt.Sprintf("loop_rows_%d", rowsN), func(b *testing.B) {
			benchmarkDotProductBatch768Rows(b, rowsN, false)
		})
	}
}

func benchmarkDotProductBatch768Rows(b *testing.B, rowsN int, batch bool) {
	b.Helper()
	const dim = 768
	vec := deterministicF64Vector(17, dim)
	rows := make([][]float64, rowsN)
	for i := range rows {
		rows[i] = deterministicF64Vector(1000+i, dim)
	}
	results := make([]float64, rowsN)
	b.ReportAllocs()
	b.ReportMetric(float64(rowsN), "dots/op")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if batch {
			DotProductBatch(results, rows, vec)
		} else {
			for j, row := range rows {
				results[j] = DotProduct(row, vec)
			}
		}
	}
	sinkBatch4Results64 = results
}

func deterministicF64Vector(seed, n int) []float64 {
	out := make([]float64, n)
	x := uint32(seed)*747796405 + 2891336453
	for i := range out {
		x = x*1664525 + 1013904223
		out[i] = float64(int32(x>>9)%2000-1000) / 1000
	}
	return out
}

func closeFloat64(got, want float64) bool {
	diff := math.Abs(got - want)
	return diff <= 1e-9*(1+math.Abs(want))
}
