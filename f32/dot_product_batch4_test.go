package f32

import (
	"fmt"
	"math"
	"testing"
)

var sinkBatch4Results []float32

func TestDotProductBatchMatchesDotProductWideShapes(t *testing.T) {
	dims := []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 64, 127, 128, 255, 256, 768}
	rowCounts := []int{0, 1, 2, 3, 4, 5, 7, 8, 16, 128}
	for _, dim := range dims {
		for _, rowCount := range rowCounts {
			t.Run("shape", func(t *testing.T) {
				vec := deterministicF32Vector(11, dim)
				rows := make([][]float32, rowCount)
				for i := range rows {
					rows[i] = deterministicF32Vector(100+i, dim)
				}
				got := make([]float32, rowCount)
				DotProductBatch(got, rows, vec)
				for i := range rows {
					want := DotProduct(rows[i], vec)
					if !closeFloat32(got[i], want) {
						t.Fatalf("dim=%d rows=%d row=%d got=%g want=%g", dim, rowCount, i, got[i], want)
					}
				}
			})
		}
	}
}

func TestDotProductBatchVariedRowLengths(t *testing.T) {
	vec := deterministicF32Vector(7, 768)
	rows := [][]float32{
		deterministicF32Vector(1, 0),
		deterministicF32Vector(2, 1),
		deterministicF32Vector(3, 15),
		deterministicF32Vector(4, 16),
		deterministicF32Vector(5, 255),
		deterministicF32Vector(6, 768),
		deterministicF32Vector(7, 769),
	}
	got := make([]float32, len(rows))
	DotProductBatch(got, rows, vec)
	for i := range rows {
		want := DotProduct(rows[i], vec)
		if !closeFloat32(got[i], want) {
			t.Fatalf("row=%d len=%d got=%g want=%g", i, len(rows[i]), got[i], want)
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
	vec := deterministicF32Vector(17, dim)
	rows := make([][]float32, rowsN)
	for i := range rows {
		rows[i] = deterministicF32Vector(1000+i, dim)
	}
	results := make([]float32, rowsN)
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
	sinkBatch4Results = results
}

func deterministicF32Vector(seed, n int) []float32 {
	out := make([]float32, n)
	x := uint32(seed)*747796405 + 2891336453
	for i := range out {
		x = x*1664525 + 1013904223
		out[i] = float32(int32(x>>9)%2000-1000) / 1000
	}
	return out
}

func closeFloat32(got, want float32) bool {
	diff := math.Abs(float64(got - want))
	return diff <= 1e-3*(1+math.Abs(float64(want)))
}
