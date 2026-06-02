package f32

import (
	"runtime"
	"testing"

	"github.com/tphakala/simd/cpu"
)

func TestDotProductIndexedRowMajorParity(t *testing.T) {
	dimsList := []int{1, 2, 7, 8, 15, 16, 31, 64, 65, 128, 768}
	rowCounts := []int{0, 1, 2, 4, 5, 8, 13, 16}
	for _, dims := range dimsList {
		for _, rows := range rowCounts {
			t.Run("shape", func(t *testing.T) {
				baseRows := rows + 11
				base := deterministicF32Vector(100+dims+rows, baseRows*dims)
				query := deterministicF32Vector(200+dims, dims)
				rowIDs := make([]uint32, rows)
				for i := range rowIDs {
					rowIDs[i] = uint32((i*7 + 3) % baseRows)
				}

				got := make([]float32, rows)
				DotProductIndexed(got, base, query, rowIDs, dims)
				// Compare against a pure-scalar oracle that never touches the
				// SIMD dispatch path, so a bug shared by the batched and
				// single-row SIMD kernels cannot slip through.
				assertCloseSlice(t, got, scalarIndexedOracle(base, query, rowIDs, dims))
			})
		}
	}
}

func TestDotProductStridedRowMajorParity(t *testing.T) {
	dimsList := []int{1, 2, 7, 8, 15, 16, 31, 64, 65, 128, 768}
	rowCounts := []int{0, 1, 2, 4, 5, 8, 13, 16}
	for _, dims := range dimsList {
		for _, rows := range rowCounts {
			t.Run("shape", func(t *testing.T) {
				stride := dims + 3
				base := deterministicF32Vector(300+dims+rows, rows*stride)
				query := deterministicF32Vector(400+dims, dims)

				got := make([]float32, rows)
				DotProductStrided(got, base, query, rows, dims, stride)
				assertCloseSlice(t, got, scalarStridedOracle(base, query, rows, dims, stride))
			})
		}
	}
}

func TestDotProductRowMajorRaggedAndTails(t *testing.T) {
	base := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, // truncated row 2
	}
	query := []float32{1, 10, 100}

	t.Run("indexed", func(t *testing.T) {
		rowIDs := []uint32{0, 1, 2, 99}
		got := []float32{-1, -1, -1, -1, 123}
		used := DotProductIndexed(got[:4], base, query, rowIDs, 4)
		if used {
			t.Fatalf("ragged indexed shape should use fallback")
		}
		want := []float32{321, 765, 109, 0, 123}
		assertCloseSlice(t, got, want)
	})

	t.Run("strided", func(t *testing.T) {
		got := []float32{-1, -1, -1, -1, 123}
		used := DotProductStrided(got[:4], base, query, 4, 4, 4)
		if used {
			t.Fatalf("ragged strided shape should use fallback")
		}
		want := []float32{321, 765, 109, 0, 123}
		assertCloseSlice(t, got, want)
	})

	t.Run("dst shorter", func(t *testing.T) {
		got := []float32{-1, -1}
		DotProductIndexed(got, base, query, []uint32{0, 1, 2}, 4)
		assertCloseSlice(t, got, []float32{321, 765})
	})

	t.Run("invalid shape zeros processed dst", func(t *testing.T) {
		got := []float32{7, 8, 9}
		DotProductStrided(got, base, query, 3, 0, 4)
		assertCloseSlice(t, got, []float32{0, 0, 0})

		got = []float32{7, 8, 9}
		DotProductStrided(got, base, query, 3, 4, 0)
		assertCloseSlice(t, got, []float32{0, 0, 0})
	})
}

func TestDotProductRowMajorOptimizedStatus(t *testing.T) {
	const dims = 64
	const rows = 8
	base := deterministicF32Vector(501, rows*dims)
	query := deterministicF32Vector(502, dims)
	rowIDs := []uint32{7, 0, 5, 2, 6, 1, 4, 3}
	indexedDst := make([]float32, rows)
	stridedDst := make([]float32, rows)

	indexedUsed := DotProductIndexed(indexedDst, base, query, rowIDs, dims)
	stridedUsed := DotProductStrided(stridedDst, base, query, rows, dims, dims)
	wantOptimized := (runtime.GOARCH == "amd64" && ((cpu.X86.AVX512F && cpu.X86.AVX512VL) || (cpu.X86.AVX && cpu.X86.FMA))) ||
		(runtime.GOARCH == "arm64" && cpu.HasNEON())
	if indexedUsed != wantOptimized {
		t.Fatalf("indexed optimized status = %v, want %v (cpu=%s)", indexedUsed, wantOptimized, cpu.Info())
	}
	if stridedUsed != wantOptimized {
		t.Fatalf("strided optimized status = %v, want %v (cpu=%s)", stridedUsed, wantOptimized, cpu.Info())
	}

	if DotProductIndexed(make([]float32, 2), base, query, rowIDs[:2], dims) {
		t.Fatalf("rows<4 should not report optimized")
	}
	if DotProductStrided(make([]float32, rows), base, query[:32], rows, dims, dims) {
		t.Fatalf("query shorter than dims should not report optimized")
	}
}

func TestDotProductRowMajorAllocs(t *testing.T) {
	const dims = 64
	const rows = 8
	base := deterministicF32Vector(601, rows*dims)
	query := deterministicF32Vector(602, dims)
	rowIDs := []uint32{7, 0, 5, 2, 6, 1, 4, 3}
	dst := make([]float32, rows)

	indexedAllocs := testing.AllocsPerRun(1000, func() {
		DotProductIndexed(dst, base, query, rowIDs, dims)
	})
	if indexedAllocs != 0 {
		t.Fatalf("DotProductIndexed allocations = %v, want 0", indexedAllocs)
	}

	stridedAllocs := testing.AllocsPerRun(1000, func() {
		DotProductStrided(dst, base, query, rows, dims, dims)
	})
	if stridedAllocs != 0 {
		t.Fatalf("DotProductStrided allocations = %v, want 0", stridedAllocs)
	}
}

func assertCloseSlice(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d len(want)=%d", len(got), len(want))
	}
	for i := range got {
		if !closeFloat32(got[i], want[i]) {
			t.Fatalf("[%d] got=%g want=%g", i, got[i], want[i])
		}
	}
}

// scalarIndexedOracle and scalarStridedOracle recompute the expected scores
// with the pure-scalar dotProductGo, mirroring the public APIs' row selection,
// ragged-input clamping, and out-of-range zeroing. They never call the
// dispatched dotProduct, so they stay independent of the SIMD kernels under
// test.
func scalarIndexedOracle(base, query []float32, rowIDs []uint32, dims int) []float32 {
	dst := make([]float32, len(rowIDs))
	if dims <= 0 || len(query) == 0 {
		return dst
	}
	for i, id := range rowIDs {
		off := int(id) * dims
		if off < 0 || off >= len(base) {
			continue
		}
		n := min(dims, len(query))
		if rem := len(base) - off; rem < n {
			n = rem
		}
		if n <= 0 {
			continue
		}
		dst[i] = dotProductGo(base[off:off+n], query[:n])
	}
	return dst
}

func scalarStridedOracle(base, query []float32, rowCount, dims, stride int) []float32 {
	dst := make([]float32, rowCount)
	if dims <= 0 || stride <= 0 || len(query) == 0 {
		return dst
	}
	for i := range rowCount {
		off := i * stride
		if off < 0 || off >= len(base) {
			continue
		}
		n := min(dims, len(query))
		if rem := len(base) - off; rem < n {
			n = rem
		}
		if n <= 0 {
			continue
		}
		dst[i] = dotProductGo(base[off:off+n], query[:n])
	}
	return dst
}
