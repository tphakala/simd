package i8

import (
	"testing"
)

// naiveDotProductBatch is the independent scalar oracle for DotProductBatch:
// out[i] = sum_j int32(rows[i][j])*int32(vec[j]) over m = min(len(row), len(vec))
// elements, int32 two's-complement wraparound, with an empty row or empty vec
// scoring 0. Written straight from the contract, it shares no code with the
// kernels or dispatch under test.
func naiveDotProductBatch(rows [][]int8, vec []int8) []int32 {
	out := make([]int32, len(rows))
	for i, row := range rows {
		m := min(len(row), len(vec))
		var s int32
		for j := range m {
			s += int32(row[j]) * int32(vec[j])
		}
		out[i] = s
	}
	return out
}

// TestDotProductBatch sweeps vector lengths and row counts that straddle the
// AVX2/NEON dispatch gates (vecLen >= 16, rows >= 4), the 16-wide loop, the
// 8-wide AVX2 prelude, and the scalar tail, checking every result against the
// independent oracle. Full-length rows keep every group on the 4-row kernel.
func TestDotProductBatch(t *testing.T) {
	vecLens := []int{0, 1, 7, 8, 15, 16, 17, 23, 24, 31, 32, 33, 63, 64, 100, 256}
	rowCounts := []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17}
	seed := uint32(0)
	for _, vl := range vecLens {
		for _, nr := range rowCounts {
			seed++
			vec := genI8(vl, seed)
			rows := make([][]int8, nr)
			for r := range rows {
				seed++
				rows[r] = genI8(vl, seed)
			}
			results := make([]int32, nr)
			DotProductBatch(results, rows, vec)
			want := naiveDotProductBatch(rows, vec)
			for r := range results {
				if results[r] != want[r] {
					t.Fatalf("vecLen=%d rows=%d row %d: got %d want %d", vl, nr, r, results[r], want[r])
				}
			}
		}
	}
}

// TestDotProductBatchRagged mixes rows shorter than, equal to, and longer than
// vec, plus nil and empty rows, inside and across 4-row groups, so the
// per-row fallback for ragged groups and trailing rows is exercised.
func TestDotProductBatchRagged(t *testing.T) {
	vec := genI8(20, 7)
	rows := [][]int8{
		genI8(20, 11), // == len(vec): kernel-eligible
		genI8(5, 12),  // shorter: forces the group onto the per-row fallback
		nil,           // nil row -> 0
		{},            // empty row -> 0
		genI8(40, 13), // longer than vec, clamped to 20
		genI8(16, 14), // shorter, crosses a 16-byte block boundary
		genI8(1, 15),  // single element
		genI8(20, 16), // == len(vec)
	}
	results := make([]int32, len(rows))
	DotProductBatch(results, rows, vec)
	want := naiveDotProductBatch(rows, vec)
	for r := range results {
		if results[r] != want[r] {
			t.Fatalf("row %d (len %d): got %d want %d", r, len(rows[r]), results[r], want[r])
		}
	}
}

// TestDotProductBatchFullGroups uses only full-length rows so consecutive 4-row
// groups run entirely through the kernel, and checks a row count that leaves a
// non-multiple-of-4 trailing remainder.
func TestDotProductBatchFullGroups(t *testing.T) {
	for _, vl := range []int{16, 24, 48, 64} {
		vec := genI8(vl, uint32(vl))
		rows := make([][]int8, 10) // two full groups + a 2-row tail
		for r := range rows {
			rows[r] = genI8(vl, uint32(vl*100+r))
		}
		results := make([]int32, len(rows))
		DotProductBatch(results, rows, vec)
		want := naiveDotProductBatch(rows, vec)
		for r := range results {
			if results[r] != want[r] {
				t.Fatalf("vecLen=%d row %d: got %d want %d", vl, r, results[r], want[r])
			}
		}
	}
}

// TestDotProductBatchEmptyVec verifies the contract that an empty vec zeroes
// results[:n] rather than leaving stale values (the dot of an empty vector is 0,
// consistent with the empty-row rule).
func TestDotProductBatchEmptyVec(t *testing.T) {
	rows := [][]int8{genI8(10, 1), genI8(20, 2), nil, genI8(5, 3)}
	for _, vec := range [][]int8{nil, {}} {
		results := []int32{111, 222, 333, 444}
		DotProductBatch(results, rows, vec)
		for r := range results {
			if results[r] != 0 {
				t.Fatalf("empty vec (len %d): results[%d] = %d, want 0", len(vec), r, results[r])
			}
		}
	}
}

// TestDotProductBatchLengthClamp checks n = min(len(results), len(rows)): results
// beyond n stay untouched, and rows beyond n are ignored.
func TestDotProductBatchLengthClamp(t *testing.T) {
	vec := genI8(16, 1)
	rows := [][]int8{genI8(16, 2), genI8(16, 3), genI8(16, 4)}
	want := naiveDotProductBatch(rows, vec)

	results := []int32{9, 9, 9, 9, 9} // longer than rows
	DotProductBatch(results, rows, vec)
	for r := range rows {
		if results[r] != want[r] {
			t.Fatalf("row %d: got %d want %d", r, results[r], want[r])
		}
	}
	if results[3] != 9 || results[4] != 9 {
		t.Fatalf("trailing results must be untouched, got %v", results[3:])
	}

	short := make([]int32, 2) // shorter than rows
	DotProductBatch(short, rows, vec)
	for r := range short {
		if short[r] != want[r] {
			t.Fatalf("clamped row %d: got %d want %d", r, short[r], want[r])
		}
	}
}

// TestDotProductBatchAllocFree asserts the operation writes into the caller's
// results slice with no heap allocation on the kernel path.
func TestDotProductBatchAllocFree(t *testing.T) {
	vec := genI8(64, 1)
	rows := make([][]int8, 8)
	for r := range rows {
		rows[r] = genI8(64, uint32(r+2))
	}
	results := make([]int32, len(rows))
	if got := testing.AllocsPerRun(10, func() { DotProductBatch(results, rows, vec) }); got != 0 {
		t.Fatalf("DotProductBatch allocated %v times, want 0", got)
	}
}

// TestDotProductBatchTrailingEmpty exercises the trailing-row fallback of the
// kernel dispatch: two full 4-row groups followed by a nil and an empty row.
func TestDotProductBatchTrailingEmpty(t *testing.T) {
	vec := genI8(24, 5)
	rows := make([][]int8, 10)
	for r := range 8 {
		rows[r] = genI8(24, uint32(r+10))
	}
	rows[8] = nil
	rows[9] = []int8{}
	results := make([]int32, len(rows))
	DotProductBatch(results, rows, vec)
	want := naiveDotProductBatch(rows, vec)
	for r := range results {
		if results[r] != want[r] {
			t.Fatalf("row %d: got %d want %d", r, results[r], want[r])
		}
	}
	if results[8] != 0 || results[9] != 0 {
		t.Fatalf("trailing empty rows must score 0, got %d %d", results[8], results[9])
	}
}

// TestDotProductBatchWraparound forces int32 two's-complement wraparound
// (127*127 = 16129 per element, so ~133k elements overflow int32) and checks the
// kernel wraps identically to the scalar oracle. A saturating kernel would
// diverge here; the deterministic sweeps above stay well below 2^31.
func TestDotProductBatchWraparound(t *testing.T) {
	const n = 200000
	vec := make([]int8, n)
	for i := range vec {
		vec[i] = 127
	}
	rows := [][]int8{vec, vec, vec, vec, vec} // 5 rows: one kernel group + trailing
	results := make([]int32, len(rows))
	DotProductBatch(results, rows, vec)
	want := naiveDotProductBatch(rows, vec)
	for r := range results {
		if results[r] != want[r] {
			t.Fatalf("row %d: got %d want %d (wraparound)", r, results[r], want[r])
		}
	}
	if want[0] >= 0 {
		t.Fatalf("test setup error: expected the overflowed sum to be negative, got %d", want[0])
	}
}
