package f32

import (
	"math"
	"testing"
)

// These tests pin the tie-break and NaN contract documented on MinIdx and
// MaxIdx in f32.go. They characterize existing behavior (strict comparison,
// first-index-wins on ties, NaN never displaces the incumbent) so a future
// vectorized implementation cannot silently change it. See issue #153.

func TestMinIdxContract(t *testing.T) {
	nan := float32(math.NaN())

	tests := []struct {
		name string
		a    []float32
		want int
	}{
		// Tie cases: expected is the index of the FIRST occurrence.
		{"tie_repeated_at_0_and_7_of_16", []float32{0, 5, 5, 5, 5, 5, 5, 0, 5, 5, 5, 5, 5, 5, 5, 5}, 0},
		{"tie_repeated_at_3_5_11", []float32{9, 9, 9, 0, 9, 0, 9, 9, 9, 9, 9, 0, 9, 9, 9, 9}, 3},
		{"tie_all_equal", []float32{4, 4, 4, 4, 4}, 0},
		{"tie_min_at_first_and_last", []float32{0, 5, 5, 5, 5, 0}, 0},
		{"tie_adjacent_minima_at_4_and_5", []float32{9, 9, 9, 9, 0, 0, 9, 9}, 4},
		{"tie_len33_equal_minima_at_15_and_16", make33(15, 16), 15},
		{"tie_len33_equal_minima_at_16_and_17", make33(16, 17), 16},

		// NaN cases.
		{"nan_all_nan", []float32{nan, nan, nan, nan}, 0},
		{"nan_at_0_sticky_incumbent", []float32{nan, -100, -50, -1}, 0},
		{"nan_mid_slice_extreme_after", []float32{5, 3, nan, 1, 8}, 3},
		{"nan_mid_slice_extreme_before", []float32{1, 3, nan, 5, 8}, 0},
		{"nan_at_last_index_unaffected", []float32{5, 1, 3, 8, nan}, 1},

		// Infinity sanity.
		{"inf_minus_inf_over_finite", []float32{1, 2, float32(math.Inf(-1)), 3}, 2},
		{"inf_repeated_minus_inf_first_occurrence", []float32{float32(math.Inf(-1)), 1, float32(math.Inf(-1)), 2}, 0},

		// Edge cases.
		{"edge_empty", nil, -1},
		{"edge_single_element", []float32{5}, 0},
		{"edge_two_equal_elements", []float32{5, 5}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MinIdx(tt.a)
			if got != tt.want {
				t.Errorf("MinIdx(%v) = %v, want %v", tt.a, got, tt.want)
			}
		})
	}
}

func TestMaxIdxContract(t *testing.T) {
	nan := float32(math.NaN())

	tests := []struct {
		name string
		a    []float32
		want int
	}{
		// Tie cases: expected is the index of the FIRST occurrence.
		{"tie_repeated_at_0_and_7_of_16", []float32{9, 5, 5, 5, 5, 5, 5, 9, 5, 5, 5, 5, 5, 5, 5, 5}, 0},
		{"tie_repeated_at_3_5_11", []float32{0, 0, 0, 9, 0, 9, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0}, 3},
		{"tie_all_equal", []float32{4, 4, 4, 4, 4}, 0},
		{"tie_max_at_first_and_last", []float32{9, 5, 5, 5, 5, 9}, 0},
		{"tie_adjacent_maxima_at_4_and_5", []float32{0, 0, 0, 0, 9, 9, 0, 0}, 4},
		{"tie_len33_equal_maxima_at_15_and_16", makeMax33(15, 16), 15},
		{"tie_len33_equal_maxima_at_16_and_17", makeMax33(16, 17), 16},

		// NaN cases.
		{"nan_all_nan", []float32{nan, nan, nan, nan}, 0},
		{"nan_at_0_sticky_incumbent", []float32{nan, 100, 50, 1}, 0},
		{"nan_mid_slice_extreme_after", []float32{5, 3, nan, 8, 1}, 3},
		{"nan_mid_slice_extreme_before", []float32{8, 3, nan, 5, 1}, 0},
		{"nan_at_last_index_unaffected", []float32{5, 8, 3, 1, nan}, 1},

		// Infinity sanity.
		{"inf_plus_inf_over_finite", []float32{1, 2, float32(math.Inf(1)), 3}, 2},
		{"inf_repeated_plus_inf_first_occurrence", []float32{float32(math.Inf(1)), 1, float32(math.Inf(1)), 2}, 0},

		// Edge cases.
		{"edge_empty", nil, -1},
		{"edge_single_element", []float32{5}, 0},
		{"edge_two_equal_elements", []float32{5, 5}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MaxIdx(tt.a)
			if got != tt.want {
				t.Errorf("MaxIdx(%v) = %v, want %v", tt.a, got, tt.want)
			}
		})
	}
}

// make33 returns a 33-element slice filled with a distinct high value, with
// the minimum (0) placed at indices i and j so the pair straddles any future
// 8/16-lane vector boundary.
func make33(i, j int) []float32 {
	a := make([]float32, 33)
	for k := range a {
		a[k] = 9
	}
	a[i] = 0
	a[j] = 0
	return a
}

// makeMax33 returns a 33-element slice filled with a distinct low value,
// with the maximum (9) placed at indices i and j so the pair straddles any
// future 8/16-lane vector boundary.
func makeMax33(i, j int) []float32 {
	a := make([]float32, 33)
	for k := range a {
		a[k] = 0
	}
	a[i] = 9
	a[j] = 9
	return a
}
