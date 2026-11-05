package f32

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// composeRowsAdd is the independent oracle for MinIdxOfSumRowsAdd when len(a) > 0:
// MinIdxOfSumRows into fresh scratch, then the limit gate loop. It is not valid for
// len(a) == 0, where MinIdxOfSumRows reports v = 0 and the gate would call the row
// live. Scratch is sized to the processed row count.
func composeRowsAdd(dst []float32, idxs []int32, node, a, k []float32, base, slide int, limit, sentinel float32) {
	m := min(len(dst), len(idxs), len(node))
	vals := make([]float32, m)
	ix := make([]int32, m)
	MinIdxOfSumRows(vals, ix, a, k, base, slide)
	for r := range m {
		if vals[r] < limit {
			dst[r] = node[r] + vals[r]
			idxs[r] = ix[r]
		} else {
			dst[r] = sentinel
			idxs[r] = -1
		}
	}
}

// rowsAddInputs builds a, k and node for the given shape. k mixes quantized
// values with +Inf padding, NaN and MaxFloat32 (the "dead previous row" value);
// node mixes finite values with signed zeros and infinities.
func rowsAddInputs(rng *rand.Rand, rows, n, slide int) (a, k, node []float32, base int) {
	base, klen := windowLayout(rows, n, slide)
	quant := func() float32 { return float32(rng.Intn(33)-16) / 8 }
	a = make([]float32, n)
	for i := range a {
		a[i] = quant()
	}
	k = make([]float32, klen)
	for i := range k {
		switch rng.Intn(14) {
		case 0:
			k[i] = float32(math.Inf(1))
		case 1:
			k[i] = float32(math.NaN())
		case 2:
			k[i] = math.MaxFloat32
		default:
			k[i] = quant()
		}
	}
	node = make([]float32, rows)
	for i := range node {
		switch rng.Intn(12) {
		case 0:
			node[i] = float32(math.Copysign(0, -1))
		case 1:
			node[i] = float32(math.Inf(-1))
		default:
			node[i] = float32(rng.Intn(65)-32) / 4
		}
	}
	return a, k, node, base
}

func rowsAddEqual(gd []float32, gi []int32, wd []float32, wi []int32) (int, bool) {
	for r := range gd {
		if gi[r] != wi[r] || !bitsEqF32(gd[r], wd[r]) {
			return r, false
		}
	}
	return 0, true
}

func TestMinIdxOfSumRowsAdd_MatchesComposition(t *testing.T) {
	rowCounts := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 21, 22, 23, 31, 33, 96}
	widths := []int{1, 2, 7, 15, 31, 121}
	slides := []int{1, -1, 2}
	limits := []float32{math.MaxFloat32, float32(math.Inf(1)), 0, float32(math.NaN())}
	const sentinel = float32(99)
	rng := rand.New(rand.NewSource(3))
	for _, rows := range rowCounts {
		for _, n := range widths {
			for _, slide := range slides {
				for _, limit := range limits {
					a, k, node, base := rowsAddInputs(rng, rows, n, slide)
					gd := make([]float32, rows)
					gi := make([]int32, rows)
					MinIdxOfSumRowsAdd(gd, gi, node, a, k, base, slide, limit, sentinel)
					rd := make([]float32, rows)
					ri := make([]int32, rows)
					minIdxOfSumRowsAddGo(rd, ri, node, a, k, base, slide, limit, sentinel)
					cd := make([]float32, rows)
					ci := make([]int32, rows)
					composeRowsAdd(cd, ci, node, a, k, base, slide, limit, sentinel)
					if r, ok := rowsAddEqual(gd, gi, rd, ri); !ok {
						t.Fatalf("rows=%d n=%d slide=%d limit=%v row %d: got (%v,%d) go (%v,%d)",
							rows, n, slide, limit, r, gd[r], gi[r], rd[r], ri[r])
					}
					if r, ok := rowsAddEqual(gd, gi, cd, ci); !ok {
						t.Fatalf("rows=%d n=%d slide=%d limit=%v row %d: got (%v,%d) composition (%v,%d)",
							rows, n, slide, limit, r, gd[r], gi[r], cd[r], ci[r])
					}
				}
			}
		}
	}
}

func TestMinIdxOfSumRowsAdd_Boundaries(t *testing.T) {
	const sentinel = float32(99)
	negZero := float32(math.Copysign(0, -1))
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	justBelow3 := math.Nextafter32(3, float32(math.Inf(-1)))
	tiny := float32(1.0 / (1 << 24))
	var sum1 = float32(1)
	sum1 += tiny // node 1 plus 2^-24 ties to even: exactly 1
	var overflow = float32(math.MaxFloat32)
	overflow += math.MaxFloat32
	// a is a single -0 element, so row r's minimum is exactly k[window r].
	tests := []struct {
		name     string
		node, v  float32
		limit    float32
		want     float32
		wantLive bool
	}{
		{"v equals limit is dead", 10, 3, 3, sentinel, false},
		{"v just below limit is live", 10, justBelow3, 3, 10 + justBelow3, true},
		{"all +Inf window is dead", 10, inf, math.MaxFloat32, sentinel, false},
		{"leading NaN is dead", 10, nan, 3, sentinel, false},
		{"node plus v rounds once, tie to even", 1, tiny, 3, sum1, true},
		{"negative zero node and v", negZero, negZero, 1, negZero, true},
		{"node equals minus v gives positive zero", -2, 2, 3, 0, true},
		{"overflowing live sum is +Inf", math.MaxFloat32, math.MaxFloat32, inf, overflow, true},
		{"gate is on v, not on the sum", 10, 2, 3, 12, true},
		{"dead row with -Inf node does not leak NaN", float32(math.Inf(-1)), inf, 3, sentinel, false},
		{"minus zero is not below plus zero", 5, negZero, 0, sentinel, false},
	}
	const rows = 12 // an 8-row block then a 4-row block
	for _, slide := range []int{1, -1} {
		for _, tc := range tests {
			for lane := range rows {
				t.Run(fmt.Sprintf("%s/slide%d/row%d", tc.name, slide, lane), func(t *testing.T) {
					base, klen := windowLayout(rows, 1, slide)
					k := make([]float32, klen)
					node := make([]float32, rows)
					for r := range rows {
						k[base+r*slide] = 2 // live filler: 2 < limit when limit is 3
						node[r] = 5
					}
					k[base+lane*slide] = tc.v
					node[lane] = tc.node
					a := []float32{negZero}
					dst := make([]float32, rows)
					idxs := make([]int32, rows)
					MinIdxOfSumRowsAdd(dst, idxs, node, a, k, base, slide, tc.limit, sentinel)
					if math.Float32bits(dst[lane]) != math.Float32bits(tc.want) {
						t.Fatalf("dst: got %#x want %#x", math.Float32bits(dst[lane]), math.Float32bits(tc.want))
					}
					wantIdx := int32(-1)
					if tc.wantLive {
						wantIdx = 0
					}
					if idxs[lane] != wantIdx {
						t.Fatalf("idx: got %d want %d", idxs[lane], wantIdx)
					}
				})
			}
		}
	}
}

func TestMinIdxOfSumRowsAdd_InPlaceNode(t *testing.T) {
	const sentinel = float32(99)
	rng := rand.New(rand.NewSource(4))
	for _, slide := range []int{1, -1} {
		for _, n := range []int{1, 7, 15} {
			for m := 1; m <= 40; m++ {
				a, k, node, base := rowsAddInputs(rng, m, n, slide)
				want := make([]float32, m)
				wi := make([]int32, m)
				MinIdxOfSumRowsAdd(want, wi, node, a, k, base, slide, 1, sentinel)
				got := append([]float32(nil), node...)
				gi := make([]int32, m)
				MinIdxOfSumRowsAdd(got, gi, got, a, k, base, slide, 1, sentinel)
				if r, ok := rowsAddEqual(got, gi, want, wi); !ok {
					t.Fatalf("slide=%d n=%d m=%d row %d: in place (%v,%d) separate (%v,%d)",
						slide, n, m, r, got[r], gi[r], want[r], wi[r])
				}
			}
		}
	}
}

func TestMinIdxOfSumRowsAdd_EmptyA(t *testing.T) {
	const sentinel = float32(-5)
	const rows = 9
	for _, limit := range []float32{float32(math.Inf(1)), float32(math.NaN()), math.MaxFloat32} {
		dst := make([]float32, rows+2)
		idxs := make([]int32, rows+2)
		node := make([]float32, rows)
		for i := range dst {
			dst[i] = 7
			idxs[i] = 7
		}
		MinIdxOfSumRowsAdd(dst[:rows+2], idxs[:rows+2], node, nil, nil, 1<<40, -(1 << 40), limit, sentinel)
		for r := range rows {
			if dst[r] != sentinel || idxs[r] != -1 {
				t.Fatalf("limit=%v row %d: got (%v,%d) want (sentinel,-1)", limit, r, dst[r], idxs[r])
			}
		}
		for r := rows; r < rows+2; r++ {
			if dst[r] != 7 || idxs[r] != 7 {
				t.Fatalf("limit=%v row %d beyond m was written", limit, r)
			}
		}
	}
}

func TestMinIdxOfSumRowsAdd_ShortSlicesAndPanics(t *testing.T) {
	const sentinel = float32(99)
	const rows, n = 13, 4
	a := []float32{1, 2, 3, 4}
	for _, short := range []string{"dst", "idxs", "node"} {
		const m = 6
		lens := map[string]int{"dst": rows, "idxs": rows, "node": rows}
		lens[short] = m
		// k only fits the first m rows' windows plus slack, so a window that is
		// out of range only for a row >= m must not panic.
		base, klen := windowLayout(m, n, 1)
		k := make([]float32, klen)
		dst := make([]float32, lens["dst"])
		idxs := make([]int32, lens["idxs"])
		node := make([]float32, lens["node"])
		for i := range dst {
			dst[i] = 7
		}
		for i := range idxs {
			idxs[i] = 7
		}
		MinIdxOfSumRowsAdd(dst, idxs, node, a, k, base, 1, 100, sentinel)
		for r := range dst {
			if (r < m) != (dst[r] != 7) {
				t.Fatalf("short %s: dst[%d]=%v written=%v want %v", short, r, dst[r], dst[r] != 7, r < m)
			}
		}
		for r := range idxs {
			if (r < m) != (idxs[r] != 7) {
				t.Fatalf("short %s: idxs[%d]=%d", short, r, idxs[r])
			}
		}
	}

	// Out-of-range window: panics, and dst and idxs stay untouched.
	dst := []float32{7, 7, 7, 7}
	idxs := []int32{7, 7, 7, 7}
	node := make([]float32, 4)
	k := make([]float32, n+2) // rows 0..3 need n+3
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected a panic for an out-of-range window")
			}
		}()
		MinIdxOfSumRowsAdd(dst, idxs, node, a, k, 0, 1, 100, sentinel)
	}()
	for r := range dst {
		if dst[r] != 7 || idxs[r] != 7 {
			t.Fatalf("row %d was written before the panic", r)
		}
	}
}

// TestMinIdxOfSumRowsAdd_AllocFree declares the buffers inside the measured closure
// so a dispatch that leaks its parameters (a function variable, a missing
// //go:noescape) forces the caller to heap-allocate and fails here.
func TestMinIdxOfSumRowsAdd_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(100, func() {
		var dst [12]float32
		var idxs [12]int32
		var node [12]float32
		var a [12]float32
		var k [64]float32
		MinIdxOfSumRowsAdd(dst[:], idxs[:], node[:], a[:], k[:], 0, 1, 1, 0)
	}); n != 0 {
		t.Errorf("MinIdxOfSumRowsAdd forces %v caller allocations per run, want 0", n)
	}
}

func BenchmarkMinIdxOfSumRowsAdd(b *testing.B) {
	const rows = 96
	for _, n := range []int{15, 31, 61, 121} {
		rng := rand.New(rand.NewSource(5))
		a, k, node, base := rowsAddInputs(rng, rows, n, 1)
		dst := make([]float32, rows)
		idxs := make([]int32, rows)
		vals := make([]float32, rows)
		ix := make([]int32, rows)
		b.Run(fmt.Sprintf("Fused_n%d", n), func(b *testing.B) {
			for range b.N {
				MinIdxOfSumRowsAdd(dst, idxs, node, a, k, base, 1, 1, 99)
			}
		})
		b.Run(fmt.Sprintf("Composition_n%d", n), func(b *testing.B) {
			for range b.N {
				MinIdxOfSumRows(vals, ix, a, k, base, 1)
				for r := range vals {
					if vals[r] < 1 {
						dst[r] = node[r] + vals[r]
						idxs[r] = ix[r]
					} else {
						dst[r] = 99
						idxs[r] = -1
					}
				}
			}
		})
	}
}
