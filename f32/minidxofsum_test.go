package f32

import (
	"math"
	"math/rand"
	"testing"
)

// oracleMinIdxOfSum is an independent argmin-of-sum reference, written
// differently from minIdxOfSumGo: it materializes every candidate a[i]+b[i]
// into a scratch slice first, then scans that slice with a strict less-than.
// Same contract (single-rounded add, ties to the lowest index, NaN never wins),
// different code shape, so a shared bug in the production reference does not hide
// in the oracle too.
func oracleMinIdxOfSum(a, b []float32) (int, float32) {
	n := min(len(a), len(b))
	if n == 0 {
		return -1, 0
	}
	c := make([]float32, n)
	for i := range c {
		c[i] = a[i] + b[i]
	}
	idx := 0
	for i := 1; i < n; i++ {
		if c[i] < c[idx] {
			idx = i
		}
	}
	return idx, c[idx]
}

// quant025Set returns the quantized value pool: every multiple of 0.25 in
// [-2, 2]. Drawing operands from this dense grid makes ties common so the
// first-index-wins rule is exercised hard.
func quant025Set() []float32 {
	const lo, hi = -8, 8 // -8*0.25 = -2, 8*0.25 = 2
	s := make([]float32, 0, hi-lo+1)
	for v := lo; v <= hi; v++ {
		s = append(s, float32(v)*0.25)
	}
	return s
}

// fillQuant025 fills dst with random draws from the quantized pool.
func fillQuant025(rng *rand.Rand, dst, set []float32) {
	for i := range dst {
		dst[i] = set[rng.Intn(len(set))]
	}
}

// injectSpecials32 sprinkles +Inf (padding that must never win) and NaN
// (candidates that must never displace the incumbent) into k at random
// positions, mimicking the sliding-window caller whose signal is +Inf-padded.
func injectSpecials32(rng *rand.Rand, k []float32, nan, pinf float32) {
	for i := range k {
		switch rng.Intn(6) {
		case 0:
			k[i] = pinf
		case 1:
			k[i] = nan
		}
	}
}

// windowLayout returns a base offset and a k length that keep every row's
// window in range for the given row count, window width, and slide. base is the
// smallest non-negative start (so the lowest reached offset is 0) and klen
// covers the highest reached offset plus a small pad.
func windowLayout(rows, n, slide int) (base, klen int) {
	minMul, maxMul := 0, 0
	if slide >= 0 {
		maxMul = slide * (rows - 1)
	} else {
		minMul = slide * (rows - 1)
	}
	base = -minMul
	klen = base + maxMul + n + 3
	return base, klen
}

func TestMinIdxOfSum_Contract(t *testing.T) {
	nan := float32(math.NaN())
	ninf := float32(math.Inf(-1))
	pinf := float32(math.Inf(1))

	// With a all-zeros the candidate a[i]+b[i] equals b[i] exactly (including
	// signed specials), so wantVal below is just the winning b entry.
	z8 := []float32{0, 0, 0, 0, 0, 0, 0, 0}
	z6 := []float32{0, 0, 0, 0, 0, 0}
	z5 := []float32{0, 0, 0, 0, 0}
	z4 := []float32{0, 0, 0, 0}
	z12 := make([]float32, 12)

	tests := []struct {
		name    string
		a, b    []float32
		wantIdx int
		wantVal float32
	}{
		{"tie_at_0_and_7", z8, []float32{0, 5, 5, 5, 5, 5, 5, 0}, 0, 0},
		{"tie_at_3_5_11", z12, []float32{9, 9, 9, 0, 9, 0, 9, 9, 9, 9, 9, 0}, 3, 0},
		{"all_equal", z5, []float32{4, 4, 4, 4, 4}, 0, 4},
		{"first_and_last", z6, []float32{0, 5, 5, 5, 5, 0}, 0, 0},
		{"adjacent_4_and_5", z8, []float32{9, 9, 9, 9, 0, 0, 9, 9}, 4, 0},

		{"all_nan", z4, []float32{nan, nan, nan, nan}, 0, nan},
		{"nan_first_smaller_later", z4, []float32{nan, -100, -50, -1}, 0, nan},
		{"nan_mid_skipped", z5, []float32{5, 3, nan, 1, 8}, 3, 1},

		{"neg_inf_wins", z4, []float32{1, 2, ninf, 3}, 2, ninf},
		{"pos_inf_never_wins", z4, []float32{5, pinf, 1, pinf}, 2, 1},
		// Pins the documented promise: candidates all +Inf yield (0, +Inf)
		// (ties, first wins; nothing ever compares strictly less).
		{"all_pos_inf", z4, []float32{pinf, pinf, pinf, pinf}, 0, pinf},

		{"empty_nil_nil", nil, nil, -1, 0},
		{"empty_nil_nonempty", nil, []float32{1, 2, 3}, -1, 0},

		{"single_element", []float32{3}, []float32{4}, 0, 7},

		// len(a)=4, len(b)=2 -> n=2; the a[2:] entries must be ignored.
		{"mismatched_uses_min", []float32{5, 1, 0, 0}, []float32{5, 5}, 1, 6},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotIdx, gotVal := MinIdxOfSum(tt.a, tt.b)
			if gotIdx != tt.wantIdx || !bitsEqF32(gotVal, tt.wantVal) {
				t.Errorf("MinIdxOfSum = (%d, %#x), want (%d, %#x)",
					gotIdx, math.Float32bits(gotVal), tt.wantIdx, math.Float32bits(tt.wantVal))
			}
			// The independent oracle must agree on every case too.
			oi, ov := oracleMinIdxOfSum(tt.a, tt.b)
			if oi != tt.wantIdx || !bitsEqF32(ov, tt.wantVal) {
				t.Errorf("oracle = (%d, %#x), want (%d, %#x)",
					oi, math.Float32bits(ov), tt.wantIdx, math.Float32bits(tt.wantVal))
			}
		})
	}
}

func TestMinIdxOfSum_ParityWithOracle(t *testing.T) {
	set := quant025Set()
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	rng := rand.New(rand.NewSource(0xA26310))
	lengths := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 17, 31, 32, 33, 96}

	for _, n := range lengths {
		for trial := range 40 {
			a := make([]float32, n)
			b := make([]float32, n)
			fillQuant025(rng, a, set)
			fillQuant025(rng, b, set)
			// Inject specials into both operands. a stays finite-or-special;
			// no -Inf is injected, so no Inf-minus-Inf NaN surprises arise.
			injectSpecials32(rng, a, nan, pinf)
			injectSpecials32(rng, b, nan, pinf)

			gi, gv := MinIdxOfSum(a, b)
			oi, ov := oracleMinIdxOfSum(a, b)
			if gi != oi || !bitsEqF32(gv, ov) {
				t.Fatalf("n=%d trial=%d: MinIdxOfSum=(%d,%#x) oracle=(%d,%#x)\n a=%v\n b=%v",
					n, trial, gi, math.Float32bits(gv), oi, math.Float32bits(ov), a, b)
			}
		}
	}
}

// TestMinIdxOfSumRows_MatchesPairwisePerRow pins the defining property: each row
// of MinIdxOfSumRows equals the standalone MinIdxOfSum over that row's window.
// This mirrors TestXCorr_MatchesDotProductAtEveryLag.
func TestMinIdxOfSumRows_MatchesPairwisePerRow(t *testing.T) {
	set := quant025Set()
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	rng := rand.New(rand.NewSource(0xBADC0DE))
	slides := []int{-2, -1, 0, 1, 2}

	for trial := range 300 {
		n := 1 + rng.Intn(20)
		rows := 1 + rng.Intn(12)
		slide := slides[rng.Intn(len(slides))]
		base, klen := windowLayout(rows, n, slide)

		a := make([]float32, n)
		k := make([]float32, klen)
		fillQuant025(rng, a, set)
		fillQuant025(rng, k, set)
		injectSpecials32(rng, k, nan, pinf)

		vals := make([]float32, rows)
		idxs := make([]int32, rows)
		MinIdxOfSumRows(vals, idxs, a, k, base, slide)

		for r := range rows {
			off := base + r*slide
			wi, wv := MinIdxOfSum(a, k[off:off+n])
			if int(idxs[r]) != wi || !bitsEqF32(vals[r], wv) {
				t.Fatalf("trial=%d rows=%d n=%d slide=%d row=%d: got (%d,%#x) want (%d,%#x)",
					trial, rows, n, slide, r, idxs[r], math.Float32bits(vals[r]), wi, math.Float32bits(wv))
			}
		}
	}
}

// TestMinIdxOfSumRows_ParityMatrix sweeps rows x n across the tail-boundary
// sizes for slide +1 and -1, positioning the extreme rows exactly at the ends
// of k (k[0] for slide -1, k[len(k)-n] for slide +1) with a +Inf pad tail like
// the motivating caller, and checks against both the batched reference and the
// independent per-row oracle.
func TestMinIdxOfSumRows_ParityMatrix(t *testing.T) {
	set := quant025Set()
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	rng := rand.New(rand.NewSource(0x312D0F5))
	sizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 17, 31, 32, 33, 96}
	const padTail = 5

	for _, rows := range sizes {
		for _, n := range sizes {
			for _, slide := range []int{1, -1} {
				span := rows - 1 + n // windows over slide -1 exactly cover k[0:span]
				k := make([]float32, span+padTail)
				fillQuant025(rng, k, set)
				injectSpecials32(rng, k, nan, pinf)
				for i := span; i < len(k); i++ {
					k[i] = pinf // +Inf padding tail
				}

				var base int
				if slide == -1 {
					base = rows - 1
				} else {
					base = len(k) - n - (rows - 1)
				}

				a := make([]float32, n)
				fillQuant025(rng, a, set)

				vals := make([]float32, rows)
				idxs := make([]int32, rows)
				MinIdxOfSumRows(vals, idxs, a, k, base, slide)

				refVals := make([]float32, rows)
				refIdxs := make([]int32, rows)
				minIdxOfSumRowsGo(refVals, refIdxs, a, k, base, slide)

				for r := range rows {
					if idxs[r] != refIdxs[r] || !bitsEqF32(vals[r], refVals[r]) {
						t.Fatalf("rows=%d n=%d slide=%d row=%d: got (%d,%#x) ref (%d,%#x)",
							rows, n, slide, r, idxs[r], math.Float32bits(vals[r]), refIdxs[r], math.Float32bits(refVals[r]))
					}
					off := base + r*slide
					oi, ov := oracleMinIdxOfSum(a, k[off:off+n])
					if int(idxs[r]) != oi || !bitsEqF32(vals[r], ov) {
						t.Fatalf("rows=%d n=%d slide=%d row=%d: got (%d,%#x) oracle (%d,%#x)",
							rows, n, slide, r, idxs[r], math.Float32bits(vals[r]), oi, math.Float32bits(ov))
					}
				}
			}
		}
	}
}

func TestMinIdxOfSumRows_TailUntouched(t *testing.T) {
	const (
		sentinelBits = uint32(0x7fc00001)
		sentinelIdx  = int32(-42)
	)
	sentinelVal := math.Float32frombits(sentinelBits)
	set := quant025Set()
	rng := rand.New(rand.NewSource(0x7A11))

	// check runs one clamping direction: m = min(len(vals), len(idxs)).
	check := func(t *testing.T, valsLen, idxsLen int) {
		t.Helper()
		m := min(valsLen, idxsLen)
		n, slide := 4, 1
		base, klen := windowLayout(m, n, slide)

		a := make([]float32, n)
		k := make([]float32, klen)
		fillQuant025(rng, a, set)
		fillQuant025(rng, k, set) // finite only, so results never equal the NaN sentinel

		vals := make([]float32, valsLen)
		idxs := make([]int32, idxsLen)
		for i := range vals {
			vals[i] = sentinelVal
		}
		for i := range idxs {
			idxs[i] = sentinelIdx
		}

		MinIdxOfSumRows(vals, idxs, a, k, base, slide)

		// Processed region matches the reference.
		refVals := make([]float32, m)
		refIdxs := make([]int32, m)
		minIdxOfSumRowsGo(refVals, refIdxs, a, k, base, slide)
		for r := range m {
			if idxs[r] != refIdxs[r] || !bitsEqF32(vals[r], refVals[r]) {
				t.Fatalf("row %d: got (%d,%#x) ref (%d,%#x)", r, idxs[r], math.Float32bits(vals[r]), refIdxs[r], math.Float32bits(refVals[r]))
			}
		}
		// Tails beyond m are untouched.
		for r := m; r < valsLen; r++ {
			if math.Float32bits(vals[r]) != sentinelBits {
				t.Errorf("vals[%d] = %#x, want untouched sentinel %#x", r, math.Float32bits(vals[r]), sentinelBits)
			}
		}
		for r := m; r < idxsLen; r++ {
			if idxs[r] != sentinelIdx {
				t.Errorf("idxs[%d] = %d, want untouched sentinel %d", r, idxs[r], sentinelIdx)
			}
		}
	}

	t.Run("vals_longer", func(t *testing.T) { check(t, 10, 6) })
	t.Run("idxs_longer", func(t *testing.T) { check(t, 6, 10) })
}

func TestMinIdxOfSumRows_PanicsOnOutOfRange(t *testing.T) {
	const sentinelBits = uint32(0x7fc00001)
	const sentinelIdx = int32(-42)
	sentinelVal := math.Float32frombits(sentinelBits)

	newFinite := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(i) * 0.5
		}
		return s
	}

	cases := []struct {
		name        string
		rows        int
		a, k        []float32
		base, slide int
	}{
		// slide +1, offsets 5,6,7,8; lim = 10-3 = 7, so row 3 (off 8) overruns.
		{"high_extreme_overrun", 4, newFinite(3), newFinite(10), 5, 1},
		// slide -1, offsets 2,1,0,-1; row 3 (off -1) underruns.
		{"low_extreme_underrun", 4, newFinite(3), newFinite(8), 2, -1},
		// len(a) > len(k): lim = 3-5 = -2, so row 0 (off 0) already overruns.
		{"len_a_gt_len_k", 4, newFinite(5), newFinite(3), 0, 1},
		// Adversarial slide: base 0 passes row 0, then off jumps to MaxInt and
		// the incremental check must catch it (the closed-form product would wrap).
		{"adversarial_overflow_slide", 3, newFinite(2), newFinite(4), 0, math.MaxInt},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			vals := make([]float32, c.rows)
			idxs := make([]int32, c.rows)
			for i := range vals {
				vals[i] = sentinelVal
			}
			for i := range idxs {
				idxs[i] = sentinelIdx
			}

			func() {
				defer func() {
					if recover() == nil {
						t.Fatalf("expected panic, got none")
					}
				}()
				MinIdxOfSumRows(vals, idxs, c.a, c.k, c.base, c.slide)
				t.Fatalf("MinIdxOfSumRows returned without panicking")
			}()

			// The panic must precede any write: sentinels fully intact.
			for i := range vals {
				if math.Float32bits(vals[i]) != sentinelBits {
					t.Errorf("vals[%d] = %#x written before panic, want sentinel %#x", i, math.Float32bits(vals[i]), sentinelBits)
				}
			}
			for i := range idxs {
				if idxs[i] != sentinelIdx {
					t.Errorf("idxs[%d] = %d written before panic, want sentinel %d", i, idxs[i], sentinelIdx)
				}
			}
		})
	}
}

func TestMinIdxOfSumRows_SignedZeroAndSubnormal(t *testing.T) {
	negZero := math.Float32frombits(0x80000000)
	const negZeroBits = uint32(0x80000000)

	// Incumbent -0.0 at index 0, candidate +0.0 at index 1:
	//   -0.0 + -0.0 = -0.0 (index 0)
	//    1.0 + -1.0 = +0.0 (index 1)
	// +0.0 < -0.0 is false, so the incumbent -0.0 keeps its exact bits.
	t.Run("signed_zero_incumbent_wins", func(t *testing.T) {
		a := []float32{negZero, 1.0}
		b := []float32{negZero, -1.0}

		gi, gv := MinIdxOfSum(a, b)
		if gi != 0 || math.Float32bits(gv) != negZeroBits {
			t.Errorf("MinIdxOfSum = (%d, %#x), want (0, %#x)", gi, math.Float32bits(gv), negZeroBits)
		}

		vals := make([]float32, 1)
		idxs := make([]int32, 1)
		MinIdxOfSumRows(vals, idxs, a, b, 0, 1)
		if idxs[0] != 0 || math.Float32bits(vals[0]) != negZeroBits {
			t.Errorf("MinIdxOfSumRows row 0 = (%d, %#x), want (0, %#x)", idxs[0], math.Float32bits(vals[0]), negZeroBits)
		}
	})

	// Subnormal candidates compared bit-exactly against the references.
	t.Run("subnormal", func(t *testing.T) {
		subs := []float32{
			math.SmallestNonzeroFloat32,
			2 * math.SmallestNonzeroFloat32,
			math.Float32frombits(0x00000005),
			math.Float32frombits(0x007fffff), // largest subnormal
			0,
			negZero,
		}
		rng := rand.New(rand.NewSource(0x50B))
		for trial := range 200 {
			n := 1 + rng.Intn(6)
			a := make([]float32, n)
			b := make([]float32, n)
			for i := range a {
				a[i] = subs[rng.Intn(len(subs))]
				b[i] = subs[rng.Intn(len(subs))]
			}
			gi, gv := MinIdxOfSum(a, b)
			oi, ov := oracleMinIdxOfSum(a, b)
			if gi != oi || math.Float32bits(gv) != math.Float32bits(ov) {
				t.Fatalf("trial=%d: MinIdxOfSum=(%d,%#x) oracle=(%d,%#x)\n a=%v b=%v",
					trial, gi, math.Float32bits(gv), oi, math.Float32bits(ov), a, b)
			}

			base, klen := windowLayout(1, n, 1)
			k := make([]float32, klen)
			copy(k[base:], b)
			vals := make([]float32, 1)
			idxs := make([]int32, 1)
			MinIdxOfSumRows(vals, idxs, a, k, base, 1)
			if int(idxs[0]) != oi || math.Float32bits(vals[0]) != math.Float32bits(ov) {
				t.Fatalf("trial=%d rows: got (%d,%#x) want (%d,%#x)",
					trial, idxs[0], math.Float32bits(vals[0]), oi, math.Float32bits(ov))
			}
		}
	})
}

func TestMinIdxOfSumRows_EmptyA(t *testing.T) {
	const sentinelBits = uint32(0x7fc00001)
	const sentinelIdx = int32(-42)
	sentinelVal := math.Float32frombits(sentinelBits)

	var a []float32 // len 0
	k := []float32{1, 2, 3}
	const m = 5

	vals := make([]float32, m+2)
	idxs := make([]int32, m+2)
	for i := range vals {
		vals[i] = sentinelVal
		idxs[i] = sentinelIdx
	}

	// base is wildly out of range and slide is arbitrary; because len(a) == 0 no
	// k index is ever reached, so this must fill (0, -1) and never panic.
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("unexpected panic with empty a: %v", r)
			}
		}()
		MinIdxOfSumRows(vals[:m], idxs[:m], a, k, math.MaxInt, -13)
	}()

	for r := range m {
		if vals[r] != 0 || idxs[r] != -1 {
			t.Errorf("row %d = (val %#x, idx %d), want (0, -1)", r, math.Float32bits(vals[r]), idxs[r])
		}
	}
	for r := m; r < len(vals); r++ {
		if math.Float32bits(vals[r]) != sentinelBits || idxs[r] != sentinelIdx {
			t.Errorf("tail row %d = (%#x, %d), want untouched sentinel (%#x, %d)", r, math.Float32bits(vals[r]), idxs[r], sentinelBits, sentinelIdx)
		}
	}
}

func TestMinIdxOfSumRows_SlideVariants(t *testing.T) {
	set := quant025Set()
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	rng := rand.New(rand.NewSource(0x51DE))

	for _, slide := range []int{0, 8, -3} {
		const rows, n = 6, 5
		base, klen := windowLayout(rows, n, slide)

		a := make([]float32, n)
		k := make([]float32, klen)
		fillQuant025(rng, a, set)
		fillQuant025(rng, k, set)
		injectSpecials32(rng, k, nan, pinf)

		vals := make([]float32, rows)
		idxs := make([]int32, rows)
		MinIdxOfSumRows(vals, idxs, a, k, base, slide)

		refVals := make([]float32, rows)
		refIdxs := make([]int32, rows)
		minIdxOfSumRowsGo(refVals, refIdxs, a, k, base, slide)

		for r := range rows {
			if idxs[r] != refIdxs[r] || !bitsEqF32(vals[r], refVals[r]) {
				t.Fatalf("slide=%d row=%d: got (%d,%#x) ref (%d,%#x)",
					slide, r, idxs[r], math.Float32bits(vals[r]), refIdxs[r], math.Float32bits(refVals[r]))
			}
			off := base + r*slide
			oi, ov := oracleMinIdxOfSum(a, k[off:off+n])
			if int(idxs[r]) != oi || !bitsEqF32(vals[r], ov) {
				t.Fatalf("slide=%d row=%d: got (%d,%#x) oracle (%d,%#x)",
					slide, r, idxs[r], math.Float32bits(vals[r]), oi, math.Float32bits(ov))
			}
		}
	}
}

// TestMinIdxOfSumRows_UnalignedOperands runs the parity check on offset
// sub-slices so no path may assume aligned operands.
func TestMinIdxOfSumRows_UnalignedOperands(t *testing.T) {
	set := quant025Set()
	nan := float32(math.NaN())
	pinf := float32(math.Inf(1))
	rng := rand.New(rand.NewSource(0xA11A))

	const rows, n, slide = 6, 5, 1
	base, klen := windowLayout(rows, n, slide)

	aBack := make([]float32, n+1)
	kBack := make([]float32, klen+3)
	valsBack := make([]float32, rows+1)
	idxsBack := make([]int32, rows+1)
	fillQuant025(rng, aBack, set)
	fillQuant025(rng, kBack, set)
	injectSpecials32(rng, kBack, nan, pinf)

	a := aBack[1:]
	k := kBack[3:]
	vals := valsBack[1:]
	idxs := idxsBack[1:]

	MinIdxOfSumRows(vals, idxs, a, k, base, slide)

	refVals := make([]float32, rows)
	refIdxs := make([]int32, rows)
	minIdxOfSumRowsGo(refVals, refIdxs, a, k, base, slide)

	for r := range rows {
		if idxs[r] != refIdxs[r] || !bitsEqF32(vals[r], refVals[r]) {
			t.Fatalf("row=%d: got (%d,%#x) ref (%d,%#x)",
				r, idxs[r], math.Float32bits(vals[r]), refIdxs[r], math.Float32bits(refVals[r]))
		}
		off := base + r*slide
		oi, ov := oracleMinIdxOfSum(a, k[off:off+n])
		if int(idxs[r]) != oi || !bitsEqF32(vals[r], ov) {
			t.Fatalf("row=%d: got (%d,%#x) oracle (%d,%#x)",
				r, idxs[r], math.Float32bits(vals[r]), oi, math.Float32bits(ov))
		}
	}
}

// TestMinIdxOfSum_AllocFree and TestMinIdxOfSumRows_AllocFree pin the
// zero-allocation contract from the CALLER's side. The buffers are declared
// INSIDE the measured closure deliberately: hoisting them out measures only the
// function's own allocations and passes even when it leaks its parameters,
// forcing every caller to heap-allocate. See i16/xcorr_test.go:284-291.
func TestMinIdxOfSum_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(100, func() {
		var a [17]float32
		var b [17]float32
		MinIdxOfSum(a[:], b[:])
	}); n != 0 {
		t.Errorf("MinIdxOfSum forces %v caller allocations per run, want 0", n)
	}
}

func TestMinIdxOfSumRows_AllocFree(t *testing.T) {
	if n := testing.AllocsPerRun(100, func() {
		var vals [8]float32
		var idxs [8]int32
		var a [12]float32
		var k [64]float32
		MinIdxOfSumRows(vals[:], idxs[:], a[:], k[:], 0, 1)
	}); n != 0 {
		t.Errorf("MinIdxOfSumRows forces %v caller allocations per run, want 0", n)
	}
}
