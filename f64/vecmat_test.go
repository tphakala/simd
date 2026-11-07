package f64

import (
	"math"
	"math/rand/v2"
	"testing"
)

// vecMatPath is one directly callable VecMat implementation. minC is the
// smallest len(dst) the path accepts; below it only the public VecMat runs.
type vecMatPath struct {
	name string
	fn   func(dst, x, m []float64, stride int)
	minC int
}

// vecMatNaive is the oracle: a transcription of the scalar loops VecMat
// replaces (column outer, row inner, sum from +0.0, product rounded before the
// add). It shares no code with vecMatGo.
func vecMatNaive(x, m []float64, stride, c int) float64 {
	sum := 0.0
	for r := range x {
		sum += float64(x[r] * m[r*stride+c])
	}
	return sum
}

// vecMatRandom returns finite values over several binades with some +-0.
func vecMatRandom(rng *rand.Rand, n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		switch rng.IntN(16) {
		case 0:
			v[i] = 0
		case 1:
			v[i] = math.Copysign(0, -1)
		default:
			v[i] = (rng.Float64() - 0.5) * math.Ldexp(1, rng.IntN(41)-20)
		}
	}
	return v
}

// vecMatCase is one shape: r rows, c columns, pad extra columns per row, and an
// element offset of the matrix and dst inside their backing arrays.
type vecMatCase struct{ r, c, pad, off int }

// vecMatShapes returns the consumer shapes plus a ladder across every block
// boundary of every path.
func vecMatShapes() []vecMatCase {
	shapes := make([]vecMatCase, 0, 128)
	shapes = append(shapes,
		vecMatCase{64, 32, 0, 0}, vecMatCase{36, 32, 0, 0}, vecMatCase{36, 18, 0, 0}, vecMatCase{62, 62, 0, 0},
		vecMatCase{59, 59, 3, 0}, vecMatCase{41, 41, 3, 0}, vecMatCase{39, 39, 5, 0}, vecMatCase{44, 44, 0, 0},
		vecMatCase{36, 32, 0, 1}, vecMatCase{59, 59, 3, 1}, vecMatCase{39, 39, 5, 1},
	)
	for _, c := range []int{1, 2, 3, 4, 5, 8, 12, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 128, 129} {
		for _, r := range []int{1, 7} {
			for _, pad := range []int{0, 3} {
				shapes = append(shapes, vecMatCase{r, c, pad, 0})
			}
		}
		shapes = append(shapes, vecMatCase{7, c, 3, 1})
	}
	return shapes
}

const (
	vecMatSentinel = 12345.5
	vecMatGuard    = 8
)

func TestVecMatMatchesOracle(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	run := func(t *testing.T, name string, fn func(dst, x, m []float64, stride int), cs vecMatCase) {
		t.Helper()
		stride := cs.c + cs.pad
		mLen := (cs.r-1)*stride + cs.c
		// The matrix is the exact-length window of a longer backing array; padding
		// columns and everything past the window hold NaN, so any use of them shows.
		backing := make([]float64, cs.off+mLen+vecMatGuard)
		for i := range backing {
			backing[i] = math.NaN()
		}
		m := backing[cs.off : cs.off+mLen : cs.off+mLen]
		vals := vecMatRandom(rng, mLen)
		for r := range cs.r {
			copy(m[r*stride:r*stride+cs.c], vals[r*stride:r*stride+cs.c])
		}
		x := vecMatRandom(rng, cs.r)
		dbuf := make([]float64, cs.off+cs.c+vecMatGuard)
		for i := range dbuf {
			dbuf[i] = vecMatSentinel
		}
		dst := dbuf[cs.off : cs.off+cs.c : cs.off+cs.c]
		for i := range dst {
			dst[i] = math.NaN()
		}
		fn(dst, x, m, stride)
		for c := range dst {
			if want := vecMatNaive(x, m, stride, c); !bitsEqF64(dst[c], want) {
				t.Fatalf("%s %+v: dst[%d] = %x, want %x", name, cs, c, math.Float64bits(dst[c]), math.Float64bits(want))
			}
		}
		for i, v := range dbuf {
			if (i < cs.off || i >= cs.off+cs.c) && v != vecMatSentinel {
				t.Fatalf("%s %+v: store outside window at %d", name, cs, i)
			}
		}
	}
	for _, cs := range vecMatShapes() {
		run(t, "VecMat", VecMat, cs)
		for _, p := range vecMatPaths() {
			if cs.c >= p.minC {
				run(t, p.name, p.fn, cs)
			}
		}
	}

	// x may be a slice of m's backing array (both are read-only).
	mm := vecMatRandom(rng, 9*16)
	x := mm[:9]
	dst := make([]float64, 16)
	VecMat(dst, x, mm, 16)
	for c := range dst {
		if want := vecMatNaive(x, mm, 16, c); !bitsEqF64(dst[c], want) {
			t.Fatalf("x aliasing m: dst[%d] = %x, want %x", c, math.Float64bits(dst[c]), math.Float64bits(want))
		}
	}
}

// vecMatRun runs the public VecMat and every direct path on one input, calling
// check with the result.
func vecMatRun(t *testing.T, c int, x, m []float64, stride int, check func(name string, dst []float64)) {
	t.Helper()
	runs := []vecMatPath{{"VecMat", VecMat, 0}}
	for _, p := range vecMatPaths() {
		if c >= p.minC && len(x) > 0 { // direct paths require len(x) >= 1
			runs = append(runs, p)
		}
	}
	for _, r := range runs {
		dst := make([]float64, c)
		for i := range dst {
			dst[i] = math.Copysign(0, -1)
		}
		r.fn(dst, x, m, stride)
		check(r.name, dst)
	}
}

// vecMatFill returns an r x c matrix (stride c) with every row filled by col(r).
func vecMatFill(r, c int, col func(row int) float64) []float64 {
	m := make([]float64, r*c)
	for i := range r {
		for j := range c {
			m[i*c+j] = col(i)
		}
	}
	return m
}

func TestVecMatSpecialValues(t *testing.T) {
	same := func(v float64) func(int) float64 { return func(int) float64 { return v } }
	negZero := math.Copysign(0, -1)
	for _, c := range []int{1, 2, 3, 5, 17, 33, 62} {
		expect := func(t *testing.T, label string, x, m []float64, want func(col int) float64) {
			t.Helper()
			vecMatRun(t, c, x, m, c, func(name string, dst []float64) {
				for j := range dst {
					if w := want(j); !bitsEqF64(dst[j], w) {
						t.Errorf("%s C=%d %s: dst[%d] = %x, want %x", name, c, label, j, math.Float64bits(dst[j]), math.Float64bits(w))
					}
				}
			})
		}

		// Unfused the sum is exactly +0.0; fused it would be 2^-60.
		e := math.Ldexp(1, -30)
		expect(t, "no fusion", []float64{1, 1 + e}, vecMatFill(2, c, func(r int) float64 {
			if r == 0 {
				return -(1 + math.Ldexp(1, -29))
			}
			return 1 + e
		}), same(0))

		// Ascending order absorbs the 1; reversed or pairwise order keeps it.
		vals := []float64{1e16, 1, -1e16}
		expect(t, "order", []float64{1, 1, 1}, vecMatFill(3, c, func(r int) float64 { return vals[r] }), same(0))

		// All products -0.0 still sum to +0.0 from the +0.0 seed.
		expect(t, "negative zero x", []float64{negZero, negZero}, vecMatFill(2, c, func(int) float64 { return 1 }), same(0))
		expect(t, "negative zero m", []float64{1, 1}, vecMatFill(2, c, func(int) float64 { return negZero }), same(0))
		expect(t, "empty x", nil, nil, same(0))

		// Subnormal result is exact (no flush to zero).
		tiny := math.Ldexp(1, -540)
		expect(t, "subnormal", []float64{tiny}, vecMatFill(1, c, func(int) float64 { return tiny }), same(math.Ldexp(1, -1080)))

		inf := math.Inf(1)
		expect(t, "Inf times zero", []float64{inf}, vecMatFill(1, c, func(int) float64 { return 0 }), same(math.NaN()))
		expect(t, "Inf minus Inf", []float64{inf, inf}, vecMatFill(2, c, func(r int) float64 { return float64(1 - 2*r) }), same(math.NaN()))
		expect(t, "single Inf", []float64{inf, 1}, vecMatFill(2, c, func(int) float64 { return 1 }), same(inf))
		expect(t, "NaN in x", []float64{1, math.NaN()}, vecMatFill(2, c, func(int) float64 { return 1 }), same(math.NaN()))

		// A NaN in exactly one column of m makes exactly that column NaN, at the
		// first column, the last, and the start of every block ladder's overlap.
		cols := map[int]bool{0: true, c - 1: true}
		for _, w := range []int{2, 4, 16, 32} {
			if c >= w {
				cols[c-w] = true
			}
		}
		for nc := range cols {
			m := vecMatFill(3, c, func(int) float64 { return 2 })
			m[c+nc] = math.NaN()
			expect(t, "NaN column", []float64{1, 1, 1}, m, func(j int) float64 {
				if j == nc {
					return math.NaN()
				}
				return 6
			})
		}
	}
}

func TestVecMatPanics(t *testing.T) {
	mk := func(n int) []float64 { return make([]float64, n) }
	cases := []struct {
		name      string
		dst, x, m int
		nilM      bool
		stride    int
		mustPanic bool
	}{
		{"stride below columns", 4, 2, 100, false, 3, true},
		{"matrix one short", 4, 3, 2*5 + 4 - 1, false, 5, true},
		{"single row short matrix", 4, 1, 3, false, 4, true},
		{"single row nil matrix", 4, 1, 0, true, 4, true},
		{"row offset overflow", 4, 3, 100, false, math.MaxInt / 2, true},
		{"empty dst ignores stride", 0, 3, 0, true, -1, false},
		{"empty dst empty x", 0, 0, 0, true, -1, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dstBuf := mk(tc.dst + 1)
			for i := range dstBuf {
				dstBuf[i] = vecMatSentinel
			}
			dst := dstBuf[:tc.dst]
			var m []float64
			if !tc.nilM {
				m = mk(tc.m)
			}
			x := mk(tc.x)
			panicked := func() (p bool) {
				defer func() { p = recover() != nil }()
				VecMat(dst, x, m, tc.stride)
				return
			}()
			if panicked != tc.mustPanic {
				t.Fatalf("panicked = %v, want %v", panicked, tc.mustPanic)
			}
			for i, v := range dstBuf {
				if v != vecMatSentinel {
					t.Fatalf("dst[%d] written", i)
				}
			}
		})
	}
}

func TestVecMatAllocs(t *testing.T) {
	// Every array is declared inside the measured closure: a function pointer or
	// function-value helper in the dispatch would make the caller's arrays escape.
	if n := testing.AllocsPerRun(100, func() {
		var dst, x [62]float64
		var m [62 * 62]float64
		VecMat(dst[:], x[:], m[:], 62)
	}); n != 0 {
		t.Errorf("62x62: %v allocs per run, want 0", n)
	}
	if n := testing.AllocsPerRun(100, func() {
		var dst [18]float64
		var x [36]float64
		var m [36 * 18]float64
		VecMat(dst[:], x[:], m[:], 18)
	}); n != 0 {
		t.Errorf("36x18: %v allocs per run, want 0", n)
	}
}
