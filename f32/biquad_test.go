package f32

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

// biquadFunc is the shape shared by the public BiquadCascade, the Go fallback and
// the per-architecture direct-kernel adapters, so one suite drives every path.
type biquadFunc func(dst, src, coeffs, state []float32)

// biquadOracle is an independent oracle for BiquadCascade. It runs the Direct
// Form I recurrence sample by sample with the sections inside, in float64, and
// rounds to float32 after every product and every sum. A float32 product is
// exact in float64 and float64 holds the float32 sum with innocuous double
// rounding (53 >= 2*24+2), so the result is the float32 recurrence with each
// operation rounded once. Every intermediate goes through an explicit float32
// conversion, so no compiler can fuse a multiply into an add.
func biquadOracle(dst, src, coeffs, state []float32) {
	n := min(len(dst), len(src))
	rnd := func(v float64) float64 { return float64(float32(v)) }
	for i := range n {
		x := float64(src[i])
		for k := range len(coeffs) / biquadCoeffs {
			c := coeffs[biquadCoeffs*k : biquadCoeffs*(k+1)]
			s := state[biquadStateLen*k : biquadStateLen*(k+1)]
			acc := rnd(float64(c[0]) * x)
			acc = rnd(acc + rnd(float64(c[1])*float64(s[0])))
			acc = rnd(acc + rnd(float64(c[2])*float64(s[1])))
			acc = rnd(acc - rnd(float64(c[3])*float64(s[2])))
			y := rnd(acc - rnd(float64(c[4])*float64(s[3])))
			s[1], s[0] = s[0], float32(x)
			s[3], s[2] = s[2], float32(y)
			x = y
		}
		dst[i] = float32(x)
	}
}

// biquadSame compares two float32 values as the contract promises: equal bits,
// or both NaN (the NaN payload and sign differ between architectures).
func biquadSame(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
}

// biquadCompare fails the test on the first mismatches between a result and the
// oracle, for both the output and the final state.
func biquadCompare(t *testing.T, tag string, wantDst, gotDst, wantSt, gotSt []float32) {
	t.Helper()
	bad := 0
	report := func(what string, i int, got, want float32) {
		if bad < 4 {
			t.Errorf("%s: %s[%d] = %v (%#08x), want %v (%#08x)", tag, what, i,
				got, math.Float32bits(got), want, math.Float32bits(want))
		}
		bad++
	}
	for i := range wantDst {
		if !biquadSame(wantDst[i], gotDst[i]) {
			report("dst", i, gotDst[i], wantDst[i])
		}
	}
	for i := range wantSt {
		if !biquadSame(wantSt[i], gotSt[i]) {
			report("state", i, gotSt[i], wantSt[i])
		}
	}
	if bad > 0 {
		t.Fatalf("%s: %d mismatches", tag, bad)
	}
}

// biquadRBJ returns the float32 coefficients of a random stable cascade: RBJ
// cookbook peaking sections with random centre frequency, Q and gain at 48 kHz.
func biquadRBJ(sections int, r *rand.Rand) []float32 {
	coeffs := make([]float32, biquadCoeffs*sections)
	for k := range sections {
		f0 := 50 + r.Float64()*15000
		q := 0.5 + r.Float64()*3
		gain := r.Float64()*24 - 12
		a := math.Pow(10, gain/40)
		w := 2 * math.Pi * f0 / 48000
		alpha := math.Sin(w) / (2 * q)
		cw := math.Cos(w)
		a0 := 1 + alpha/a
		c := [biquadCoeffs]float64{
			(1 + alpha*a) / a0, -2 * cw / a0, (1 - alpha*a) / a0, -2 * cw / a0, (1 - alpha/a) / a0,
		}
		for j, v := range c {
			coeffs[biquadCoeffs*k+j] = float32(v)
		}
	}
	return coeffs
}

// biquadKWeighting48k returns the two ITU-R BS.1770 K-weighting stages (shelving
// pre-filter, then the high-pass) for 48 kHz, rounded to float32.
func biquadKWeighting48k() []float32 {
	const (
		fs     = 48000.0
		f1     = 1681.974450955533
		q1     = 0.7071752369554196
		gain1  = 3.999843853973347
		vbExp  = 0.4996667741545416
		f2     = 38.13547087602444
		q2     = 0.5003270373238773
		dbDiv  = 20
		ten    = 10
		twoVal = 2.0
	)
	k := math.Tan(math.Pi * f1 / fs)
	vh := math.Pow(ten, gain1/dbDiv)
	vb := math.Pow(vh, vbExp)
	k2 := k * k
	a0 := 1.0 + k/q1 + k2
	s1 := [biquadCoeffs]float64{
		(vh + vb*k/q1 + k2) / a0,
		twoVal * (k2 - vh) / a0,
		(vh - vb*k/q1 + k2) / a0,
		twoVal * (k2 - 1.0) / a0,
		(1.0 - k/q1 + k2) / a0,
	}
	k = math.Tan(math.Pi * f2 / fs)
	k2 = k * k
	a0 = 1.0 + k/q2 + k2
	s2 := [biquadCoeffs]float64{1, -2, 1, twoVal * (k2 - 1.0) / a0, (1.0 - k/q2 + k2) / a0}
	out := make([]float32, 0, 2*biquadCoeffs)
	for _, v := range s1 {
		out = append(out, float32(v))
	}
	for _, v := range s2 {
		out = append(out, float32(v))
	}
	return out
}

func biquadNoise(n int, r *rand.Rand) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(r.Float64()*0.2 - 0.1)
	}
	return s
}

// biquadWide draws magnitudes 10^U(-44, 2) with random sign and one exact zero in
// sixteen, so subnormal inputs and wide dynamic range both appear.
func biquadWide(n int, r *rand.Rand) []float32 {
	const exact0 = 16
	s := make([]float32, n)
	for i := range s {
		v := float32(math.Pow(10, r.Float64()*46-44))
		if r.IntN(2) == 0 {
			v = -v
		}
		if r.IntN(exact0) == 0 {
			v = 0
		}
		s[i] = v
	}
	return s
}

// biquadSilence is noise for the first tenth of the signal and exact zeros after.
func biquadSilence(n int, r *rand.Rand) []float32 {
	s := biquadNoise(n, r)
	clear(s[n/10:])
	return s
}

func biquadRandState(sections int, r *rand.Rand) []float32 {
	st := make([]float32, biquadStateLen*sections)
	for i := range st {
		st[i] = float32(r.Float64() - 0.5)
	}
	return st
}

var biquadSignals = []struct {
	name string
	gen  func(int, *rand.Rand) []float32
}{
	{"noise", biquadNoise},
	{"wide", biquadWide},
	{"silence", biquadSilence},
}

// biquadSectionCounts covers the single-section case, every count up to one pass
// of eight, a count that is not a lane multiple, two full 8-section passes and
// two full passes plus a 1-section block.
var biquadSectionCounts = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17}

// biquadLengths is every length from 0 to 100, the 256 boundary and a few long
// ones. 48000 is added for a subset of section counts by the caller.
var biquadLengths = func() []int {
	var l []int
	for n := 0; n <= 100; n++ {
		l = append(l, n)
	}
	return append(l, 255, 256, 257, 1000, 4801)
}()

const biquadLongLength = 48000

func biquadClone(s []float32) []float32 { return slices.Clone(s) }

// biquadCheckExact runs fn against the oracle for every section count, length,
// signal and initial state, out of place and in place.
func biquadCheckExact(t *testing.T, fn biquadFunc) {
	t.Helper()
	r := rand.New(rand.NewPCG(1, 2))
	for _, sections := range biquadSectionCounts {
		coeffs := biquadRBJ(sections, r)
		lengths := biquadLengths
		switch sections {
		case 1, 2, 8, 17:
			lengths = append(slices.Clone(lengths), biquadLongLength)
		}
		for _, sig := range biquadSignals {
			for _, n := range lengths {
				src := sig.gen(n, r)
				for _, zeroState := range []bool{true, false} {
					st0 := make([]float32, biquadStateLen*sections)
					if !zeroState {
						st0 = biquadRandState(sections, r)
					}
					want := make([]float32, n)
					wantSt := biquadClone(st0)
					biquadOracle(want, src, coeffs, wantSt)

					tag := fmt.Sprintf("N=%d n=%d %s zeroState=%v", sections, n, sig.name, zeroState)
					got := make([]float32, n)
					gotSt := biquadClone(st0)
					fn(got, src, coeffs, gotSt)
					biquadCompare(t, tag+" out-of-place", want, got, wantSt, gotSt)

					inPlace := biquadClone(src)
					gotSt = biquadClone(st0)
					fn(inPlace, inPlace, coeffs, gotSt)
					biquadCompare(t, tag+" in-place", want, inPlace, wantSt, gotSt)
				}
			}
		}
	}
}

// biquadChunkLen picks the next chunk length: empty, a single sample, shorter
// than the pipeline depth, around the kernel threshold, or long.
func biquadChunkLen(r *rand.Rand, sections int) int {
	const (
		mixes     = 5
		shortMax  = 80
		longChunk = 300
	)
	switch r.IntN(mixes) {
	case 0:
		return 0
	case 1:
		return 1
	case 2:
		return r.IntN(2*sections + 1)
	case 3:
		return r.IntN(shortMax)
	default:
		return r.IntN(longChunk)
	}
}

// biquadCheckChunked splits a stream at random boundaries and compares the
// concatenated output and the final state with one oracle run over the whole
// stream.
func biquadCheckChunked(t *testing.T, fn biquadFunc) {
	t.Helper()
	r := rand.New(rand.NewPCG(3, 4))
	for _, sections := range biquadSectionCounts {
		coeffs := biquadRBJ(sections, r)
		for trial := range 30 {
			n := r.IntN(900)
			var src []float32
			switch trial % 3 {
			case 0:
				src = biquadWide(n, r)
			case 1:
				src = biquadSilence(n, r)
			default:
				src = biquadNoise(n, r)
			}
			st0 := biquadRandState(sections, r)
			want := make([]float32, n)
			wantSt := biquadClone(st0)
			biquadOracle(want, src, coeffs, wantSt)

			got := make([]float32, n)
			gotSt := biquadClone(st0)
			calls := 0
			for i := 0; i < n; {
				c := min(biquadChunkLen(r, sections), n-i)
				fn(got[i:i+c], src[i:i+c], coeffs, gotSt)
				i += c
				calls++
			}
			biquadCompare(t, fmt.Sprintf("chunked N=%d trial=%d n=%d calls=%d", sections, trial, n, calls),
				want, got, wantSt, gotSt)
		}
	}
}

// biquadCheckUnaligned runs fn with dst and src at different offsets into larger
// buffers, so no vector load or store can rely on 16- or 32-byte alignment, and
// checks that the elements around dst stay untouched.
func biquadCheckUnaligned(t *testing.T, fn biquadFunc) {
	t.Helper()
	r := rand.New(rand.NewPCG(5, 6))
	const (
		n      = 301
		offs   = 4
		guard  = float32(-7.25)
		bufLen = n + 2*offs
	)
	for _, sections := range []int{1, 2, 5, 8, 9} {
		coeffs := biquadRBJ(sections, r)
		for srcOff := range offs {
			for dstOff := range offs {
				srcBuf := biquadNoise(bufLen, r)
				src := srcBuf[srcOff : srcOff+n]
				st0 := biquadRandState(sections, r)
				want := make([]float32, n)
				wantSt := biquadClone(st0)
				biquadOracle(want, src, coeffs, wantSt)

				dstBuf := make([]float32, bufLen)
				for i := range dstBuf {
					dstBuf[i] = guard
				}
				gotSt := biquadClone(st0)
				fn(dstBuf[dstOff:dstOff+n], src, coeffs, gotSt)
				tag := fmt.Sprintf("unaligned N=%d srcOff=%d dstOff=%d", sections, srcOff, dstOff)
				biquadCompare(t, tag, want, dstBuf[dstOff:dstOff+n], wantSt, gotSt)
				for i, v := range dstBuf {
					if (i < dstOff || i >= dstOff+n) && v != guard {
						t.Fatalf("%s: dstBuf[%d] outside dst was written: %v", tag, i, v)
					}
				}
			}
		}
	}
}

// biquadCheckLengthMismatch pins the package convention: min(len(dst), len(src))
// samples are processed, the rest of a longer dst is untouched, and the state
// advances by exactly that many samples.
func biquadCheckLengthMismatch(t *testing.T, fn biquadFunc) {
	t.Helper()
	r := rand.New(rand.NewPCG(7, 8))
	const sentinel = float32(123.5)
	for _, sections := range []int{1, 2, 8, 9} {
		coeffs := biquadRBJ(sections, r)
		for _, lens := range [][2]int{{0, 40}, {40, 0}, {40, 100}, {100, 40}, {17, 300}, {300, 17}, {1, 90}, {90, 1}} {
			nd, ns := lens[0], lens[1]
			n := min(nd, ns)
			src := biquadNoise(ns, r)
			st0 := biquadRandState(sections, r)
			want := make([]float32, n)
			wantSt := biquadClone(st0)
			biquadOracle(want, src[:n], coeffs, wantSt)

			dst := make([]float32, nd)
			for i := range dst {
				dst[i] = sentinel
			}
			gotSt := biquadClone(st0)
			fn(dst, src, coeffs, gotSt)
			tag := fmt.Sprintf("mismatch N=%d len(dst)=%d len(src)=%d", sections, nd, ns)
			biquadCompare(t, tag, want, dst[:n], wantSt, gotSt)
			for i := n; i < nd; i++ {
				if dst[i] != sentinel {
					t.Fatalf("%s: dst[%d] past the processed prefix was written: %v", tag, i, dst[i])
				}
			}
		}
	}
}

// biquadCheckNoFlushToZero feeds the smallest normal float32 through a gain of
// one half, so every output is a nonzero subnormal. A path that flushed
// subnormals to zero (FTZ/DAZ) would produce zeros.
func biquadCheckNoFlushToZero(t *testing.T, fn biquadFunc) {
	t.Helper()
	const n = 256
	minNormal := math.Float32frombits(0x00800000)
	for _, sections := range []int{1, 8} {
		coeffs := make([]float32, biquadCoeffs*sections)
		coeffs[0] = 0.5 // b0 only: y = x/2
		for k := 1; k < sections; k++ {
			coeffs[biquadCoeffs*k] = 1 // identity sections
		}
		src := make([]float32, n)
		for i := range src {
			src[i] = minNormal
		}
		want := make([]float32, n)
		wantSt := make([]float32, biquadStateLen*sections)
		biquadOracle(want, src, coeffs, wantSt)
		for i, w := range want {
			if w == 0 || math.Float32bits(w) >= 0x00800000 {
				t.Fatalf("N=%d: oracle output [%d] = %#08x is not a nonzero subnormal (test premise)", sections, i, math.Float32bits(w))
			}
		}
		got := make([]float32, n)
		gotSt := make([]float32, biquadStateLen*sections)
		fn(got, src, coeffs, gotSt)
		biquadCompare(t, fmt.Sprintf("no-flush N=%d", sections), want, got, wantSt, gotSt)
	}
}

// biquadCheckSubnormalFixedPoint runs a K-weighting pair over a noise burst and a
// long run of zeros. The recurrence does not decay to exact zero: it settles in a
// nonzero subnormal state, which every path has to reproduce bit for bit.
func biquadCheckSubnormalFixedPoint(t *testing.T, fn biquadFunc) {
	t.Helper()
	const burst, silence = 480, 48000
	r := rand.New(rand.NewPCG(9, 10))
	coeffs := biquadKWeighting48k()
	src := make([]float32, burst+silence)
	copy(src, biquadNoise(burst, r))
	want := make([]float32, len(src))
	wantSt := make([]float32, biquadStateLen*2)
	biquadOracle(want, src, coeffs, wantSt)

	subnormalState := false
	for _, v := range wantSt {
		if v != 0 && math.Abs(float64(v)) < 0x1p-126 {
			subnormalState = true
		}
	}
	if !subnormalState {
		t.Fatalf("oracle final state %v holds no nonzero subnormal: the test would be vacuous", wantSt)
	}
	got := make([]float32, len(src))
	gotSt := make([]float32, biquadStateLen*2)
	fn(got, src, coeffs, gotSt)
	biquadCompare(t, "subnormal fixed point", want, got, wantSt, gotSt)
}

// runBiquadSuite drives every behavioral check against one implementation.
func runBiquadSuite(t *testing.T, fn biquadFunc) {
	t.Helper()
	t.Run("Exact", func(t *testing.T) { biquadCheckExact(t, fn) })
	t.Run("Chunked", func(t *testing.T) { biquadCheckChunked(t, fn) })
	t.Run("Unaligned", func(t *testing.T) { biquadCheckUnaligned(t, fn) })
	t.Run("LengthMismatch", func(t *testing.T) { biquadCheckLengthMismatch(t, fn) })
	t.Run("NoFlushToZero", func(t *testing.T) { biquadCheckNoFlushToZero(t, fn) })
	t.Run("SubnormalFixedPoint", func(t *testing.T) { biquadCheckSubnormalFixedPoint(t, fn) })
}

func TestBiquadCascade(t *testing.T) {
	runBiquadSuite(t, BiquadCascade)
}

// TestBiquadCascadeGo runs the whole suite against the Go fallback directly, so
// it is covered when dispatch lands on a SIMD kernel and so a compiler-fused
// multiply-add in the fallback (arm64) is caught.
func TestBiquadCascadeGo(t *testing.T) {
	runBiquadSuite(t, biquadCascadeGo)
}

// TestBiquadOracleIsNotFusedRecurrence guards the premise of the suite: the
// oracle must differ from the single-expression form a compiler may fuse, on
// inputs where the fused rounding differs. Without such a case the exactness
// checks above could pass against an oracle that shares the fallback's fusion.
func TestBiquadOracleIsNotFusedRecurrence(t *testing.T) {
	// y = b0*x + b1*x1 with b0*x not representable in float32: the fused form
	// rounds once, the oracle rounds the product first.
	b0 := float32(1 + 3*0x1p-23)
	x := b0
	b1 := float32(-1)
	x1 := float32(1)
	exactProduct := float64(b0) * float64(x)
	if float64(float32(exactProduct)) == exactProduct {
		t.Fatal("test premise: the product must not be exactly representable")
	}
	coeffs := []float32{b0, b1, 0, 0, 0}
	state := []float32{x1, 0, 0, 0}
	dst := make([]float32, 1)
	biquadOracle(dst, []float32{x}, coeffs, state)
	fused := float32(math.FMA(float64(b0), float64(x), float64(b1)*float64(x1)))
	rounded := float32(float64(float32(exactProduct)) + float64(b1)*float64(x1))
	if fused == rounded {
		t.Fatal("test premise: fused and separately rounded results must differ")
	}
	if dst[0] != rounded {
		t.Fatalf("oracle = %v, want the separately rounded %v (fused would be %v)", dst[0], rounded, fused)
	}
}

func TestBiquadCascade_PanicsOnBadShape(t *testing.T) {
	mustPanic := func(t *testing.T, name string, dst, src, coeffs, state []float32) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Fatalf("%s: no panic", name)
			}
			msg, ok := r.(string)
			if !ok || !strings.HasPrefix(msg, "simd: ") {
				t.Fatalf("%s: panic value %v, want a \"simd: \" message", name, r)
			}
		}()
		BiquadCascade(dst, src, coeffs, state)
	}
	type shape struct{ coeffs, state int }
	bad := []shape{
		{0, 0}, {0, 4}, {4, 4}, {6, 4}, {9, 8}, {4, 0}, // coeffs not a positive multiple of 5
		{5, 3}, {5, 5}, {10, 7}, {10, 9}, {10, 0}, // state not 4 per section
	}
	for _, s := range bad {
		name := fmt.Sprintf("coeffs=%d state=%d", s.coeffs, s.state)
		coeffs := make([]float32, s.coeffs)
		state := make([]float32, s.state)
		mustPanic(t, name+" non-empty", make([]float32, 20), make([]float32, 20), coeffs, state)
		mustPanic(t, name+" nil", nil, nil, coeffs, state) // checked even with nothing to process
	}
	for _, sections := range []int{1, 2, 8, 9} {
		coeffs := make([]float32, biquadCoeffs*sections)
		state := make([]float32, biquadStateLen*sections)
		BiquadCascade(nil, nil, coeffs, state)
		BiquadCascade([]float32{}, make([]float32, 10), coeffs, state)
		BiquadCascade(make([]float32, 10), []float32{}, coeffs, state)
	}
}

func TestBiquadCascade_AllocFree(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 12))
	const n = 480
	src := biquadNoise(n, r)
	dst := make([]float32, n)
	for _, sections := range []int{2, 12} { // one pass, and two passes
		coeffs := biquadRBJ(sections, r)
		state := make([]float32, biquadStateLen*sections)
		allocs := testing.AllocsPerRun(100, func() {
			BiquadCascade(dst, src, coeffs, state)
		})
		if allocs != 0 {
			t.Fatalf("N=%d: AllocsPerRun = %v, want 0", sections, allocs)
		}
	}
}

// FuzzBiquadCascade compares the public path, chunked at random boundaries, with
// the oracle. Mode 0 draws stable RBJ sections and a random state; mode 1 draws
// raw bit patterns for coefficients, state and input, so NaN, Inf, overflow and
// subnormals all occur.
func FuzzBiquadCascade(f *testing.F) {
	f.Add(uint64(1), uint16(300), uint8(2), uint8(0))
	f.Add(uint64(2), uint16(257), uint8(8), uint8(0))
	f.Add(uint64(3), uint16(120), uint8(9), uint8(1))
	f.Add(uint64(4), uint16(64), uint8(1), uint8(1))
	f.Fuzz(func(t *testing.T, seed uint64, nRaw uint16, nSec, mode uint8) {
		const maxSections = 20
		r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
		n := int(nRaw) % 1200
		sections := 1 + int(nSec)%maxSections
		var coeffs, state, src []float32
		if mode%2 == 0 {
			coeffs = biquadRBJ(sections, r)
			state = biquadRandState(sections, r)
			src = biquadWide(n, r)
		} else {
			raw := func(count int) []float32 {
				s := make([]float32, count)
				for i := range s {
					s[i] = math.Float32frombits(r.Uint32())
				}
				return s
			}
			coeffs, state, src = raw(biquadCoeffs*sections), raw(biquadStateLen*sections), raw(n)
		}
		want := make([]float32, n)
		wantSt := biquadClone(state)
		biquadOracle(want, src, coeffs, wantSt)

		got := make([]float32, n)
		gotSt := biquadClone(state)
		for i := 0; i < n; {
			c := min(biquadChunkLen(r, sections), n-i)
			BiquadCascade(got[i:i+c], src[i:i+c], coeffs, gotSt)
			i += c
		}
		biquadCompare(t, fmt.Sprintf("fuzz N=%d n=%d mode=%d", sections, n, mode), want, got, wantSt, gotSt)
	})
}

// biquadConsumerFused is the loop a consumer writes by hand: one expression per
// section, with the state in a slice. A compiler may fuse its multiply-adds, so it
// is a speed baseline, not a bit-exact reference.
func biquadConsumerFused(dst, src, coeffs, state []float32) {
	for i, x := range src[:min(len(dst), len(src))] {
		for k := range len(coeffs) / biquadCoeffs {
			c := coeffs[biquadCoeffs*k : biquadCoeffs*(k+1)]
			s := state[biquadStateLen*k : biquadStateLen*(k+1)]
			y := c[0]*x + c[1]*s[0] + c[2]*s[1] - c[3]*s[2] - c[4]*s[3]
			s[1], s[0] = s[0], x
			s[3], s[2] = s[2], y
			x = y
		}
		dst[i] = x
	}
}

func BenchmarkBiquadCascade(b *testing.B) {
	r := rand.New(rand.NewPCG(13, 14))
	variants := []struct {
		name string
		fn   biquadFunc
	}{
		{"Kernel", BiquadCascade},
		{"GoExact", biquadCascadeGo},
		{"ConsumerFused", biquadConsumerFused},
	}
	for _, sections := range []int{1, 2, 8} {
		coeffs := biquadRBJ(sections, r)
		if sections == 2 {
			coeffs = biquadKWeighting48k()
		}
		for _, n := range []int{480, 4800, 48000} {
			for _, sig := range []string{"noise", "silence"} {
				src := biquadNoise(n, r)
				if sig == "silence" {
					clear(src[n/10:])
				}
				dst := make([]float32, n)
				state := make([]float32, biquadStateLen*sections)
				for _, v := range variants {
					b.Run(fmt.Sprintf("N=%d/n=%d/%s/%s", sections, n, sig, v.name), func(b *testing.B) {
						for b.Loop() {
							clear(state)
							v.fn(dst, src, coeffs, state)
						}
						b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(n), "ns/sample")
					})
				}
			}
		}
	}
}
