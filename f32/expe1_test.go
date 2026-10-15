package f32

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sync"
	"testing"
)

// Tolerances for ExpE1 on every dispatch path (Go fallback, NEON, AVX2+FMA).
//
// expE1RelTol32 is the documented bound. It is the largest relative error
// measured over every positive float32 x up to 88 whose exact E1 is a normal
// float32 (x up to 82.9069748), rounded up to two significant digits. That
// exhaustive sweep gave 4.621e-7 at x = 7.5932312 (bits 0x40f2fbc0) for both the
// NEON and the AVX2+FMA kernels, and 2.505e-7 for the Go fallback. The sampled
// sets below stay under the bound.
//
// Where the exact E1 is subnormal (x above 82.9069748) a relative error is not
// meaningful. There the result must be finite and non-negative and within
// expE1SubnormalAbsTol32 of the exact value. The sweep's largest absolute error
// in that range was 9.37e-41 for the kernels (the value they flush at
// x = 87.6831284) and 6.8e-41 for the Go fallback (the value it flushes at 88).
const (
	expE1RelTol32          = 4.7e-7
	expE1SubnormalAbsTol32 = 1e-40
)

// expE1Zero is the argument from which every tier returns exactly +0. It is
// the same threshold as expE1ZeroCutoff, restated so the test does not follow a
// change to the constant.
const expE1Zero = 88.0

// expE1WorstBits are float32 bit patterns of the largest relative error seen
// by the exhaustive sweeps; the accuracy set samples +-2048 ulp around each.
var expE1WorstBits = []uint32{0x40f2fbc0, 0x42782292}

// e1Series is -gamma - ln x - sum_{k>=1} (-x)^k / (k k!), used below x = 2.
func e1Series(x float64) float64 {
	const eulerGamma = 0.57721566490153286060651209008240243
	sum, term := 0.0, 1.0
	for k := 1; k < 200; k++ {
		term *= -x / float64(k)
		d := term / float64(k)
		sum += d
		if math.Abs(d) < 1e-18*math.Abs(sum) && k > 5 {
			break
		}
	}
	return -eulerGamma - math.Log(x) - sum
}

// e1ContinuedFraction is E1(x) = e^-x / (x+1 - 1/(x+3 - 4/(x+5 - ...))) by the
// modified Lentz method, used from x = 2.
func e1ContinuedFraction(x float64) float64 {
	const tiny = 1e-300
	b := x + 1
	c := 1 / tiny
	d := 1 / b
	h := d
	for i := 1; i < 10000; i++ {
		an := -float64(i) * float64(i)
		b += 2
		d = 1 / (an*d + b)
		c = b + an/c
		del := c * d
		h *= del
		if math.Abs(del-1) < 1e-16 {
			break
		}
	}
	return h * math.Exp(-x)
}

// e1Oracle is an independent float64 reference for E1(x), x > 0: the power
// series below 2 and the continued fraction above. It shares nothing with the
// Abramowitz and Stegun form under test; TestExpE1OracleMatchesPublished pins it.
func e1Oracle(x float64) float64 {
	if x < 2 {
		return e1Series(x)
	}
	return e1ContinuedFraction(x)
}

// TestExpE1OracleMatchesPublished pins e1Oracle against E1 at nine decimal
// arguments, evaluated to 20 digits in arbitrary precision (mpmath), so the oracle
// is not the thing under test. The 1e-13 tolerance absorbs the rounding of each
// argument to float64.
func TestExpE1OracleMatchesPublished(t *testing.T) {
	for _, c := range []struct{ x, want float64 }{
		{1e-6, 13.238295893062491244},
		{0.01, 4.0379295765381138318},
		{0.5, 0.55977359477616081175},
		{1, 0.21938393439552027368},
		{2, 0.048900510708061119567},
		{10, 4.1569689296853242774e-6},
		{31, 1.0767670386162382617e-15},
		{62.03, 1.8250385663523651574e-29},
		{80, 2.2285432586884729112e-37},
	} {
		got := e1Oracle(c.x)
		if re := math.Abs(got-c.want) / c.want; re > 1e-13 {
			t.Errorf("e1Oracle(%g) = %.17g, want %.17g (rel %g)", c.x, got, c.want, re)
		}
	}
}

// e1Check compares one ExpE1 result with the contract and returns "" when it
// holds. The classes are exact: NaN in gives NaN out, x <= 0 (-0 and -Inf
// included) gives +Inf, x >= 88 (+Inf included) gives +0 by bits. In between the
// result is checked against the oracle, relatively where the exact value is a
// normal float32 and absolutely where it is subnormal.
func e1Check(x, got float32) string {
	switch {
	case x != x:
		if got == got {
			return fmt.Sprintf("E1(NaN) = %v, want NaN", got)
		}
	case x <= 0:
		if math.Float32bits(got) != math.Float32bits(float32(math.Inf(1))) {
			return fmt.Sprintf("E1(%v) = %v [%#08x], want +Inf", x, got, math.Float32bits(got))
		}
	case x >= expE1Zero:
		if math.Float32bits(got) != 0 {
			return fmt.Sprintf("E1(%v) = %v [%#08x], want +0", x, got, math.Float32bits(got))
		}
	default:
		return e1CheckValue(x, got, e1Oracle(float64(x)))
	}
	return ""
}

// e1CheckValue is the numeric arm of e1Check against a precomputed oracle value.
func e1CheckValue(x, got float32, want float64) string {
	g := float64(got)
	if want >= fltMinNormal32 {
		if re := math.Abs(g-want) / want; !(re <= expE1RelTol32) {
			return fmt.Sprintf("E1(%v) [%#08x] = %v, want %v (rel err %g > %g)",
				x, math.Float32bits(x), got, want, re, expE1RelTol32)
		}
		return ""
	}
	if !(g >= 0) || !(math.Abs(g-want) <= expE1SubnormalAbsTol32) {
		return fmt.Sprintf("E1(%v) [%#08x] = %v, want %v within %g absolute",
			x, math.Float32bits(x), got, want, expE1SubnormalAbsTol32)
	}
	return ""
}

// expE1Sample is the accuracy input set and the oracle value of each element.
type expE1Sample struct {
	x    []float32
	want []float64
}

var (
	expE1SampleOnce  sync.Once
	expE1SampleCache expE1Sample
)

// expE1AccuracySet returns log-spaced inputs over [1e-45, 88], a dense stretch
// across the x = 1 branch point, a stride through the subnormals, and the
// neighbourhoods of the worst-error arguments. 2^18 log-spaced points, or 2^14
// under -short (precedent: sqrtSweepInputs). The oracle values are computed once.
func expE1AccuracySet() expE1Sample {
	expE1SampleOnce.Do(func() {
		nLog := 1 << 18
		if testing.Short() {
			nLog = 1 << 14
		}
		xs := make([]float32, 0, nLog+20000)
		lo, hi := math.Log(1e-45), math.Log(expE1Zero)
		for i := 0; i < nLog; i++ {
			xs = append(xs, float32(math.Exp(lo+(hi-lo)*float64(i)/float64(nLog-1))))
		}
		for b := math.Float32bits(0.99); b <= math.Float32bits(1.01); b += 7 {
			xs = append(xs, math.Float32frombits(b))
		}
		for b := uint32(1); b < 0x00800000; b += 997 {
			xs = append(xs, math.Float32frombits(b))
		}
		for _, w := range expE1WorstBits {
			for b := w - 2048; b <= w+2048; b++ {
				xs = append(xs, math.Float32frombits(b))
			}
		}
		want := make([]float64, len(xs))
		for i, x := range xs {
			want[i] = e1Oracle(float64(x))
		}
		expE1SampleCache = expE1Sample{x: xs, want: want}
	})
	return expE1SampleCache
}

// expE1Func is the signature shared by ExpE1, expE1Go and the direct kernels.
type expE1Func func(dst, src []float32)

// expE1Lengths are the slice lengths the ragged-length and in-place subtests
// cover: every length through two vector widths of the widest kernel, a few
// block boundaries, and sizes around the denoiser's bin counts.
func expE1Lengths() []int {
	ns := make([]int, 0, 80)
	for n := 0; n <= 70; n++ {
		ns = append(ns, n)
	}
	return append(ns, 255, 256, 257, 513, 1025, 1031)
}

// expE1RandomInputs returns n values: mostly in (0, 90], a few non-positive, so
// the x <= 0 mask and the zero flush are exercised in every block position.
func expE1RandomInputs(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	fillExpE1Random(r, v)
	return v
}

// fillExpE1Random overwrites v with the values expE1RandomInputs draws.
func fillExpE1Random(r *rand.Rand, v []float32) {
	for i := range v {
		v[i] = float32(r.Float64()*92 - 2)
	}
}

// runExpE1Suite runs the whole ExpE1 contract against fn.
func runExpE1Suite(t *testing.T, fn expE1Func) {
	t.Helper()
	t.Run("Accuracy", func(t *testing.T) { expE1Accuracy(t, fn) })
	t.Run("Edges", func(t *testing.T) { expE1Edges(t, fn) })
	t.Run("ElementIndependence", func(t *testing.T) { expE1ElementIndependence(t, fn) })
	t.Run("RaggedLengthsAndBounds", func(t *testing.T) { expE1Ragged(t, fn) })
	t.Run("InPlace", func(t *testing.T) { expE1InPlace(t, fn) })
}

func expE1Accuracy(t *testing.T, fn expE1Func) {
	t.Helper()
	s := expE1AccuracySet()
	dst := make([]float32, len(s.x))
	fn(dst, s.x)
	var maxRel, maxAt float64
	fails := 0
	for i, x := range s.x {
		if s.want[i] >= fltMinNormal32 {
			if re := math.Abs(float64(dst[i])-s.want[i]) / s.want[i]; re > maxRel {
				maxRel, maxAt = re, float64(x)
			}
		}
		if msg := e1CheckValue(x, dst[i], s.want[i]); msg != "" {
			fails++
			if fails <= 5 {
				t.Error(msg)
			}
		}
	}
	if fails > 5 {
		t.Errorf("%d more failures", fails-5)
	}
	t.Logf("%d points, max relative error %.3g at x = %.9g (bits %#08x)",
		len(s.x), maxRel, maxAt, math.Float32bits(float32(maxAt)))
}

func expE1Edges(t *testing.T, fn expE1Func) {
	t.Helper()
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	negZero := math.Float32frombits(0x80000000)
	specials := []float32{
		nan, inf, -inf, negZero, 0, -1, -1e-45, -88, -1e30, 1e-45, 1e-40,
		1, 87.68, 87.69, 88, 88.0001, 89, 1e30, math.MaxFloat32,
	}
	const fill = 2.0
	wantFill := e1Oracle(fill)
	srcBuf := make([]float32, 40)
	dstBuf := make([]float32, 40)
	for _, x := range specials {
		for n := 1; n <= len(srcBuf); n++ {
			src, dst := srcBuf[:n], dstBuf[:n]
			for p := range n {
				for i := range src {
					src[i] = fill
				}
				src[p] = x
				fn(dst, src)
				for i := range dst {
					msg := e1CheckValue(fill, dst[i], wantFill)
					if i == p {
						msg = e1Check(x, dst[i])
					}
					if msg != "" {
						t.Fatalf("n=%d special %v at %d, lane %d: %s", n, x, p, i, msg)
					}
				}
			}
		}
	}
	// E1(1e-45) is 102.70171: the smallest subnormal goes through the log core's
	// pre-scale.
	var one [4]float32
	fn(one[:], []float32{1e-45, 1e-45, 1e-45, 1e-45})
	if re := math.Abs(float64(one[0])-102.70171) / 102.70171; re > expE1RelTol32 {
		t.Errorf("E1(1e-45) = %v, want 102.70171 (rel err %g)", one[0], re)
	}
}

// expE1ElementIndependence pins invariant 1: a lane's result does not depend on
// the lane position, the slice offset or the neighbours, and, once the slice is
// at least one widest vector (8 lanes) long, not on the length either. Slices
// shorter than a vector take the Go path on a SIMD host, which may differ from
// the kernel in the last bit, so below 8 the reference is the same length with
// the value at position 0; from 8 on it is position 0 of a 64-element slice. A
// kernel whose overlapping final block reads or writes the wrong lanes fails here.
func expE1ElementIndependence(t *testing.T, fn expE1Func) {
	t.Helper()
	const wide = 8
	r := rand.New(rand.NewPCG(311, 1))
	probes := make([]float32, 0, 64)
	for range 60 {
		probes = append(probes, float32(r.Float64()*92-2))
	}
	probes = append(probes, 0.5, 1, 87.9, -3)
	const maxLen, maxOff = 40, 3
	backing := make([]float32, maxLen+maxOff)
	out := make([]float32, maxLen+maxOff)
	for _, x := range probes {
		base := expE1RandomInputs(r, 64)
		base[0] = x
		ref64 := make([]float32, 64)
		fn(ref64, base)
		for n := 4; n <= maxLen; n++ {
			want := ref64[0]
			if n < wide {
				short := append([]float32{x}, expE1RandomInputs(r, n-1)...)
				shortOut := make([]float32, n)
				fn(shortOut, short)
				want = shortOut[0]
			}
			for off := range 4 {
				for p := 0; p < n; p++ {
					src := backing[off : off+n]
					fillExpE1Random(r, src)
					src[p] = x
					dst := out[off : off+n]
					fn(dst, src)
					if math.Float32bits(dst[p]) != math.Float32bits(want) {
						t.Fatalf("x=%v n=%d off=%d pos=%d: got %#08x, want %#08x",
							x, n, off, p, math.Float32bits(dst[p]), math.Float32bits(want))
					}
				}
			}
		}
	}
}

// expE1Ragged pins invariant 3 and ragged-length correctness: every length and
// offset, canaries on both sides of dst untouched, every element right. Inputs
// are random, since a ramp hides an in-place or lane-mixup defect by coincidence.
func expE1Ragged(t *testing.T, fn expE1Func) {
	t.Helper()
	r := rand.New(rand.NewPCG(311, 2))
	const pad = 8
	canary := math.Float32frombits(0x7fc0dead)
	for _, n := range expE1Lengths() {
		for off := range 4 {
			srcBuf := expE1RandomInputs(r, n+off+pad)
			src := srcBuf[off : off+n]
			buf := make([]float32, n+off+2*pad)
			for i := range buf {
				buf[i] = canary
			}
			dst := buf[pad+off : pad+off+n]
			fn(dst, src)
			for i, c := range buf {
				if i >= pad+off && i < pad+off+n {
					continue
				}
				if math.Float32bits(c) != math.Float32bits(canary) {
					t.Fatalf("n=%d off=%d: wrote outside dst at buf[%d]", n, off, i)
				}
			}
			for i := range dst {
				if msg := e1Check(src[i], dst[i]); msg != "" {
					t.Fatalf("n=%d off=%d lane %d: %s", n, off, i, msg)
				}
			}
		}
	}
}

// expE1InPlace pins invariant 2: fn(a, a) equals fn(b, a) bit for bit.
func expE1InPlace(t *testing.T, fn expE1Func) {
	t.Helper()
	r := rand.New(rand.NewPCG(311, 3))
	for _, n := range expE1Lengths() {
		for off := range 4 {
			src := expE1RandomInputs(r, n+off)[off:]
			want := make([]float32, n)
			fn(want, src)
			a := append([]float32(nil), src...)
			fn(a, a)
			for i := range a {
				if math.Float32bits(a[i]) != math.Float32bits(want[i]) {
					t.Fatalf("n=%d off=%d lane %d (x=%v): in-place %#08x, separate %#08x",
						n, off, i, src[i], math.Float32bits(a[i]), math.Float32bits(want[i]))
				}
			}
		}
	}
}

func TestExpE1Dispatch(t *testing.T) { runExpE1Suite(t, ExpE1) }

func TestExpE1Go(t *testing.T) { runExpE1Suite(t, expE1Go) }

func TestExpE1ClampsLength(t *testing.T) {
	src := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	dst := make([]float32, 3)
	ExpE1(dst, src) // src longer than dst: only 3 written
	for i := range dst {
		if msg := e1Check(src[i], dst[i]); msg != "" {
			t.Errorf("short dst lane %d: %s", i, msg)
		}
	}
	long := make([]float32, 11)
	for i := range long {
		long[i] = -1
	}
	ExpE1(long, src[:9]) // dst longer than src: only 9 written
	for i := range long {
		switch {
		case i < 9:
			if msg := e1Check(src[i], long[i]); msg != "" {
				t.Errorf("short src lane %d: %s", i, msg)
			}
		case long[i] != -1:
			t.Errorf("lane %d beyond src was written: %v", i, long[i])
		}
	}
	ExpE1(nil, src)
	ExpE1(dst, nil)
	ExpE1InPlace(nil)
}

func TestExpE1InPlaceMatchesExpE1(t *testing.T) {
	r := rand.New(rand.NewPCG(311, 4))
	for _, n := range []int{1, 3, 4, 7, 8, 9, 31, 257} {
		src := expE1RandomInputs(r, n)
		want := make([]float32, n)
		ExpE1(want, src)
		got := append([]float32(nil), src...)
		ExpE1InPlace(got)
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("n=%d lane %d: ExpE1InPlace %#08x, ExpE1 %#08x",
					n, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
}

// TestExpE1TierParity compares the dispatched ExpE1 with the Go fallback on the
// accuracy set. It is vacuous on a leg where the dispatcher selects the Go path
// (both sides are then the same function); the direct-kernel tests are what
// cover the kernels there.
func TestExpE1TierParity(t *testing.T) {
	s := expE1AccuracySet()
	got := make([]float32, len(s.x))
	want := make([]float32, len(s.x))
	ExpE1(got, s.x)
	expE1Go(want, s.x)
	for i, x := range s.x {
		g, w := float64(got[i]), float64(want[i])
		switch {
		case w >= fltMinNormal32:
			if re := math.Abs(g-w) / w; !(re <= 2*expE1RelTol32) {
				t.Fatalf("E1(%v) [%#08x]: dispatched %v, Go %v (rel diff %g)", x, math.Float32bits(x), g, w, re)
			}
		default:
			if !(g >= 0) || !(math.Abs(g-w) <= expE1SubnormalAbsTol32) {
				t.Fatalf("E1(%v) [%#08x]: dispatched %v, Go %v", x, math.Float32bits(x), g, w)
			}
		}
	}
}

func TestExpE1AllocFree(t *testing.T) {
	for _, n := range []int{3, 128, 257} {
		src := make([]float32, n)
		dst := make([]float32, n)
		if a := testing.AllocsPerRun(5, func() { ExpE1(dst, src) }); a != 0 {
			t.Errorf("ExpE1/n=%d allocated %v times per run, want 0", n, a)
		}
		if a := testing.AllocsPerRun(5, func() { ExpE1InPlace(dst) }); a != 0 {
			t.Errorf("ExpE1InPlace/n=%d allocated %v times per run, want 0", n, a)
		}
	}
}

// FuzzF32ExpE1 fuzzes ExpE1 over arbitrary bit patterns (NaN, infinities,
// negatives, zeros, subnormals) against the oracle and the exact special-value
// classes.
func FuzzF32ExpE1(f *testing.F) {
	addByteLenSeeds(f)
	f.Fuzz(func(t *testing.T, raw []byte) {
		v := f32sBits(raw)
		if len(v) == 0 {
			return
		}
		got := make([]float32, len(v))
		ExpE1(got, v)
		for i := range got {
			if msg := e1Check(v[i], got[i]); msg != "" {
				t.Fatalf("lane %d of %d: %s", i, len(v), msg)
			}
		}
	})
}

var expE1BenchSizes = []int{257, 513, 1025, 4096, 16384}

// BenchmarkExpE1 compares the kernel with the per-element float64 evaluation of
// the same approximations (Scalar64, listed first so it is the benchstat base).
// Read it with `benchstat -col /op`.
func BenchmarkExpE1(b *testing.B) {
	variants := []struct {
		name string
		fn   func(dst, src []float32)
	}{
		{"Scalar64", expE1Go},
		{"ExpE1", ExpE1},
	}
	for _, n := range expE1BenchSizes {
		src := make([]float32, n)
		dst := make([]float32, n)
		for i := range src {
			src[i] = genF32Pos(i) * 10 // spread over (0.3, 80]
		}
		for _, v := range variants {
			b.Run(fmt.Sprintf("n=%d/op=%s", n, v.name), func(b *testing.B) {
				b.SetBytes(int64(n) * 4 * 2) // read src, write dst
				for b.Loop() {
					v.fn(dst, src)
				}
			})
		}
	}
}
