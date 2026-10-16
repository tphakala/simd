package f32

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

// biquadNFunc is the shape shared by the public BiquadCascadeN and the
// per-architecture direct adapters, so one suite drives every path.
type biquadNFunc func(dst, src [][]float32, coeffs []float32, state [][]float32)

// biquadNSentinel fills the part of a dst slice that must stay untouched.
const biquadNSentinel = float32(123.5)

// biquadNInput is one multichannel call's inputs. Every channel has its own
// signal and its own initial state, so a swap of two channels is visible.
type biquadNInput struct {
	src    [][]float32
	dstLen []int
	st0    [][]float32
}

func biquadNFilled(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = biquadNSentinel
	}
	return s
}

// biquadNMake draws per-channel signals of the given lengths and per-channel
// states, random or all zero.
func biquadNMake(r *rand.Rand, sections int, gen func(int, *rand.Rand) []float32, srcLens, dstLens []int, zeroState bool) biquadNInput {
	in := biquadNInput{dstLen: dstLens}
	for _, n := range srcLens {
		in.src = append(in.src, gen(n, r))
		if zeroState {
			in.st0 = append(in.st0, make([]float32, biquadStateLen*sections))
		} else {
			in.st0 = append(in.st0, biquadRandState(sections, r))
		}
	}
	return in
}

// biquadNInPlace chooses which channels run in place (dst[c] the same slice as
// src[c]).
var biquadNInPlace = []struct {
	name string
	is   func(c int) bool
}{
	{"out-of-place", func(int) bool { return false }},
	{"in-place", func(int) bool { return true }},
	{"mixed", func(c int) bool { return c%2 == 0 }},
}

// biquadNCheck runs fn on in and compares every channel's dst and state with the
// per-channel oracle. The whole of each dst is compared, so a write past the
// processed prefix shows up as a sentinel mismatch. An in-place channel needs
// dstLen equal to its src length.
func biquadNCheck(t *testing.T, fn biquadNFunc, tag string, coeffs []float32, in biquadNInput, inPlace func(c int) bool) {
	t.Helper()
	chans := len(in.src)
	dst := make([][]float32, chans)
	src := make([][]float32, chans)
	state := make([][]float32, chans)
	want := make([][]float32, chans)
	wantSt := make([][]float32, chans)
	for c := range chans {
		if inPlace(c) {
			dst[c] = slices.Clone(in.src[c])
			src[c] = dst[c]
		} else {
			dst[c] = biquadNFilled(in.dstLen[c])
			src[c] = slices.Clone(in.src[c])
		}
		state[c] = slices.Clone(in.st0[c])
		want[c] = biquadNFilled(len(dst[c]))
		wantSt[c] = slices.Clone(in.st0[c])
		biquadOracle(want[c], in.src[c], coeffs, wantSt[c])
	}
	fn(dst, src, coeffs, state)
	for c := range chans {
		biquadCompare(t, fmt.Sprintf("%s ch=%d", tag, c), want[c], dst[c], wantSt[c], state[c])
	}
}

// biquadNLengthSets returns per-channel length vectors: equal lengths across the
// channels, independent random lengths, and pairs whose lengths differ by 1 and
// by 2 so the longer channel has a 1 to 2 sample tail after the pair kernel.
func biquadNLengthSets(chans int, r *rand.Rand) [][]int {
	equal := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 63, 64, 65, 255, 256, 257, 1000}
	const raggedSets, deltaSets = 4, 4
	sets := make([][]int, 0, len(equal)+raggedSets+deltaSets)
	for _, n := range equal {
		sets = append(sets, slices.Repeat([]int{n}, chans))
	}
	const maxRagged = 300
	for range raggedSets {
		l := make([]int, chans)
		for c := range l {
			l[c] = r.IntN(maxRagged + 1)
		}
		sets = append(sets, l)
	}
	for _, delta := range []int{1, 2, -1, -2} {
		base := 3 + r.IntN(200)
		l := make([]int, chans)
		for c := range l {
			l[c] = base
			if c%2 == 1 {
				l[c] = base + delta
			}
		}
		sets = append(sets, l)
	}
	return sets
}

// biquadNCheckExact runs fn against the per-channel oracle for several channel
// counts, section counts, signals, states and lengths, out of place, in place
// and mixed.
func biquadNCheckExact(t *testing.T, fn biquadNFunc) {
	t.Helper()
	r := rand.New(rand.NewPCG(21, 22))
	for _, chans := range []int{1, 2, 3, 5, 6} {
		for _, sections := range []int{1, 2, 3, 4, 8, 9} {
			coeffs := biquadRBJ(sections, r)
			for _, sig := range biquadSignals {
				for _, lens := range biquadNLengthSets(chans, r) {
					for _, zeroState := range []bool{true, false} {
						in := biquadNMake(r, sections, sig.gen, lens, lens, zeroState)
						for _, mode := range biquadNInPlace {
							tag := fmt.Sprintf("chans=%d N=%d %s lens=%v zeroState=%v %s",
								chans, sections, sig.name, lens, zeroState, mode.name)
							biquadNCheck(t, fn, tag, coeffs, in, mode.is)
						}
					}
				}
			}
		}
	}
}

// biquadNCheckFillDrainEdges runs every pair of lengths in [0, 10] on two
// channels, so the pair kernel meets its shortest legal inputs (one sample for
// one section, two for two), odd and even remainders, and an empty channel next
// to a long one. The extra-length variants make len(dst) and len(src) differ.
func biquadNCheckFillDrainEdges(t *testing.T, fn biquadNFunc) {
	t.Helper()
	r := rand.New(rand.NewPCG(23, 24))
	const maxLen = 10
	for _, sections := range []int{1, 2, 3} {
		coeffs := biquadRBJ(sections, r)
		for n0 := 0; n0 <= maxLen; n0++ {
			for n1 := 0; n1 <= maxLen; n1++ {
				for extra := range 3 {
					for _, longSrc := range []bool{false, true} {
						srcLens := []int{n0, n1}
						dstLens := []int{n0 + extra, n1 + extra}
						if longSrc {
							srcLens, dstLens = dstLens, srcLens
						}
						in := biquadNMake(r, sections, biquadNoise, srcLens, dstLens, false)
						tag := fmt.Sprintf("edges N=%d n0=%d n1=%d extra=%d longSrc=%v", sections, n0, n1, extra, longSrc)
						biquadNCheck(t, fn, tag, coeffs, in, func(int) bool { return false })
					}
				}
			}
		}
	}
}

// biquadNCheckLengthMismatch pins min(len(dst[c]), len(src[c])) per channel: the
// rest of dst[c] is untouched and state[c] advances by exactly that many samples.
func biquadNCheckLengthMismatch(t *testing.T, fn biquadNFunc) {
	t.Helper()
	r := rand.New(rand.NewPCG(25, 26))
	lens := [][2]int{{0, 40}, {40, 0}, {40, 100}, {100, 40}, {17, 300}, {300, 17}, {1, 90}, {90, 1}}
	for _, sections := range []int{1, 2, 3, 8} {
		coeffs := biquadRBJ(sections, r)
		for _, chans := range []int{2, 3} {
			for shift := range lens {
				var srcLens, dstLens []int
				for c := range chans {
					l := lens[(shift+3*c)%len(lens)]
					dstLens = append(dstLens, l[0])
					srcLens = append(srcLens, l[1])
				}
				in := biquadNMake(r, sections, biquadNoise, srcLens, dstLens, false)
				tag := fmt.Sprintf("mismatch chans=%d N=%d len(dst)=%v len(src)=%v", chans, sections, dstLens, srcLens)
				biquadNCheck(t, fn, tag, coeffs, in, func(int) bool { return false })
			}
		}
	}
}

// biquadNStreamCheck feeds one stream per channel through fn in chunks. Every
// call gives each channel its own random chunk length, so a pair constantly has
// ragged lengths and a tail. With subsets, each call passes a random subset of
// the channels in random order, so a channel meets different partners, becomes
// the odd leftover, and leaves and rejoins; with monoMix, some calls run mono
// BiquadCascade per channel instead, so a stream switches between the two forms
// at chunk boundaries. Each channel's concatenated output and final state must
// equal one oracle run over its whole stream.
func biquadNStreamCheck(t *testing.T, fn biquadNFunc, tag string, r *rand.Rand, coeffs []float32, src, st0 [][]float32, subsets, monoMix bool) {
	t.Helper()
	chans := len(src)
	sections := len(coeffs) / biquadCoeffs
	want := make([][]float32, chans)
	wantSt := make([][]float32, chans)
	got := make([][]float32, chans)
	gotSt := make([][]float32, chans)
	pos := make([]int, chans)
	for c := range chans {
		want[c] = make([]float32, len(src[c]))
		wantSt[c] = slices.Clone(st0[c])
		biquadOracle(want[c], src[c], coeffs, wantSt[c])
		got[c] = make([]float32, len(src[c]))
		gotSt[c] = slices.Clone(st0[c])
	}
	remaining := func() bool {
		for c := range chans {
			if pos[c] < len(src[c]) {
				return true
			}
		}
		return false
	}
	const maxCalls = 100000
	calls := 0
	for ; remaining() && calls < maxCalls; calls++ {
		order := r.Perm(chans)
		if subsets {
			order = order[:1+r.IntN(chans)]
		}
		d := make([][]float32, 0, len(order))
		s := make([][]float32, 0, len(order))
		st := make([][]float32, 0, len(order))
		for _, c := range order {
			n := min(biquadChunkLen(r, sections), len(src[c])-pos[c])
			d = append(d, got[c][pos[c]:pos[c]+n])
			s = append(s, src[c][pos[c]:pos[c]+n])
			st = append(st, gotSt[c])
			pos[c] += n
		}
		if monoMix && r.IntN(4) == 0 {
			for i := range d {
				BiquadCascade(d[i], s[i], coeffs, st[i])
			}
			continue
		}
		fn(d, s, coeffs, st)
	}
	if remaining() {
		t.Fatalf("%s: stream not consumed after %d calls", tag, maxCalls)
	}
	for c := range chans {
		biquadCompare(t, fmt.Sprintf("%s ch=%d calls=%d", tag, c, calls), want[c], got[c], wantSt[c], gotSt[c])
	}
}

func biquadNCheckChunked(t *testing.T, fn biquadNFunc) {
	t.Helper()
	r := rand.New(rand.NewPCG(27, 28))
	for _, chans := range []int{2, 3, 4} {
		for _, sections := range []int{1, 2, 3} {
			coeffs := biquadRBJ(sections, r)
			for trial := range 12 {
				lens := make([]int, chans)
				for c := range lens {
					lens[c] = r.IntN(900)
				}
				sig := biquadSignals[trial%len(biquadSignals)]
				in := biquadNMake(r, sections, sig.gen, lens, lens, false)
				tag := fmt.Sprintf("chunked chans=%d N=%d trial=%d lens=%v", chans, sections, trial, lens)
				biquadNStreamCheck(t, fn, tag, r, coeffs, in.src, in.st0, false, false)
			}
		}
	}
}

// biquadNCheckChannelSetChanges is the transition test for channel independence
// and for state compatibility with mono: the channel set, the order and the form
// (N or mono) change from call to call while every stream stays exact.
func biquadNCheckChannelSetChanges(t *testing.T, fn biquadNFunc) {
	t.Helper()
	r := rand.New(rand.NewPCG(29, 30))
	for _, chans := range []int{3, 5, 6} {
		for _, sections := range []int{1, 2, 3} {
			coeffs := biquadRBJ(sections, r)
			for trial := range 12 {
				lens := make([]int, chans)
				for c := range lens {
					lens[c] = r.IntN(600)
				}
				sig := biquadSignals[trial%len(biquadSignals)]
				in := biquadNMake(r, sections, sig.gen, lens, lens, false)
				tag := fmt.Sprintf("set-changes chans=%d N=%d trial=%d lens=%v", chans, sections, trial, lens)
				biquadNStreamCheck(t, fn, tag, r, coeffs, in.src, in.st0, true, true)
			}
		}
	}
}

// biquadNCheckCrossTalk runs one live channel next to silent channels with zero
// state. A silent channel must come out as exact zeros with a zero state, so any
// feed of the live channel's samples or section outputs into a neighbour, a swap
// of the two output streams, or a wrong lane in the state transfer shows up
// directly. The silent dst starts at a sentinel, so zeros prove it was written.
func biquadNCheckCrossTalk(t *testing.T, fn biquadNFunc) {
	t.Helper()
	r := rand.New(rand.NewPCG(31, 32))
	for _, chans := range []int{2, 3, 4} {
		for _, sections := range []int{1, 2} {
			coeffs := biquadRBJ(sections, r)
			for _, n := range []int{1, 2, 3, 4, 5, 64, 65, 301} {
				for live := range chans {
					lens := slices.Repeat([]int{n}, chans)
					in := biquadNMake(r, sections, biquadNoise, lens, lens, true)
					for c := range chans {
						if c == live {
							in.st0[c] = biquadRandState(sections, r)
						} else {
							clear(in.src[c])
						}
					}
					tag := fmt.Sprintf("crosstalk chans=%d N=%d n=%d live=%d", chans, sections, n, live)
					biquadNCheck(t, fn, tag, coeffs, in, func(int) bool { return false })
					biquadNCheckSilent(t, fn, tag, coeffs, in, live)
				}
			}
		}
	}
}

// biquadNCheckSilent runs fn on in and requires every channel but live to leave
// +0 bits in its dst and state, and the live channel to have produced a nonzero
// output (the premise that makes the zeros informative).
func biquadNCheckSilent(t *testing.T, fn biquadNFunc, tag string, coeffs []float32, in biquadNInput, live int) {
	t.Helper()
	chans := len(in.src)
	dst := make([][]float32, chans)
	state := make([][]float32, chans)
	for c := range chans {
		dst[c] = biquadNFilled(len(in.src[c]))
		state[c] = slices.Clone(in.st0[c])
	}
	fn(dst, in.src, coeffs, state)
	if !slices.ContainsFunc(dst[live], func(v float32) bool { return v != 0 }) {
		t.Fatalf("%s: the live channel's output is all zero (test premise)", tag)
	}
	notPlusZero := func(v float32) bool { return math.Float32bits(v) != 0 }
	for c := range chans {
		if c == live {
			continue
		}
		if slices.ContainsFunc(dst[c], notPlusZero) {
			t.Fatalf("%s: silent ch=%d dst = %v, want all +0", tag, c, dst[c])
		}
		if slices.ContainsFunc(state[c], notPlusZero) {
			t.Fatalf("%s: silent ch=%d state = %v, want all +0", tag, c, state[c])
		}
	}
}

// biquadNCheckDistinctChannels gives every channel the same src slice (allowed
// overlap) but a different state. The premise is that the oracle outputs differ,
// so a kernel that loaded one channel's state into both cannot pass.
func biquadNCheckDistinctChannels(t *testing.T, fn biquadNFunc) {
	t.Helper()
	r := rand.New(rand.NewPCG(33, 34))
	for _, chans := range []int{2, 3, 4} {
		for _, sections := range []int{1, 2, 3} {
			coeffs := biquadRBJ(sections, r)
			for _, n := range []int{1, 2, 5, 40, 301} {
				shared := biquadNoise(n, r)
				in := biquadNInput{dstLen: slices.Repeat([]int{n}, chans)}
				for range chans {
					in.src = append(in.src, shared)
					in.st0 = append(in.st0, biquadRandState(sections, r))
				}
				var outs [][]float32
				for c := range chans {
					o := make([]float32, n)
					biquadOracle(o, shared, coeffs, slices.Clone(in.st0[c]))
					outs = append(outs, o)
				}
				for c := 1; c < chans; c++ {
					if slices.EqualFunc(outs[0], outs[c], bitsEqF32) {
						t.Fatalf("chans=%d N=%d n=%d: oracle outputs of channels 0 and %d are equal (test premise)", chans, sections, n, c)
					}
				}
				// Call fn directly: every channel reads the very same src slice.
				dst := make([][]float32, chans)
				state := make([][]float32, chans)
				for c := range chans {
					dst[c] = make([]float32, n)
					state[c] = slices.Clone(in.st0[c])
				}
				fn(dst, in.src, coeffs, state)
				for c := range chans {
					wantSt := slices.Clone(in.st0[c])
					want := make([]float32, n)
					biquadOracle(want, shared, coeffs, wantSt)
					tag := fmt.Sprintf("distinct chans=%d N=%d n=%d ch=%d", chans, sections, n, c)
					biquadCompare(t, tag, want, dst[c], wantSt, state[c])
				}
			}
		}
	}
}

// biquadNCheckUnaligned places every channel's src and dst at different offsets
// into larger buffers, so no vector access can rely on alignment, and checks that
// the guard values around each dst stay untouched.
func biquadNCheckUnaligned(t *testing.T, fn biquadNFunc) {
	t.Helper()
	r := rand.New(rand.NewPCG(35, 36))
	const (
		n      = 301
		offs   = 4
		bufLen = n + 2*offs
	)
	for _, chans := range []int{2, 3} {
		for _, sections := range []int{1, 2, 3, 9} {
			coeffs := biquadRBJ(sections, r)
			for base := range offs {
				srcBuf := make([][]float32, chans)
				dstBuf := make([][]float32, chans)
				src := make([][]float32, chans)
				dst := make([][]float32, chans)
				state := make([][]float32, chans)
				want := make([][]float32, chans)
				wantSt := make([][]float32, chans)
				for c := range chans {
					nc := n - c
					srcOff, dstOff := (base+c)%offs, (base+2*c+1)%offs
					srcBuf[c] = biquadNoise(bufLen, r)
					src[c] = srcBuf[c][srcOff : srcOff+nc]
					dstBuf[c] = biquadNFilled(bufLen)
					dst[c] = dstBuf[c][dstOff : dstOff+nc]
					state[c] = biquadRandState(sections, r)
					wantSt[c] = slices.Clone(state[c])
					want[c] = biquadNFilled(bufLen)
					biquadOracle(want[c][dstOff:dstOff+nc], src[c], coeffs, wantSt[c])
				}
				fn(dst, src, coeffs, state)
				for c := range chans {
					tag := fmt.Sprintf("unaligned chans=%d N=%d base=%d ch=%d", chans, sections, base, c)
					biquadCompare(t, tag, want[c], dstBuf[c], wantSt[c], state[c])
				}
			}
		}
	}
}

// biquadNCheckNoFlushToZero feeds each channel a different smallest-normal
// multiple through a gain of one half, so every output is a nonzero subnormal.
// A path that flushed subnormals would produce zeros.
func biquadNCheckNoFlushToZero(t *testing.T, fn biquadNFunc) {
	t.Helper()
	const n = 256
	minNormal := math.Float32frombits(0x00800000)
	for _, sections := range []int{1, 2, 8} {
		coeffs := make([]float32, biquadCoeffs*sections)
		coeffs[0] = 0.5
		for k := 1; k < sections; k++ {
			coeffs[biquadCoeffs*k] = 1
		}
		in := biquadNInput{dstLen: []int{n, n, n}}
		for c, scale := range []float32{1, 1.5, 1.25} {
			s := make([]float32, n)
			for i := range s {
				s[i] = minNormal * scale
			}
			in.src = append(in.src, s)
			in.st0 = append(in.st0, make([]float32, biquadStateLen*sections))
			want := make([]float32, n)
			biquadOracle(want, s, coeffs, make([]float32, biquadStateLen*sections))
			for i, w := range want {
				if w == 0 || math.Float32bits(w) >= 0x00800000 {
					t.Fatalf("N=%d ch=%d: oracle output [%d] = %#08x is not a nonzero subnormal (test premise)",
						sections, c, i, math.Float32bits(w))
				}
			}
		}
		biquadNCheck(t, fn, fmt.Sprintf("no-flush N=%d", sections), coeffs, in, func(int) bool { return false })
	}
}

// biquadNCheckSubnormalFixedPoint runs a K-weighting pair over a different noise
// burst on each channel followed by a long run of zeros. Each channel settles in
// its own nonzero subnormal state, which every path must reproduce.
func biquadNCheckSubnormalFixedPoint(t *testing.T, fn biquadNFunc) {
	t.Helper()
	const burst, silence = 480, 48000
	r := rand.New(rand.NewPCG(37, 38))
	coeffs := biquadKWeighting48k()
	in := biquadNInput{dstLen: slices.Repeat([]int{burst + silence}, 3)}
	for range 3 {
		s := make([]float32, burst+silence)
		copy(s, biquadNoise(burst, r))
		in.src = append(in.src, s)
		in.st0 = append(in.st0, make([]float32, biquadStateLen*2))
		wantSt := make([]float32, biquadStateLen*2)
		biquadOracle(make([]float32, len(s)), s, coeffs, wantSt)
		subnormal := false
		for _, v := range wantSt {
			if v != 0 && math.Abs(float64(v)) < 0x1p-126 {
				subnormal = true
			}
		}
		if !subnormal {
			t.Fatalf("oracle final state %v holds no nonzero subnormal: the test would be vacuous", wantSt)
		}
	}
	biquadNCheck(t, fn, "subnormal fixed point", coeffs, in, func(int) bool { return false })
}

// runBiquadNSuite drives every behavioral check against one implementation.
func runBiquadNSuite(t *testing.T, fn biquadNFunc) {
	t.Helper()
	t.Run("Exact", func(t *testing.T) { biquadNCheckExact(t, fn) })
	t.Run("FillDrainEdges", func(t *testing.T) { biquadNCheckFillDrainEdges(t, fn) })
	t.Run("LengthMismatch", func(t *testing.T) { biquadNCheckLengthMismatch(t, fn) })
	t.Run("Chunked", func(t *testing.T) { biquadNCheckChunked(t, fn) })
	t.Run("ChannelSetChanges", func(t *testing.T) { biquadNCheckChannelSetChanges(t, fn) })
	t.Run("CrossTalk", func(t *testing.T) { biquadNCheckCrossTalk(t, fn) })
	t.Run("DistinctChannels", func(t *testing.T) { biquadNCheckDistinctChannels(t, fn) })
	t.Run("Unaligned", func(t *testing.T) { biquadNCheckUnaligned(t, fn) })
	t.Run("NoFlushToZero", func(t *testing.T) { biquadNCheckNoFlushToZero(t, fn) })
	t.Run("SubnormalFixedPoint", func(t *testing.T) { biquadNCheckSubnormalFixedPoint(t, fn) })
}

func TestBiquadCascadeN(t *testing.T) {
	runBiquadNSuite(t, BiquadCascadeN)
}

func TestBiquadCascadeN_PanicsOnBadShape(t *testing.T) {
	mustPanic := func(t *testing.T, name string, dst, src [][]float32, coeffs []float32, state [][]float32) {
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
		BiquadCascadeN(dst, src, coeffs, state)
	}
	mk := func(chans, n int) [][]float32 {
		if chans == 0 {
			return nil
		}
		out := make([][]float32, chans)
		for c := range out {
			out[c] = make([]float32, n)
		}
		return out
	}
	for chans := range 4 {
		for _, n := range []int{20, 0} {
			// coeffs not a positive multiple of 5, whatever the channel count.
			for _, cl := range []int{0, 4, 6, 9} {
				name := fmt.Sprintf("chans=%d n=%d coeffs=%d", chans, n, cl)
				mustPanic(t, name, mk(chans, n), mk(chans, n), make([]float32, cl), mk(chans, biquadStateLen))
			}
			// state[c] not 4 per section (only reachable with a channel).
			if chans == 0 {
				continue
			}
			for _, sl := range [][2]int{{5, 3}, {5, 5}, {10, 7}, {10, 9}, {10, 0}, {5, 8}} {
				name := fmt.Sprintf("chans=%d n=%d coeffs=%d state=%d", chans, n, sl[0], sl[1])
				mustPanic(t, name, mk(chans, n), mk(chans, n), make([]float32, sl[0]), mk(chans, sl[1]))
			}
		}
	}

	coeffs := make([]float32, 2*biquadCoeffs)
	good := func(chans int) [][]float32 { return mk(chans, 2*biquadStateLen) }
	mustPanic(t, "len(src) = len(dst)+1", mk(2, 20), mk(3, 20), coeffs, good(2))
	mustPanic(t, "len(src) = len(dst)-1", mk(2, 20), mk(1, 20), coeffs, good(2))
	mustPanic(t, "len(state) = len(dst)-1", mk(2, 20), mk(2, 20), coeffs, good(1))
	mustPanic(t, "len(state) = len(dst)+1", mk(2, 20), mk(2, 20), coeffs, good(3))
	mustPanic(t, "nil state with 2 channels", mk(2, 20), mk(2, 20), coeffs, nil)
	mustPanic(t, "nil src with 2 channels", mk(2, 20), nil, coeffs, good(2))

	// A bad state in the last channel only: nothing may have been written to
	// the earlier, valid channels (validate before any write).
	r := rand.New(rand.NewPCG(39, 40))
	dst := [][]float32{biquadNFilled(200), biquadNFilled(200), biquadNFilled(200)}
	src := [][]float32{biquadNoise(200, r), biquadNoise(200, r), biquadNoise(200, r)}
	state := [][]float32{biquadRandState(2, r), biquadRandState(2, r), make([]float32, 2*biquadStateLen-1)}
	st0 := slices.Clone(state[0])
	mustPanic(t, "bad state in the last channel only", dst, src, biquadRBJ(2, r), state)
	for c, d := range dst {
		if !slices.Equal(d, biquadNFilled(200)) {
			t.Fatalf("dst[%d] was written before the panic", c)
		}
	}
	if !slices.Equal(state[0], st0) {
		t.Fatalf("state[0] changed before the panic: %v, want %v", state[0], st0)
	}

	// Valid shapes never panic, with nothing to process or no channels.
	for _, sections := range []int{1, 2, 8, 9} {
		cf := make([]float32, biquadCoeffs*sections)
		BiquadCascadeN(nil, nil, cf, nil)
		BiquadCascadeN([][]float32{}, [][]float32{}, cf, [][]float32{})
		st := mk(2, biquadStateLen*sections)
		BiquadCascadeN(mk(2, 0), mk(2, 10), cf, st)
		BiquadCascadeN(mk(2, 10), mk(2, 0), cf, st)
		BiquadCascadeN([][]float32{nil, nil}, [][]float32{nil, nil}, cf, st)
	}
}

func TestBiquadCascadeN_AllocFree(t *testing.T) {
	r := rand.New(rand.NewPCG(41, 42))
	const n = 480
	for _, chans := range []int{2, 3, 6} {
		src := make([][]float32, chans)
		dst := make([][]float32, chans)
		for c := range chans {
			src[c] = biquadNoise(n, r)
			dst[c] = make([]float32, n)
		}
		for _, sections := range []int{1, 2, 12} {
			coeffs := biquadRBJ(sections, r)
			state := make([][]float32, chans)
			for c := range state {
				state[c] = make([]float32, biquadStateLen*sections)
			}
			allocs := testing.AllocsPerRun(100, func() {
				BiquadCascadeN(dst, src, coeffs, state)
			})
			if allocs != 0 {
				t.Fatalf("chans=%d N=%d: AllocsPerRun = %v, want 0", chans, sections, allocs)
			}
		}
	}
}

// FuzzBiquadCascadeN compares the public path with the oracle per channel, with
// every call giving each channel a random chunk and a random subset of the
// channels in random order, some calls running mono instead. Mode 0 draws stable
// RBJ sections and random states; mode 1 draws raw bit patterns for coefficients,
// states and inputs, so NaN, Inf, overflow and subnormals all occur.
func FuzzBiquadCascadeN(f *testing.F) {
	f.Add(uint64(1), uint16(300), uint8(2), uint8(1), uint8(0))
	f.Add(uint64(2), uint16(257), uint8(5), uint8(2), uint8(0))
	f.Add(uint64(3), uint16(120), uint8(3), uint8(3), uint8(1))
	f.Add(uint64(4), uint16(64), uint8(6), uint8(0), uint8(1))
	f.Fuzz(func(t *testing.T, seed uint64, nRaw uint16, chRaw, nSec, mode uint8) {
		const maxChans, maxSections, maxLen = 8, 6, 700
		r := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
		chans := int(chRaw) % maxChans
		sections := 1 + int(nSec)%maxSections
		maxN := int(nRaw)%maxLen + 1
		raw := func(count int) []float32 {
			s := make([]float32, count)
			for i := range s {
				s[i] = math.Float32frombits(r.Uint32())
			}
			return s
		}
		var coeffs []float32
		var src, st0 [][]float32
		if mode%2 == 0 {
			coeffs = biquadRBJ(sections, r)
		} else {
			coeffs = raw(biquadCoeffs * sections)
		}
		for range chans {
			n := r.IntN(maxN)
			if mode%2 == 0 {
				src = append(src, biquadWide(n, r))
				st0 = append(st0, biquadRandState(sections, r))
			} else {
				src = append(src, raw(n))
				st0 = append(st0, raw(biquadStateLen*sections))
			}
		}
		tag := fmt.Sprintf("fuzz chans=%d N=%d maxN=%d mode=%d", chans, sections, maxN, mode)
		biquadNStreamCheck(t, BiquadCascadeN, tag, r, coeffs, src, st0, true, true)
	})
}

func BenchmarkBiquadCascadeN(b *testing.B) {
	r := rand.New(rand.NewPCG(43, 44))
	// N=4 sections is the control: both implementations take the same
	// per-channel route, so a delta there is harness noise.
	for _, sections := range []int{1, 2, 4} {
		coeffs := biquadRBJ(sections, r)
		for _, chans := range []int{2, 6} {
			for _, n := range []int{480, 4800, 48000} {
				src := make([][]float32, chans)
				dst := make([][]float32, chans)
				state := make([][]float32, chans)
				for c := range chans {
					src[c] = biquadNoise(n, r)
					dst[c] = make([]float32, n)
					state[c] = make([]float32, biquadStateLen*sections)
				}
				variants := []struct {
					name string
					fn   func()
				}{
					{"N", func() { BiquadCascadeN(dst, src, coeffs, state) }},
					{"MonoLoop", func() {
						for c := range chans {
							BiquadCascade(dst[c], src[c], coeffs, state[c])
						}
					}},
				}
				for _, v := range variants {
					b.Run(fmt.Sprintf("sec=%d/ch=%d/n=%d/impl=%s", sections, chans, n, v.name), func(b *testing.B) {
						for b.Loop() {
							v.fn()
						}
						b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)/float64(n*chans), "ns/chsample")
					})
				}
			}
		}
	}
}
