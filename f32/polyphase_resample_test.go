package f32

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// ---------------------------------------------------------------------------
// Shared helpers for the fused polyphase cubic resampler tests.
// ---------------------------------------------------------------------------

// polyDeriveStep reproduces go-audio-resampler's fixed-point step derivation:
// step = round((1/ratio) * numPhases * 2^fracBits), ratio = outRate/inRate.
func polyDeriveStep(inRate, outRate float64, numPhases, fracBits int) int64 {
	ratio := outRate / inRate
	scale := float64(int64(1) << uint(fracBits))
	return int64(math.Round((1.0 / ratio) * float64(numPhases) * scale))
}

type polyRate struct {
	name            string
	inRate, outRate float64
}

var polyRates = []polyRate{
	{"44k_to_48k_up", 44100, 48000},
	{"48k_to_44k_down", 48000, 44100},
	{"44k_to_64k_sub", 44100, 64000},
	{"96k_to_48k_2to1", 96000, 48000},
	{"16k_to_48k_3x", 16000, 48000},
}

func polyMakeBanks(numPhases, taps int, rng *rand.Rand) (a, b, c, d [][]float32) {
	mk := func() [][]float32 {
		rows := make([][]float32, numPhases)
		for p := range numPhases {
			row := make([]float32, taps)
			for t := range taps {
				row[t] = float32(rng.NormFloat64()) * 0.25
			}
			rows[p] = row
		}
		return rows
	}
	return mk(), mk(), mk(), mk()
}

func polyMakeHist(n int, rng *rand.Rand) []float32 {
	h := make([]float32, n)
	for i := range h {
		h[i] = float32(rng.NormFloat64())
	}
	return h
}

// polySizeHist returns a history length guaranteeing numOut outputs are produced
// (the window bound never fires before out is full), for at==0.
func polySizeHist(step int64, numOut, numPhases, taps, fracBits int) int {
	lastDiv := int((int64(numOut-1) * step >> uint(fracBits)) / int64(numPhases))
	return lastDiv + taps + 8
}

// polyRefLoop is the faithful per-output div/mod reference. dot selects the tier
// (CubicInterpDotUnsafe for tier-aware parity, cubicInterpDotGo for the
// arch-independent Go check). It returns the number of outputs written.
func polyRefLoop(out, hist []float32, a, b, c, d [][]float32, at, step int64, numPhases, taps, fracBits int, dot func(h, ca, cb, cc, cd []float32, x float32) float32) int {
	numPhases64 := int64(numPhases)
	fracMask := int64(1)<<uint(fracBits) - 1
	fracScale := float32(1.0 / float64(int64(1)<<uint(fracBits)))
	histLen := len(hist)
	k := 0
	for k < len(out) {
		full := at >> uint(fracBits)
		div := int(full / numPhases64)
		phase := int(full % numPhases64)
		frac := at & fracMask
		if div+taps > histLen {
			break
		}
		x := float32(frac) * fracScale
		out[k] = dot(hist[div:div+taps], a[phase][:taps], b[phase][:taps], c[phase][:taps], d[phase][:taps], x)
		k++
		at += step
	}
	return k
}

func polyBitsEqual(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length got=%d want=%d", label, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: mismatch at %d: got=%v (0x%08x) want=%v (0x%08x)",
				label, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

var polyGridPhases = []int{64, 80, 128, 256}

// polyGridTaps spans the tap counts and, within the vectorized kernels, the loop
// tiers: 16/32/64 are pure 16-wide-block multiples, 20/100 add a scalar tail, and
// 24 (16+8 on amd64 f32, 16+4+4 on amd64 f64) exercises the secondary
// 8-wide/4-wide block after the 16-wide block in the fused path directly.
var polyGridTaps = []int{16, 20, 24, 32, 64, 100}

const polyFracBits = 16

// ---------------------------------------------------------------------------
// Tier-aware parity: the public fused API vs a div/mod reference loop calling
// CubicInterpDotUnsafe. The dispatch guards are aligned, so both pick the same
// dot tier and the results are bit-identical on every CPU.
// ---------------------------------------------------------------------------

func TestPolyphaseResampleCubicParity(t *testing.T) {
	const outLen = 1024
	rng := rand.New(rand.NewSource(0xC0FFEE))
	for _, numPhases := range polyGridPhases {
		for _, taps := range polyGridTaps {
			a, b, c, d := polyMakeBanks(numPhases, taps, rng)
			for _, r := range polyRates {
				step := polyDeriveStep(r.inRate, r.outRate, numPhases, polyFracBits)
				histLen := polySizeHist(step, outLen, numPhases, taps, polyFracBits)
				hist := polyMakeHist(histLen, rng)

				gotOut := make([]float32, outLen)
				wantOut := make([]float32, outLen)
				nGot, atGot := PolyphaseResampleCubic(gotOut, hist, a, b, c, d, 0, step, numPhases, taps, polyFracBits)
				nWant := polyRefLoop(wantOut, hist, a, b, c, d, 0, step, numPhases, taps, polyFracBits, CubicInterpDotUnsafe)

				label := polyLabel(numPhases, taps, r.name)
				if nGot != nWant {
					t.Fatalf("%s: n got=%d want=%d", label, nGot, nWant)
				}
				if atGot != int64(nGot)*step {
					t.Fatalf("%s: atOut got=%d want=%d", label, atGot, int64(nGot)*step)
				}
				polyBitsEqual(t, label, gotOut[:nGot], wantOut[:nWant])

				// The Unsafe entry point must produce the identical result and atOut.
				unsafeOut := make([]float32, outLen)
				nU, atU := PolyphaseResampleCubicUnsafe(unsafeOut, hist, a, b, c, d, 0, step, numPhases, taps, polyFracBits)
				if nU != nGot || atU != atGot {
					t.Fatalf("%s: Unsafe (n=%d,at=%d) != safe (n=%d,at=%d)", label, nU, atU, nGot, atGot)
				}
				polyBitsEqual(t, label+"/unsafe", unsafeOut[:nU], gotOut[:nGot])
			}
		}
	}
}

func polyLabel(numPhases, taps int, rate string) string {
	return fmt.Sprintf("nP%d_taps%d/%s", numPhases, taps, rate)
}

// ---------------------------------------------------------------------------
// Arch-independent Go check: polyphaseResampleCubicGo vs a div/mod reference
// loop calling cubicInterpDotGo. Both use the scalar Go dot, so they are
// bit-identical at any tap count on any host, isolating the state machine from
// the asm.
// ---------------------------------------------------------------------------

func TestPolyphaseResampleCubicGoStateMachine(t *testing.T) {
	const outLen = 1024
	rng := rand.New(rand.NewSource(0x1234))
	for _, numPhases := range polyGridPhases {
		for _, taps := range polyGridTaps {
			a, b, c, d := polyMakeBanks(numPhases, taps, rng)
			for _, r := range polyRates {
				step := polyDeriveStep(r.inRate, r.outRate, numPhases, polyFracBits)
				histLen := polySizeHist(step, outLen, numPhases, taps, polyFracBits)
				hist := polyMakeHist(histLen, rng)

				gotOut := make([]float32, outLen)
				wantOut := make([]float32, outLen)
				nGot := polyphaseResampleCubicGo(gotOut, hist, a, b, c, d, 0, step, numPhases, taps, polyFracBits)
				nWant := polyRefLoop(wantOut, hist, a, b, c, d, 0, step, numPhases, taps, polyFracBits, cubicInterpDotGo)
				label := polyLabel(numPhases, taps, r.name)
				if nGot != nWant {
					t.Fatalf("%s: n got=%d want=%d", label, nGot, nWant)
				}
				polyBitsEqual(t, label, gotOut[:nGot], wantOut[:nWant])
			}
		}
	}
}

// TestPolyphaseResampleCubicGoHistExhaust drives polyphaseResampleCubicGo directly
// with a history shorter than the requested block so the window-exhaustion break
// fires and returns a partial n. On a SIMD host the public-API exhaustion path runs
// the asm kernel, so this is the only test that reaches the Go state machine's
// break; it must still match the div/mod oracle (partial n and bit-exact output).
func TestPolyphaseResampleCubicGoHistExhaust(t *testing.T) {
	const outLen = 1024
	rng := rand.New(rand.NewSource(0xE0F))
	for _, numPhases := range polyGridPhases {
		for _, taps := range polyGridTaps {
			a, b, c, d := polyMakeBanks(numPhases, taps, rng)
			step := polyDeriveStep(44100, 48000, numPhases, polyFracBits)
			// Short history: enough for some outputs, far fewer than outLen.
			hist := polyMakeHist(taps+37, rng)
			gotOut := make([]float32, outLen)
			wantOut := make([]float32, outLen)
			nGot := polyphaseResampleCubicGo(gotOut, hist, a, b, c, d, 0, step, numPhases, taps, polyFracBits)
			nWant := polyRefLoop(wantOut, hist, a, b, c, d, 0, step, numPhases, taps, polyFracBits, cubicInterpDotGo)
			label := polyLabel(numPhases, taps, "hist_exhaust")
			if nGot != nWant {
				t.Fatalf("%s: n got=%d want=%d", label, nGot, nWant)
			}
			if nGot >= outLen {
				t.Fatalf("%s: expected the history to exhaust before outLen, got n=%d", label, nGot)
			}
			polyBitsEqual(t, label, gotOut[:nGot], wantOut[:nWant])
		}
	}
}

// TestPolyphaseResampleCubicGoForcedCarries drives configurations where the frac
// carry and the phase wrap fire, including on the same output, and checks the Go
// state machine still matches the div/mod reference. Small numPhases and fracBits
// pack many carries into a short run, and a nonzero starting at exercises the
// seeding divisions.
func TestPolyphaseResampleCubicGoForcedCarries(t *testing.T) {
	rng := rand.New(rand.NewSource(0x9))
	cases := []struct {
		numPhases, taps, fracBits int
		at, step                  int64
	}{
		{3, 16, 4, 0, 22},      // fracMask=15; step frac=6, sPhase=1: carries interleave
		{3, 16, 4, 37, 25},     // nonzero at, both carries recur
		{5, 20, 8, 0, 617},     // fracMask=255
		{7, 16, 3, 11, 45},     // fracMask=7, tight
		{2, 16, 5, 0, 97},      // numPhases=2 so phase wraps almost every step
		{80, 20, 16, 0, 60073}, // realistic active sub-phase
	}
	for ci, tc := range cases {
		a, b, c, d := polyMakeBanks(tc.numPhases, tc.taps, rng)
		const outLen = 300
		lastDiv := int((tc.at + int64(outLen-1)*tc.step) >> uint(tc.fracBits) / int64(tc.numPhases))
		hist := polyMakeHist(lastDiv+tc.taps+8, rng)
		gotOut := make([]float32, outLen)
		wantOut := make([]float32, outLen)
		nGot := polyphaseResampleCubicGo(gotOut, hist, a, b, c, d, tc.at, tc.step, tc.numPhases, tc.taps, tc.fracBits)
		nWant := polyRefLoop(wantOut, hist, a, b, c, d, tc.at, tc.step, tc.numPhases, tc.taps, tc.fracBits, cubicInterpDotGo)
		if nGot != nWant {
			t.Fatalf("case %d: n got=%d want=%d", ci, nGot, nWant)
		}
		polyBitsEqual(t, fmt.Sprintf("forced_carry_case%d", ci), gotOut[:nGot], wantOut[:nWant])
	}
}

// ---------------------------------------------------------------------------
// Streaming continuity: chunked output blocks carrying atOut forward over the
// same hist must concatenate bit-identically to one shot.
// ---------------------------------------------------------------------------

func TestPolyphaseResampleCubicStreamingChunks(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5EED))
	const numPhases, taps = 80, 32
	step := polyDeriveStep(44100, 64000, numPhases, polyFracBits) // active sub-phase
	a, b, c, d := polyMakeBanks(numPhases, taps, rng)
	const total = 1024
	histLen := polySizeHist(step, total, numPhases, taps, polyFracBits)
	hist := polyMakeHist(histLen, rng)

	oneShot := make([]float32, total)
	nOne, _ := PolyphaseResampleCubic(oneShot, hist, a, b, c, d, 0, step, numPhases, taps, polyFracBits)

	for _, chunk := range []int{1, 7, 64, 1024} {
		chunked := make([]float32, total)
		at := int64(0)
		produced := 0
		for produced < nOne {
			end := min(produced+chunk, total)
			n, atOut := PolyphaseResampleCubic(chunked[produced:end], hist, a, b, c, d, at, step, numPhases, taps, polyFracBits)
			if n == 0 {
				break
			}
			at = atOut
			produced += n
		}
		if produced != nOne {
			t.Fatalf("chunk=%d produced=%d want=%d", chunk, produced, nOne)
		}
		polyBitsEqual(t, fmt.Sprintf("chunk%d", chunk), chunked[:nOne], oneShot[:nOne])
	}
}

// TestPolyphaseResampleCubicConsumerRebase mirrors processZeroCopy: between calls
// it consumes input from hist and rebases at by consumed*numPhases<<fracBits, and
// the concatenated output must match the unchunked result over the same stream.
func TestPolyphaseResampleCubicConsumerRebase(t *testing.T) {
	rng := rand.New(rand.NewSource(0xABCD))
	const numPhases, taps, fracBits = 80, 20, 16
	step := polyDeriveStep(44100, 48000, numPhases, fracBits)

	a, b, c, d := polyMakeBanks(numPhases, taps, rng)
	const total = 4096
	fullHist := polyMakeHist(polySizeHist(step, total, numPhases, taps, fracBits)+64, rng)

	// One-shot over the full history.
	oneShot := make([]float32, total)
	nOne, _ := PolyphaseResampleCubic(oneShot, fullHist, a, b, c, d, 0, step, numPhases, taps, fracBits)

	// Consumer-style: a sliding window into fullHist with rebasing at.
	got := make([]float32, 0, nOne)
	at := int64(0)
	base := 0 // index in fullHist that the current window starts at
	block := make([]float32, 200)
	for len(got) < nOne {
		window := fullHist[base:]
		n, atOut := PolyphaseResampleCubic(block, window, a, b, c, d, at, step, numPhases, taps, fracBits)
		if n == 0 {
			break
		}
		got = append(got, block[:n]...)
		at = atOut
		// Consume: drop consumed input samples, rebase at (matches processZeroCopy).
		consumed := int((at >> uint(fracBits)) / int64(numPhases))
		if consumed > 0 {
			base += consumed
			at -= int64(consumed) * int64(numPhases) << uint(fracBits)
		}
	}
	if len(got) > nOne {
		got = got[:nOne]
	}
	polyBitsEqual(t, "consumer_rebase", got, oneShot[:len(got)])
	if len(got) != nOne {
		t.Fatalf("consumer_rebase produced=%d want=%d", len(got), nOne)
	}
}

// ---------------------------------------------------------------------------
// Termination and edge cases.
// ---------------------------------------------------------------------------

func TestPolyphaseResampleCubicTermination(t *testing.T) {
	rng := rand.New(rand.NewSource(0x7))
	const numPhases, taps, fracBits = 80, 20, 16
	step := polyDeriveStep(44100, 48000, numPhases, fracBits)
	a, b, c, d := polyMakeBanks(numPhases, taps, rng)

	// out full before hist exhausted: big hist, small out.
	t.Run("out_full_first", func(t *testing.T) {
		hist := polyMakeHist(polySizeHist(step, 1024, numPhases, taps, fracBits), rng)
		out := make([]float32, 10)
		n, atOut := PolyphaseResampleCubic(out, hist, a, b, c, d, 0, step, numPhases, taps, fracBits)
		if n != 10 {
			t.Fatalf("n=%d want 10", n)
		}
		if atOut != int64(n)*step {
			t.Fatalf("atOut=%d want %d", atOut, int64(n)*step)
		}
	})

	// hist exhausted before out: small hist, big out.
	t.Run("hist_exhausted_first", func(t *testing.T) {
		hist := polyMakeHist(taps+30, rng)
		out := make([]float32, 1024)
		n, _ := PolyphaseResampleCubic(out, hist, a, b, c, d, 0, step, numPhases, taps, fracBits)
		want := polyRefLoop(make([]float32, 1024), hist, a, b, c, d, 0, step, numPhases, taps, fracBits, CubicInterpDotUnsafe)
		if n != want {
			t.Fatalf("n=%d want %d", n, want)
		}
		if n >= 1024 {
			t.Fatalf("expected early termination, got n=%d", n)
		}
	})

	// n==0: empty out, hist shorter than taps, at already past hist.
	t.Run("zero_outputs", func(t *testing.T) {
		hist := polyMakeHist(polySizeHist(step, 100, numPhases, taps, fracBits), rng)
		if n, at := PolyphaseResampleCubic(nil, hist, a, b, c, d, 0, step, numPhases, taps, fracBits); n != 0 || at != 0 {
			t.Fatalf("empty out: (%d,%d) want (0,0)", n, at)
		}
		short := polyMakeHist(taps-1, rng)
		out := make([]float32, 16)
		if n, at := PolyphaseResampleCubic(out, short, a, b, c, d, 0, step, numPhases, taps, fracBits); n != 0 || at != 0 {
			t.Fatalf("short hist: (%d,%d) want (0,0)", n, at)
		}
		// at already past hist: start beyond the last valid window.
		hist2 := polyMakeHist(taps+5, rng)
		startAt := int64(len(hist2)) * int64(numPhases) << uint(fracBits)
		if n, at := PolyphaseResampleCubic(out, hist2, a, b, c, d, startAt, step, numPhases, taps, fracBits); n != 0 || at != startAt {
			t.Fatalf("past-hist at: (%d,%d) want (0,%d)", n, at, startAt)
		}
	})
}

func TestPolyphaseResampleCubicValidation(t *testing.T) {
	rng := rand.New(rand.NewSource(0x11))
	const numPhases, taps, fracBits = 8, 16, 16
	step := polyDeriveStep(44100, 48000, numPhases, fracBits)
	a, b, c, d := polyMakeBanks(numPhases, taps, rng)
	hist := polyMakeHist(polySizeHist(step, 64, numPhases, taps, fracBits), rng)
	out := make([]float32, 64)

	// Each variant must return (0, at) with at unchanged.
	assertNoop := func(name string, at int64, fn func() (int, int64)) {
		n, atOut := fn()
		if n != 0 || atOut != at {
			t.Fatalf("%s: got (%d,%d) want (0,%d)", name, n, atOut, at)
		}
	}
	assertNoop("numPhases<1", 5, func() (int, int64) {
		return PolyphaseResampleCubic(out, hist, a, b, c, d, 5, step, 0, taps, fracBits)
	})
	assertNoop("taps<1", 5, func() (int, int64) {
		return PolyphaseResampleCubic(out, hist, a, b, c, d, 5, step, numPhases, 0, fracBits)
	})
	assertNoop("step<1", 5, func() (int, int64) {
		return PolyphaseResampleCubic(out, hist, a, b, c, d, 5, 0, numPhases, taps, fracBits)
	})
	assertNoop("at<0", -3, func() (int, int64) {
		return PolyphaseResampleCubic(out, hist, a, b, c, d, -3, step, numPhases, taps, fracBits)
	})
	assertNoop("fracBits<0", 5, func() (int, int64) {
		return PolyphaseResampleCubic(out, hist, a, b, c, d, 5, step, numPhases, taps, -1)
	})
	assertNoop("fracBits>max", 5, func() (int, int64) {
		return PolyphaseResampleCubic(out, hist, a, b, c, d, 5, step, numPhases, taps, polyphaseMaxFracBits32+1)
	})
	assertNoop("short_bank", 5, func() (int, int64) {
		return PolyphaseResampleCubic(out, hist, a[:numPhases-1], b, c, d, 5, step, numPhases, taps, fracBits)
	})
	// short row: shrink one row below taps.
	shortRows := make([][]float32, numPhases)
	copy(shortRows, a)
	shortRows[numPhases-1] = shortRows[numPhases-1][:taps-1]
	assertNoop("short_row", 5, func() (int, int64) {
		return PolyphaseResampleCubic(out, hist, shortRows, b, c, d, 5, step, numPhases, taps, fracBits)
	})
	// Accumulator overflow: a step so large that at+len(out)*step overflows int64
	// must be rejected as (0, at) rather than panicking or reading out of range.
	assertNoop("step_overflow", 5, func() (int, int64) {
		return PolyphaseResampleCubic(out, hist, a, b, c, d, 5, math.MaxInt64, numPhases, taps, fracBits)
	})
}

// TestPolyphaseResampleCubicMaxFracBits checks the valid side of the fracBits
// boundary: at exactly polyphaseMaxFracBits32 the fused kernel still produces
// output bit-identical to the div/mod oracle, confirming float32(frac)*2^-fracBits
// is exact at the maximum (the constant is not off by one).
func TestPolyphaseResampleCubicMaxFracBits(t *testing.T) {
	rng := rand.New(rand.NewSource(0x2C))
	const numPhases, taps, outLen = 80, 32, 256
	fracBits := polyphaseMaxFracBits32
	step := polyDeriveStep(44100, 48000, numPhases, fracBits)
	a, b, c, d := polyMakeBanks(numPhases, taps, rng)
	hist := polyMakeHist(polySizeHist(step, outLen, numPhases, taps, fracBits), rng)
	gotOut := make([]float32, outLen)
	wantOut := make([]float32, outLen)
	n, _ := PolyphaseResampleCubic(gotOut, hist, a, b, c, d, 0, step, numPhases, taps, fracBits)
	nWant := polyRefLoop(wantOut, hist, a, b, c, d, 0, step, numPhases, taps, fracBits, CubicInterpDotUnsafe)
	if n == 0 || n != nWant {
		t.Fatalf("max fracBits: n got=%d want=%d (expected nonzero)", n, nWant)
	}
	polyBitsEqual(t, "max_fracbits", gotOut[:n], wantOut[:nWant])
}

// ---------------------------------------------------------------------------
// Zero allocation on the safe path.
// ---------------------------------------------------------------------------

func TestPolyphaseResampleCubicNoAlloc(t *testing.T) {
	rng := rand.New(rand.NewSource(0x33))
	const numPhases, taps, fracBits = 80, 32, 16
	step := polyDeriveStep(44100, 48000, numPhases, fracBits)
	a, b, c, d := polyMakeBanks(numPhases, taps, rng)
	hist := polyMakeHist(polySizeHist(step, 1024, numPhases, taps, fracBits), rng)
	out := make([]float32, 1024)
	// Guard against a vacuous pass: confirm this config actually produces output,
	// so a regression that early-returned (0, at) could not read as zero-alloc.
	if n, _ := PolyphaseResampleCubic(out, hist, a, b, c, d, 0, step, numPhases, taps, fracBits); n == 0 {
		t.Fatalf("no-alloc config produced no output; the allocation check would be vacuous")
	}
	if allocs := testing.AllocsPerRun(50, func() {
		PolyphaseResampleCubic(out, hist, a, b, c, d, 0, step, numPhases, taps, fracBits)
	}); allocs != 0 {
		t.Fatalf("PolyphaseResampleCubic allocated %v times, want 0", allocs)
	}
	if allocs := testing.AllocsPerRun(50, func() {
		PolyphaseResampleCubicUnsafe(out, hist, a, b, c, d, 0, step, numPhases, taps, fracBits)
	}); allocs != 0 {
		t.Fatalf("PolyphaseResampleCubicUnsafe allocated %v times, want 0", allocs)
	}
}
