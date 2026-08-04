package f32

import (
	"math/rand"
	"testing"
)

// polyStepLoopUnsafe is the best pure-Go resampler-side alternative to the fused
// kernel: the same incremental phase-stepping state machine, but calling the
// dispatched CubicInterpDotUnsafe once per output instead of running the whole
// block in one asm call. It is the baseline the fused kernel must beat to justify
// the assembly (issue #52). It returns the number of outputs written.
func polyStepLoopUnsafe(out, hist []float32, a, b, c, d [][]float32, at, step int64, numPhases, taps, fracBits int) int {
	numPhases64 := int64(numPhases)
	fracMask := int64(1)<<uint(fracBits) - 1
	fracScale := float32(1.0 / float64(int64(1)<<uint(fracBits)))
	full := at >> uint(fracBits)
	div := int(full / numPhases64)
	phase := int(full - int64(div)*numPhases64)
	frac := at & fracMask
	sFull := step >> uint(fracBits)
	sDiv := int(sFull / numPhases64)
	sPhase := int(sFull - int64(sDiv)*numPhases64)
	sFrac := step & fracMask
	histLen := len(hist)
	k := 0
	for k < len(out) {
		if div+taps > histLen {
			break
		}
		x := float32(frac) * fracScale
		out[k] = CubicInterpDotUnsafe(
			hist[div:div+taps],
			a[phase][:taps], b[phase][:taps],
			c[phase][:taps], d[phase][:taps], x)
		k++
		frac += sFrac
		if frac > fracMask {
			frac -= fracMask + 1
			phase++
		}
		phase += sPhase
		div += sDiv
		if phase >= numPhases {
			phase -= numPhases
			div++
		}
	}
	return k
}

// polyBenchConfigs covers the QualityMedium regime that gates issue #52: the tap
// counts a real polyphase resampler runs at, all at numPhases 80.
var polyBenchConfigs = []struct{ taps, numPhases int }{
	{16, 80}, {20, 80}, {32, 80},
}

const polyBenchOut = 1024

type polyBenchData struct {
	out, hist  []float32
	a, b, c, d [][]float32
	step       int64
}

func benchPolySetup(taps, numPhases int) polyBenchData {
	rng := rand.New(rand.NewSource(1))
	step := polyDeriveStep(44100, 48000, numPhases, polyFracBits)
	a, b, c, d := polyMakeBanks(numPhases, taps, rng)
	hist := polyMakeHist(polySizeHist(step, polyBenchOut, numPhases, taps, polyFracBits), rng)
	return polyBenchData{make([]float32, polyBenchOut), hist, a, b, c, d, step}
}

// BenchmarkPolyphaseResampleCubicFused measures the fused kernel (dispatched to
// asm where available) over a 1024-output block.
func BenchmarkPolyphaseResampleCubicFused(b *testing.B) {
	for _, cfg := range polyBenchConfigs {
		d := benchPolySetup(cfg.taps, cfg.numPhases)
		b.Run(polyLabel(cfg.numPhases, cfg.taps, "44k_to_48k_up"), func(b *testing.B) {
			for b.Loop() {
				PolyphaseResampleCubicUnsafe(d.out, d.hist, d.a, d.b, d.c, d.d, 0, d.step, cfg.numPhases, cfg.taps, polyFracBits)
			}
		})
	}
}

// BenchmarkPolyphaseResampleCubicStepLoop measures the pure-Go per-output
// StepLoop over the same block, the baseline for the issue #52 merge gate.
func BenchmarkPolyphaseResampleCubicStepLoop(b *testing.B) {
	for _, cfg := range polyBenchConfigs {
		d := benchPolySetup(cfg.taps, cfg.numPhases)
		b.Run(polyLabel(cfg.numPhases, cfg.taps, "44k_to_48k_up"), func(b *testing.B) {
			for b.Loop() {
				polyStepLoopUnsafe(d.out, d.hist, d.a, d.b, d.c, d.d, 0, d.step, cfg.numPhases, cfg.taps, polyFracBits)
			}
		})
	}
}
