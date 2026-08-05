package f64

import (
	"math"
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// Aliasing sweep for the f64 exact-overlay contract (issue #221). See the f32
// suite for the method: each op is run once into a separate destination and once
// with the destination overlaid on an input, bit-compared under each kernel tier
// forTiers can force (every amd64 function-pointer tier; both the Go and NEON
// arm64 paths). The transcendentals dispatch through an inline CPU-feature branch
// the harness cannot rebind, but they have no distinct SSE kernel, so the size
// sweep still runs both their Go and native-SIMD kernels. It asserts nothing about
// how a shifted overlay (dst offset from an input) corrupts, which is undefined.

func aliasEqF64(x, y float64) bool { return math.Float64bits(x) == math.Float64bits(y) }

func aliasHashF64(i int) float64 {
	// A full-width 64-bit mix (the golden-ratio multiplier), so small indices
	// still spread across the whole [0,1) range. A 32-bit step over uint64 would
	// leave every practical index near 0, clustering aliasGenF64 around -4.
	u := uint64(i)*0x9e3779b97f4a7c15 + 1013904223
	return float64(u>>11) / float64(1<<53)
}

// aliasGenF64 spreads values over [-4,4), including negatives and near-zero.
func aliasGenF64(i int) float64 { return aliasHashF64(i)*8 - 4 }

// aliasGenF64Pos spreads values over (0,8] for domain-restricted ops.
func aliasGenF64Pos(i int) float64 { return aliasHashF64(i)*8 + 0.03125 }

// Scalar parameters for the scalar-taking ops. Their exact values do not affect
// the overlay property.
const (
	aliasScaleK  = 1.5
	aliasAddK    = 0.75
	aliasClampLo = -1.25
	aliasClampHi = 2.5
	aliasScaleC  = 0.5
	aliasPowExp  = 0.75
)

func f64AliasCases() []aliastest.Case {
	return []aliastest.Case{
		aliastest.UnaryCase("Abs", aliasEqF64, aliasGenF64, Abs),
		aliastest.UnaryCase("Neg", aliasEqF64, aliasGenF64, Neg),
		aliastest.UnaryCase("Round", aliasEqF64, aliasGenF64, Round),
		aliastest.UnaryCase("Reciprocal", aliasEqF64, aliasGenF64, Reciprocal),
		aliastest.UnaryCase("ReLU", aliasEqF64, aliasGenF64, ReLU),
		aliastest.UnaryCase("Sigmoid", aliasEqF64, aliasGenF64, Sigmoid),
		aliastest.UnaryCase("Tanh", aliasEqF64, aliasGenF64, Tanh),
		aliastest.UnaryCase("Exp", aliasEqF64, aliasGenF64, Exp),
		aliastest.UnaryCase("Sqrt", aliasEqF64, aliasGenF64Pos, Sqrt),
		aliastest.UnaryCase("Log", aliasEqF64, aliasGenF64Pos, Log),
		aliastest.UnaryCase("Log2", aliasEqF64, aliasGenF64Pos, Log2),
		aliastest.UnaryCase("Log10", aliasEqF64, aliasGenF64Pos, Log10),
		aliastest.UnaryCase("CumulativeSum", aliasEqF64, aliasGenF64, CumulativeSum),
		aliastest.UnaryCase("Normalize", aliasEqF64, aliasGenF64, Normalize),

		aliastest.UnaryCase("Scale", aliasEqF64, aliasGenF64, func(dst, a []float64) { Scale(dst, a, aliasScaleK) }),
		aliastest.UnaryCase("AddScalar", aliasEqF64, aliasGenF64, func(dst, a []float64) { AddScalar(dst, a, aliasAddK) }),
		aliastest.UnaryCase("SubFromScalar", aliasEqF64, aliasGenF64, func(dst, a []float64) { SubFromScalar(dst, a, aliasAddK) }),
		aliastest.UnaryCase("Clamp", aliasEqF64, aliasGenF64, func(dst, a []float64) { Clamp(dst, a, aliasClampLo, aliasClampHi) }),
		aliastest.UnaryCase("ClampScale", aliasEqF64, aliasGenF64, func(dst, a []float64) { ClampScale(dst, a, aliasClampLo, aliasClampHi, aliasScaleC) }),
		aliastest.UnaryCase("Pow", aliasEqF64, aliasGenF64Pos, func(dst, a []float64) { Pow(dst, a, aliasPowExp) }),

		aliastest.BinaryCase("Add", aliasEqF64, aliasGenF64, Add),
		aliastest.BinaryCase("Sub", aliasEqF64, aliasGenF64, Sub),
		aliastest.BinaryCase("Mul", aliasEqF64, aliasGenF64, Mul),
		aliastest.BinaryCase("Div", aliasEqF64, aliasGenF64, Div),
		aliastest.BinaryCase("PowElem", aliasEqF64, aliasGenF64Pos, PowElem),

		aliastest.TernaryCase("FMA", aliasEqF64, aliasGenF64, FMA),
	}
}

// TestAliasingSweep drives the exact-overlay sweep across every bound kernel.
func TestAliasingSweep(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.Sweep(t, f64AliasCases())
		t.Run("AddScaled", sweepAddScaled)
	})
}

// TestAliasingZeroAlloc asserts the in-place overlay path is allocation-free for
// every swept op under each bound tier, enforcing the package zero-allocation
// contract on the aliasing path.
func TestAliasingZeroAlloc(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.SweepAlloc(t, f64AliasCases())
		t.Run("AddScaled", func(t *testing.T) {
			a := make([]float64, 64)
			for i := range a {
				a[i] = aliasGenF64(i)
			}
			aliastest.ZeroAlloc(t, "AddScaled s==dst", func() { AddScaled(a, 1.5, a) })
		})
	})
}

// sweepAddScaled checks AddScaled's documented in-place overlay: s may equal dst
// exactly (the self-scaling dst[i] += alpha*dst[i]). AddScaled does not fit the
// generic Unary/Binary shapes because dst is a read-modify-write accumulator, so
// it gets a bespoke check. AddScaled is function-pointer dispatched, so forTiers
// forces its tiers.
func sweepAddScaled(t *testing.T) {
	t.Helper()
	const alpha = 1.5
	for _, n := range aliastest.Sizes {
		data := make([]float64, n)
		for i := range data {
			data[i] = aliasGenF64(i)
		}
		want := append([]float64(nil), data...)
		AddScaled(want, alpha, append([]float64(nil), data...))
		got := append([]float64(nil), data...)
		AddScaled(got, alpha, got)
		aliastest.Report(t, n, "AddScaled s==dst", aliasEqF64, want, got)
	}
}
