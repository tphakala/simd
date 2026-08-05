package c128

import (
	"math"
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// Aliasing sweep for the c128 exact-overlay contract (issue #221). Each element-
// wise op is run once into a separate destination and once with the destination
// overlaid on an input, then bit-compared (real and imaginary parts) under every
// bound kernel tier. It asserts nothing about how a shifted overlay (dst offset
// from an input) corrupts: that pattern is undefined and varies with kernel
// width and length.

func aliasEqC128(x, y complex128) bool {
	return math.Float64bits(real(x)) == math.Float64bits(real(y)) &&
		math.Float64bits(imag(x)) == math.Float64bits(imag(y))
}

func aliasHashF64(i int) float64 {
	// A full-width 64-bit mix (the golden-ratio multiplier) so small indices
	// still spread across [-4,4); a 32-bit step over uint64 would cluster every
	// practical index near -4.
	u := uint64(i)*0x9e3779b97f4a7c15 + 1013904223
	return float64(u>>11)/float64(1<<53)*8 - 4
}

// aliasGenC128 builds a deterministic complex128 with independent real and
// imaginary spreads over [-4,4).
func aliasGenC128(i int) complex128 { return complex(aliasHashF64(i), aliasHashF64(i+9973)) }

// aliasScaleC128 is the scalar for the Scale overlay check; its value does not
// affect the overlay property.
const aliasScaleC128 = complex(1.25, -0.5)

func c128AliasCases() []aliastest.Case {
	return []aliastest.Case{
		aliastest.BinaryCase("Add", aliasEqC128, aliasGenC128, Add),
		aliastest.BinaryCase("Sub", aliasEqC128, aliasGenC128, Sub),
		aliastest.BinaryCase("Mul", aliasEqC128, aliasGenC128, Mul),
		aliastest.BinaryCase("MulConj", aliasEqC128, aliasGenC128, MulConj),
		aliastest.UnaryCase("Conj", aliasEqC128, aliasGenC128, Conj),
		aliastest.UnaryCase("Scale", aliasEqC128, aliasGenC128, func(dst, a []complex128) { Scale(dst, a, aliasScaleC128) }),
	}
}

// TestAliasingSweep drives the exact-overlay sweep across every bound kernel.
func TestAliasingSweep(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.Sweep(t, c128AliasCases())
	})
}

// TestAliasingZeroAlloc asserts the in-place overlay path is allocation-free for
// every swept op under each bound tier.
func TestAliasingZeroAlloc(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.SweepAlloc(t, c128AliasCases())
	})
}
