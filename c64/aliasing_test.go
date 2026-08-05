package c64

import (
	"math"
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// Aliasing sweep for the c64 exact-overlay contract (issue #221). Each element-
// wise op is run once into a separate destination and once with the destination
// overlaid on an input, then bit-compared (real and imaginary parts) under every
// bound kernel tier. It asserts nothing about how a shifted overlay (dst offset
// from an input) corrupts: that pattern is undefined and varies with kernel
// width and length.

func aliasEqC64(x, y complex64) bool {
	return math.Float32bits(real(x)) == math.Float32bits(real(y)) &&
		math.Float32bits(imag(x)) == math.Float32bits(imag(y))
}

func aliasHashF32(i int) float32 {
	u := uint32(i)*2654435761 + 1013904223
	return float32(u)/float32(1<<32)*8 - 4
}

// aliasGenC64 builds a deterministic complex64 with independent real and
// imaginary spreads over [-4,4).
func aliasGenC64(i int) complex64 { return complex(aliasHashF32(i), aliasHashF32(i+9973)) }

// aliasScaleC64 is the scalar for the Scale overlay check; its value does not
// affect the overlay property.
const aliasScaleC64 = complex64(complex(1.25, -0.5))

func c64AliasCases() []aliastest.Case {
	return []aliastest.Case{
		aliastest.BinaryCase("Add", aliasEqC64, aliasGenC64, Add),
		aliastest.BinaryCase("Sub", aliasEqC64, aliasGenC64, Sub),
		aliastest.BinaryCase("Mul", aliasEqC64, aliasGenC64, Mul),
		aliastest.BinaryCase("MulConj", aliasEqC64, aliasGenC64, MulConj),
		aliastest.UnaryCase("Conj", aliasEqC64, aliasGenC64, Conj),
		aliastest.UnaryCase("Scale", aliasEqC64, aliasGenC64, func(dst, a []complex64) { Scale(dst, a, aliasScaleC64) }),
	}
}

// TestAliasingSweep drives the exact-overlay sweep across every bound kernel.
func TestAliasingSweep(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.Sweep(t, c64AliasCases())
	})
}

// TestAliasingZeroAlloc asserts the in-place overlay path is allocation-free for
// every swept op under each bound tier.
func TestAliasingZeroAlloc(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.SweepAlloc(t, c64AliasCases())
	})
}
