package i16

import (
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// Aliasing sweep for the i16 exact-overlay contract (issue #221). Each element-
// wise op is run once into a separate destination and once with the destination
// overlaid on an input, then compared under both the Go and SIMD kernels. The
// widening ops (DotProduct, XCorr) and the length-changing shuffles (Interleave2,
// Deinterleave2) cannot take an element-for-element overlay and are not swept
// here. It asserts nothing about how a shifted overlay (dst offset from an input)
// corrupts, which is undefined.

func aliasEqI16(x, y int16) bool { return x == y }

// aliasGenI16 spreads values over the full int16 range, including -32768 (the
// wrapping edge for Abs and the MinInt16*MinInt16 case for MulQ15).
func aliasGenI16(i int) int16 {
	u := uint32(i)*2654435761 + 1013904223
	return int16(u >> 16) //nolint:gosec // deliberate wrap to cover [-32768,32767]
}

func i16AliasCases() []aliastest.Case {
	return []aliastest.Case{
		aliastest.UnaryCase("Abs", aliasEqI16, aliasGenI16, Abs),
		aliastest.BinaryCase("MulQ15", aliasEqI16, aliasGenI16, MulQ15),
	}
}

// TestAliasingSweep drives the exact-overlay sweep across every bound kernel.
func TestAliasingSweep(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.Sweep(t, i16AliasCases())
	})
}

// TestAliasingZeroAlloc asserts the in-place overlay path is allocation-free for
// every swept op under both the Go and SIMD kernels.
func TestAliasingZeroAlloc(t *testing.T) {
	forTiers(t, func(t *testing.T) {
		t.Helper()
		aliastest.SweepAlloc(t, i16AliasCases())
	})
}
