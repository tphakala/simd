//go:build arm64

package f16

import (
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// forTiers runs the aliasing sweep on both the pure-Go reference and the native
// arm64 kernels. f16 arithmetic dispatches on hasFP16 (the FEAT_FP16 half-precision
// kernels) with ReLU/Neg on hasNEON, so both gates are flipped together: one pass
// with the host's native kernels bound and one with both forced off (the Go path
// at every length). The overlay property is per-kernel, so covering the native and
// the Go tier for every op is sufficient.
func forTiers(t *testing.T, run func(t *testing.T)) {
	t.Helper()
	savedFP16, savedNEON := hasFP16, hasNEON
	aliastest.ForTiers(t, []aliastest.Tier{
		{
			Name:      "native",
			Bind:      func() { hasFP16, hasNEON = savedFP16, savedNEON },
			Supported: savedFP16 || savedNEON,
		},
		{
			Name:      "Go",
			Bind:      func() { hasFP16, hasNEON = false, false },
			Supported: true,
		},
	}, run)
}
