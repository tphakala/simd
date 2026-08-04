//go:build arm64

package f32

import (
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// forTiers runs the aliasing sweep on both the Go and NEON arm64 paths by
// flipping the package hasNEON gate, so the sweep sees the two kernels that #215
// showed can disagree. On a NEON-less arm64 host the NEON tier is skipped and the
// Go path runs once.
func forTiers(t *testing.T, run func(t *testing.T)) {
	t.Helper()
	aliastest.ForGate(t, &hasNEON, "NEON", run)
}
