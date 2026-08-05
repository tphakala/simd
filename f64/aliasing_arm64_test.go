//go:build arm64

package f64

import (
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// forTiers runs the aliasing sweep on both the Go and NEON arm64 paths by
// flipping the package hasNEON gate, so the sweep sees the two kernels that #215
// showed can disagree.
func forTiers(t *testing.T, run func(t *testing.T)) {
	t.Helper()
	aliastest.ForGate(t, &hasNEON, "NEON", run)
}
