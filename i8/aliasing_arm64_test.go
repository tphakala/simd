//go:build arm64

package i8

import (
	"testing"

	"github.com/tphakala/simd/internal/aliastest"
)

// forTiers runs the aliasing sweep on both the Go and NEON arm64 paths by
// flipping the package hasNEON gate, so the sweep sees both int8 kernels: the
// NEON path and the pure-Go fallback can absorb their tail differently, so an
// in-place overlay must hold on each.
func forTiers(t *testing.T, run func(t *testing.T)) {
	t.Helper()
	aliastest.ForGate(t, &hasNEON, "NEON", run)
}
