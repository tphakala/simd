//go:build !amd64 && !arm64

package f64

import "testing"

// forTiers runs the aliasing sweep once on architectures with only the pure-Go
// path (no tier to force).
func forTiers(t *testing.T, run func(t *testing.T)) {
	t.Helper()
	run(t)
}
