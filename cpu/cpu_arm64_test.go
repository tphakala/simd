//go:build arm64

package cpu

import "testing"

// TestCpuInfoAllBranches tests all branches of cpuInfo by temporarily
// modifying the global feature flags.
func TestCpuInfoAllBranches(t *testing.T) {
	// Save original values
	origNEON := ARM64.NEON
	origSVE := ARM64.SVE
	origSVE2 := ARM64.SVE2

	// Restore at end
	defer func() {
		ARM64.NEON = origNEON
		ARM64.SVE = origSVE
		ARM64.SVE2 = origSVE2
	}()

	tests := []struct {
		name            string
		neon, sve, sve2 bool
		want            string
	}{
		{"SVE2", true, true, true, "ARM64 SVE2"},
		{"SVE", true, true, false, "ARM64 SVE"},
		{"NEON", true, false, false, "ARM64 NEON"},
		{"no_SIMD", false, false, false, "ARM64 (no SIMD)"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ARM64.NEON = tt.neon
			ARM64.SVE = tt.sve
			ARM64.SVE2 = tt.sve2

			got := cpuInfo()
			if got != tt.want {
				t.Errorf("cpuInfo() = %q, want %q", got, tt.want)
			}
		})
	}
}
