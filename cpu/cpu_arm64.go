//go:build arm64 && !darwin

package cpu

import (
	"os"

	"golang.org/x/sys/cpu"
)

func init() {
	ARM64.NEON = cpu.ARM64.HasASIMD
	ARM64.FP16 = cpu.ARM64.HasASIMDHP // FEAT_FP16 - half-precision SIMD
	ARM64.SVE = cpu.ARM64.HasSVE
	ARM64.SVE2 = cpu.ARM64.HasSVE2
	ARM64.PMULL = cpu.ARM64.HasPMULL // FEAT_PMULL - polynomial multiply

	// Honor SIMD_DISABLE last, so the env var can mask any detected feature.
	applyDisable(&ARM64, os.Getenv("SIMD_DISABLE"))
}

func cpuInfo() string {
	switch {
	case ARM64.SVE2:
		return "ARM64 SVE2"
	case ARM64.SVE:
		return "ARM64 SVE"
	case ARM64.NEON && ARM64.FP16:
		return "ARM64 NEON+FP16"
	case ARM64.NEON:
		return "ARM64 NEON"
	default:
		return "ARM64 (no SIMD)"
	}
}
