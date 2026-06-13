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
	ARM64.PMULL = cpu.ARM64.HasPMULL     // FEAT_PMULL - polynomial multiply
	ARM64.DOTPROD = cpu.ARM64.HasASIMDDP // FEAT_DotProd - SDOT/UDOT int8 dot product

	// Honor SIMD_DISABLE last, so the env var can mask any detected feature.
	applyDisable(&ARM64, os.Getenv("SIMD_DISABLE"))
}

// cpuInfo is shared with the darwin build; see cpu_arm64_info.go.
