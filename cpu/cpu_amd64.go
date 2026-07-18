//go:build amd64

package cpu

import (
	"os"

	"golang.org/x/sys/cpu"
)

func init() {
	X86.SSE = true // SSE is baseline for amd64
	X86.SSE2 = cpu.X86.HasSSE2
	X86.SSE3 = cpu.X86.HasSSE3
	X86.SSSE3 = cpu.X86.HasSSSE3
	X86.SSE41 = cpu.X86.HasSSE41
	X86.SSE42 = cpu.X86.HasSSE42
	X86.AVX = cpu.X86.HasAVX
	X86.AVX2 = cpu.X86.HasAVX2
	X86.AVX512F = cpu.X86.HasAVX512F
	X86.AVX512VL = cpu.X86.HasAVX512VL
	X86.FMA = cpu.X86.HasFMA
	X86.BMI1 = cpu.X86.HasBMI1
	X86.BMI2 = cpu.X86.HasBMI2
	X86.POPCNT = cpu.X86.HasPOPCNT
	X86.PCLMULQDQ = cpu.X86.HasPCLMULQDQ

	// F16C (half<->single conversion) is not exposed by golang.org/x/sys/cpu, so
	// read it directly from CPUID leaf 1, ECX bit 29. Gate on the x/sys AVX flag,
	// which already includes the OSXSAVE/XGETBV OS-support check that F16C needs
	// because it operates on VEX/YMM state.
	_, _, ecx, _ := cpuid(1, 0)
	X86.F16C = X86.AVX && ecx&cpuidF16CBit != 0

	// AVX-VNNI (VEX-encoded VPDPWSSD etc.) is not exposed by golang.org/x/sys/cpu
	// either, so read it directly from CPUID leaf 7 sub-leaf 1, EAX bit 4. Gate on
	// AVX2: the VEX form operates on YMM state, so it needs the same
	// OSXSAVE/XGETBV OS support that AVX2 detection already establishes, and the
	// kernel that consumes it is an AVX2 kernel.
	eaxVNNI, _, _, _ := cpuid(cpuidExtdFeatureLeaf, cpuidExtdFeatureSubleaf1)
	X86.AVXVNNI = X86.AVX2 && eaxVNNI&cpuidAVXVNNIBit != 0

	// Honor SIMD_DISABLE last, so the env var can mask any detected feature
	// (including F16C and AVXVNNI via the "all" token).
	applyDisable(&X86, os.Getenv("SIMD_DISABLE"))
}

// cpuidF16CBit is CPUID leaf 1 ECX bit 29, set when F16C is supported.
const cpuidF16CBit = 1 << 29

// CPUID leaf 7 sub-leaf 1 addresses the structured extended feature flags whose
// EAX reports AVX-VNNI; cpuidAVXVNNIBit is that EAX's bit 4, set when the
// VEX-encoded AVX-VNNI instructions are supported. Named rather than inlined so
// the leaf/sub-leaf selectors read as identifiers, not magic numbers.
const (
	cpuidExtdFeatureLeaf     = 7
	cpuidExtdFeatureSubleaf1 = 1
	cpuidAVXVNNIBit          = 1 << 4
)

// cpuid is the raw CPUID wrapper implemented in cpuid_amd64.s.
//
//go:noescape
func cpuid(leaf, subleaf uint32) (eax, ebx, ecx, edx uint32)

func cpuInfo() string {
	switch {
	case X86.AVX512F && X86.AVX512VL:
		return "AMD64 AVX-512"
	case X86.AVX && X86.FMA:
		return "AMD64 AVX+FMA"
	case X86.AVX:
		return "AMD64 AVX"
	case X86.SSE2:
		return "AMD64 SSE2"
	default:
		return "AMD64 (scalar)"
	}
}
