//go:build amd64

package cpu

import "golang.org/x/sys/cpu"

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
}

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
