package cpu

import (
	"testing"
)

// TestHasAVX tests the HasAVX function
func TestHasAVX(_ *testing.T) {
	// Just verify it returns a boolean without panicking
	got := HasAVX()
	// On ARM64, this should be false
	// On AMD64, it depends on the CPU
	_ = got
}

// TestHasAVX2 tests the HasAVX2 function
func TestHasAVX2(_ *testing.T) {
	got := HasAVX2()
	_ = got
}

// TestHasFMA tests the HasFMA function
func TestHasFMA(_ *testing.T) {
	got := HasFMA()
	_ = got
}

// TestHasNEON tests the HasNEON function
func TestHasNEON(_ *testing.T) {
	got := HasNEON()
	_ = got
}

// TestHasFP16 tests the HasFP16 function
func TestHasFP16(_ *testing.T) {
	got := HasFP16()
	_ = got
}

// TestHasPCLMULQDQ tests the HasPCLMULQDQ function
func TestHasPCLMULQDQ(_ *testing.T) {
	got := HasPCLMULQDQ()
	_ = got
}

// TestHasPMULL tests the HasPMULL function
func TestHasPMULL(_ *testing.T) {
	got := HasPMULL()
	_ = got
}

// TestHasAVX512VL tests the HasAVX512VL function
func TestHasAVX512VL(_ *testing.T) {
	got := HasAVX512VL()
	_ = got
}

// TestInfo tests the Info function
func TestInfo(t *testing.T) {
	info := Info()
	if info == "" {
		t.Error("Info() returned empty string")
	}
	// Surface the selected tier in `go test -v` output so CI logs show which
	// dispatch path each runner exercised (e.g. the native linux/arm64 leg
	// reporting an ARM64 NEON configuration rather than the darwin override).
	t.Logf("cpu.Info() = %q", info)
}

// TestCpuInfo tests the cpuInfo function directly
func TestCpuInfo(t *testing.T) {
	info := cpuInfo()
	if info == "" {
		t.Error("cpuInfo() returned empty string")
	}
}

// TestFeatures tests that Features struct fields are accessible
func TestFeatures(_ *testing.T) {
	// Test X86 features struct
	_ = X86.SSE
	_ = X86.SSE2
	_ = X86.SSE3
	_ = X86.SSSE3
	_ = X86.SSE41
	_ = X86.SSE42
	_ = X86.AVX
	_ = X86.AVX2
	_ = X86.AVX512F
	_ = X86.AVX512VL
	_ = X86.FMA
	_ = X86.BMI1
	_ = X86.BMI2
	_ = X86.POPCNT
	_ = X86.PCLMULQDQ

	// Test ARM64 features struct
	_ = ARM64.NEON
	_ = ARM64.FP16
	_ = ARM64.SVE
	_ = ARM64.SVE2
	_ = ARM64.PMULL
}
