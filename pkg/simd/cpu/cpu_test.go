package cpu

import (
	"testing"
)

// TestHasAVX tests the HasAVX function
func TestHasAVX(t *testing.T) {
	// Just verify it returns a boolean without panicking
	got := HasAVX()
	// On ARM64, this should be false
	// On AMD64, it depends on the CPU
	_ = got
}

// TestHasAVX2 tests the HasAVX2 function
func TestHasAVX2(t *testing.T) {
	got := HasAVX2()
	_ = got
}

// TestHasFMA tests the HasFMA function
func TestHasFMA(t *testing.T) {
	got := HasFMA()
	_ = got
}

// TestHasNEON tests the HasNEON function
func TestHasNEON(t *testing.T) {
	got := HasNEON()
	_ = got
}

// TestHasAVX512VL tests the HasAVX512VL function
func TestHasAVX512VL(t *testing.T) {
	got := HasAVX512VL()
	_ = got
}

// TestInfo tests the Info function
func TestInfo(t *testing.T) {
	info := Info()
	if info == "" {
		t.Error("Info() returned empty string")
	}
}

// TestCpuInfo tests the cpuInfo function directly
func TestCpuInfo(t *testing.T) {
	info := cpuInfo()
	if info == "" {
		t.Error("cpuInfo() returned empty string")
	}
}

// TestFeatures tests that Features struct fields are accessible
func TestFeatures(t *testing.T) {
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

	// Test ARM64 features struct
	_ = ARM64.NEON
	_ = ARM64.SVE
	_ = ARM64.SVE2
}
