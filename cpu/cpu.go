// Package cpu provides CPU feature detection for SIMD operations.
package cpu

// Features contains detected CPU SIMD capabilities.
type Features struct {
	// x86/AMD64 features
	SSE       bool
	SSE2      bool
	SSE3      bool
	SSSE3     bool
	SSE41     bool
	SSE42     bool
	AVX       bool
	AVX2      bool
	AVX512F   bool
	AVX512VL  bool
	FMA       bool
	BMI1      bool
	BMI2      bool
	POPCNT    bool
	PCLMULQDQ bool // carry-less multiply (CLMUL) - used for CRC folding

	// ARM64 features
	NEON  bool
	FP16  bool // ARM64 half-precision floating point (FEAT_FP16)
	SVE   bool
	SVE2  bool
	PMULL bool // polynomial multiply (FEAT_PMULL) - used for CRC folding
}

// X86 contains x86/AMD64 CPU features (populated on amd64).
var X86 Features

// ARM64 contains ARM64 CPU features (populated on arm64).
var ARM64 Features

// HasAVX returns true if AVX is available.
func HasAVX() bool { return X86.AVX }

// HasAVX2 returns true if AVX2 is available.
func HasAVX2() bool { return X86.AVX2 }

// HasFMA returns true if FMA is available.
func HasFMA() bool { return X86.FMA }

// HasNEON returns true if ARM NEON is available.
func HasNEON() bool { return ARM64.NEON }

// HasPCLMULQDQ returns true if the x86 carry-less multiply instruction
// (PCLMULQDQ) is available. It is used to accelerate CRC folding.
func HasPCLMULQDQ() bool { return X86.PCLMULQDQ }

// HasPMULL returns true if the ARM64 polynomial multiply instruction (PMULL)
// is available. It is used to accelerate CRC folding.
func HasPMULL() bool { return ARM64.PMULL }

// HasFP16 returns true if ARM FP16 (half-precision) is available.
func HasFP16() bool { return ARM64.FP16 }

// HasAVX512VL returns true if AVX-512VL is available.
func HasAVX512VL() bool { return X86.AVX512VL }

// Info returns a string describing the available SIMD features.
func Info() string {
	return cpuInfo()
}
