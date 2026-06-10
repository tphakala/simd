// Package cpu provides CPU feature detection for SIMD operations.
package cpu

import "strings"

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
	F16C      bool // half<->single float conversion (VCVTPH2PS/VCVTPS2PH)

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

// HasF16C returns true if the x86 F16C half-precision conversion instructions
// (VCVTPH2PS / VCVTPS2PH) are available. They accelerate Float16 <-> float32
// slice conversion. F16C provides conversion only, not half-precision arithmetic.
func HasF16C() bool { return X86.F16C }

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

// applyDisable clears CPU feature flags in f according to the comma-separated,
// case-insensitive token list in spec (the value of the SIMD_DISABLE env var).
// Each token clears its own flag plus every flag that depends on it, so the
// resulting Features value never describes an impossible CPU (for example, AVX2
// set while AVX is cleared). Unknown and empty tokens are ignored: a library must
// not panic or write to stderr in response to environment input.
//
// Recognized tokens:
//
//	avx512     AVX512F, AVX512VL
//	avx2       AVX2 and the avx512 set
//	avx        AVX, FMA and the avx2 set
//	fma        FMA only
//	sse42      SSE42 and the avx set
//	sse41      SSE41 and the sse42 set
//	ssse3      SSSE3 and the sse41 set
//	sse3       SSE3 and the ssse3 set
//	pclmulqdq  PCLMULQDQ only
//	neon       NEON, FP16, SVE, SVE2, PMULL
//	fp16       FP16 only
//	sve        SVE, SVE2
//	pmull      PMULL only
//	all        every flag (forces the pure-Go path everywhere)
func applyDisable(f *Features, spec string) {
	if spec == "" {
		return // common case: SIMD_DISABLE unset, nothing to clear
	}
	for tok := range strings.SplitSeq(spec, ",") {
		switch strings.ToLower(strings.TrimSpace(tok)) {
		case "":
			// Empty token (e.g. trailing comma): ignore.
		case "avx512":
			clearAVX512(f)
		case "avx2":
			clearAVX2(f)
		case "avx":
			clearAVX(f)
		case "fma":
			f.FMA = false
		case "sse42":
			clearSSE42(f)
		case "sse41":
			clearSSE41(f)
		case "ssse3":
			clearSSSE3(f)
		case "sse3":
			clearSSE3(f)
		case "pclmulqdq":
			f.PCLMULQDQ = false
		case "neon":
			clearNEON(f)
		case "fp16":
			f.FP16 = false
		case "sve":
			clearSVE(f)
		case "pmull":
			f.PMULL = false
		case "all":
			*f = Features{}
		default:
			// Unknown token: ignore.
		}
	}
}

// The clearXxx helpers encode the x86 tier dependency chain: disabling a lower
// tier also disables every higher tier that requires it, because the runtime
// dispatch checks the highest tier first.

func clearAVX512(f *Features) {
	f.AVX512F = false
	f.AVX512VL = false
}

func clearAVX2(f *Features) {
	f.AVX2 = false
	clearAVX512(f)
}

func clearAVX(f *Features) {
	f.AVX = false
	f.FMA = false
	// F16C is VEX-encoded and only detected when AVX is present, so disabling the
	// AVX family must also drop the F16C conversion path back to pure Go. avx2/fma/
	// avx512 sit above AVX and do not cascade here, so they correctly leave it set.
	f.F16C = false
	clearAVX2(f)
}

func clearSSE42(f *Features) {
	f.SSE42 = false
	clearAVX(f)
}

func clearSSE41(f *Features) {
	f.SSE41 = false
	clearSSE42(f)
}

func clearSSSE3(f *Features) {
	f.SSSE3 = false
	clearSSE41(f)
}

func clearSSE3(f *Features) {
	f.SSE3 = false
	clearSSSE3(f)
}

// clearNEON disables the entire ARM64 SIMD feature set.
func clearNEON(f *Features) {
	f.NEON = false
	f.FP16 = false
	f.SVE = false
	f.SVE2 = false
	f.PMULL = false
}

// clearSVE disables the scalable-vector tiers without touching NEON.
func clearSVE(f *Features) {
	f.SVE = false
	f.SVE2 = false
}
