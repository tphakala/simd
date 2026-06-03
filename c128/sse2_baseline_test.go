//go:build amd64

package c128

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// postSSE2Mnemonics lists legacy (non-VEX) x86 instruction mnemonics that are
// NOT part of the SSE2 baseline: they require SSE3, SSSE3, SSE4.1 or SSE4.2.
// VEX-encoded forms (V-prefixed, e.g. VBLENDPD) are AVX and never appear in an
// SSE2 kernel, so exact-token matching against this set does not flag them.
var postSSE2Mnemonics = map[string]string{
	// SSE3
	"ADDSUBPD": "SSE3", "ADDSUBPS": "SSE3",
	"HADDPD": "SSE3", "HADDPS": "SSE3", "HSUBPD": "SSE3", "HSUBPS": "SSE3",
	"MOVDDUP": "SSE3", "MOVSHDUP": "SSE3", "MOVSLDUP": "SSE3",
	"LDDQU": "SSE3", "FISTTP": "SSE3",
	// SSSE3
	"PABSB": "SSSE3", "PABSW": "SSSE3", "PABSD": "SSSE3", "PALIGNR": "SSSE3",
	"PHADDW": "SSSE3", "PHADDD": "SSSE3", "PHADDSW": "SSSE3",
	"PHSUBW": "SSSE3", "PHSUBD": "SSSE3", "PHSUBSW": "SSSE3",
	"PMADDUBSW": "SSSE3", "PMULHRSW": "SSSE3", "PSHUFB": "SSSE3",
	"PSIGNB": "SSSE3", "PSIGNW": "SSSE3", "PSIGND": "SSSE3",
	// SSE4.1
	"BLENDPD": "SSE4.1", "BLENDPS": "SSE4.1", "BLENDVPD": "SSE4.1", "BLENDVPS": "SSE4.1",
	"DPPD": "SSE4.1", "DPPS": "SSE4.1", "EXTRACTPS": "SSE4.1", "INSERTPS": "SSE4.1",
	"MOVNTDQA": "SSE4.1", "MPSADBW": "SSE4.1", "PACKUSDW": "SSE4.1",
	"PBLENDVB": "SSE4.1", "PBLENDW": "SSE4.1", "PCMPEQQ": "SSE4.1",
	"PEXTRB": "SSE4.1", "PEXTRD": "SSE4.1", "PEXTRQ": "SSE4.1", "PEXTRW": "SSE4.1",
	"PHMINPOSUW": "SSE4.1", "PINSRB": "SSE4.1", "PINSRD": "SSE4.1", "PINSRQ": "SSE4.1",
	"PMAXSB": "SSE4.1", "PMAXSD": "SSE4.1", "PMAXUD": "SSE4.1", "PMAXUW": "SSE4.1",
	"PMINSB": "SSE4.1", "PMINSD": "SSE4.1", "PMINUD": "SSE4.1", "PMINUW": "SSE4.1",
	"PMOVSXBW": "SSE4.1", "PMOVSXBD": "SSE4.1", "PMOVSXBQ": "SSE4.1",
	"PMOVSXWD": "SSE4.1", "PMOVSXWQ": "SSE4.1", "PMOVSXDQ": "SSE4.1",
	"PMOVZXBW": "SSE4.1", "PMOVZXBD": "SSE4.1", "PMOVZXBQ": "SSE4.1",
	"PMOVZXWD": "SSE4.1", "PMOVZXWQ": "SSE4.1", "PMOVZXDQ": "SSE4.1",
	"PMULDQ": "SSE4.1", "PMULLD": "SSE4.1", "PTEST": "SSE4.1",
	"ROUNDPD": "SSE4.1", "ROUNDPS": "SSE4.1", "ROUNDSD": "SSE4.1", "ROUNDSS": "SSE4.1",
	// SSE4.2
	"PCMPGTQ": "SSE4.2", "PCMPESTRI": "SSE4.2", "PCMPESTRM": "SSE4.2",
	"PCMPISTRI": "SSE4.2", "PCMPISTRM": "SSE4.2", "CRC32": "SSE4.2", "POPCNT": "SSE4.2",
}

// TestSSE2KernelsAreBaselineSSE2 guards the c128 dispatch contract: c128_amd64.go
// routes the *SSE2 kernels through `case cpu.X86.SSE2` (genuine SSE2 baseline),
// unlike c64 which gates its similarly named kernels on SSE4.1. So every
// instruction in a c128 *SSE2 TEXT block must be SSE2 (or older). A post-SSE2
// instruction here SIGILLs on a CPU that lacks the feature (reproduced under
// `qemu-x86_64 -cpu Conroe`), which is the regression this test prevents.
func TestSSE2KernelsAreBaselineSSE2(t *testing.T) {
	// Locate the .s file relative to this test's source, not the working
	// directory: when this test runs from a precompiled binary (e.g. the
	// qemu cross-ISA CI leg, run from the repo root) the cwd is not the
	// package dir, so a bare relative open would spuriously fail.
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller(0) failed; cannot locate source dir")
	}
	asmFile := filepath.Join(filepath.Dir(thisFile), "c128_amd64.s")
	src, err := os.ReadFile(asmFile)
	if err != nil {
		t.Fatalf("read %s: %v", asmFile, err)
	}

	var curFunc string
	inSSE2 := false
	scanned := 0
	for lineNo, raw := range strings.Split(string(src), "\n") {
		line := raw
		// Strip line comment, then trim.
		if idx := strings.Index(line, "//"); idx >= 0 {
			line = line[:idx]
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if name, ok := textSymbol(line); ok {
			curFunc = name
			inSSE2 = strings.HasSuffix(name, "SSE2")
			if inSSE2 {
				scanned++
			}
			continue
		}
		if !inSSE2 {
			continue
		}

		mnemonic := strings.Fields(line)[0]
		if strings.HasSuffix(mnemonic, ":") { // label
			continue
		}
		// VEX/EVEX (V-prefixed) and AVX-512 opmask (K-prefixed) instructions
		// require AVX / AVX-512 and SIGILL on plain SSE2. No SSE2 mnemonic or
		// Plan9 pseudo-op starts with V or K, so a blanket prefix check catches
		// any AVX instruction the named denylist below would miss.
		switch {
		case strings.HasPrefix(mnemonic, "V"):
			t.Errorf("%s:%d: %s uses VEX/EVEX instruction %s; an SSE2 kernel must not use AVX (SIGILLs on CPUs without AVX)",
				asmFile, lineNo+1, curFunc, mnemonic)
		case strings.HasPrefix(mnemonic, "K"):
			t.Errorf("%s:%d: %s uses AVX-512 opmask instruction %s; an SSE2 kernel must not use AVX-512 (SIGILLs on CPUs without AVX-512)",
				asmFile, lineNo+1, curFunc, mnemonic)
		default:
			if isa, bad := postSSE2Mnemonics[mnemonic]; bad {
				t.Errorf("%s:%d: %s uses %s (%s); c128 dispatches *SSE2 kernels on plain SSE2, so this SIGILLs on CPUs without %s",
					asmFile, lineNo+1, curFunc, mnemonic, isa, isa)
			}
		}
	}
	if scanned == 0 {
		t.Fatalf("scanned no *SSE2 TEXT blocks in %s; parser likely broken", asmFile)
	}
}

// textSymbol returns the function name from a Plan9 `TEXT ·name(SB), ...` line.
func textSymbol(line string) (string, bool) {
	if !strings.HasPrefix(line, "TEXT") {
		return "", false
	}
	rest := strings.TrimSpace(strings.TrimPrefix(line, "TEXT"))
	rest = strings.TrimPrefix(rest, "·") // middle dot prefixes the symbol
	idx := strings.Index(rest, "(")
	if idx <= 0 {
		return "", false
	}
	return rest[:idx], true
}
