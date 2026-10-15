package simd

// This test makes the AVX dispatch guard enforced rather than documented, the
// third axis next to AVX2 (asmcheck_amd64_gate_test.go) and FMA3
// (asmcheck_amd64_fma_test.go).
//
// The bug class is a kernel with a VEX or EVEX encoding dispatched without a
// condition that implies cpu.X86.AVX. On a CPU without AVX, or under an OS that
// has not enabled YMM state, the first such instruction faults with SIGILL.
// It has shipped before: birdnet-go #3353 reported interleave2AVX dispatched on
// length alone and faulting on a Celeron 887, fixed in #87
// (f32/interleave2_gate_amd64_test.go).
//
// A kernel counts as needing AVX when asmcheck.X86Kernel.NeedsAVX says so: a
// V-prefixed mnemonic (so a 128-bit VMOVD counts), a YMM or ZMM operand, or
// anything the level ladder puts above AVX1. AVX-512 kernels are included. The
// AVX2 axis skips them (see TestAmd64KernelDispatchRequiresAVX2); here they are
// checked because the init tier that selects them names AVX512F.
//
// This check does not cover VEX-encoded general-purpose instructions (BMI1 and
// BMI2), which need their own CPUID bits and not AVX, and like the other axes it
// cannot see a dispatch from another package: only <pkg>/*_amd64.go is parsed.

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/tphakala/simd/asmcheck"
)

// avxGateIdents are the identifiers that establish AVX, restricted to flags that
// imply it in this repo.
//
// AVX2, AVX512F and AVX512VL are detected by golang.org/x/sys/cpu v0.47.0 only
// when the OS supports AVX (cpu_x86.go:180 for AVX, :188 for AVX2, and :194-201
// for AVX512F and AVX512VL via osSupportsAVX512, which requires osSupportsAVX).
// SIMD_DISABLE=avx clears AVX2 and the AVX-512 flags with AVX (clearAVX calls
// clearAVX2 calls clearAVX512, cpu/cpu.go:166-188). AVXVNNI is defined as
// X86.AVX2 plus the VNNI bit (cpu/cpu_amd64.go:41), and F16C as X86.AVX plus its
// bit (cpu/cpu_amd64.go:33).
//
// FMA is deliberately absent. x/sys sets X86.HasFMA from the CPUID bit alone
// (cpu_x86.go:155), with none of the OS-support check HasAVX gets (:180), so
// cpu.X86.FMA can be true while cpu.X86.AVX is false.
var avxGateIdents = map[string]bool{
	"AVX": true, "AVX2": true, "AVX512F": true, "AVX512VL": true,
	"AVXVNNI": true, "F16C": true,
}

// avxAxis is the AVX dispatch axis.
var avxAxis = gateAxis{
	name:   "AVX",
	idents: avxGateIdents,
	remedy: "A CPU without AVX (Core 2, Nehalem and Westmere, and Celeron or Pentium " +
		"parts such as the Celeron 887 in birdnet-go #3353), or an OS that has not " +
		"enabled YMM state, faults with SIGILL on the first VEX instruction. Add the " +
		"cpu.X86.AVX conjunct to this branch, or move the kernel behind one.",
	firstLine: func(k asmcheck.X86Kernel) int { return k.AVX.Line },
	firstText: func(k asmcheck.X86Kernel) string { return k.AVX.Text + "  (" + k.AVX.Reason + ")" },
}

// TestAmd64KernelDispatchRequiresAVX asserts that every AMD64 kernel whose body
// contains a VEX or EVEX instruction is dispatched only from a branch whose
// condition requires AVX or a flag implying it.
func TestAmd64KernelDispatchRequiresAVX(t *testing.T) {
	parsed := map[string]*pkgFuncs{}
	checked := 0

	for _, file := range findAmd64Asm(t) {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		pkgDir := filepath.Dir(file)

		for _, k := range asmcheck.ScanX86Source(string(src)) {
			if !k.NeedsAVX() {
				continue
			}
			pf := parsed[pkgDir]
			if pf == nil {
				pf = parseAmd64Package(t, pkgDir)
				parsed[pkgDir] = pf
			}
			checked++
			checkKernelGated(t, file, pkgDir, k, pf, avxAxis)
		}
	}

	// A classifier that stopped recognising VEX would leave every assertion above
	// vacuous while the test still reported success.
	if checked == 0 {
		t.Error("no AVX-using kernels found in any amd64 assembly, but this repo has " +
			"many. The AVX classifier has stopped matching and this test is checking " +
			"nothing.")
	}
}

// TestAVXGateIdentsExcludeFMA pins the deliberate omission. FMA3 is a CPUID bit
// that x/sys reports without checking OS support for YMM state, so a branch
// testing only cpu.X86.FMA is not an AVX guard.
func TestAVXGateIdentsExcludeFMA(t *testing.T) {
	for _, ident := range []string{"FMA", "SSE2", "SSE41", "SSE42"} {
		if avxGateIdents[ident] {
			t.Errorf("avxGateIdents accepts %q as establishing AVX, but it does not imply it.", ident)
		}
	}
	if !avxGateIdents["AVX"] {
		t.Error("avxGateIdents does not accept AVX, so no branch can ever satisfy the " +
			"AVX dispatch check and TestAmd64KernelDispatchRequiresAVX would fail wholesale")
	}
}
