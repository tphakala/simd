package asmcheck

import "testing"

func TestX86InstrLevelClassification(t *testing.T) {
	cases := []struct {
		instr string
		want  X86Level
	}{
		// AVX1-legal, or not SIMD at all.
		{"VMOVUPS 0(SI), Y0", X86LevelAVX},
		{"VDIVPS Y0, Y7, Y0", X86LevelAVX},
		{"VMULPD Y1, Y2, Y3", X86LevelAVX},
		{"VBROADCASTSS recip32_one<>(SB), Y7", X86LevelAVX},
		{"VBROADCASTSD s+48(FP), Y1", X86LevelAVX},
		{"VBROADCASTSS alpha+24(FP), Y3", X86LevelAVX},
		{"VPERM2F128 $0x01, Y0, Y1, Y2", X86LevelAVX},
		{"VPERMILPS $0x1B, Y0, Y1", X86LevelAVX},
		{"VPTEST Y0, Y1", X86LevelAVX},
		{"VPERMILPD $0x5, Y0, Y1", X86LevelAVX},
		{"VINSERTF128 $1, X1, Y0, Y0", X86LevelAVX},
		{"VEXTRACTF128 $1, Y0, X1", X86LevelAVX},
		{"VPAND X1, X2, X3", X86LevelAVX}, // 128-bit integer VEX op is AVX1
		{"MOVQ dst_base+0(FP), DX", X86LevelAVX},
		{"RET", X86LevelAVX},
		{"recip32_avx_loop32:", X86LevelAVX},

		// AVX2: register-source broadcast.
		{"VBROADCASTSS X7, Y7", X86LevelAVX2},
		{"VBROADCASTSD X3, Y3", X86LevelAVX2},

		// AVX2: 256-bit integer operations.
		{"VPADDD Y10, Y4, Y4", X86LevelAVX2},
		{"VPCMPEQD Y14, Y14, Y14", X86LevelAVX2},
		{"VPSLLD $31, Y14, Y14", X86LevelAVX2},
		{"VPSUBQ log64_off<>(SB), Y0, Y2", X86LevelAVX2},
		{"VPMOVSXWD (SI), Y0", X86LevelAVX2},
		{"VPABSB (SI), Y1", X86LevelAVX2},

		// AVX2: mnemonics that are AVX2-only in every form.
		{"VPERMPD $0x1B, Y2, Y2", X86LevelAVX2},
		{"VPERMD Y3, Y9, Y3", X86LevelAVX2},
		{"VPERMPS Y0, Y3, Y12", X86LevelAVX2},
		{"VPBLENDD $0xAA, Y5, Y4, Y6", X86LevelAVX2},
		{"VEXTRACTI128 $1, Y0, X3", X86LevelAVX2},
		{"VPBROADCASTB lo+48(FP), Y3", X86LevelAVX2},

		// AVX2: 256-bit integer ops whose mnemonic does not start with VP, so
		// only the explicit list can catch them.
		{"VMPSADBW $0, Y1, Y2, Y3", X86LevelAVX2},
		{"VMOVNTDQA (SI), Y0", X86LevelAVX2},

		// Hand-encoded bytes are opaque to source analysis, so they must not
		// read as AVX1-legal. This is how i16 spells AVX-VNNI VPDPWSSD (#169).
		{"BYTE $0xC4", X86LevelAVX2},
		{"WORD $0x1234", X86LevelAVX2},

		// AVX-512: ZMM or opmask operands.
		{"VPBROADCASTD AX, Z7", X86LevelAVX512},
		{"VMOVUPS 0(SI), Z0", X86LevelAVX512},
		{"VDIVPS Z0, Z7, Z0", X86LevelAVX512},
		{"VMOVUPS Y0, Y1, K1", X86LevelAVX512},

		// AVX-512: an EVEX decorator is AVX-512 even with X/Y operands. Without
		// the decorator in the mnemonic pattern these lines match nothing and
		// are skipped as though they were not instructions.
		{"VADDPD.RN_SAE Z1, Z2, Z3", X86LevelAVX512},
		{"VMOVUPS.Z (SI), K1, Y0", X86LevelAVX512},
		{"VADDPS.BCST (SI), Y1, Y2", X86LevelAVX512},

		// A symbol that merely happens to be named K1 is not an opmask operand.
		{"MOVOU K1<>(SB), X1", X86LevelAVX},
		{"VMOVUPS zmm_const<>(SB), Y1", X86LevelAVX},
	}
	for _, c := range cases {
		got, reason := x86InstrLevel(c.instr)
		if got != c.want {
			t.Errorf("x86InstrLevel(%q) = %v (%s), want %v", c.instr, got, reason, c.want)
		}
		if got != X86LevelAVX && reason == "" {
			t.Errorf("x86InstrLevel(%q) reported %v with no reason", c.instr, got)
		}
	}
}

// TestX86InstrLevelIgnoresRegisterNamesInMnemonic guards against a classifier
// that looks for a YMM register anywhere in the line: only operands count, so a
// float mnemonic on YMM registers must stay AVX1.
func TestX86InstrLevelClassificationIgnoresRegisterNamesInMnemonic(t *testing.T) {
	for _, instr := range []string{"VADDPS Y0, Y1, Y2", "VXORPD Y3, Y3, Y3", "VSHUFPS $0xB1, Y1, Y1, Y4"} {
		if got, _ := x86InstrLevel(instr); got != X86LevelAVX {
			t.Errorf("x86InstrLevel(%q) = %v, want %v", instr, got, X86LevelAVX)
		}
	}
}

const scanFixture = `#include "textflag.h"

DATA one<>+0x00(SB)/4, $0x3f800000
GLOBL one<>(SB), RODATA|NOPTR, $4

TEXT ·cleanAVX(SB), NOSPLIT, $0-48
    VBROADCASTSS one<>(SB), Y7   // VBROADCASTSS X7, Y7 would be AVX2
    VDIVPS Y0, Y7, Y0
    RET

TEXT ·dirtyAVX(SB), NOSPLIT, $0-48
    VBROADCASTSS X7, Y7
    VPADDD Y1, Y2, Y3
    RET

TEXT ·wideAVX512(SB), NOSPLIT, $0-48
    VPBROADCASTD AX, Z7
    RET
`

func TestScanX86Source(t *testing.T) {
	kernels := ScanX86Source(scanFixture)
	if len(kernels) != 3 {
		t.Fatalf("got %d kernels, want 3: %+v", len(kernels), kernels)
	}

	if got := kernels[0]; got.Name != "cleanAVX" || got.Level != X86LevelAVX || len(got.Uses) != 0 {
		t.Errorf("kernel 0 = %+v, want cleanAVX at AVX with no uses "+
			"(the comment naming an AVX2 form must be stripped)", got)
	}

	dirty := kernels[1]
	if dirty.Name != "dirtyAVX" || dirty.Level != X86LevelAVX2 {
		t.Errorf("kernel 1 = %+v, want dirtyAVX at AVX2", dirty)
	}
	if len(dirty.Uses) != 2 {
		t.Fatalf("dirtyAVX uses = %+v, want 2", dirty.Uses)
	}
	// Line numbers are 1-based and point at the offending instruction.
	if dirty.Uses[0].Line != 12 || dirty.Uses[1].Line != 13 {
		t.Errorf("dirtyAVX use lines = %d, %d; want 12, 13", dirty.Uses[0].Line, dirty.Uses[1].Line)
	}

	if got := kernels[2]; got.Name != "wideAVX512" || got.Level != X86LevelAVX512 {
		t.Errorf("kernel 2 = %+v, want wideAVX512 at AVX512", got)
	}
}

// TestScanX86SourceEmpty checks a source with no TEXT symbols, and that
// instructions appearing before the first TEXT are not attributed to a kernel.
func TestScanX86SourceEmpty(t *testing.T) {
	if got := ScanX86Source("#include \"textflag.h\"\n    VPADDD Y1, Y2, Y3\n"); len(got) != 0 {
		t.Errorf("ScanX86Source with no TEXT = %+v, want none", got)
	}
}

func TestDeclaredX86Level(t *testing.T) {
	cases := map[string]X86Level{
		"reciprocalAVX":    X86LevelAVX,
		"addScaledSSE":     X86LevelAVX,
		"interleave2_32":   X86LevelAVX,
		"minMaxAVX2":       X86LevelAVX2,
		"xcorr4AVXVNNI":    X86LevelAVX2,
		"addScaledAVX512":  X86LevelAVX512,
		"dotProductAVX512": X86LevelAVX512,
	}
	for name, want := range cases {
		if got := DeclaredX86Level(name); got != want {
			t.Errorf("DeclaredX86Level(%q) = %v, want %v", name, got, want)
		}
	}
}

func TestX86LevelString(t *testing.T) {
	cases := map[X86Level]string{
		X86LevelAVX:    "AVX",
		X86LevelAVX2:   "AVX2",
		X86LevelAVX512: "AVX512",
	}
	for level, want := range cases {
		if got := level.String(); got != want {
			t.Errorf("X86Level(%d).String() = %q, want %q", int(level), got, want)
		}
	}
}

func TestStripAsmCommentHelper(t *testing.T) {
	cases := map[string]string{
		"    VPADDD Y1, Y2, Y3   // comment": "VPADDD Y1, Y2, Y3",
		"// whole line":                      "",
		"    VZEROUPPER":                     "VZEROUPPER",
		"":                                   "",
	}
	for in, want := range cases {
		if got := stripAsmComment(in); got != want {
			t.Errorf("stripAsmComment(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestScanX86SourceTakesLevelMaximum pins that a kernel reports the HIGHEST
// level any instruction needs, not the last one seen. Without the maximum, an
// AVX-512 instruction followed by an AVX2 one would report AVX2, so an
// ...AVX2-named kernel containing an AVX-512 instruction would pass.
func TestScanX86SourceTakesLevelMaximum(t *testing.T) {
	const src = `TEXT ·mixedAVX2(SB), NOSPLIT, $0-0
    VPBROADCASTD AX, Z7
    VPADDD Y1, Y2, Y3
    RET
`
	kernels := ScanX86Source(src)
	if len(kernels) != 1 {
		t.Fatalf("got %d kernels, want 1", len(kernels))
	}
	if kernels[0].Level != X86LevelAVX512 {
		t.Errorf("Level = %v, want %v (the AVX-512 use must not be overwritten by "+
			"the later AVX2 one)", kernels[0].Level, X86LevelAVX512)
	}
	if len(kernels[0].Uses) != 2 {
		t.Errorf("Uses = %d, want 2", len(kernels[0].Uses))
	}
}

// TestScanX86SourceSplitsStatements pins that several statements on one line,
// separated by ";", are each classified. i16 writes its hand-encoded AVX-VNNI
// instruction that way, so classifying only the first statement would hide the
// rest as though they were operands.
func TestScanX86SourceSplitsStatements(t *testing.T) {
	const src = `TEXT ·joinedAVX(SB), NOSPLIT, $0-0
    MOVQ AX, BX; VPADDD Y1, Y2, Y3
    RET
`
	kernels := ScanX86Source(src)
	if len(kernels) != 1 {
		t.Fatalf("got %d kernels, want 1", len(kernels))
	}
	if kernels[0].Level != X86LevelAVX2 {
		t.Errorf("Level = %v, want %v; the second statement on the line was not "+
			"classified", kernels[0].Level, X86LevelAVX2)
	}
}
