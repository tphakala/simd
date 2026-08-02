package asmcheck

// This file adds a static ISA-level check for the AMD64 assembly. For every TEXT
// symbol it reports the instructions that need more than AVX1, so a test can
// assert that a kernel's body never outruns the CPU feature its dispatch guard
// requires.
//
// The bug class it exists to catch is real and has shipped twice (#196, #197): a
// kernel named ...AVX, selected behind a plain cpu.X86.AVX guard, whose body in
// fact contains AVX2-only encodings. Such a kernel faults with #UD (SIGILL) on
// an AVX1-only CPU (Intel Sandy/Ivy Bridge, AMD Bulldozer) or on an
// AVX+FMA3-without-AVX2 CPU (AMD Piledriver, Steamroller). The trap is that the
// offending instruction
// often looks innocuous: VBROADCASTSS/VBROADCASTSD accept a register source only
// under AVX2, and every 256-bit integer operation is AVX2 because AVX1's YMM
// support is float-only.
//
// The check is pure source analysis: it needs no x86 hardware, no assembler and
// no build, so it runs everywhere the test suite does.

import (
	"regexp"
	"slices"
	"strings"
)

// X86Level is the minimum x86 SIMD feature level an instruction requires.
type X86Level int

const (
	// X86LevelAVX covers AVX1 and everything below it (SSE, scalar, and any
	// non-SIMD instruction). It is the zero value, so an unclassified
	// instruction never inflates a kernel's level.
	X86LevelAVX X86Level = iota
	// X86LevelAVX2 means the instruction was introduced by AVX2.
	X86LevelAVX2
	// X86LevelAVX512 means the instruction needs an AVX-512 extension.
	X86LevelAVX512
)

// String names the feature level as it appears in this repo's kernel suffixes.
func (l X86Level) String() string {
	switch l {
	case X86LevelAVX2:
		return "AVX2"
	case X86LevelAVX512:
		return "AVX512"
	default:
		return "AVX"
	}
}

// X86Feature names a CPUID feature bit that is ORTHOGONAL to the AVX level
// ladder, so it cannot be expressed as an X86Level. A CPU can have AVX1 with or
// without it, which is why it needs its own axis rather than a rung.
type X86Feature string

// X86FeatureFMA is FMA3, CPUID.01H:ECX.FMA[bit 12], reported by Go as
// cpu.X86.HasFMA. It is a separate bit from AVX: Sandy Bridge and Ivy Bridge
// have AVX and no FMA3, and this repo keeps an initAVXNoFMA dispatch tier for
// exactly those parts. A VFMADD-family instruction reaching such a CPU faults
// with #UD (SIGILL), the same failure #196 and #197 were about on the AVX2 axis.
const X86FeatureFMA X86Feature = "FMA"

// X86Use is one instruction that requires a feature level above AVX1, or a
// feature off the level ladder, or both.
type X86Use struct {
	Line    int        // 1-based line number in the scanned source
	Text    string     // the instruction as written, with any comment stripped
	Level   X86Level   // the level the instruction needs
	Feature X86Feature // an off-ladder feature it needs, or "" for none
	Reason  string     // why it needs that level or feature
}

// X86Kernel is one TEXT symbol together with the instructions in it that need
// more than plain AVX1.
//
// Off-ladder features are derived from Uses rather than stored alongside Level,
// so there is one place a feature can be recorded and no second field to keep in
// step with it.
type X86Kernel struct {
	Name  string   // symbol name, without the leading interpunct
	Line  int      // 1-based line of the TEXT directive
	Level X86Level // the highest level any instruction in the body needs
	Uses  []X86Use // qualifying instructions, in source order; nil when clean
}

// Needs reports whether the kernel's body uses an instruction requiring f.
func (k X86Kernel) Needs(f X86Feature) bool {
	return slices.ContainsFunc(k.Uses, func(u X86Use) bool { return u.Feature == f })
}

// FeatureUses returns the instructions in the kernel that require f, in source
// order.
func (k X86Kernel) FeatureUses(f X86Feature) []X86Use {
	out := make([]X86Use, 0, len(k.Uses))
	for _, u := range k.Uses {
		if u.Feature == f {
			out = append(out, u)
		}
	}
	return out
}

// avx2Mnemonics are AVX2-only in every VEX form, whatever their operands.
var avx2Mnemonics = map[string]bool{
	// Cross-lane permutes with an integer or full-width element selector.
	"VPERMPD": true, "VPERMQ": true, "VPERMD": true, "VPERMPS": true,
	"VPERM2I128": true, "VINSERTI128": true, "VEXTRACTI128": true,
	"VBROADCASTI128": true,
	// Integer broadcasts.
	"VPBROADCASTB": true, "VPBROADCASTW": true,
	"VPBROADCASTD": true, "VPBROADCASTQ": true,
	// Per-element variable shifts, integer blend, masked move.
	"VPSLLVD": true, "VPSLLVQ": true, "VPSRLVD": true, "VPSRLVQ": true,
	"VPSRAVD": true, "VPBLENDD": true, "VPMASKMOVD": true, "VPMASKMOVQ": true,
	// Gathers.
	"VPGATHERDD": true, "VPGATHERDQ": true, "VPGATHERQD": true, "VPGATHERQQ": true,
	"VGATHERDPS": true, "VGATHERDPD": true, "VGATHERQPS": true, "VGATHERQPD": true,
	// AVX2 256-bit integer ops whose mnemonic does not start with VP, so the
	// VP-on-YMM rule below cannot catch them. Both have a 128-bit SSE4.1 form.
	"VMPSADBW": true, "VMOVNTDQA": true,
}

// opaqueDirectives emit instruction bytes the assembler never sees as a
// mnemonic. This repo hand-encodes on amd64 deliberately: go1.26 assembles
// VPDPWSSD to the EVEX (AVX-512) form, which faults on Alder Lake, so i16's
// AVX-VNNI kernel spells it as BYTE directives instead (#169). Such bytes are
// opaque to source analysis, so treating them as AVX1-legal would let the
// repo's own documented workaround smuggle an arbitrarily high instruction into
// a kernel named for a lower tier. They report AVX2 instead: not because the
// bytes are known to be AVX2, but because a hand encoding always needs an
// explicit tier claim above plain AVX1.
var opaqueDirectives = map[string]bool{
	"BYTE": true, "WORD": true, "LONG": true, "QUAD": true,
}

// avx1YmmVPMnemonics are the only VP-prefixed instructions AVX1 defines on a YMM
// operand. AVX1's 256-bit support is float-only, so every other VP... form on a
// YMM register was introduced by AVX2. Keeping this as a closed whitelist (the
// AVX1 instruction set is frozen) makes the VP rule fail safe: a 256-bit integer
// operation nobody thought to enumerate is still reported. That safety is
// specific to the VP rule; see the scope note on X86LevelAVX for what the
// classifier as a whole does not model.
var avx1YmmVPMnemonics = map[string]bool{
	"VPERM2F128": true, // float lane permute
	"VPERMILPS":  true, // in-lane float permute
	"VPERMILPD":  true,
	"VPTEST":     true, // 256-bit form is AVX1
}

var (
	// textRe matches a TEXT directive and captures the symbol name. Go asm
	// spells the package qualifier with an interpunct; a bare "." is accepted
	// too so a hand-written or transformed source still parses.
	textRe = regexp.MustCompile(`^TEXT\s+[·.]([A-Za-z_]\w*)`)
	// instrRe captures an instruction mnemonic, any EVEX decorator suffix, and
	// the operand text. Go's assembler spells AVX-512 masking, broadcast and
	// embedded rounding as a dotted suffix (VADDPD.RN_SAE, VMOVUPS.Z), so the
	// suffix has to be part of the pattern; without it those lines match nothing
	// and are skipped as though they were not instructions at all.
	instrRe = regexp.MustCompile(`^([A-Z][A-Z0-9]*)((?:\.[A-Z0-9_]+)*)(?:\s+(.*))?$`)
	// ymmRe matches a YMM register operand.
	ymmRe = regexp.MustCompile(`\bY\d+\b`)
	// zmmRe matches a ZMM register operand.
	zmmRe = regexp.MustCompile(`\bZ\d+\b`)
	// maskOperandRe matches an operand that is exactly an AVX-512 opmask
	// register. It is applied per operand, not to the whole operand string, so
	// a symbol that merely happens to be named K1 cannot look like a mask.
	maskOperandRe = regexp.MustCompile(`^K[0-7]$`)
	// xmmOperandRe matches an operand that is exactly an XMM register.
	xmmOperandRe = regexp.MustCompile(`^X\d+$`)
	// fmaRe matches the FMA3 multiply-add family. The set is closed and regular:
	// an optional N for the negated forms, ADD or SUB, an optional second ADD or
	// SUB for the alternating VFMADDSUB/VFMSUBADD pair, one of the three operand
	// orders, then packed or scalar and single or double. Every member is FMA3,
	// in both its 128-bit and 256-bit form, so no operand inspection is needed:
	// unlike VBROADCASTSS, there is no FMA mnemonic with an AVX1-legal form.
	fmaRe = regexp.MustCompile(`^VF(N)?M(ADD|SUB)(ADD|SUB)?(132|213|231)[PS][SD]$`)
)

// noFeatureMnemonics are directives and instructions that carry no SIMD feature
// level, so they are classified without inspecting their operands.
var noFeatureMnemonics = map[string]bool{
	"TEXT": true, "DATA": true, "GLOBL": true, "FUNCDATA": true,
	"PCDATA": true, "RET": true, "NOP": true,
}

// stripAsmComment removes a trailing // comment and surrounding whitespace from
// one line of Go assembly.
func stripAsmComment(line string) string {
	if i := strings.Index(line, "//"); i >= 0 {
		line = line[:i]
	}
	return strings.TrimSpace(line)
}

// hasMaskOperand reports whether any operand is an AVX-512 opmask register.
func hasMaskOperand(operands string) bool {
	for op := range strings.SplitSeq(operands, ",") {
		if maskOperandRe.MatchString(strings.TrimSpace(op)) {
			return true
		}
	}
	return false
}

// x86InstrLevel classifies one AMD64 instruction, returning the feature level it
// needs and a short reason. Anything AVX1-legal, and anything that is not an
// instruction at all, reports X86LevelAVX with an empty reason.
func x86InstrLevel(instr string) (level X86Level, reason string) {
	m := instrRe.FindStringSubmatch(instr)
	if m == nil {
		return X86LevelAVX, ""
	}
	mnem, decorator, operands := m[1], m[2], m[3]
	if noFeatureMnemonics[mnem] {
		return X86LevelAVX, ""
	}
	if opaqueDirectives[mnem] {
		return X86LevelAVX2, "hand-encoded " + mnem + " directive (opaque to source analysis)"
	}
	if decorator != "" {
		return X86LevelAVX512, "EVEX decorator " + decorator
	}
	if zmmRe.MatchString(operands) {
		return X86LevelAVX512, "ZMM operand"
	}
	if hasMaskOperand(operands) {
		return X86LevelAVX512, "opmask operand"
	}
	if avx2Mnemonics[mnem] {
		return X86LevelAVX2, "AVX2-only mnemonic"
	}
	if strings.HasPrefix(mnem, "VP") && ymmRe.MatchString(operands) && !avx1YmmVPMnemonics[mnem] {
		return X86LevelAVX2, "256-bit integer operation (AVX1 YMM support is float-only)"
	}
	// VBROADCASTSS/VBROADCASTSD take an m32/m64 source under AVX1; the
	// register-source form was added by AVX2. Go assembly puts the source first.
	if mnem == "VBROADCASTSS" || mnem == "VBROADCASTSD" {
		if src, _, ok := strings.Cut(operands, ","); ok && xmmOperandRe.MatchString(strings.TrimSpace(src)) {
			return X86LevelAVX2, "register-source " + mnem + " (AVX1 defines only the memory-source form)"
		}
	}
	return X86LevelAVX, ""
}

// x86InstrFeature reports the off-ladder CPUID feature one AMD64 instruction
// needs, or "" if it needs none. It is deliberately independent of
// x86InstrLevel: an FMA3 instruction is AVX1-legal as far as the level ladder is
// concerned, and an AVX-512 kernel's EVEX-encoded FMA needs no separate claim
// because every AVX-512 part implements FMA3. What this catches is the middle
// case the ladder cannot express, an AVX1 CPU without FMA3.
func x86InstrFeature(instr string) (feature X86Feature, reason string) {
	m := instrRe.FindStringSubmatch(instr)
	if m == nil {
		return "", ""
	}
	if fmaRe.MatchString(m[1]) {
		return X86FeatureFMA, "FMA3 multiply-add (a CPUID bit separate from AVX)"
	}
	return "", ""
}

// ScanX86Source parses every TEXT symbol in an AMD64 assembly source and reports
// each one's required feature level together with the instructions that set it.
// Kernels are returned in source order. Content before the first TEXT directive
// (constant pools, includes) is ignored.
func ScanX86Source(src string) []X86Kernel {
	var out []X86Kernel
	cur := -1
	for i, raw := range strings.Split(src, "\n") {
		line := stripAsmComment(raw)
		if line == "" {
			continue
		}
		if m := textRe.FindStringSubmatch(line); m != nil {
			out = append(out, X86Kernel{Name: m[1], Line: i + 1})
			cur = len(out) - 1
			continue
		}
		if cur < 0 {
			continue
		}
		// Go assembly allows several statements on one line separated by ";",
		// which the hand-encoded byte runs in i16 use. Classifying the raw line
		// would see only the first statement and treat the rest as operands.
		for stmt := range strings.SplitSeq(line, ";") {
			stmt = strings.TrimSpace(stmt)
			if stmt == "" {
				continue
			}
			level, reason := x86InstrLevel(stmt)
			feature, freason := x86InstrFeature(stmt)
			if level == X86LevelAVX && feature == "" {
				continue
			}
			if reason == "" {
				reason = freason
			}
			k := &out[cur]
			k.Uses = append(k.Uses, X86Use{
				Line: i + 1, Text: stmt, Level: level, Feature: feature, Reason: reason,
			})
			if level > k.Level {
				k.Level = level
			}
		}
	}
	return out
}

// DeclaredX86Level reports the feature level a kernel's name claims, from its
// suffix. Names ending in AVX512 claim AVX-512 and names ending in AVX2 claim
// AVX2. AVXVNNI also claims AVX2: AVX-VNNI is the VEX-encoded form of the
// dot-product instructions and every part that implements it also implements
// AVX2. Everything else (an ...AVX, ...SSE2 or unsuffixed kernel) claims no more
// than AVX1.
func DeclaredX86Level(name string) X86Level {
	switch {
	case strings.HasSuffix(name, "AVX512"):
		return X86LevelAVX512
	case strings.HasSuffix(name, "AVX2"), strings.HasSuffix(name, "AVXVNNI"):
		return X86LevelAVX2
	default:
		return X86LevelAVX
	}
}
