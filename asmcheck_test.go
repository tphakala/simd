package simd

import (
	"context"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/tphakala/simd/asmcheck"
)

// requireObjdumpEnv, when set in the environment, makes a missing aarch64
// objdump a hard failure instead of a skip-with-warning. CI sets it so the
// FP16 (.8H) cross-check cannot silently regress to "unchecked" if the
// binutils install ever breaks. Local runs without it stay green.
const requireObjdumpEnv = "SIMD_REQUIRE_OBJDUMP"

// findArm64Asm returns every *_arm64.s file under the module root.
func findArm64Asm(t *testing.T) []string {
	t.Helper()
	var files []string
	err := filepath.WalkDir(".", func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(p, "_arm64.s") {
			// Normalize to forward slashes so reporting is identical on
			// Windows (WalkDir yields backslashes).
			files = append(files, filepath.ToSlash(p))
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	if len(files) == 0 {
		t.Fatal("no *_arm64.s files found; is the test running from the module root?")
	}
	return files
}

// objdumpDirective is a WORD directive that golang.org/x/arch/arm64asm cannot
// decode (an ARMv8.2 half-precision .8H FP16 SIMD instruction, or a FEAT_DotProd
// SDOT/UDOT dot product), deferred to the objdump cross-check.
type objdumpDirective struct {
	line    int
	hex     uint32
	comment string
}

// deferredToObjdump reports whether a WORD directive that arm64asm cannot decode
// is nonetheless a sanctioned encoding to be cross-checked with objdump: an
// ARMv8.2 FP16 (.8H) SIMD instruction, or a FEAT_DotProd dot product (SDOT/UDOT,
// which arm64asm does not know). Anything else is treated as a real error.
func deferredToObjdump(comment string) bool {
	u := strings.ToUpper(comment)
	if strings.Contains(u, ".8H") {
		return true
	}
	mnem, _, _ := strings.Cut(strings.TrimSpace(u), " ")
	return mnem == "SDOT" || mnem == "UDOT"
}

// TestArm64WordEncodings decodes every hand-encoded WORD directive in the ARM64
// assembly and asserts it matches the instruction named in its comment.
// Instructions arm64asm can decode are checked directly; ARMv8.2 FP16 (.8H) and
// FEAT_DotProd (SDOT/UDOT) instructions, which it cannot decode, are
// cross-checked with an aarch64 objdump when one is available. Without objdump
// these directives are accepted unchecked (so the test stays green on machines
// lacking cross binutils) unless SIMD_REQUIRE_OBJDUMP is set, which CI does.
func TestArm64WordEncodings(t *testing.T) {
	tool := asmcheck.FindObjdump()
	if os.Getenv(requireObjdumpEnv) != "" && tool == "" {
		t.Fatalf("%s is set but no aarch64-capable objdump was found; install binutils-aarch64-linux-gnu", requireObjdumpEnv)
	}
	if tool == "" {
		t.Logf("no aarch64 objdump found; FP16 (.8H) encodings are accepted unchecked. "+
			"Install binutils-aarch64-linux-gnu, or set %s=1 to require the cross-check.", requireObjdumpEnv)
	}

	for _, file := range findArm64Asm(t) {
		checkArm64File(t, file, tool)
	}
}

// checkArm64File validates every WORD directive in one assembly file.
func checkArm64File(t *testing.T, file, tool string) {
	t.Helper()
	src, err := os.ReadFile(file)
	if err != nil {
		t.Fatalf("read %s: %v", file, err)
	}
	directives := asmcheck.ScanSource(string(src))

	// matched holds hexes proven to match their comment, whether decoded by
	// arm64asm or cross-checked by objdump. Uncommented repeats of a matched
	// hex are then accepted.
	matched := map[uint32]bool{}
	var deferred []objdumpDirective

	// Pass 1: every commented directive. Decodable ones are verified now;
	// undecodable ones (FP16 .8H, or SDOT/UDOT) are deferred to the objdump
	// cross-check.
	for _, d := range directives {
		if d.Source == asmcheck.NoComment {
			continue
		}
		res := asmcheck.Verify(d.Hex, d.Comment)
		switch res.Status {
		case asmcheck.Match:
			matched[d.Hex] = true
		case asmcheck.Mismatch:
			t.Errorf("%s:%d  0x%08X  claims=%q  decodes=%q", file, d.Line, d.Hex, res.Claimed, res.Decoded)
		case asmcheck.Undecodable:
			// ARMv8.2 FP16 (.8H) SIMD and FEAT_DotProd (SDOT/UDOT) are the only
			// sanctioned reasons arm64asm cannot decode a directive here; both are
			// cross-checked with objdump. Any other undecodable WORD (a malformed
			// encoding, or a future extension) is a real problem: fail loudly
			// instead of funneling it into the objdump fallback, which is lenient
			// when no objdump is installed.
			if !deferredToObjdump(d.Comment) {
				t.Errorf("%s:%d  0x%08X  undecodable WORD that is neither FP16 (.8H) nor DotProd (SDOT/UDOT): %q", file, d.Line, d.Hex, d.Comment)
				continue
			}
			deferred = append(deferred, objdumpDirective{line: d.Line, hex: d.Hex, comment: d.Comment})
		}
	}

	crossCheckObjdump(t, file, tool, deferred, matched)

	// Pass 2: uncommented directives must reuse a hex proven above.
	for _, d := range directives {
		if d.Source != asmcheck.NoComment {
			continue
		}
		if matched[d.Hex] {
			continue
		}
		t.Errorf("%s:%d  0x%08X  uncommented WORD, cannot validate (add a comment naming the instruction)", file, d.Line, d.Hex)
	}
}

// crossCheckObjdump verifies directives that arm64asm cannot decode (FP16 .8H,
// or SDOT/UDOT) by disassembling them with aarch64 objdump and comparing against
// their comments. When no objdump is available it accepts them (marking each hex
// matched so uncommented repeats pass) and relies on the warning and
// SIMD_REQUIRE_OBJDUMP gate in TestArm64WordEncodings.
func crossCheckObjdump(t *testing.T, file, tool string, deferred []objdumpDirective, matched map[uint32]bool) {
	t.Helper()
	if len(deferred) == 0 {
		return
	}
	if tool == "" {
		for _, d := range deferred {
			matched[d.hex] = true
		}
		return
	}

	hexes := make([]uint32, 0, len(deferred))
	for _, d := range deferred {
		hexes = append(hexes, d.hex)
	}
	decoded, err := asmcheck.DisassembleWords(context.Background(), tool, hexes)
	if err != nil {
		t.Fatalf("objdump cross-check of %s failed: %v", file, err)
	}

	for _, d := range deferred {
		got, ok := decoded[d.hex]
		if !ok {
			t.Errorf("%s:%d  0x%08X  objdump produced no disassembly", file, d.line, d.hex)
			continue
		}
		if res := asmcheck.VerifyDecoded(got, d.comment); res.Status == asmcheck.Match {
			matched[d.hex] = true
		} else {
			t.Errorf("%s:%d  0x%08X  claims=%q  objdump=%q", file, d.line, d.hex, res.Claimed, res.Decoded)
		}
	}
}

// TestNoUncheckedAmd64Encodings is a warn-only tripwire: amd64 currently has no
// hand-encoded instructions. If any appear (for example from AVX-512 work that
// the Go assembler cannot express as mnemonics), this logs them so they can be
// brought under a decoder-based check too. It does not fail the build.
func TestNoUncheckedAmd64Encodings(t *testing.T) {
	var hits []string
	err := filepath.WalkDir(".", func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(p, "_amd64.s") {
			return nil
		}
		src, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		for i, line := range strings.Split(string(src), "\n") {
			trimmed := strings.TrimSpace(line)
			for _, kw := range []string{"BYTE $0x", "WORD $0x", "LONG $0x", "QUAD $0x"} {
				if strings.HasPrefix(trimmed, kw) {
					hits = append(hits, p+":"+strconv.Itoa(i+1)+"  "+trimmed)
				}
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	if len(hits) > 0 {
		t.Logf("WARNING: %d hand-encoded amd64 directive(s) are not decoder-checked:\n  %s",
			len(hits), strings.Join(hits, "\n  "))
	}
}

// TestNoGoroutineRegisterClobber fails if any hand-written assembly uses the
// register that holds the current goroutine pointer g as an instruction
// operand. Clobbering g is the single biggest footgun in hand-written Go
// assembly (see CLAUDE.md): a latent, GC-time crash the moment the function
// gains a CALL into the runtime or a stack map. This is the regression guard
// for the hazard fixed in #57 (ConvolveDecimate) and #58 (realFFTUnpackAVX):
// scratch math must live in a non-reserved register (e.g. BX on amd64).
//
// Only operands are flagged; comments that name the register for documentation
// (the "BX (not R14) ..." convention notes) are ignored, so those reminders
// stay legal. R15 on amd64 is intentionally not flagged: it is sanctioned
// static scratch in this repo (see CLAUDE.md), unlike g.
func TestNoGoroutineRegisterClobber(t *testing.T) {
	// g lives in R14 on amd64 and R28 on arm64 (W28 is the 32-bit view; a
	// write to it zero-extends over g all the same).
	gReg := []struct {
		suffix string
		name   string
		re     *regexp.Regexp
	}{
		{"_amd64.s", "R14", regexp.MustCompile(`\bR14\b`)},
		{"_arm64.s", "R28/W28", regexp.MustCompile(`\b[RW]28\b`)},
	}

	var violations []string
	err := filepath.WalkDir(".", func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		// Normalize to forward slashes at the boundary so the suffix checks and
		// reporting are identical on Windows (WalkDir yields backslashes).
		p = filepath.ToSlash(p)
		if d.IsDir() || !strings.HasSuffix(p, ".s") {
			return nil
		}
		var name string
		var re *regexp.Regexp
		for _, g := range gReg {
			if strings.HasSuffix(p, g.suffix) {
				name, re = g.name, g.re
				break
			}
		}
		if re == nil {
			return nil
		}
		src, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		for i, line := range strings.Split(string(src), "\n") {
			code := line
			// Drop the line comment; documentation references to the register
			// are fine, only operands are a hazard.
			if j := strings.Index(code, "//"); j >= 0 {
				code = code[:j]
			}
			if re.MatchString(code) {
				violations = append(violations, p+":"+strconv.Itoa(i+1)+
					"  uses "+name+" (goroutine g):  "+strings.TrimSpace(line))
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk: %v", err)
	}
	if len(violations) > 0 {
		t.Errorf("goroutine-pointer register used as an operand in %d location(s); move scratch "+
			"to a non-reserved register (BX on amd64) and keep g untouched, see CLAUDE.md:\n  %s",
			len(violations), strings.Join(violations, "\n  "))
	}
}

// singleRoundingKernel names a kernel whose scalar reference computes
// float32(a*b)+c as two separate roundings (the product rounds to float32 before
// the add). Its body must emit a distinct multiply and add, never a fused
// multiply-add: a consumer that reproduces the reference bit-for-bit (go-aac's
// quantize path, #155) depends on the intermediate rounding. See the f32 package
// doc and #156.
type singleRoundingKernel struct {
	file, fn string
	mul, add string // the two separate mnemonics that must both appear
}

var singleRoundingKernels = []singleRoundingKernel{
	{"f32/f32_amd64.s", "float32ToInt32ScaleClampAVX", "VMULPS", "VADDPS"},
	{"f32/f32_arm64.s", "float32ToInt32ScaleClampNEON", "FMUL", "FADD"},
}

// asmFuncBody returns the lines of the TEXT ·fn(...) block, from its TEXT line to
// the next TEXT line (or EOF). The header doc comment sits above the TEXT line
// and is deliberately excluded, so a comment mentioning "VFMADD" or "FMLA" (as
// the no-FMA kernels' own comments do) is not scanned.
func asmFuncBody(src, fn string) ([]string, bool) {
	lines := strings.Split(src, "\n")
	start := -1
	for i, l := range lines {
		if strings.HasPrefix(strings.TrimSpace(l), "TEXT ·"+fn+"(") {
			start = i
			break
		}
	}
	if start < 0 {
		return nil, false
	}
	end := len(lines)
	for i := start + 1; i < len(lines); i++ {
		if strings.HasPrefix(strings.TrimSpace(lines[i]), "TEXT ·") {
			end = i
			break
		}
	}
	return lines[start:end], true
}

// TestNoFMAContract enforces the f32 single-rounding contract (#156): the
// listed kernels must contain a separate multiply and add and no fused
// multiply-add. amd64 instructions are read as mnemonics; arm64 vector-float
// ops are WORD-encoded, so each WORD is decoded through arm64asm and the decoded
// mnemonic is checked (a WORD-encoded FMLA is caught even though the source text
// is a hex literal).
func TestNoFMAContract(t *testing.T) {
	// VFMADD/VFNMADD/VFMSUB/VFNMSUB (amd64); FMADD/FMSUB/FNMADD/FNMSUB and the
	// FMLA/FMLS vector forms (arm64). Deliberately excludes FMUL/FADD/FMAX/FMIN.
	fmaRe := regexp.MustCompile(`^(VFN?M(ADD|SUB)|FN?M(ADD|SUB)|FML[AS])`)

	for _, k := range singleRoundingKernels {
		src, err := os.ReadFile(k.file)
		if err != nil {
			t.Fatalf("%s: %v", k.file, err)
		}
		body, ok := asmFuncBody(string(src), k.fn)
		if !ok {
			t.Fatalf("%s: TEXT ·%s not found (renamed? update singleRoundingKernels)", k.file, k.fn)
		}

		var mulSeen, addSeen bool
		for _, line := range body {
			// Resolve the effective mnemonic: the decoded op for a WORD directive,
			// otherwise the first token of the (comment-stripped) instruction.
			var mnem string
			if hex, _, isWord := asmcheck.ParseWordLine(line); isWord {
				dec, derr := asmcheck.Decode(hex)
				if derr != nil {
					continue // undecodable words are TestArm64WordEncodings' job
				}
				if f := strings.Fields(dec); len(f) > 0 {
					mnem = strings.ToUpper(f[0])
				}
			} else {
				code := line
				if idx := strings.Index(code, "//"); idx >= 0 {
					code = code[:idx]
				}
				if f := strings.Fields(code); len(f) > 0 {
					mnem = strings.ToUpper(f[0])
				}
			}
			if mnem == "" {
				continue
			}
			if fmaRe.MatchString(mnem) {
				t.Errorf("%s ·%s: forbidden fused %s in a single-rounding kernel; emit a "+
					"separate %s then %s so the product rounds to float32 first (see #156)",
					k.file, k.fn, mnem, k.mul, k.add)
			}
			switch mnem {
			case k.mul:
				mulSeen = true
			case k.add:
				addSeen = true
			}
		}
		if !mulSeen || !addSeen {
			t.Errorf("%s ·%s: the two-rounding multiply+add must be present and unfused, "+
				"but got %s=%v %s=%v (see #156)", k.file, k.fn, k.mul, mulSeen, k.add, addSeen)
		}
	}
}
