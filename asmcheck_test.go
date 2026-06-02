package simd

import (
	"context"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/tphakala/simd/internal/asmencoding"
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

// fp16Directive is a WORD directive that golang.org/x/arch/arm64asm cannot
// decode (an ARMv8.2 half-precision .8H FP16 SIMD instruction), deferred to the
// objdump cross-check.
type fp16Directive struct {
	line    int
	hex     uint32
	comment string
}

// TestArm64WordEncodings decodes every hand-encoded WORD directive in the ARM64
// assembly and asserts it matches the instruction named in its comment.
// Instructions arm64asm can decode are checked directly; ARMv8.2 FP16 (.8H)
// instructions, which it cannot decode, are cross-checked with an aarch64
// objdump when one is available. Without objdump the FP16 directives are
// accepted unchecked (so the test stays green on machines lacking cross
// binutils) unless SIMD_REQUIRE_OBJDUMP is set, which CI does.
func TestArm64WordEncodings(t *testing.T) {
	tool := asmencoding.FindObjdump()
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
	directives := asmencoding.ScanSource(string(src))

	// matched holds hexes proven to match their comment, whether decoded by
	// arm64asm or cross-checked by objdump. Uncommented repeats of a matched
	// hex are then accepted.
	matched := map[uint32]bool{}
	var fp16 []fp16Directive

	// Pass 1: every commented directive. Decodable ones are verified now;
	// undecodable ones (FP16 .8H) are deferred to the objdump cross-check.
	for _, d := range directives {
		if d.Source == asmencoding.NoComment {
			continue
		}
		res := asmencoding.Verify(d.Hex, d.Comment)
		switch res.Status {
		case asmencoding.Match:
			matched[d.Hex] = true
		case asmencoding.Mismatch:
			t.Errorf("%s:%d  0x%08X  claims=%q  decodes=%q", file, d.Line, d.Hex, res.Claimed, res.Decoded)
		case asmencoding.Undecodable:
			// ARMv8.2 FP16 (.8H) SIMD is the only sanctioned reason arm64asm
			// cannot decode a directive here. Any other undecodable WORD (a
			// malformed encoding, or a future extension) is a real problem:
			// fail loudly instead of funneling it into the objdump fallback,
			// which is lenient when no objdump is installed.
			if !strings.Contains(strings.ToUpper(d.Comment), ".8H") {
				t.Errorf("%s:%d  0x%08X  undecodable WORD that is not FP16 (.8H): %q", file, d.Line, d.Hex, d.Comment)
				continue
			}
			fp16 = append(fp16, fp16Directive{line: d.Line, hex: d.Hex, comment: d.Comment})
		}
	}

	crossCheckFP16(t, file, tool, fp16, matched)

	// Pass 2: uncommented directives must reuse a hex proven above.
	for _, d := range directives {
		if d.Source != asmencoding.NoComment {
			continue
		}
		if matched[d.Hex] {
			continue
		}
		t.Errorf("%s:%d  0x%08X  uncommented WORD, cannot validate (add a comment naming the instruction)", file, d.Line, d.Hex)
	}
}

// crossCheckFP16 verifies FP16 (.8H) directives that arm64asm cannot decode by
// disassembling them with aarch64 objdump and comparing against their comments.
// When no objdump is available it accepts them (marking each hex matched so
// uncommented repeats pass) and relies on the warning and SIMD_REQUIRE_OBJDUMP
// gate in TestArm64WordEncodings.
func crossCheckFP16(t *testing.T, file, tool string, fp16 []fp16Directive, matched map[uint32]bool) {
	t.Helper()
	if len(fp16) == 0 {
		return
	}
	if tool == "" {
		for _, d := range fp16 {
			matched[d.hex] = true
		}
		return
	}

	hexes := make([]uint32, 0, len(fp16))
	for _, d := range fp16 {
		hexes = append(hexes, d.hex)
	}
	decoded, err := asmencoding.DisassembleWords(context.Background(), tool, hexes)
	if err != nil {
		t.Fatalf("objdump cross-check of %s failed: %v", file, err)
	}

	for _, d := range fp16 {
		got, ok := decoded[d.hex]
		if !ok {
			t.Errorf("%s:%d  0x%08X  objdump produced no disassembly", file, d.line, d.hex)
			continue
		}
		if res := asmencoding.VerifyDecoded(got, d.comment); res.Status == asmencoding.Match {
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
				violations = append(violations, filepath.ToSlash(p)+":"+strconv.Itoa(i+1)+
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
