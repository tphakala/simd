package simd

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tphakala/simd/internal/asmencoding"
)

// asmException documents a WORD directive the checker cannot mechanically
// verify, with the reason it is exempt. The current exemptions are all ARMv8.2
// half-precision (.8H) SIMD instructions, which golang.org/x/arch/arm64asm
// cannot decode. These are covered by the on-hardware numeric tests instead.
type asmException struct {
	file   string // path suffix relative to the module root
	hex    uint32
	expect string // intended instruction, for human reference
	reason string
}

// fp16Reason applies to every entry below: golang.org/x/arch/arm64asm cannot
// decode ARMv8.2 half-precision (.8H) SIMD instructions. Each encoding here was
// instead verified with aarch64 GNU objdump (binutils 2.44) to match its
// comment, and is exercised by the on-hardware numeric tests in f16.
const fp16Reason = "ARMv8.2 FP16 (.8H) SIMD; not decodable by arm64asm; verified via aarch64 objdump"

var asmAllowlist = []asmException{
	{"f16/f16_arm64.s", 0x4E403484, "FMAX V4.8H, V4.8H, V0.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x4E410C02, "FMLA V2.8H, V0.8H, V1.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x4E411400, "FADD V0.8H, V0.8H, V1.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x4E411402, "FADD V2.8H, V0.8H, V1.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x4E413402, "FMAX V2.8H, V0.8H, V1.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x4E421400, "FADD V0.8H, V0.8H, V2.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x4E423401, "FMAX V1.8H, V0.8H, V2.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x4EC03484, "FMIN V4.8H, V4.8H, V0.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x4EC11402, "FSUB V2.8H, V0.8H, V1.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x4EC33421, "FMIN V1.8H, V1.8H, V3.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x4EF8F801, "FABS V1.8H, V0.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x6E403C41, "FDIV V1.8H, V2.8H, V0.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x6E411C02, "FMUL V2.8H, V0.8H, V1.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x6E411C62, "FMUL V2.8H, V3.8H, V1.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x6E413C02, "FDIV V2.8H, V0.8H, V1.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x6E443484, "FMAXP V4.8H, V4.8H, V4.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x6EC43484, "FMINP V4.8H, V4.8H, V4.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x6EF8F801, "FNEG V1.8H, V0.8H", fp16Reason},
	{"f16/f16_arm64.s", 0x6EF9F801, "FSQRT V1.8H, V0.8H", fp16Reason},
}

// findArm64Asm returns every *_arm64.s file under the module root.
func findArm64Asm(t *testing.T) []string {
	t.Helper()
	var files []string
	err := filepath.WalkDir(".", func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(p, "_arm64.s") {
			// Normalize to forward slashes so allow-list matching and
			// reporting are identical on Windows (WalkDir yields backslashes).
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

func allowed(file string, hex uint32, used map[int]bool) bool {
	for i, e := range asmAllowlist {
		if e.hex == hex && strings.HasSuffix(file, e.file) {
			used[i] = true
			return true
		}
	}
	return false
}

// TestArm64WordEncodings decodes every hand-encoded WORD directive in the ARM64
// assembly and asserts it matches the instruction named in its comment.
func TestArm64WordEncodings(t *testing.T) {
	usedExceptions := map[int]bool{}

	for _, file := range findArm64Asm(t) {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		directives := asmencoding.ScanSource(string(src))

		// Hexes that verified cleanly somewhere in this file; used to accept
		// uncommented repeats of an already-validated instruction.
		matched := map[uint32]bool{}
		for _, d := range directives {
			if d.Source != asmencoding.NoComment {
				if asmencoding.Verify(d.Hex, d.Comment).Status == asmencoding.Match {
					matched[d.Hex] = true
				}
			}
		}

		for _, d := range directives {
			if d.Source == asmencoding.NoComment {
				if matched[d.Hex] || allowed(file, d.Hex, usedExceptions) {
					continue
				}
				t.Errorf("%s:%d  0x%08X  uncommented WORD, cannot validate (add a comment)", file, d.Line, d.Hex)
				continue
			}
			res := asmencoding.Verify(d.Hex, d.Comment)
			switch res.Status {
			case asmencoding.Match:
				// ok
			case asmencoding.Mismatch:
				if allowed(file, d.Hex, usedExceptions) {
					continue
				}
				t.Errorf("%s:%d  0x%08X  claims=%q  decodes=%q", file, d.Line, d.Hex, res.Claimed, res.Decoded)
			case asmencoding.Undecodable:
				if allowed(file, d.Hex, usedExceptions) {
					continue
				}
				t.Errorf("%s:%d  0x%08X  claims=%q  not decodable and not allow-listed (%v)", file, d.Line, d.Hex, d.Comment, res.Err)
			}
		}
	}

	for i, e := range asmAllowlist {
		if !usedExceptions[i] {
			t.Errorf("stale allow-list entry never matched: %s 0x%08X (%s)", e.file, e.hex, e.expect)
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
