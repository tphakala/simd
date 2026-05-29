// Package asmencoding validates hand-encoded ARM64 WORD directives against
// the instruction described in their comment. It decodes each 32-bit word with
// the same disassembler go tool objdump uses, so it runs on any architecture
// with no ARM hardware.
package asmencoding

import (
	"encoding/binary"
	"regexp"
	"strconv"
	"strings"

	"golang.org/x/arch/arm64/arm64asm"
)

// Status is the outcome of verifying one WORD directive.
type Status int

const (
	// Match means the decoded instruction matches the comment.
	Match Status = iota + 1
	// Mismatch means the bytes decode to a different instruction than claimed.
	Mismatch
	// Undecodable means the disassembler could not decode the bytes.
	Undecodable
)

// CommentSource records where a directive's describing comment came from.
type CommentSource int

const (
	// NoComment means the WORD directive has no associated comment.
	NoComment CommentSource = iota
	// InlineComment means the comment is on the same line as the WORD.
	InlineComment
	// PrecedingComment means the comment is on the line directly above.
	PrecedingComment
)

// Directive is one parsed WORD $0x... line with its associated comment.
type Directive struct {
	Line    int    // 1-based line number
	Hex     uint32 // the 32-bit instruction word
	Comment string // the claimed instruction text, empty if NoComment
	Source  CommentSource
}

// ScanSource parses every WORD $0x... directive in src, associating each with
// its inline comment, or the comment on the line directly above it.
func ScanSource(src string) []Directive {
	lines := strings.Split(src, "\n")
	out := make([]Directive, 0, len(lines))
	for i, line := range lines {
		hex, inline, ok := ParseWordLine(line)
		if !ok {
			continue
		}
		d := Directive{Line: i + 1, Hex: hex}
		switch {
		case inline != "":
			d.Comment = inline
			d.Source = InlineComment
		case i > 0:
			if prev := strings.TrimSpace(lines[i-1]); strings.HasPrefix(prev, "//") {
				// Use the line above as the description only if it is a
				// non-empty comment that is not itself a commented-out WORD
				// directive (matched case-insensitively).
				text := strings.TrimSpace(strings.TrimPrefix(prev, "//"))
				if text != "" && !strings.HasPrefix(strings.ToUpper(text), "WORD") {
					d.Comment = text
					d.Source = PrecedingComment
				}
			}
		}
		out = append(out, d)
	}
	return out
}

// Result is the outcome of Verify.
type Result struct {
	Status  Status
	Decoded string // normalized decoded instruction (empty when Undecodable)
	Claimed string // normalized claimed instruction
	Err     error  // decode error when Undecodable
}

// aliases maps mnemonics the comments use to the form the disassembler emits.
// INS (element) and MOV (element) are documented aliases; objdump prefers MOV.
var aliases = map[string]string{
	"INS": "MOV",
}

var wordLineRe = regexp.MustCompile(`WORD\s+\$0x([0-9A-Fa-f]{1,8})\b`)
var commentRe = regexp.MustCompile(`//\s*(.*)$`)

// ParseWordLine extracts the hex value and inline comment from a single
// assembly line. ok is false when the line is not a WORD $0x... directive.
func ParseWordLine(line string) (hex uint32, comment string, ok bool) {
	// Ignore commented-out directives like "// WORD $0x...": if WORD appears
	// only after a // on the line, it is disabled code, not an instruction.
	if c := strings.Index(line, "//"); c != -1 {
		if w := strings.Index(line, "WORD"); w > c {
			return 0, "", false
		}
	}
	m := wordLineRe.FindStringSubmatch(line)
	if m == nil {
		return 0, "", false
	}
	v, err := strconv.ParseUint(m[1], 16, 32)
	if err != nil {
		return 0, "", false
	}
	if cm := commentRe.FindStringSubmatch(line); cm != nil {
		comment = strings.TrimSpace(cm[1])
	}
	return uint32(v), comment, true
}

// Decode returns the GNU-syntax disassembly of a 32-bit ARM64 instruction word.
func Decode(hex uint32) (string, error) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], hex)
	inst, err := arm64asm.Decode(b[:])
	if err != nil {
		return "", err
	}
	return arm64asm.GNUSyntax(inst), nil
}

// Normalize canonicalizes an instruction string for comparison: upper case,
// single-spaced, no space after commas, with aliased mnemonics folded.
func Normalize(s string) string {
	s = strings.ToUpper(strings.TrimSpace(s))
	s = strings.Join(strings.Fields(s), " ")
	s = strings.ReplaceAll(s, " ,", ",")
	s = strings.ReplaceAll(s, ", ", ",")
	mnem, rest, hasRest := strings.Cut(s, " ")
	if canon, found := aliases[mnem]; found {
		mnem = canon
	}
	if hasRest {
		return mnem + " " + rest
	}
	return mnem
}

// Verify decodes hex and checks it against the claimed instruction text. The
// decoded instruction must equal the claim or be a token-boundary prefix of it,
// so comments may carry a trailing free-form annotation after the operands.
func Verify(hex uint32, claimed string) Result {
	gnu, err := Decode(hex)
	if err != nil {
		return Result{Status: Undecodable, Err: err}
	}
	nd := Normalize(gnu)
	nc := Normalize(claimed)
	if nc == nd || strings.HasPrefix(nc, nd+" ") {
		return Result{Status: Match, Decoded: nd, Claimed: nc}
	}
	return Result{Status: Mismatch, Decoded: nd, Claimed: nc}
}
