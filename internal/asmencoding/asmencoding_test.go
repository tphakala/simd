package asmencoding

import (
	"strings"
	"testing"
)

func TestScanSource(t *testing.T) {
	src := strings.Join([]string{
		"TEXT ·foo(SB), NOSPLIT, $0-48",                     // 1
		"    WORD $0x4E24CC40  // FMLA V0.4S, V2.4S, V4.4S", // 2 inline comment
		"    // FDIV V1.8H, V2.8H, V0.8H (1.0 / x)",         // 3 comment line
		"    WORD $0x6E403C41",                              // 4 preceding-line comment
		"    WORD $0x6E443484",                              // 5 no comment (prev line is a WORD)
		"    FADD V0.4S, V0.4S, V1.4S",                      // 6 not a WORD directive
	}, "\n")

	got := ScanSource(src)
	want := []Directive{
		{Line: 2, Hex: 0x4E24CC40, Comment: "FMLA V0.4S, V2.4S, V4.4S", Source: InlineComment},
		{Line: 4, Hex: 0x6E403C41, Comment: "FDIV V1.8H, V2.8H, V0.8H (1.0 / x)", Source: PrecedingComment},
		{Line: 5, Hex: 0x6E443484, Comment: "", Source: NoComment},
	}
	if len(got) != len(want) {
		t.Fatalf("ScanSource returned %d directives, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("directive[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestScanSourceIgnoresCommentedOutWord(t *testing.T) {
	// Line 1 is a commented-out directive; line 2 is the real one.
	src := "    // WORD $0x4EC03480\n    WORD $0x4E24CC40"
	got := ScanSource(src)
	if len(got) != 1 {
		t.Fatalf("expected 1 directive (commented-out one ignored), got %d: %+v", len(got), got)
	}
	if got[0].Hex != 0x4E24CC40 {
		t.Errorf("hex = 0x%08X, want 0x4E24CC40", got[0].Hex)
	}
	if got[0].Source != NoComment {
		t.Errorf("Source = %v, want NoComment (commented-out WORD must not become a comment)", got[0].Source)
	}
}

func TestScanSourceEmptyCommentAbove(t *testing.T) {
	// An empty "//" line above a directive must not become an empty comment.
	src := "    //\n    WORD $0x4E24CC40"
	got := ScanSource(src)
	if len(got) != 1 {
		t.Fatalf("expected 1 directive, got %d: %+v", len(got), got)
	}
	if got[0].Source != NoComment {
		t.Errorf("Source = %v, want NoComment (empty comment line must not be used)", got[0].Source)
	}
}

func TestParseWordLine(t *testing.T) {
	tests := []struct {
		name        string
		line        string
		wantHex     uint32
		wantComment string
		wantOK      bool
	}{
		{
			name:        "inline comment",
			line:        "    WORD $0x4E24CC40           // FMLA V0.4S, V2.4S, V4.4S",
			wantHex:     0x4E24CC40,
			wantComment: "FMLA V0.4S, V2.4S, V4.4S",
			wantOK:      true,
		},
		{
			name:        "bare word without inline comment",
			line:        "    WORD $0x6E403C41",
			wantHex:     0x6E403C41,
			wantComment: "",
			wantOK:      true,
		},
		{
			name:   "non-word instruction line",
			line:   "    FADD V0.4S, V0.4S, V1.4S",
			wantOK: false,
		},
		{
			name:   "comment only line",
			line:   "    // FADD V2.8H, V0.8H, V1.8H",
			wantOK: false,
		},
		{
			name:   "commented-out word directive",
			line:   "    // WORD $0x4EC03480",
			wantOK: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hex, comment, ok := ParseWordLine(tt.line)
			if ok != tt.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tt.wantOK)
			}
			if !ok {
				return
			}
			if hex != tt.wantHex {
				t.Errorf("hex = 0x%08X, want 0x%08X", hex, tt.wantHex)
			}
			if comment != tt.wantComment {
				t.Errorf("comment = %q, want %q", comment, tt.wantComment)
			}
		})
	}
}

func TestDecode(t *testing.T) {
	gnu, err := Decode(0x4E24CC40)
	if err != nil {
		t.Fatalf("Decode(0x4E24CC40) error: %v", err)
	}
	if gnu != "fmla v0.4s, v2.4s, v4.4s" {
		t.Errorf("Decode(0x4E24CC40) = %q, want %q", gnu, "fmla v0.4s, v2.4s, v4.4s")
	}

	// FP16 .8H FMUL is not decodable by arm64asm; must report an error.
	if _, err := Decode(0x6E411C02); err == nil {
		t.Errorf("Decode(0x6E411C02) err = nil, want non-nil (FP16 .8H unsupported)")
	}
}

func TestNormalize(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"FMLA V0.4S, V2.4S, V4.4S", "FMLA V0.4S,V2.4S,V4.4S"},
		{"fmla v0.4s,  v2.4s,v4.4s", "FMLA V0.4S,V2.4S,V4.4S"},
		{"FMLA V0.4S , V2.4S", "FMLA V0.4S,V2.4S"},
	}
	for _, tt := range tests {
		if got := Normalize(tt.in); got != tt.want {
			t.Errorf("Normalize(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}

	// INS and MOV (element) are aliases; they must normalize equal.
	if Normalize("INS V2.D[1], V3.D[0]") != Normalize("MOV V2.D[1], V3.D[0]") {
		t.Errorf("INS/MOV element forms should normalize equal: %q vs %q",
			Normalize("INS V2.D[1], V3.D[0]"), Normalize("MOV V2.D[1], V3.D[0]"))
	}
}

func TestVerify(t *testing.T) {
	tests := []struct {
		name    string
		hex     uint32
		claimed string
		want    Status
	}{
		{"exact match", 0x4E24CC40, "FMLA V0.4S, V2.4S, V4.4S", Match},
		{"match with trailing annotation", 0x6E34DC01, "FMUL V1.4S, V0.4S, V20.4S V1 = -X * LOG2E", Match},
		{"match via INS/MOV alias", 0x6E180462, "INS V2.D[1], V3.D[0]", Match},
		{"corrected FMINP matches", 0x6EA0F400, "FMINP V0.4S, V0.4S, V0.4S", Match},
		{"wrong register mismatch (FMINP V16)", 0x6EB0F400, "FMINP V0.4S, V0.4S, V0.4S", Mismatch},
		{"INS encoded as TBL mismatch", 0x4E032042, "INS V2.D[1], V3.D[0]", Mismatch},
		{"register prefix must not false-match (V1 vs V10)", 0x6E21DC02, "FMUL V2.4S, V0.4S, V10.4S", Mismatch},
		{"FP16 undecodable", 0x6E411C02, "FMUL V2.8H, V0.8H, V1.8H", Undecodable},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Verify(tt.hex, tt.claimed)
			if got.Status != tt.want {
				t.Errorf("Verify(0x%08X, %q).Status = %v, want %v (decoded=%q)",
					tt.hex, tt.claimed, got.Status, tt.want, got.Decoded)
			}
		})
	}
}
