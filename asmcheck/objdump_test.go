package asmcheck

import "testing"

// fp16Samples are FP16 (.8H) encodings arm64asm cannot decode, paired with the
// disassembly aarch64 objdump produces for them.
var fp16Samples = []struct {
	hex    uint32
	expect string
}{
	{0x6E411C02, "FMUL V2.8H, V0.8H, V1.8H"},
	{0x4E411400, "FADD V0.8H, V0.8H, V1.8H"},
	{0x4EF8F801, "FABS V1.8H, V0.8H"},
	{0x6EF9F801, "FSQRT V1.8H, V0.8H"},
	{0x6EC43484, "FMINP V4.8H, V4.8H, V4.8H"},
}

func TestObjdumpLineRe(t *testing.T) {
	// A representative `objdump -D -b binary -m aarch64` transcript. Header
	// lines must be ignored; only instruction rows are captured.
	sample := "\nfp16.bin:     file format binary\n\n" +
		"Disassembly of section .data:\n\n" +
		"0000000000000000 <.data>:\n" +
		"   0:\t6e411c02 \tfmul\tv2.8h, v0.8h, v1.8h\n" +
		"   4:\t4ef8f801 \tfabs\tv1.8h, v0.8h\n" +
		"   8:\t6ef9f801 \tfsqrt\tv1.8h, v0.8h\n"

	ms := objdumpLineRe.FindAllStringSubmatch(sample, -1)
	if len(ms) != 3 {
		t.Fatalf("expected 3 instruction rows, got %d: %v", len(ms), ms)
	}
	want := []struct{ hex, insn string }{
		{"6e411c02", "fmul\tv2.8h, v0.8h, v1.8h"},
		{"4ef8f801", "fabs\tv1.8h, v0.8h"},
		{"6ef9f801", "fsqrt\tv1.8h, v0.8h"},
	}
	for i, m := range ms {
		if m[1] != want[i].hex || m[2] != want[i].insn {
			t.Errorf("row %d: got hex=%q insn=%q, want hex=%q insn=%q", i, m[1], m[2], want[i].hex, want[i].insn)
		}
	}
}

func TestObjdumpCandidatesEnvOverride(t *testing.T) {
	t.Setenv(SIMDObjdumpEnv, "/opt/my/objdump")
	got := objdumpCandidates()
	if len(got) == 0 || got[0] != "/opt/my/objdump" {
		t.Fatalf("SIMD_OBJDUMP override should be tried first, got %v", got)
	}
}

// TestDisassembleWordsWithTool exercises the real objdump path end to end. It
// skips cleanly when no aarch64 objdump is installed, matching the production
// gate in asmcheck_test.go.
func TestDisassembleWordsWithTool(t *testing.T) {
	tool := FindObjdump()
	if tool == "" {
		t.Skip("no aarch64 objdump available; install binutils-aarch64-linux-gnu")
	}

	hexes := make([]uint32, len(fp16Samples))
	for i, s := range fp16Samples {
		hexes[i] = s.hex
	}
	decoded, err := DisassembleWords(t.Context(), tool, hexes)
	if err != nil {
		t.Fatalf("DisassembleWords: %v", err)
	}
	for _, s := range fp16Samples {
		got, ok := decoded[s.hex]
		if !ok {
			t.Errorf("0x%08X: no disassembly returned", s.hex)
			continue
		}
		if res := VerifyDecoded(got, s.expect); res.Status != Match {
			t.Errorf("0x%08X: objdump=%q does not match claim=%q", s.hex, res.Decoded, res.Claimed)
		}
	}
}

// TestFindObjdumpProbe checks that FindObjdump only returns a tool that
// actually decodes the FP16 probe instruction.
func TestFindObjdumpProbe(t *testing.T) {
	tool := FindObjdump()
	if tool == "" {
		t.Skip("no aarch64 objdump available")
	}
	decoded, err := DisassembleWords(t.Context(), tool, []uint32{probeWord})
	if err != nil {
		t.Fatalf("DisassembleWords(probe): %v", err)
	}
	if res := VerifyDecoded(decoded[probeWord], probeExpect); res.Status != Match {
		t.Fatalf("located objdump %q failed the probe: got %q", tool, res.Decoded)
	}
}

// TestDisassembleWordsEmpty checks that an empty word slice returns an empty
// map without invoking objdump (which rejects empty input files).
func TestDisassembleWordsEmpty(t *testing.T) {
	got, err := DisassembleWords(t.Context(), "/nonexistent-objdump", nil)
	if err != nil {
		t.Fatalf("empty input should not error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected empty map, got %v", got)
	}
}
