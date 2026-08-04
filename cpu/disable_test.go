package cpu

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"
)

// fullFeatures returns a Features value with every boolean field set to true,
// so a disable spec can be checked by which fields it turns back off.
func fullFeatures() Features {
	var f Features
	v := reflect.ValueOf(&f).Elem()
	for i := 0; i < v.NumField(); i++ {
		v.Field(i).SetBool(true)
	}
	return f
}

// disabledFields returns the sorted names of the boolean fields that are false in f.
func disabledFields(f Features) []string {
	var names []string
	v := reflect.ValueOf(f)
	t := v.Type()
	for i := 0; i < v.NumField(); i++ {
		if !v.Field(i).Bool() {
			names = append(names, t.Field(i).Name)
		}
	}
	sort.Strings(names)
	return names
}

func TestApplyDisable(t *testing.T) {
	tests := []struct {
		spec     string
		disabled []string
	}{
		{"", nil},
		{"avx512", []string{"AVX512F", "AVX512VL"}},
		// AVXVNNI is VEX-encoded and its dispatch tier sits above AVX2, so clearing
		// AVX2 (and every token that cascades through clearAVX2) must also clear it.
		// The avxvnni token itself clears only AVXVNNI.
		{"avxvnni", []string{"AVXVNNI"}},
		{"avx2", []string{"AVX2", "AVX512F", "AVX512VL", "AVXVNNI"}},
		// F16C is VEX-encoded and gated on AVX, so the avx cascade (and every SSE
		// token that cascades through clearAVX) must also clear F16C. avx2/fma/avx512
		// sit above AVX and correctly leave it set.
		{"avx", []string{"AVX", "AVX2", "AVX512F", "AVX512VL", "AVXVNNI", "F16C", "FMA"}},
		{"fma", []string{"FMA"}},
		{"sse42", []string{"AVX", "AVX2", "AVX512F", "AVX512VL", "AVXVNNI", "F16C", "FMA", "SSE42"}},
		{"sse41", []string{"AVX", "AVX2", "AVX512F", "AVX512VL", "AVXVNNI", "F16C", "FMA", "SSE41", "SSE42"}},
		{"ssse3", []string{"AVX", "AVX2", "AVX512F", "AVX512VL", "AVXVNNI", "F16C", "FMA", "SSE41", "SSE42", "SSSE3"}},
		{"sse3", []string{"AVX", "AVX2", "AVX512F", "AVX512VL", "AVXVNNI", "F16C", "FMA", "SSE3", "SSE41", "SSE42", "SSSE3"}},
		{"pclmulqdq", []string{"PCLMULQDQ"}},
		{"neon", []string{"DOTPROD", "FP16", "NEON", "PMULL", "SVE", "SVE2"}},
		{"fp16", []string{"FP16"}},
		{"sve", []string{"SVE", "SVE2"}},
		{"pmull", []string{"PMULL"}},
		{"dotprod", []string{"DOTPROD"}},
		// Case-insensitivity and surrounding whitespace.
		{"AVX512", []string{"AVX512F", "AVX512VL"}},
		{"  avx512  ", []string{"AVX512F", "AVX512VL"}},
		{"Avx2", []string{"AVX2", "AVX512F", "AVX512VL", "AVXVNNI"}},
		// Unknown tokens are ignored.
		{"foobar", nil},
		{"avx512,foobar", []string{"AVX512F", "AVX512VL"}},
		// Empty tokens between commas are ignored.
		{"avx512,,neon", []string{"AVX512F", "AVX512VL", "DOTPROD", "FP16", "NEON", "PMULL", "SVE", "SVE2"}},
		// Multiple tokens combine.
		{"avx512,neon", []string{"AVX512F", "AVX512VL", "DOTPROD", "FP16", "NEON", "PMULL", "SVE", "SVE2"}},
	}
	for _, tt := range tests {
		f := fullFeatures()
		applyDisable(&f, tt.spec)
		got := disabledFields(f)
		want := append([]string(nil), tt.disabled...)
		sort.Strings(want)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("applyDisable(%q): disabled = %v, want %v", tt.spec, got, want)
		}
	}
}

// TestApplyDisableAll verifies the "all" token forces every feature off.
func TestApplyDisableAll(t *testing.T) {
	f := fullFeatures()
	applyDisable(&f, "all")
	if f != (Features{}) {
		t.Errorf("applyDisable(\"all\") left fields set: %+v", f)
	}
}

// TestHelperInfo is the helper process for TestSIMDDisableAllIntegration. It only
// runs when re-executed with SIMD_DISABLE_HELPER=1 and prints cpu.Info().
//
// It has a second caller outside this package: the cross-isa job in
// .github/workflows/ci.yml runs it under qemu and parses the CPUINFO= line to
// assert each CPU model reports the tier its matrix entry is named for. Renaming
// it, or changing that output format, breaks every leg of that job. Keep the
// bare fmt.Printf; a t.Logf would prefix the file:line and defeat the parse.
func TestHelperInfo(t *testing.T) {
	if os.Getenv("SIMD_DISABLE_HELPER") != "1" {
		t.Skip("not the helper process")
	}
	fmt.Printf("CPUINFO=%s\n", Info())
}

// subprocessTimeout bounds the SIMD_DISABLE=all re-exec below. Natively the
// helper finishes in a few milliseconds, so 30s is a generous ceiling that
// still trips on a genuine hang. Under whole-program CPU emulation the child's
// runtime init and CPUID probe are interpreted instruction by instruction,
// inflating wall time by orders of magnitude; the AVX-512 (Intel SDE) and
// cross-isa (qemu-user) CI legs occasionally blew the fixed 30s budget for
// exactly this reason (#237). Those legs export SIMD_TEST_EMULATED=1, so key a
// larger budget off that one signal. The native path never sets it and keeps
// the tight 30s bound, so a real native hang is still caught quickly.
func subprocessTimeout() time.Duration {
	if os.Getenv("SIMD_TEST_EMULATED") == "1" {
		return 5 * time.Minute
	}
	return 30 * time.Second
}

// TestSIMDDisableAllIntegration re-execs the test binary with SIMD_DISABLE=all
// and asserts cpu.Info() reports the no-SIMD string for this architecture, which
// proves the env var is read and applied during package init.
func TestSIMDDisableAllIntegration(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), subprocessTimeout())
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestHelperInfo$", "-test.v")
	cmd.Env = append(os.Environ(), "SIMD_DISABLE_HELPER=1", "SIMD_DISABLE=all")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("helper process failed: %v\n%s", err, out)
	}

	var want string
	switch runtime.GOARCH {
	case "amd64":
		want = "AMD64 (scalar)"
	case "arm64":
		want = "ARM64 (no SIMD)"
	default:
		want = "Generic (no SIMD)"
	}

	got := ""
	for line := range strings.SplitSeq(string(out), "\n") {
		if v, ok := strings.CutPrefix(strings.TrimSpace(line), "CPUINFO="); ok {
			got = v
			break
		}
	}
	if got != want {
		t.Errorf("Info() under SIMD_DISABLE=all = %q, want %q\nfull output:\n%s", got, want, out)
	}
}
