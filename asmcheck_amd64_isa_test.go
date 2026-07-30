package simd

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tphakala/simd/asmcheck"
)

// findAmd64Asm returns every *_amd64.s file under the module root.
func findAmd64Asm(t *testing.T) []string {
	t.Helper()
	var files []string
	err := filepath.WalkDir(".", func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(p, "_amd64.s") {
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
		t.Fatal("no *_amd64.s files found; is the test running from the module root?")
	}
	return files
}

// avx2GatedAVXKernels lists kernels whose name claims only AVX but whose body
// needs AVX2, and which are safe anyway because the Go dispatch that selects them
// gates on cpu.X86.AVX2 (or the package's hasAVX2) rather than cpu.X86.AVX. The
// value records that gate, so the guard is stated where the exemption is granted.
//
// A kernel may be added here ONLY together with its AVX2 gate. Without one, an
// AVX1-only CPU (Intel Sandy/Ivy Bridge, AMD Bulldozer) or an
// AVX+FMA3-without-AVX2 CPU (AMD Piledriver, Steamroller) reaches an AVX2
// encoding and faults with SIGILL;
// that is exactly what #196 and #197 were. Renaming an entry's kernel to the
// ...AVX2 suffix its body requires is the better fix and lets the entry go.
//
// Keys are "file:kernel", because the same kernel name exists in several packages.
// The log core and its two pow entry points share one gate per package.
const (
	f32LogGate = "f32_amd64.go logSIMDOK32: cpu.X86.AVX2 && cpu.X86.FMA"
	f64LogGate = "f64_amd64.go logSIMDOK: cpu.X86.AVX2 && cpu.X86.FMA"
)

var avx2GatedAVXKernels = map[string]string{
	// f32: 256-bit integer ops and cross-lane permutes, all AVX2-gated.
	"f32/f32_amd64.s:interleave3AVX":         "f32_amd64.go interleaveN32: cpu.X86.AVX2",
	"f32/f32_amd64.s:interleave6AVX":         "f32_amd64.go interleaveN32: cpu.X86.AVX2",
	"f32/f32_amd64.s:deinterleave3AVX":       "f32_amd64.go deinterleaveN32: cpu.X86.AVX2",
	"f32/f32_amd64.s:deinterleave6AVX":       "f32_amd64.go deinterleaveN32: cpu.X86.AVX2",
	"f32/f32_amd64.s:sigmoidAVX":             "f32_amd64.go sigmoid32: cpu.X86.AVX2",
	"f32/f32_amd64.s:tanhAVX":                "f32_amd64.go tanh32: cpu.X86.AVX2",
	"f32/f32_amd64.s:expAVX":                 "f32_amd64.go exp32: cpu.X86.AVX2",
	"f32/f32_amd64.s:logAVX":                 f32LogGate,
	"f32/f32_amd64.s:powAVX":                 f32LogGate,
	"f32/f32_amd64.s:powElemAVX":             f32LogGate,
	"f32/f32_amd64.s:int16ToFloat32ScaleAVX": "f32_amd64.go int16ToFloat32Scale: cpu.X86.AVX2",
	"f32/f32_amd64.s:realFFTUnpackAVX":       "f32_amd64.go realFFTUnpack32: cpu.X86.AVX2 && cpu.X86.FMA",

	// f64: same shape as f32, plus VPERMPD in the autocorrelation and unpack steps.
	"f64/f64_amd64.s:interleave3AVX":   "f64_amd64.go interleaveN64: cpu.X86.AVX2",
	"f64/f64_amd64.s:deinterleave3AVX": "f64_amd64.go deinterleaveN64: cpu.X86.AVX2",
	"f64/f64_amd64.s:deinterleave6AVX": "f64_amd64.go deinterleaveN64: cpu.X86.AVX2",
	"f64/f64_amd64.s:sigmoidAVX":       "f64_amd64.go sigmoid64: cpu.X86.AVX2",
	"f64/f64_amd64.s:tanhAVX":          "f64_amd64.go tanh64: cpu.X86.AVX2",
	"f64/f64_amd64.s:expAVX":           "f64_amd64.go exp64: cpu.X86.AVX2",
	"f64/f64_amd64.s:logAVX":           f64LogGate,
	"f64/f64_amd64.s:powAVX":           f64LogGate,
	"f64/f64_amd64.s:powElemAVX":       f64LogGate,
	"f64/f64_amd64.s:autocorrStep4AVX": "f64_amd64.go autocorrelate64: hasAVX2 early-out",
	"f64/f64_amd64.s:realFFTUnpackAVX": "f64_amd64.go realFFTUnpack64: hasAVX2 && cpu.X86.FMA",
}

// TestAmd64KernelISALevel asserts that no AMD64 kernel uses an instruction above
// the feature level its name claims, so that a kernel dispatched behind a plain
// cpu.X86.AVX guard cannot contain an AVX2-only encoding. Kernels that outrun
// their name must be listed in avx2GatedAVXKernels together with the AVX2 gate
// that protects them.
func TestAmd64KernelISALevel(t *testing.T) {
	usedExemption := map[string]bool{}

	for _, file := range findAmd64Asm(t) {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		kernels := asmcheck.ScanX86Source(string(src))
		if len(kernels) == 0 {
			t.Errorf("%s: no TEXT symbols found. Every amd64 assembly file in this "+
				"repo defines kernels, so zero means the scanner stopped recognising "+
				"this file's TEXT spelling and is silently checking nothing in it.", file)
		}
		for _, k := range kernels {
			declared := asmcheck.DeclaredX86Level(k.Name)
			key := file + ":" + k.Name
			if k.Level <= declared {
				continue
			}
			if gate, ok := avx2GatedAVXKernels[key]; ok {
				usedExemption[key] = true
				t.Logf("%s: %s needs %s, exempt via %s", file, k.Name, k.Level, gate)
				continue
			}
			t.Errorf("%s:%d: kernel %s is named for %s but its body needs %s.\n"+
				"%s"+
				"Fix the assembly to stay within %s, rename the kernel to its real tier, "+
				"or add %q to avx2GatedAVXKernels with the AVX2 gate that protects it.",
				file, k.Line, k.Name, declared, k.Level,
				formatX86Uses(file, k.Uses, declared), declared, key)
		}
	}

	for key := range avx2GatedAVXKernels {
		if !usedExemption[key] {
			t.Errorf("avx2GatedAVXKernels lists %q, but that kernel no longer needs an "+
				"exemption (it was fixed, renamed or removed); drop the entry.", key)
		}
	}
}

// formatX86Uses renders the offending instructions as indented, file:line
// prefixed lines, one per distinct mnemonic and reason.
func formatX86Uses(file string, uses []asmcheck.X86Use, declared asmcheck.X86Level) string {
	var b strings.Builder
	seen := map[string]bool{}
	for _, u := range uses {
		if u.Level <= declared {
			continue
		}
		mnem, _, _ := strings.Cut(u.Text, " ")
		if seen[mnem+u.Reason] {
			continue
		}
		seen[mnem+u.Reason] = true
		b.WriteString("\t")
		b.WriteString(file)
		b.WriteString(":")
		b.WriteString(strconv.Itoa(u.Line))
		b.WriteString(": ")
		b.WriteString(u.Text)
		b.WriteString("  <- ")
		b.WriteString(u.Reason)
		b.WriteString("\n")
	}
	return b.String()
}
