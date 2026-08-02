package simd

// FMA3 is the sibling of the AVX2 axis that #200 modelled, and until #201 it was
// unguarded. cpu.X86.FMA is a CPUID bit separate from AVX: Sandy Bridge and Ivy
// Bridge have AVX and no FMA3, which is why f64 and c128 keep an initAVXNoFMA
// dispatch tier. Every FMA-using kernel in this repo is correct today only
// because a hand-written guard says so, and nothing checked those guards.
//
// Three edits used to pass the whole suite and SIGILL on Sandy Bridge, all three
// verified against the tree before these tests existed:
//
//  1. adding VFMADD231PD to f64 clampAVX, a kernel initAVXNoFMA assigns
//  2. adding VFMADD231PS to f32 copySignAVX, dispatched on plain cpu.X86.AVX
//  3. deleting `&& cpu.X86.FMA` from f32 realFFTUnpack32's guard
//
// The two tests here split the dispatch surface between them. Kernels reached
// through an init-time tier assignment are covered by TestAmd64InitTierFMA,
// which reads the tier switch; kernels reached through a per-call branch are
// covered by TestAmd64KernelDispatchRequiresFMA, which reuses the #200 dominance
// walker on the FMA identifier set. Case 1 is the first, cases 2 and 3 the second.

import (
	"go/ast"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tphakala/simd/asmcheck"
)

// initTierFMA classifies every init-time tier assigner by whether the CPU that
// reaches it is guaranteed to have FMA3. A tier mapped false must assign only
// FMA-free kernels.
//
// initAVX512 is true by architecture: FMA3 predates AVX-512 and every part
// implementing an AVX-512 extension implements it. initAVX is true by dispatch
// rather than by architecture, so it is not taken on trust; assertAVXTierNamesFMA
// reads each package's tier switch and fails if the condition selecting initAVX
// stops naming FMA.
var initTierFMA = map[string]bool{
	"initAVX512":   true,
	"initAVX":      true,
	"initAVXNoFMA": false,
	"initSSE":      false,
	"initSSE2":     false,
	"initGo":       false,
}

// noFMATierPackages are the packages that keep an explicit AVX-without-FMA tier.
// f32 and c64 do not: their init() falls from AVX+FMA straight to SSE, so an
// AVX1-only part gets SSE kernels rather than a separate AVX tier. The list is
// asserted rather than derived so that renaming initAVXNoFMA, or adding one to a
// third package, has to be a deliberate edit here.
var noFMATierPackages = []string{"c128", "f64"}

// TestAmd64InitTierFMA asserts that no kernel needing FMA3 is assigned by a tier
// that runs on CPUs which may lack it.
func TestAmd64InitTierFMA(t *testing.T) {
	var gotNoFMATier []string

	for _, pkgDir := range amd64PackageDirs(t) {
		needsFMA := fmaKernels(t, pkgDir)
		pf := parseAmd64Package(t, pkgDir)

		sawTier := false
		for _, name := range sortedFuncNames(pf.funcs) {
			if !isInitTier(name) {
				continue
			}
			// "init" itself is the entry point that picks a tier, not a tier.
			if name == "init" {
				continue
			}
			guaranteed, known := initTierFMA[name]
			if !known {
				t.Errorf("%s: %s looks like an init-time tier assigner but is not "+
					"classified in initTierFMA. Add it with true if the CCPU reaching it "+
					"always has FMA3, false otherwise. Leaving it out would let the tier "+
					"assign FMA kernels unchecked.", pkgDir, name)
				continue
			}
			sawTier = true
			if guaranteed {
				continue
			}
			for _, kernel := range referencedIdents(pf.funcs[name], needsFMA) {
				t.Errorf("%s: %s assigns %s, whose body uses FMA3.\n"+
					"\tfirst FMA use: %s\n"+
					"This tier runs on CPUs without FMA3 (Intel Sandy/Ivy Bridge, and the "+
					"AVX-without-FMA parts initAVXNoFMA exists for), so that kernel faults "+
					"with SIGILL there. Assign the SSE2 sibling in this tier instead.",
					pkgDir, name, kernel, fmaUseText(needsFMA[kernel]))
			}
		}
		// Packages without a tiered init dispatch per call instead, and
		// TestAmd64KernelDispatchRequiresFMA covers those; only the presence of
		// an unclassified tier is an error here.
		_ = sawTier
		if pf.funcs["initAVXNoFMA"] != nil {
			gotNoFMATier = append(gotNoFMATier, pkgDir)
		}
		assertAVXTierNamesFMA(t, pkgDir, pf)
	}

	slices.Sort(gotNoFMATier)
	if !slices.Equal(gotNoFMATier, noFMATierPackages) {
		t.Errorf("packages defining initAVXNoFMA are %v, expected %v.\n"+
			"That tier is the one place an FMA-free AVX path is spelled out, so this "+
			"list is what tells the check where to look. A package that gained one "+
			"needs adding; a package that lost one either dropped the tier (fine, drop "+
			"it here too) or renamed it, in which case its FMA kernels are now "+
			"unchecked.", gotNoFMATier, noFMATierPackages)
	}
}

// assertAVXTierNamesFMA verifies the one entry initTierFMA takes on dispatch
// rather than architecture: that the condition selecting initAVX requires FMA.
// Without this the classification would be a prose claim, and deleting
// `&& cpu.X86.FMA` from a tier switch would silently promote every FMA kernel in
// that tier onto AVX1-only CPUs.
func assertAVXTierNamesFMA(t *testing.T, pkgDir string, pf *pkgFuncs) {
	t.Helper()
	if pf.funcs["initAVX"] == nil {
		return // package has no AVX tier; nothing to verify
	}
	cond, bound, where := tierSelector(pf, "initAVX")
	if cond == nil {
		t.Errorf("%s: could not find the switch case that calls initAVX, so the claim "+
			"that its tier implies FMA3 is unverified. initTierFMA marks initAVX as "+
			"FMA-guaranteed only because the dispatch says so.", pkgDir)
		return
	}
	if !positiveGate(cond, bound, fmaGateIdents) {
		t.Errorf("%s: the case selecting initAVX (in %s) does not require FMA3, but "+
			"initTierFMA classifies that tier as FMA-guaranteed and every FMA kernel it "+
			"assigns relies on it. Restore the cpu.X86.FMA conjunct, or move the "+
			"FMA-using kernels into a tier that has one.", pkgDir, where)
	}
}

// tierSelector finds the switch case that calls tier and returns its condition,
// together with the name of the function holding the switch.
//
// Two shapes occur. f32, f64 and c64 switch on the feature expressions directly
// inside init(). c128 passes them as booleans into selectImpl, so the case reads
// `case avxFMA:` and the real condition is the matching argument at the single
// call site; bindParams resolves that, which is why the returned condition can
// be evaluated the same way in both shapes.
func tierSelector(pf *pkgFuncs, tier string) (ast.Expr, *pkgFuncs, string) {
	for _, name := range sortedFuncNames(pf.funcs) {
		fn := pf.funcs[name]
		if fn.Body == nil {
			continue
		}
		var found ast.Expr
		ast.Inspect(fn.Body, func(n ast.Node) bool {
			cc, ok := n.(*ast.CaseClause)
			if !ok || found != nil || len(cc.List) == 0 {
				return true
			}
			if !callsFunc(cc.Body, tier) {
				return true
			}
			found = cc.List[0]
			return false
		})
		if found != nil {
			return found, bindParams(pf, name), name
		}
	}
	return nil, pf, ""
}

// bindParams returns a view of pf in which identifiers local to the named
// function resolve to the CPU expressions behind them. Two kinds are bound, and
// both are real dispatch idioms here:
//
//   - Parameters, from the arguments at the function's single call site. This is
//     what makes c128's `case avxFMA:` resolve to the expression init() passed
//     into selectImpl.
//   - Local declarations, `useAVX := !useAVX512 && cpu.X86.AVX && cpu.X86.FMA`,
//     which f32's indexed batch dot uses before an early-out guard. Without this
//     the guard reads as a test of an opaque bool and the kernel looks ungated.
//
// Locals are collected function-wide rather than only up to the reference. That
// cannot manufacture a gate: an assignment only matters when the identifier also
// appears in a condition that dominates the reference, and dominance is decided
// separately. It does mean a reassigned or shadowed name resolves to whichever
// binding is seen last, which no dispatch function in this repo does.
//
// A parameter list with anything other than exactly one call site is left
// unbound: the condition then fails to resolve and the caller reports that
// rather than guessing.
func bindParams(pf *pkgFuncs, name string) *pkgFuncs {
	fn := pf.funcs[name]
	if fn == nil || fn.Body == nil {
		return pf
	}

	locals := map[string]ast.Expr{}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		as, ok := n.(*ast.AssignStmt)
		if !ok || len(as.Lhs) != len(as.Rhs) {
			return true
		}
		for i, lhs := range as.Lhs {
			if id, ok := lhs.(*ast.Ident); ok {
				locals[id.Name] = as.Rhs[i]
			}
		}
		return true
	})

	var params []string
	if fn.Type.Params != nil {
		for _, field := range fn.Type.Params.List {
			for _, id := range field.Names {
				params = append(params, id.Name)
			}
		}
	}
	if len(params) > 0 {
		var args []ast.Expr
		calls := 0
		for _, caller := range sortedFuncNames(pf.funcs) {
			cf := pf.funcs[caller]
			if cf.Body == nil || caller == name {
				continue
			}
			ast.Inspect(cf.Body, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				if id, ok := call.Fun.(*ast.Ident); ok && id.Name == name {
					calls++
					args = call.Args
				}
				return true
			})
		}
		if calls == 1 && len(args) == len(params) {
			for i, p := range params {
				locals[p] = args[i]
			}
		}
	}

	if len(locals) == 0 {
		return pf
	}
	bound := &pkgFuncs{funcs: pf.funcs, varInits: map[string]ast.Expr{}}
	maps.Copy(bound.varInits, pf.varInits)
	maps.Copy(bound.varInits, locals)
	return bound
}

// isInitTier reports whether a dispatch function is one of the init-time tier
// assigners rather than a per-call dispatcher.
func isInitTier(name string) bool {
	return strings.HasPrefix(name, "init")
}

// callsFunc reports whether any statement in body calls the named function.
func callsFunc(body []ast.Stmt, name string) bool {
	found := false
	for _, s := range body {
		ast.Inspect(s, func(n ast.Node) bool {
			if found {
				return false
			}
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			if id, ok := call.Fun.(*ast.Ident); ok && id.Name == name {
				found = true
			}
			return !found
		})
	}
	return found
}

// referencedIdents returns the keys of want that appear as identifiers in fn,
// sorted, so a failure reports identically on every run.
func referencedIdents(fn *ast.FuncDecl, want map[string]asmcheck.X86Kernel) []string {
	if fn == nil || fn.Body == nil {
		return nil
	}
	out := make([]string, 0, len(want))
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok {
			return true
		}
		if _, wanted := want[id.Name]; wanted && !slices.Contains(out, id.Name) {
			out = append(out, id.Name)
		}
		return true
	})
	slices.Sort(out)
	return out
}

// fmaKernels returns the FMA-using kernels defined by pkgDir's amd64 assembly,
// keyed by symbol name.
func fmaKernels(t *testing.T, pkgDir string) map[string]asmcheck.X86Kernel {
	t.Helper()
	out := map[string]asmcheck.X86Kernel{}
	paths, err := filepath.Glob(filepath.Join(pkgDir, "*_amd64.s"))
	if err != nil {
		t.Fatalf("glob %s: %v", pkgDir, err)
	}
	for _, p := range paths {
		src, rerr := os.ReadFile(p)
		if rerr != nil {
			t.Fatalf("read %s: %v", p, rerr)
		}
		for _, k := range asmcheck.ScanX86Source(string(src)) {
			if k.Needs(asmcheck.X86FeatureFMA) {
				out[k.Name] = k
			}
		}
	}
	return out
}

// fmaUseText renders a kernel's first FMA instruction for an error message.
func fmaUseText(k asmcheck.X86Kernel) string {
	uses := k.FeatureUses(asmcheck.X86FeatureFMA)
	if len(uses) == 0 {
		return "(none)"
	}
	return uses[0].Text
}

// amd64PackageDirs returns every directory holding amd64 assembly, sorted.
func amd64PackageDirs(t *testing.T) []string {
	t.Helper()
	var dirs []string
	for _, f := range findAmd64Asm(t) {
		d := filepath.Dir(f)
		if !slices.Contains(dirs, d) {
			dirs = append(dirs, d)
		}
	}
	slices.Sort(dirs)
	return dirs
}

// fmaAxis is the FMA3 dispatch axis. Unlike the AVX2 axis it also reaches the
// initXxx tier assigners, because their guard is at the call site in init() and
// checkKernelGated walks up to it. TestAmd64InitTierFMA covers the same ground
// from the other direction, naming the tier rather than the kernel, so a tier
// that stops being FMA-guaranteed is reported once per tier and not once per
// kernel it assigns.
var fmaAxis = gateAxis{
	name:   "FMA3",
	idents: fmaGateIdents,
	remedy: "An AVX1-only CPU without FMA3 (Intel Sandy/Ivy Bridge, and the parts " +
		"initAVXNoFMA exists for) reaching this kernel faults with SIGILL, the same " +
		"way #196/#197 did on the AVX2 axis. Add the cpu.X86.FMA conjunct to this " +
		"branch, or move the kernel behind one.",
	firstLine: firstFMAUseLine,
	firstText: firstFMAUseText,
}

func firstFMAUseLine(k asmcheck.X86Kernel) int {
	uses := k.FeatureUses(asmcheck.X86FeatureFMA)
	if len(uses) == 0 {
		return k.Line
	}
	return uses[0].Line
}

func firstFMAUseText(k asmcheck.X86Kernel) string {
	uses := k.FeatureUses(asmcheck.X86FeatureFMA)
	if len(uses) == 0 {
		return ""
	}
	return uses[0].Text + "  (" + uses[0].Reason + ")"
}

// TestAmd64KernelDispatchRequiresFMA asserts that every AMD64 kernel whose body
// uses FMA3 and which is dispatched from a per-call branch sits behind a
// condition requiring FMA3.
func TestAmd64KernelDispatchRequiresFMA(t *testing.T) {
	parsed := map[string]*pkgFuncs{}
	checked := 0

	for _, file := range findAmd64Asm(t) {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		pkgDir := filepath.Dir(file)

		for _, k := range asmcheck.ScanX86Source(string(src)) {
			if !k.Needs(asmcheck.X86FeatureFMA) {
				continue
			}
			pf := parsed[pkgDir]
			if pf == nil {
				pf = parseAmd64Package(t, pkgDir)
				parsed[pkgDir] = pf
			}
			checked++
			checkKernelGated(t, file, pkgDir, k, pf, fmaAxis)
		}
	}

	// A scanner that stopped recognising the FMA mnemonics would leave every
	// assertion above vacuous while the test still reported success.
	if checked == 0 {
		t.Error("no FMA-using kernels found in any amd64 assembly, but this repo has " +
			"many. The FMA classifier has stopped matching and this test is checking " +
			"nothing.")
	}
}

// fmaMnemonics is every FMA3 instruction Go's assembler accepts on amd64: the
// four multiply-add bases in packed and scalar, single and double, across the
// three operand orders, plus the two alternating forms which are packed-only.
//
// The list is not a sample. It was checked against the toolchain's own opcode
// table, where it is exactly the set whose name begins VFM or VFNM:
//
//	grep -ohE '\bVF[A-Z0-9]*\b' "$(go env GOROOT)"/src/cmd/internal/obj/x86/anames.go |
//	  sort -u | grep -E '^VF(N)?M'
//
// On go1.26 that yields 60 names, all 60 of which the classifier's pattern
// matches, and it matches none of the other 1544 amd64 mnemonics in the table.
// Regenerate and recompare if a future Go adds an encoding.
func fmaMnemonics() []string {
	out := make([]string, 0, 60)
	for _, base := range []string{"VFMADD", "VFMSUB", "VFNMADD", "VFNMSUB"} {
		for _, order := range []string{"132", "213", "231"} {
			for _, size := range []string{"PD", "PS", "SD", "SS"} {
				out = append(out, base+order+size)
			}
		}
	}
	// VFMADDSUB and VFMSUBADD alternate add and subtract across lanes, which has
	// no meaning for a single scalar, so they exist only in packed form.
	for _, base := range []string{"VFMADDSUB", "VFMSUBADD"} {
		for _, order := range []string{"132", "213", "231"} {
			for _, size := range []string{"PD", "PS"} {
				out = append(out, base+order+size)
			}
		}
	}
	return out
}

// TestFMAMnemonicCoverage pins the classifier against every FMA3 mnemonic Go's
// assembler accepts, so a narrowing of the pattern is caught here rather than by
// a SIGILL on hardware nobody in CI owns.
func TestFMAMnemonicCoverage(t *testing.T) {
	names := fmaMnemonics()
	if len(names) != 60 {
		t.Fatalf("built %d FMA mnemonics, expected the 60 in Go's amd64 opcode table; "+
			"the generator above no longer matches the ISA it documents", len(names))
	}
	fma := make([]string, 0, len(names))
	for _, n := range names {
		fma = append(fma, n+" Y1, Y2, Y0")
	}

	notFMA := []string{
		"VMULPD Y1, Y2, Y0", "VADDPS Y1, Y2, Y0", "VMOVUPS (DI), Y0",
		"VFMADD X1, X2, X0", // no operand-order digits: not an FMA3 spelling
		"MULPD X1, X0", "RET", "VPERMPD $0x1B, Y2, Y2",
	}

	for _, instr := range fma {
		src := "TEXT ·k(SB), NOSPLIT, $0-0\n    " + instr + "\n    RET\n"
		k := asmcheck.ScanX86Source(src)
		if len(k) != 1 {
			t.Fatalf("%s: expected 1 kernel, got %d", instr, len(k))
		}
		if !k[0].Needs(asmcheck.X86FeatureFMA) {
			t.Errorf("%s: not classified as needing FMA3", instr)
		}
		// An FMA instruction must not inflate the AVX level ladder: these
		// kernels are named ...AVX on purpose and the ISA test would reject them.
		if k[0].Level != asmcheck.X86LevelAVX {
			t.Errorf("%s: inflated the level to %s; FMA is an orthogonal axis",
				instr, k[0].Level)
		}
	}

	for _, instr := range notFMA {
		src := "TEXT ·k(SB), NOSPLIT, $0-0\n    " + instr + "\n    RET\n"
		k := asmcheck.ScanX86Source(src)
		if len(k) != 1 {
			t.Fatalf("%s: expected 1 kernel, got %d", instr, len(k))
		}
		if k[0].Needs(asmcheck.X86FeatureFMA) {
			t.Errorf("%s: wrongly classified as needing FMA3", instr)
		}
	}
}

// TestFMAGateIdentsExcludeAVX2 pins the deliberate omission. AVX2 and FMA3 are
// separate CPUID bits; treating AVX2 as evidence of FMA3 would make an
// AVX2-gated branch look like a valid home for an FMA kernel.
func TestFMAGateIdentsExcludeAVX2(t *testing.T) {
	for _, ident := range []string{avx2Feature, "AVX", "AVXVNNI", "SSE2"} {
		if fmaGateIdents[ident] {
			t.Errorf("fmaGateIdents accepts %q as establishing FMA3, but it does not. "+
				"Only cpu.X86.FMA does directly, and AVX512F/AVX512VL by implication.",
				strings.ToUpper(ident))
		}
	}
	if !fmaGateIdents["FMA"] {
		t.Error("fmaGateIdents does not accept FMA, so no branch can ever satisfy the " +
			"FMA dispatch check and TestAmd64KernelDispatchRequiresFMA would fail wholesale")
	}
}
