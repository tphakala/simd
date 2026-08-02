package simd

// This test makes the AVX2 dispatch guards enforced rather than documented.
//
// TestAmd64KernelISALevel proves each kernel's BODY needs AVX2. It says nothing
// about the kernel's dispatch, so on its own an exemption table entry is a prose
// claim: weaken sigmoid32's guard from cpu.X86.AVX2 back to cpu.X86.AVX and the
// ISA test still passes, silently restoring the #196/#197 SIGILL. This test
// closes that by reading the guard out of the Go source.
//
// It checks every kernel whose body needs AVX2, not only the ones the exemption
// table lists. A correctly-named ...AVX2 kernel never trips the ISA test, so
// driving this off the table would have left the 44 correctly-named AVX2 kernels
// in i8, i16, i32 and cint unchecked, and flipping one package's
// `var hasAVX2 = cpu.X86.AVX2` to `cpu.X86.AVX` would be the #196 bug class
// reintroduced wholesale with every test still green.
//
// The check is DOMINANCE-based, not "does the word AVX2 appear somewhere in the
// function". The weaker form passes several unsafe shapes that occur in this
// repo's own dispatch code: a switch whose sibling arm is AVX2-gated while the
// arm holding the kernel is not, and a negated guard that runs the kernel on
// exactly the CPUs that lack the feature.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/tphakala/simd/asmcheck"
)

// avx2GateIdents are the identifiers that establish an AVX2-or-better
// requirement. cpu.X86.AVX2 is the direct test; AVX512F/AVX512VL imply it;
// AVXVNNI is defined in cpu/cpu_amd64.go as AVX2 AND the VNNI bit, and
// cpu.clearAVX2 clears it alongside AVX2, so it implies AVX2 in this repo by
// construction. Package-level caches such as `var hasAVX2 = cpu.X86.AVX2` are
// resolved to their initializer rather than trusted by name; see varInits.
// avx2Feature is the cpu.X86 field name and the label used in failure messages.
const avx2Feature = "AVX2"

var avx2GateIdents = map[string]bool{
	avx2Feature: true, "AVX512F": true, "AVX512VL": true, "AVXVNNI": true,
}

// fmaGateIdents are the identifiers that establish FMA3. Only cpu.X86.FMA does
// directly. AVX512F/AVX512VL are included because every part implementing an
// AVX-512 extension also implements FMA3, which is what lets an AVX-512 tier
// carry FMA kernels without naming FMA. AVX2 is deliberately NOT here: it is a
// separate CPUID bit, and although no shipping part has AVX2 without FMA3, the
// architecture does not require the pairing and this repo's whole reason for an
// initAVXNoFMA tier is that the two axes are independent.
var fmaGateIdents = map[string]bool{
	"FMA": true, "AVX512F": true, "AVX512VL": true,
}

// pkgFuncs is one package's amd64 dispatch source, parsed once.
type pkgFuncs struct {
	funcs map[string]*ast.FuncDecl
	// varInits holds package-level bool vars and their initializer expressions,
	// for example `var hasAVX2 = cpu.X86.AVX2`. The expression is kept rather
	// than a precomputed verdict so the same parse serves every feature axis,
	// and so the check never trusts a name: redefining hasAVX2 as cpu.X86.AVX
	// must fail, and by name alone it would not.
	varInits map[string]ast.Expr
}

// avx2Axis is the AVX2 dispatch axis added by #200.
var avx2Axis = gateAxis{
	name:   avx2Feature,
	idents: avx2GateIdents,
	remedy: "An AVX1-only CPU (Intel Sandy/Ivy Bridge, AMD Bulldozer) or an " +
		"AVX+FMA3-without-AVX2 CPU (AMD Piledriver, Steamroller) reaching this " +
		"kernel faults with SIGILL. That is #196/#197. Restore the AVX2 guard on " +
		"this branch, or move the kernel behind one.",
	firstLine: firstUseLine,
	firstText: firstUseText,
}

// TestAmd64KernelDispatchRequiresAVX2 asserts that every AMD64 kernel whose body
// needs AVX2 is dispatched only from a branch whose condition requires AVX2.
func TestAmd64KernelDispatchRequiresAVX2(t *testing.T) {
	parsed := map[string]*pkgFuncs{}

	for _, file := range findAmd64Asm(t) {
		src, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		pkgDir := filepath.Dir(file)

		for _, k := range asmcheck.ScanX86Source(string(src)) {
			// AVX-512 kernels are selected by whole init tiers rather than by a
			// per-call branch, which this dominance check does not model. They
			// are covered by the ISA test's name-versus-body assertion.
			if k.Level != asmcheck.X86LevelAVX2 {
				continue
			}
			pf := parsed[pkgDir]
			if pf == nil {
				pf = parseAmd64Package(t, pkgDir)
				parsed[pkgDir] = pf
			}
			checkKernelGated(t, file, pkgDir, k, pf, avx2Axis)
		}
	}
}

// checkKernelGated reports every dispatch of one feature-requiring kernel that
// is not dominated by a condition establishing that feature.
func checkKernelGated(t *testing.T, file, pkgDir string, k asmcheck.X86Kernel, pf *pkgFuncs, ax gateAxis) {
	t.Helper()
	refs := 0
	for _, name := range sortedFuncNames(pf.funcs) {
		fn := pf.funcs[name]
		if name == k.Name || fn.Body == nil {
			continue
		}
		for _, path := range referencePaths(fn, k.Name) {
			refs++
			if dominatedByGate(path, bindParams(pf, name), ax.idents) {
				continue
			}
			// The guard may sit one frame up. Two shapes in this repo put it
			// there: the initXxx tier assigners, whose selecting case lives in
			// init(), and helpers such as dotProductBatchKernel that take the
			// tier decision as a bool parameter and are only ever called from an
			// already-gated branch.
			if callSitesGated(name, pf, ax.idents, 0) {
				continue
			}
			t.Errorf("%s: kernel %s needs %s, but %s.%s dispatches it from a branch "+
				"that does not require %s, and not every call site of %s requires it "+
				"either.\n\tfirst %s use: %s:%d %s\n%s",
				file, k.Name, ax.name, pkgDir, name, ax.name, name,
				ax.name, file, ax.firstLine(k), ax.firstText(k), ax.remedy)
		}
	}
	if refs == 0 {
		t.Errorf("%s: kernel %s needs %s but nothing in %s references it. "+
			"Either it is dead code, or its dispatch lives somewhere this test does not "+
			"parse (only %s/*_amd64.go is read).", file, k.Name, ax.name, pkgDir, pkgDir)
	}
}

// callSiteDepth bounds the walk up the call graph. This repo's deepest gated
// chain is two hops (init -> selectImpl -> initAVX in c128), so three leaves
// slack; the bound is what stops a recursive pair from spinning.
const callSiteDepth = 3

// callSitesGated reports whether EVERY call site of the named function is itself
// reached only when the gate held. A function nothing calls returns false: an
// unreferenced dispatcher proves nothing, and treating "no call sites" as safe
// would silently excuse a helper whose caller had been deleted.
func callSitesGated(name string, pf *pkgFuncs, idents map[string]bool, depth int) bool {
	if depth > callSiteDepth {
		return false
	}
	sites := 0
	for _, caller := range sortedFuncNames(pf.funcs) {
		cf := pf.funcs[caller]
		if cf.Body == nil || caller == name {
			continue
		}
		for _, path := range referencePaths(cf, name) {
			sites++
			if dominatedByGate(path, bindParams(pf, caller), idents) {
				continue
			}
			if callSitesGated(caller, pf, idents, depth+1) {
				continue
			}
			return false
		}
	}
	return sites > 0
}

// gateAxis describes one CPU feature axis the dispatch check can be run over.
type gateAxis struct {
	name      string          // the feature as it appears in messages
	idents    map[string]bool // identifiers whose truth establishes it
	remedy    string          // what the author should do about a failure
	firstLine func(asmcheck.X86Kernel) int
	firstText func(asmcheck.X86Kernel) string
}

func firstUseLine(k asmcheck.X86Kernel) int {
	if len(k.Uses) == 0 {
		return k.Line
	}
	return k.Uses[0].Line
}

func firstUseText(k asmcheck.X86Kernel) string {
	if len(k.Uses) == 0 {
		return ""
	}
	return k.Uses[0].Text + "  (" + k.Uses[0].Reason + ")"
}

// parseAmd64Package parses every *_amd64.go file in dir once.
func parseAmd64Package(t *testing.T, dir string) *pkgFuncs {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(dir, "*_amd64.go"))
	if err != nil {
		t.Fatalf("glob %s: %v", dir, err)
	}
	if len(paths) == 0 {
		t.Fatalf("no *_amd64.go in %s, so no dispatch could be checked", dir)
	}
	pf := &pkgFuncs{funcs: map[string]*ast.FuncDecl{}, varInits: map[string]ast.Expr{}}
	fset := token.NewFileSet()
	for _, p := range paths {
		f, perr := parser.ParseFile(fset, p, nil, parser.SkipObjectResolution)
		if perr != nil {
			t.Fatalf("parse %s: %v", p, perr)
		}
		for _, d := range f.Decls {
			switch decl := d.(type) {
			case *ast.FuncDecl:
				if decl.Recv == nil {
					pf.funcs[decl.Name.Name] = decl
				}
			case *ast.GenDecl:
				collectVarInits(decl, pf.varInits)
			}
		}
	}
	return pf
}

// collectVarInits records package-level vars together with their initializer
// expressions, so a gate spelled through a cached bool can be resolved to the
// CPU feature it actually reads.
func collectVarInits(decl *ast.GenDecl, out map[string]ast.Expr) {
	if decl.Tok != token.VAR {
		return
	}
	for _, spec := range decl.Specs {
		vs, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}
		for i, name := range vs.Names {
			if i < len(vs.Values) {
				out[name.Name] = vs.Values[i]
			}
		}
	}
}

// referencePaths returns, for each reference to name inside fn, the chain of
// enclosing nodes from the function body down to the reference.
func referencePaths(fn *ast.FuncDecl, name string) [][]ast.Node {
	var out [][]ast.Node
	var stack []ast.Node
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		stack = append(stack, n)
		if id, ok := n.(*ast.Ident); ok && id.Name == name {
			path := make([]ast.Node, len(stack))
			copy(path, stack)
			out = append(out, path)
		}
		return true
	})
	return out
}

// dominatedByGate reports whether the reference at the end of path is reached
// only when a condition in idents held.
//
// Two shapes count, and they are the two this repo uses. First, the reference
// sits inside the taken branch of an `if` (or a `case`) whose condition requires
// the feature positively. Second, an earlier statement in an enclosing block is
// an early-out guard, `if !hasAVX2 { return }`, which is a negated gate whose
// body leaves the function.
func dominatedByGate(path []ast.Node, pf *pkgFuncs, idents map[string]bool) bool {
	for i, n := range path {
		switch node := n.(type) {
		case *ast.IfStmt:
			// Only the then-branch is guarded by the condition. A reference in
			// the else-branch is reached precisely when the condition was false.
			if i+1 < len(path) && path[i+1] == ast.Node(node.Body) && positiveGate(node.Cond, pf, idents) {
				return true
			}
		case *ast.CaseClause:
			for _, expr := range node.List {
				if positiveGate(expr, pf, idents) {
					return true
				}
			}
		case *ast.BlockStmt:
			if i+1 < len(path) && earlyOutGuardBefore(node, path[i+1], pf, idents) {
				return true
			}
		}
	}
	return false
}

// earlyOutGuardBefore reports whether block contains, before stmt, an
// `if !<gate> { ... return ... }` guard.
func earlyOutGuardBefore(block *ast.BlockStmt, stmt ast.Node, pf *pkgFuncs, idents map[string]bool) bool {
	for _, s := range block.List {
		if ast.Node(s) == stmt {
			return false
		}
		ifStmt, ok := s.(*ast.IfStmt)
		if !ok || !negativeGate(ifStmt.Cond, pf, idents) {
			continue
		}
		if blockExits(ifStmt.Body) {
			return true
		}
	}
	return false
}

// blockExits reports whether every path through the block leaves the function.
// Only the shapes this repo uses are recognised: a trailing return, or a panic.
func blockExits(b *ast.BlockStmt) bool {
	if len(b.List) == 0 {
		return false
	}
	switch last := b.List[len(b.List)-1].(type) {
	case *ast.ReturnStmt:
		return true
	case *ast.ExprStmt:
		call, ok := last.X.(*ast.CallExpr)
		if !ok {
			return false
		}
		id, ok := call.Fun.(*ast.Ident)
		return ok && id.Name == "panic"
	default:
		return false
	}
}

// positiveGate reports whether cond requires the feature named by idents when
// true. A gate identifier under a `!` does not count: that is the inverted
// guard, which runs the kernel on exactly the CPUs that lack the feature.
func positiveGate(cond ast.Expr, pf *pkgFuncs, idents map[string]bool) bool {
	return gatePolarity(cond, pf, idents, false)
}

// negativeGate reports whether cond is true when the feature is ABSENT.
func negativeGate(cond ast.Expr, pf *pkgFuncs, idents map[string]bool) bool {
	return gatePolarity(cond, pf, idents, true)
}

// gateWalkDepth bounds the recursion through cached vars and predicate
// functions. Two hops covers every shape in this repo (`hasAVX2` resolving to
// cpu.X86.AVX2, and logSIMDOK32 returning a conjunction of them); the bound
// exists so a var or predicate that refers to itself cannot spin.
const gateWalkDepth = 4

// gatePolarity walks cond looking for a gate in idents whose sense matches want
// (want=false: the gate as written; want=true: the gate under a `!`).
func gatePolarity(cond ast.Expr, pf *pkgFuncs, idents map[string]bool, want bool) bool {
	found := false
	var walk func(e ast.Expr, negated bool, depth int)
	walk = func(e ast.Expr, negated bool, depth int) {
		if e == nil || found || depth > gateWalkDepth {
			return
		}
		switch x := e.(type) {
		case *ast.UnaryExpr:
			if x.Op == token.NOT {
				walk(x.X, !negated, depth)
				return
			}
			walk(x.X, negated, depth)
		case *ast.BinaryExpr:
			walk(x.X, negated, depth)
			walk(x.Y, negated, depth)
		case *ast.ParenExpr:
			walk(x.X, negated, depth)
		case *ast.SelectorExpr:
			if idents[x.Sel.Name] && negated == want {
				found = true
			}
		case *ast.Ident:
			if idents[x.Name] && negated == want {
				found = true
				return
			}
			// A cached gate such as `var hasAVX2 = cpu.X86.AVX2`. Resolving the
			// initializer rather than trusting the name is what makes redefining
			// it as cpu.X86.AVX fail the check.
			if init, ok := pf.varInits[x.Name]; ok {
				walk(init, negated, depth+1)
			}
		case *ast.CallExpr:
			// A predicate such as logSIMDOK32(n) whose body returns the gate.
			id, ok := x.Fun.(*ast.Ident)
			if ok && negated == want && predicateGates(id.Name, pf, idents, depth+1) {
				found = true
			}
		}
	}
	walk(cond, false, 0)
	return found
}

// predicateGates reports whether a same-package function returns an expression
// that requires the feature, which is how the log and pow family keeps its guard.
func predicateGates(name string, pf *pkgFuncs, idents map[string]bool, depth int) bool {
	fn := pf.funcs[name]
	if fn == nil || fn.Body == nil || depth > gateWalkDepth {
		return false
	}
	found := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if found {
			return false
		}
		ret, ok := n.(*ast.ReturnStmt)
		if !ok {
			return true
		}
		for _, r := range ret.Results {
			if gatePolarity(r, pf, idents, false) {
				found = true
			}
		}
		return true
	})
	return found
}

// sortedFuncNames returns the map's keys in a stable order so that a failure
// reports the same way on every run.
func sortedFuncNames(m map[string]*ast.FuncDecl) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}
