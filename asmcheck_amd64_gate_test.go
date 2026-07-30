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
// resolved to their initializer rather than trusted by name; see gateVars.
var avx2GateIdents = map[string]bool{
	"AVX2": true, "AVX512F": true, "AVX512VL": true, "AVXVNNI": true,
}

// pkgFuncs is one package's amd64 dispatch source, parsed once.
type pkgFuncs struct {
	funcs map[string]*ast.FuncDecl
	// gateVars holds package-level bools whose initializer establishes AVX2,
	// for example `var hasAVX2 = cpu.X86.AVX2`. Resolving the initializer is
	// what stops the check trusting a name: redefining hasAVX2 as cpu.X86.AVX
	// must fail, and by name alone it would not.
	gateVars map[string]bool
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
			checkKernelGated(t, file, pkgDir, k, pf)
		}
	}
}

// checkKernelGated reports every dispatch of one AVX2-requiring kernel that is
// not dominated by an AVX2 condition.
func checkKernelGated(t *testing.T, file, pkgDir string, k asmcheck.X86Kernel, pf *pkgFuncs) {
	t.Helper()
	refs := 0
	for _, name := range sortedFuncNames(pf.funcs) {
		fn := pf.funcs[name]
		if name == k.Name || fn.Body == nil {
			continue
		}
		for _, path := range referencePaths(fn, k.Name) {
			refs++
			if dominatedByAVX2(path, fn, pf) {
				continue
			}
			t.Errorf("%s: kernel %s needs AVX2, but %s.%s dispatches it from a branch "+
				"that does not require AVX2.\n"+
				"\tfirst AVX2 use: %s:%d %s\n"+
				"An AVX1-only CPU (Intel Sandy/Ivy Bridge, AMD Bulldozer) or an "+
				"AVX+FMA3-without-AVX2 CPU (AMD Piledriver, Steamroller) reaching this "+
				"kernel faults with SIGILL. That is #196/#197. Restore the AVX2 guard on "+
				"this branch, or move the kernel behind one.",
				file, k.Name, pkgDir, name, file, firstUseLine(k), firstUseText(k))
		}
	}
	if refs == 0 {
		t.Errorf("%s: kernel %s needs AVX2 but nothing in %s references it. "+
			"Either it is dead code, or its dispatch lives somewhere this test does not "+
			"parse (only %s/*_amd64.go is read).", file, k.Name, pkgDir, pkgDir)
	}
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
	pf := &pkgFuncs{funcs: map[string]*ast.FuncDecl{}, gateVars: map[string]bool{}}
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
				collectGateVars(decl, pf.gateVars)
			}
		}
	}
	return pf
}

// collectGateVars records package-level vars whose initializer establishes AVX2.
func collectGateVars(decl *ast.GenDecl, out map[string]bool) {
	if decl.Tok != token.VAR {
		return
	}
	for _, spec := range decl.Specs {
		vs, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}
		for i, name := range vs.Names {
			if i < len(vs.Values) && mentionsGateIdent(vs.Values[i]) {
				out[name.Name] = true
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

// dominatedByAVX2 reports whether the reference at the end of path is reached
// only when an AVX2 condition held.
//
// Two shapes count, and they are the two this repo uses. First, the reference
// sits inside the taken branch of an `if` (or a `case`) whose condition requires
// AVX2 positively. Second, an earlier statement in an enclosing block is an
// early-out guard, `if !hasAVX2 { return }`, which is a negated gate whose body
// leaves the function.
func dominatedByAVX2(path []ast.Node, fn *ast.FuncDecl, pf *pkgFuncs) bool {
	for i, n := range path {
		switch node := n.(type) {
		case *ast.IfStmt:
			// Only the then-branch is guarded by the condition. A reference in
			// the else-branch is reached precisely when the condition was false.
			if i+1 < len(path) && path[i+1] == ast.Node(node.Body) && positiveGate(node.Cond, pf) {
				return true
			}
		case *ast.CaseClause:
			for _, expr := range node.List {
				if positiveGate(expr, pf) {
					return true
				}
			}
		case *ast.BlockStmt:
			if i+1 < len(path) && earlyOutGuardBefore(node, path[i+1], pf) {
				return true
			}
		}
	}
	_ = fn
	return false
}

// earlyOutGuardBefore reports whether block contains, before stmt, an
// `if !<avx2 gate> { ... return ... }` guard.
func earlyOutGuardBefore(block *ast.BlockStmt, stmt ast.Node, pf *pkgFuncs) bool {
	for _, s := range block.List {
		if ast.Node(s) == stmt {
			return false
		}
		ifStmt, ok := s.(*ast.IfStmt)
		if !ok || !negativeGate(ifStmt.Cond, pf) {
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

// positiveGate reports whether cond requires AVX2 when true. A gate identifier
// under a `!` does not count: that is the inverted guard, which runs the kernel
// on exactly the CPUs that lack the feature.
func positiveGate(cond ast.Expr, pf *pkgFuncs) bool {
	return gatePolarity(cond, pf, false)
}

// negativeGate reports whether cond is true when AVX2 is ABSENT.
func negativeGate(cond ast.Expr, pf *pkgFuncs) bool {
	return gatePolarity(cond, pf, true)
}

// gatePolarity walks cond looking for an AVX2 gate whose sense matches want
// (want=false: the gate as written; want=true: the gate under a `!`).
func gatePolarity(cond ast.Expr, pf *pkgFuncs, want bool) bool {
	found := false
	var walk func(e ast.Expr, negated bool)
	walk = func(e ast.Expr, negated bool) {
		if e == nil || found {
			return
		}
		switch x := e.(type) {
		case *ast.UnaryExpr:
			if x.Op == token.NOT {
				walk(x.X, !negated)
				return
			}
			walk(x.X, negated)
		case *ast.BinaryExpr:
			walk(x.X, negated)
			walk(x.Y, negated)
		case *ast.ParenExpr:
			walk(x.X, negated)
		case *ast.SelectorExpr:
			if avx2GateIdents[x.Sel.Name] && negated == want {
				found = true
			}
		case *ast.Ident:
			if (avx2GateIdents[x.Name] || pf.gateVars[x.Name]) && negated == want {
				found = true
			}
		case *ast.CallExpr:
			// A predicate such as logSIMDOK32(n) whose body returns the gate.
			if id, ok := x.Fun.(*ast.Ident); ok && negated == want && predicateGates(id.Name, pf) {
				found = true
			}
		}
	}
	walk(cond, false)
	return found
}

// predicateGates reports whether a same-package function returns an expression
// that requires AVX2, which is how the log and pow family keeps its guard.
func predicateGates(name string, pf *pkgFuncs) bool {
	fn := pf.funcs[name]
	if fn == nil || fn.Body == nil {
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
			if gatePolarity(r, pf, false) {
				found = true
			}
		}
		return true
	})
	return found
}

// mentionsGateIdent reports whether e contains an AVX2 feature identifier.
func mentionsGateIdent(e ast.Expr) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		if found {
			return false
		}
		switch x := n.(type) {
		case *ast.SelectorExpr:
			if avx2GateIdents[x.Sel.Name] {
				found = true
			}
		case *ast.Ident:
			if avx2GateIdents[x.Name] {
				found = true
			}
		}
		return !found
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
