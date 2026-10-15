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
	ungated, refs := ungatedDispatches(k.Name, pf, ax.idents)
	for _, name := range ungated {
		t.Errorf("%s: kernel %s needs %s, but %s.%s dispatches it from a branch "+
			"that does not require %s, and not every call site of %s requires it "+
			"either.\n\tfirst %s use: %s:%d %s\n%s",
			file, k.Name, ax.name, pkgDir, name, ax.name, name,
			ax.name, file, ax.firstLine(k), ax.firstText(k), ax.remedy)
	}
	if refs == 0 {
		t.Errorf("%s: kernel %s needs %s but nothing in %s references it. "+
			"Either it is dead code, or its dispatch lives somewhere this test does not "+
			"parse (only %s/*_amd64.go is read).", file, k.Name, ax.name, pkgDir, pkgDir)
	}
}

// ungatedDispatches returns the name of the function holding each reference to
// kernel that is not gated, one entry per reference, and the total number of
// references found.
func ungatedDispatches(kernel string, pf *pkgFuncs, idents map[string]bool) (ungated []string, refs int) {
	for _, name := range sortedFuncNames(pf.funcs) {
		fn := pf.funcs[name]
		if name == kernel || fn.Body == nil {
			continue
		}
		for _, path := range referencePaths(fn, kernel) {
			refs++
			if dominatedByGate(path, bindParams(pf, name), idents) {
				continue
			}
			// The guard may sit one frame up. Three shapes in this repo put it
			// there: the initXxx tier assigners, whose selecting case lives in
			// init(); helpers such as dotProductBatchKernel that take the tier
			// decision as a bool parameter and are only ever called from an
			// already-gated branch; and a bool-parameter helper that is also
			// called with a literal false from the non-SIMD tiers, which
			// interleave2Kernels is in f32.
			if callSitesGated(name, pf, idents, 0) || deadArgSitesGated(fn, path, pf, idents) {
				continue
			}
			ungated = append(ungated, name)
		}
	}
	return ungated, refs
}

// guardParams returns the indices of fn's bool parameters that are a bare
// identifier among the top-level && conjuncts of an `if` condition on path whose
// then-branch holds the reference. When such a parameter is false the reference
// is unreachable. A parameter the body assigns, redefines, shadows or takes the
// address of is dropped, because then its value at the `if` is not the argument.
func guardParams(fn *ast.FuncDecl, path []ast.Node) []int {
	idx := map[string]int{}
	n := 0
	if fn.Type.Params != nil {
		for _, field := range fn.Type.Params.List {
			isBool := false
			if id, ok := field.Type.(*ast.Ident); ok {
				isBool = id.Name == "bool"
			}
			for _, id := range field.Names {
				if isBool && id.Name != "_" {
					idx[id.Name] = n
				}
				n++
			}
		}
	}
	for name := range rebound(fn) {
		delete(idx, name)
	}

	var out []int
	var conjuncts func(e ast.Expr)
	conjuncts = func(e ast.Expr) {
		switch x := e.(type) {
		case *ast.ParenExpr:
			conjuncts(x.X)
		case *ast.BinaryExpr:
			if x.Op == token.LAND {
				conjuncts(x.X)
				conjuncts(x.Y)
			}
		case *ast.Ident:
			if i, ok := idx[x.Name]; ok {
				out = append(out, i)
			}
		}
	}
	// Only an IfStmt counts. In a tagged switch the case expressions are values
	// compared against the tag, not conditions.
	for i, nd := range path {
		if ifs, ok := nd.(*ast.IfStmt); ok && i+1 < len(path) && path[i+1] == ast.Node(ifs.Body) {
			conjuncts(ifs.Cond)
		}
	}
	return out
}

// rebound returns every name fn's body assigns, declares, ranges over, binds as
// a closure parameter, or takes the address of.
func rebound(fn *ast.FuncDecl) map[string]bool {
	out := map[string]bool{}
	mark := func(e ast.Expr) {
		if id, ok := e.(*ast.Ident); ok {
			out[id.Name] = true
		}
	}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.AssignStmt:
			for _, l := range x.Lhs {
				mark(l)
			}
		case *ast.ValueSpec:
			for _, id := range x.Names {
				out[id.Name] = true
			}
		case *ast.RangeStmt:
			if x.Key != nil {
				mark(x.Key)
			}
			if x.Value != nil {
				mark(x.Value)
			}
		case *ast.FuncLit:
			for _, field := range x.Type.Params.List {
				for _, id := range field.Names {
					out[id.Name] = true
				}
			}
		case *ast.UnaryExpr:
			if x.Op == token.AND {
				mark(x.X)
			}
		}
		return true
	})
	return out
}

// deadArgSitesGated reports whether the reference at the end of path, inside fn,
// is unreachable at every call site that is not itself gated: each call site of
// fn either passes the literal false for a parameter guardParams found, or is
// gated as callSitesGated understands it. At least one call site must exist, and
// a use of fn as a value is never excused.
func deadArgSitesGated(fn *ast.FuncDecl, path []ast.Node, pf *pkgFuncs, idents map[string]bool) bool {
	guard := guardParams(fn, path)
	if len(guard) == 0 {
		return false
	}
	name := fn.Name.Name
	sites := 0
	for _, caller := range sortedFuncNames(pf.funcs) {
		cf := pf.funcs[caller]
		if cf.Body == nil || caller == name {
			continue
		}
		for _, p := range referencePaths(cf, name) {
			sites++
			if call := enclosingCall(p); call != nil && passesLiteralFalse(call, guard) {
				continue
			}
			if dominatedByGate(p, bindParams(pf, caller), idents) ||
				callSitesGated(caller, pf, idents, 1) {
				continue
			}
			return false
		}
	}
	return sites > 0
}

// enclosingCall returns the call whose function expression is the identifier at
// the end of path, or nil when that identifier is used as a value.
func enclosingCall(path []ast.Node) *ast.CallExpr {
	if len(path) < 2 {
		return nil
	}
	call, ok := path[len(path)-2].(*ast.CallExpr)
	if !ok || call.Fun != path[len(path)-1] {
		return nil
	}
	return call
}

// passesLiteralFalse reports whether call passes the literal false at any of the
// given argument indices.
func passesLiteralFalse(call *ast.CallExpr, indices []int) bool {
	for _, i := range indices {
		if i < len(call.Args) {
			if id, ok := call.Args[i].(*ast.Ident); ok && id.Name == "false" {
				return true
			}
		}
	}
	return false
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
			// A case list is an implicit ||: any one expression selects the
			// case, so every one of them has to require the feature. The
			// default clause (an empty list) is reached when none held.
			if len(node.List) > 0 && allPositiveGates(node.List, pf, idents) {
				return true
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

// allPositiveGates reports whether every expression in list requires the
// feature when true.
func allPositiveGates(list []ast.Expr, pf *pkgFuncs, idents map[string]bool) bool {
	for _, expr := range list {
		if !positiveGate(expr, pf, idents) {
			return false
		}
	}
	return true
}

// gateWalkDepth bounds the recursion through cached vars and predicate
// functions. Two hops covers every shape in this repo (`hasAVX2` resolving to
// cpu.X86.AVX2, and logSIMDOK32 returning a conjunction of them); the bound
// exists so a var or predicate that refers to itself cannot spin. The depth is
// carried into a predicate's return expressions rather than restarted there.
const gateWalkDepth = 4

// gatePolarity walks cond looking for a gate in idents whose sense matches want
// (want=false: the gate as written; want=true: the gate under a `!`).
func gatePolarity(cond ast.Expr, pf *pkgFuncs, idents map[string]bool, want bool) bool {
	return gateWalk(cond, pf, idents, want, 0)
}

// gateWalk is gatePolarity starting at a given recursion depth.
func gateWalk(cond ast.Expr, pf *pkgFuncs, idents map[string]bool, want bool, depth int) bool {
	// walk reports whether e, assumed to evaluate to the value the walk is
	// tracking, implies the feature. holds is that assumed value: true when an
	// even number of `!` sit above e and want is false, or an odd number and
	// want is true. A conjunction assumed true implies the feature when either
	// side does; a disjunction assumed true needs both sides, since either alone
	// can make it true. Assumed false, the two swap (De Morgan).
	var walk func(e ast.Expr, negated bool, depth int) bool
	walk = func(e ast.Expr, negated bool, depth int) bool {
		if e == nil || depth > gateWalkDepth {
			return false
		}
		holds := negated == want
		switch x := e.(type) {
		case *ast.UnaryExpr:
			if x.Op == token.NOT {
				return walk(x.X, !negated, depth)
			}
			return walk(x.X, negated, depth)
		case *ast.BinaryExpr:
			switch {
			case x.Op == token.LAND && holds, x.Op == token.LOR && !holds:
				return walk(x.X, negated, depth) || walk(x.Y, negated, depth)
			case x.Op == token.LOR && holds, x.Op == token.LAND && !holds:
				return walk(x.X, negated, depth) && walk(x.Y, negated, depth)
			}
			return false
		case *ast.ParenExpr:
			return walk(x.X, negated, depth)
		case *ast.SelectorExpr:
			return idents[x.Sel.Name] && holds
		case *ast.Ident:
			if idents[x.Name] {
				return holds
			}
			// A cached gate such as `var hasAVX2 = cpu.X86.AVX2`. Resolving the
			// initializer rather than trusting the name is what makes redefining
			// it as cpu.X86.AVX fail the check.
			if init, ok := pf.varInits[x.Name]; ok {
				return walk(init, negated, depth+1)
			}
		case *ast.CallExpr:
			// A predicate such as logSIMDOK32(n) whose body returns the gate.
			id, ok := x.Fun.(*ast.Ident)
			return ok && holds && predicateGates(id.Name, pf, idents, depth+1)
		}
		return false
	}
	return walk(cond, false, depth)
}

// predicateGates reports whether a same-package function only returns true when
// the feature is present, which is how the log and pow family keeps its guard.
// Every return must be a gate, except a literal false, which never makes the
// caller's condition hold; a single ungated `return true` would let the kernel
// run without the feature. Returns inside nested function literals belong to
// those literals and are skipped.
func predicateGates(name string, pf *pkgFuncs, idents map[string]bool, depth int) bool {
	fn := pf.funcs[name]
	if fn == nil || fn.Body == nil || depth > gateWalkDepth {
		return false
	}
	gated, ungated := false, false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.FuncLit:
			return false
		case *ast.ReturnStmt:
			for _, r := range x.Results {
				if id, ok := r.(*ast.Ident); ok && id.Name == "false" {
					continue
				}
				if gateWalk(r, pf, idents, false, depth) {
					gated = true
				} else {
					ungated = true
				}
			}
		}
		return true
	})
	return gated && !ungated
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
