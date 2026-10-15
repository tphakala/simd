package simd

// These tests pin the dead-argument excusal in checkKernelGated directly, on
// small parsed sources, rather than only through the f32 tree where one helper
// (interleave2Kernels) happens to use it. Each case is a shape the excusal must
// either accept or refuse.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// parseDispatchSource parses one Go source into the pkgFuncs the dispatch
// check works on.
func parseDispatchSource(t *testing.T, src string) *pkgFuncs {
	t.Helper()
	f, err := parser.ParseFile(token.NewFileSet(), "src.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse: %v\n%s", err, src)
	}
	pf := &pkgFuncs{funcs: map[string]*ast.FuncDecl{}, varInits: map[string]ast.Expr{}}
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
	return pf
}

func TestDeadArgumentExcusal(t *testing.T) {
	const prelude = "package p\n\nvar cpu struct{ X86 struct{ AVX bool } }\n\nfunc kernel() {}\n\n"
	cases := []struct {
		name string
		body string
		// wantUngated is how many references to kernel the check must report.
		wantUngated int
	}{
		{
			name: "false at an ungated site and true at a gated one is accepted",
			body: `
func k(avx bool) { if avx { kernel() } }
func ungated() { k(false) }
func gated() { if cpu.X86.AVX { k(true) } }
`,
		},
		{
			name: "a conjunct of the guarding condition is accepted",
			body: `
func k(avx bool, n int) { if n > 0 && (avx) { kernel() } }
func ungated() { k(false, 1) }
func gated() { if cpu.X86.AVX { k(true, 1) } }
`,
		},
		{
			name: "true at an ungated site is refused",
			body: `
func k(avx bool) { if avx { kernel() } }
func ungated() { k(true) }
func gated() { if cpu.X86.AVX { k(true) } }
`,
			wantUngated: 1,
		},
		{
			name: "a non-literal argument at an ungated site is refused",
			body: `
func k(avx bool) { if avx { kernel() } }
func ungated(x bool) { k(x) }
func gated() { if cpu.X86.AVX { k(true) } }
`,
			wantUngated: 1,
		},
		{
			name: "a parameter assigned inside the helper is refused",
			body: `
func k(avx bool) { avx = true; if avx { kernel() } }
func ungated() { k(false) }
func gated() { if cpu.X86.AVX { k(true) } }
`,
			wantUngated: 1,
		},
		{
			name: "a parameter shadowed inside the helper is refused",
			body: `
func k(avx bool) { avx := true; if avx { kernel() } }
func ungated() { k(false) }
func gated() { if cpu.X86.AVX { k(true) } }
`,
			wantUngated: 1,
		},
		{
			name: "a parameter whose address is taken is refused",
			body: `
func k(avx bool) { p := &avx; *p = true; if avx { kernel() } }
func ungated() { k(false) }
func gated() { if cpu.X86.AVX { k(true) } }
`,
			wantUngated: 1,
		},
		{
			name: "a parameter redeclared by a var in a nested block is refused",
			body: `
func k(avx bool) { { var avx = true; _ = avx }; if avx { kernel() } }
func ungated() { k(false) }
func gated() { if cpu.X86.AVX { k(true) } }
`,
			wantUngated: 1,
		},
		{
			name: "a parameter reused as a range variable is refused",
			body: `
func k(avx bool, xs []int) { for avx := range xs { _ = avx }; if avx { kernel() } }
func ungated() { k(false, nil) }
func gated() { if cpu.X86.AVX { k(true, nil) } }
`,
			wantUngated: 1,
		},
		{
			name: "a parameter reused as a range value is refused",
			body: `
func k(avx bool, xs []bool) { for _, avx := range xs { _ = avx }; if avx { kernel() } }
func ungated() { k(false, nil) }
func gated() { if cpu.X86.AVX { k(true, nil) } }
`,
			wantUngated: 1,
		},
		{
			name: "a parameter reused as a closure parameter is refused",
			body: `
func k(avx bool) { f := func(avx bool) {}; f(true); if avx { kernel() } }
func ungated() { k(false) }
func gated() { if cpu.X86.AVX { k(true) } }
`,
			wantUngated: 1,
		},
		{
			name: "a guard parameter that is not the first bool is accepted",
			body: `
func k(first, avx bool, n int) { if avx { kernel() } }
func ungated() { k(true, false, 1) }
func gated() { if cpu.X86.AVX { k(true, true, 1) } }
`,
		},
		{
			name: "a guard parameter after a non-bool parameter is accepted",
			body: `
func k(n int, avx bool) { if avx { kernel() } }
func ungated() { k(1, false) }
func gated() { if cpu.X86.AVX { k(1, true) } }
`,
		},
		{
			name: "unrelated rebinding does not disable the excusal",
			body: `
func k(avx bool, xs []int) {
	var other int
	for i, x := range xs { other += i + x }
	f := func(m int) int { return m + other }
	_ = f(1)
	if avx { kernel() }
}
func ungated() { k(false, nil) }
func gated() { if cpu.X86.AVX { k(true, nil) } }
`,
		},
		{
			name: "a false literal at an unrelated parameter index is refused",
			body: `
func k(first, avx bool) { if avx { kernel() } }
func ungated() { k(false, true) }
func gated() { if cpu.X86.AVX { k(true, true) } }
`,
			wantUngated: 1,
		},
		{
			name: "the else branch of the parameter test is refused",
			body: `
func k(avx bool) { if avx { return } else { kernel() } }
func ungated() { k(false) }
func gated() { if cpu.X86.AVX { k(true) } }
`,
			wantUngated: 1,
		},
		{
			name: "a disjunction with the parameter is refused",
			body: `
func k(avx, x bool) { if avx || x { kernel() } }
func ungated() { k(false, true) }
func gated() { if cpu.X86.AVX { k(true, true) } }
`,
			wantUngated: 1,
		},
		{
			name: "the helper used as a value is refused",
			body: `
func k(avx bool) { if avx { kernel() } }
func ungated() { f := k; f(false) }
func gated() { if cpu.X86.AVX { k(true) } }
`,
			wantUngated: 1,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			pf := parseDispatchSource(t, prelude+c.body)
			ungated, refs := ungatedDispatches("kernel", pf, map[string]bool{"AVX": true})
			if refs == 0 {
				t.Fatal("no reference to kernel found; the fixture is broken")
			}
			if len(ungated) != c.wantUngated {
				t.Errorf("ungated references = %v, want %d", ungated, c.wantUngated)
			}
		})
	}
}

// TestGatePolarityBooleanStructure pins how the shared gate walk reads && and
// ||. A disjunction only gates when both sides do, since either side alone can
// make it true; under a negation the roles of && and || swap. Before this rule
// was written down, `cpu.X86.AVX || x` counted as an AVX gate.
func TestGatePolarityBooleanStructure(t *testing.T) {
	idents := map[string]bool{"AVX": true, "AVX2": true}
	pf := &pkgFuncs{funcs: map[string]*ast.FuncDecl{}, varInits: map[string]ast.Expr{}}
	cases := []struct {
		cond     string
		positive bool // the then-branch of `if cond` requires the feature
		negative bool // code after `if cond { return }` requires the feature
	}{
		{cond: "cpu.X86.AVX", positive: true},
		{cond: "cpu.X86.AVX && n > 8", positive: true},
		{cond: "cpu.X86.AVX || n > 8"},
		{cond: "cpu.X86.AVX || cpu.X86.AVX2", positive: true},
		{cond: "!(!cpu.X86.AVX || n < 8)", positive: true},
		{cond: "!(!cpu.X86.AVX && n < 8)"},
		{cond: "!cpu.X86.AVX", negative: true},
		{cond: "!cpu.X86.AVX || n < 8", negative: true},
		{cond: "!cpu.X86.AVX && n < 8"},
		{cond: "!(cpu.X86.AVX && n > 8)", negative: true},
		{cond: "!(cpu.X86.AVX || n > 8)"},
	}
	for _, c := range cases {
		e, err := parser.ParseExpr(c.cond)
		if err != nil {
			t.Fatalf("parse %q: %v", c.cond, err)
		}
		if got := positiveGate(e, pf, idents); got != c.positive {
			t.Errorf("positiveGate(%s) = %v, want %v", c.cond, got, c.positive)
		}
		if got := negativeGate(e, pf, idents); got != c.negative {
			t.Errorf("negativeGate(%s) = %v, want %v", c.cond, got, c.negative)
		}
	}
}

// TestGateShapesThatOnlySometimesRequireTheFeature pins three shapes that read
// like a gate but let the kernel run without the feature: a predicate with an
// ungated `return true`, a case list where one expression is ungated (a case
// list is an implicit ||), and a predicate that calls itself, which must be
// refused rather than recursing without bound.
func TestGateShapesThatOnlySometimesRequireTheFeature(t *testing.T) {
	const prelude = "package p\n\nvar cpu struct{ X86 struct{ AVX bool } }\n\nfunc kernel() {}\n\n"
	cases := []struct {
		name        string
		body        string
		wantUngated int
	}{
		{
			name: "a predicate whose every non-false return is a gate is accepted",
			body: `
func ok(n int) bool { if n < 4 { return false }; return cpu.X86.AVX }
func d(n int) { if ok(n) { kernel() } }
`,
		},
		{
			name: "a predicate with an ungated return true is refused",
			body: `
func leaky(n int) bool { if n < 4 { return true }; return cpu.X86.AVX }
func d(n int) { if leaky(n) { kernel() } }
`,
			wantUngated: 1,
		},
		{
			name: "a return inside a nested function literal does not count",
			body: `
func p(n int) bool { f := func() bool { return true }; _ = f; return cpu.X86.AVX }
func d(n int) { if p(n) { kernel() } }
`,
		},
		{
			name: "a self-recursive predicate is refused",
			body: `
func p(n int) bool { return p(n) }
func d(n int) { if p(n) { kernel() } }
`,
			wantUngated: 1,
		},
		{
			name: "mutually recursive predicates are refused",
			body: `
func p(n int) bool { return q(n) }
func q(n int) bool { return p(n) }
func d(n int) { if p(n) { kernel() } }
`,
			wantUngated: 1,
		},
		{
			name: "a case list where every expression is a gate is accepted",
			body: `
func d(n int) { switch { case cpu.X86.AVX && n > 8, cpu.X86.AVX: kernel() } }
`,
		},
		{
			name: "a case list with one ungated expression is refused",
			body: `
func d(n int) { switch { case cpu.X86.AVX, n > 8: kernel() } }
`,
			wantUngated: 1,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			pf := parseDispatchSource(t, prelude+c.body)
			ungated, refs := ungatedDispatches("kernel", pf, map[string]bool{"AVX": true})
			if refs == 0 {
				t.Fatal("no reference to kernel found; the fixture is broken")
			}
			if len(ungated) != c.wantUngated {
				t.Errorf("ungated references = %v, want %d", ungated, c.wantUngated)
			}
		})
	}
}
