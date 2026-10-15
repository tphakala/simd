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
			pf.funcs[decl.Name.Name] = decl
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
