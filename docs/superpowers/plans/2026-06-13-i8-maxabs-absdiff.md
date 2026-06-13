# i8 MaxAbs + AbsDiff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add two int8 "absolute value" primitives to `i8`: `MaxAbs` (the Tier 1 per-tensor dynamic-quantization scale, a reduction) and `AbsDiff` (the Tier 2/3 saturating absolute difference, element-wise), with AVX2 + NEON kernels, pure-Go references, parity/bit-exact/zero-alloc tests, differential fuzzing, and asmcheck cross-checks.

**Architecture:** Same structure as the existing `i8` package. `MaxAbs` returns `int` because `|-128| = 128` does not fit `int8`; it reduces via unsigned byte max (`VPABSB`+`VPMAXUB`+horizontal `VPMAXUB`; `ABS`+`UMAX`+`UMAXV` on NEON), reading the final byte unsigned. `AbsDiff` saturates `|a-b|` to `[0,127]` consistent with `Abs` (`max(sat(a-b), sat(b-a))` on AVX2; `SABD`+`UMIN`-with-127 on NEON).

**Tech Stack:** Go 1.26, Plan9 assembly (AVX2 mnemonics on amd64; hand-encoded `WORD` NEON cross-checked by `asmcheck`).

---

## Semantics (source of truth)

- `MaxAbs(a []int8) int`: returns `max_i |int(a[i])|`, range `[0,128]`. Empty `a` returns 0. Read-only, zero-alloc.
- `AbsDiff(dst, a, b []int8)`: `dst[i] = clamp(|int(a[i]) - int(b[i])|, 0, 127)` for `i in [0,n)`, `n = min(len(dst),len(a),len(b))`. Saturates: `|127 - (-128)| = 255 -> 127`. Trailing dst capacity untouched. Zero-alloc.

## Verified instruction encodings (cross-assembler authoritative; agy cross-checked)

x86 AVX2:
- `MaxAbs`: `VPABSB` (abs; `-128` -> byte `0x80` = 128 unsigned), `VPMAXUB` (unsigned byte max) to fold blocks, horizontal `VPMAXUB`+`VPSRLDQ` cascade + `VEXTRACTI128`; final byte read zero-extended.
- `AbsDiff`: `VPSUBSB(b,a)` and `VPSUBSB(a,b)` (signed saturating), `VPMAXSB` picks the non-negative capped difference. (`SABD` is not used on x86; no such mnemonic. The max-of-sat-subs already saturates to `[0,127]`.)

NEON .16B (all base ARMv8.0 ASIMD, decodable by arm64asm -> checked directly):
- `ABS   V2.16B, V0.16B`        = `0x4E20B802`
- `ABS   V0.16B, V0.16B`        = `0x4E20B800`
- `UMAXV B3, V2.16B`            = `0x6E30A843`
- `UMAXV B4, V0.16B`            = `0x6E30A804`
- `UMAX  V0.16B, V0.16B, V2.16B`= `0x6E226400`
- `SABD  V2.16B, V0.16B, V1.16B`= `0x4E217402`
- `UMIN  V2.16B, V2.16B, V3.16B`= `0x6E236C42`
- `DUP   V3.16B, W5`            = `0x4E010CA3`

---

## Task 1: References + public API + non-arch routing (TDD: tests first)

**Files:** `i8/i8_test.go` (tests first), `i8/i8_go.go` (refs), `i8/i8.go` (API), `i8/i8_other.go` (routing).

- [ ] **Step 1 (RED):** add `TestMaxAbs` and `TestAbsDiff` with literal edge cases (`MaxAbs([-128])==128`, `MaxAbs(nil)==0`, `AbsDiff` `|127-(-128)|==127`, mismatched-length clamp) plus parity across `lengths`. Extend `TestZeroAllocations` (`MaxAbs`, `AbsDiff`) and `TestTrailingCapacityUntouched` (`AbsDiff`). Run; expect compile failure.
- [ ] **Step 2 (GREEN refs):** add to `i8_go.go`:

```go
func maxAbsGo(a []int8) int {
	m := 0
	for _, v := range a {
		av := int(v)
		if av < 0 {
			av = -av
		}
		if av > m {
			m = av
		}
	}
	return m
}

func absDiffGo(dst, a, b []int8) {
	for i := range dst {
		d := int(a[i]) - int(b[i])
		if d < 0 {
			d = -d
		}
		if d > 127 {
			d = 127
		}
		dst[i] = int8(d)
	}
}
```

(Check the modernize linter: the `if av < 0` / `if d < 0` are real branches, not min/max-able. `if av > m` may be flagged -> use `m = max(m, av)`.)

- [ ] **Step 3 (GREEN API):** add to `i8.go`:

```go
// MaxAbs returns max_i |a[i]| accumulated as int (range [0,128], since
// |-128| = 128 does not fit int8). It is the per-tensor scale for dynamic
// quantization. Empty a returns 0. a is read-only; allocates nothing.
func MaxAbs(a []int8) int {
	if len(a) == 0 {
		return 0
	}
	return maxAbsI8(a)
}

// AbsDiff writes the saturating absolute difference dst[i] = |a[i] - b[i]|,
// clamped to [0, 127], for i in [0, n), n = min(len(dst),len(a),len(b)).
// |127 - (-128)| = 255 saturates to 127. Any trailing capacity in dst is
// left untouched.
func AbsDiff(dst, a, b []int8) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	absDiffI8(dst[:n], a[:n], b[:n])
}
```

- [ ] **Step 4:** add to `i8_other.go`:

```go
func maxAbsI8(a []int8) int        { return maxAbsGo(a) }
func absDiffI8(dst, a, b []int8)    { absDiffGo(dst, a, b) }
```

- [ ] **Step 5:** `SIMD_DISABLE=avx go test ./i8/ -run 'MaxAbs|AbsDiff|ZeroAlloc|Trailing' -count=1`. Expect PASS (Go path). Commit `feat(i8): add MaxAbs/AbsDiff references and API`.

## Task 2: AMD64 AVX2 kernels

**Files:** `i8/i8_amd64.go` (dispatch + decls), `i8/i8_amd64.s` (2 kernels).

- [ ] **Step 1:** dispatch `maxAbsI8` (gate `len >= blockMinMax` 32) and `absDiffI8` (gate `len(dst) >= blockSat32` 32), else Go ref; add `//go:noescape func maxAbsAVX2(a []int8) int` and `func absDiffAVX2(dst, a, b []int8)`.
- [ ] **Step 2:** `maxAbsAVX2`: zero Y0 acc; per 32-byte block `VPABSB (SI),Y1` then `VPMAXUB Y1,Y0,Y0`; reduce with `VEXTRACTI128`+`VPMAXUB`+`VPSRLDQ` cascade to one byte; `MOVD X0,AX; ANDQ $0xFF,AX`; signed scalar tail computes `|v|` (TEST/NEG) and unsigned-compares to update AX; `MOVQ AX,ret+24(FP)`. `absDiffAVX2`: per block `VPSUBSB Y1,Y0,Y2` (a-b) and `VPSUBSB Y0,Y1,Y3` (b-a), `VPMAXSB Y3,Y2,Y4`, store; scalar tail subtract/abs/clamp-127.
- [ ] **Step 3:** `go vet ./i8/`; `go test ./i8/ -run 'MaxAbs|AbsDiff' -count=1 -v`. Expect PASS.
- [ ] **Step 4:** commit `feat(i8): AVX2 kernels for MaxAbs/AbsDiff`.

## Task 3: ARM64 NEON kernels

**Files:** `i8/i8_arm64.go` (dispatch + decls), `i8/i8_arm64.s` (2 kernels).

- [ ] **Step 1:** dispatch gating `len >= minNEON16` (16); decls `maxAbsNEON`, `absDiffNEON`.
- [ ] **Step 2:** `maxAbsNEON`: zero V_acc; per 16-byte block `VLD`, `ABS V1,V0` (`WORD 0x4E20B800`->use V-combo), `UMAX` fold; `UMAXV B,Vacc` reduce; `FMOVS`+`AND $0xFF`; signed scalar tail abs+unsigned-CSEL. `absDiffNEON`: `DUP V3.16B,W` (127), per block `SABD V2,V0,V1` (`0x4E217402`), `UMIN V2,V2,V3` (`0x6E236C42`), store; scalar tail subtract/abs/clamp-127.
- [ ] **Step 3:** `GOARCH=arm64 go build ./i8/`, `GOARCH=arm64 go vet ./i8/`, `SIMD_REQUIRE_OBJDUMP=1 go test . -run TestArm64WordEncodings`. Expect PASS.
- [ ] **Step 4:** commit `feat(i8): NEON kernels for MaxAbs/AbsDiff`.

## Task 4: Fuzz + benchmarks + cross-arch + docs

- [ ] **Step 1:** add `FuzzI8AbsOps` (MaxAbs vs ref; AbsDiff vs ref on a/b halves) to `fuzz_test.go`; `BenchmarkMaxAbs`, `BenchmarkAbsDiff` to `benchmark_test.go`.
- [ ] **Step 2:** amd64: `go test ./i8/ -count=1`, short fuzz, `golangci-lint run ./i8/...`. Expect clean.
- [ ] **Step 3:** cross-compile + run on rpi5 (native aarch64): full i8 run + fuzz seeds + benchmarks. Expect PASS, 0 allocs.
- [ ] **Step 4:** docs: add `MaxAbs`/`AbsDiff` to `doc.go` i8 line, README i8 table + example, `i8.go` package doc. Remove the "MaxAbs ... follow-up" note from the README planned-follow-ups line if present.
- [ ] **Step 5:** `go test ./... -count=1` (full repo), `go test ./i8/ -race`. Commit docs; push; PR (base = feat/i8-elementwise-tier2, stacked), referencing #132.

---

## Self-review notes

- `MaxAbs` returns `int` (not `int8`) precisely because of the `-128` case; literal test `MaxAbs([]int8{-128}) == 128` is the guard.
- `AbsDiff` saturates to `[0,127]` (not raw `|a-b|` which can be 255) for consistency with `Abs`; the `|127-(-128)|==127` literal test is the guard. On NEON `SABD` gives unsigned `|a-b|` in `[0,255]`, then `UMIN` with 127 saturates.
- Unsigned reductions: `VPMAXUB`/`UMAXV` are unsigned, so `VPABSB`/`ABS` producing `0x80` for `-128` is correctly treated as 128 (the largest), not as a negative.
</content>
