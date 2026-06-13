# i8 Tier 2 element-wise primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five single-pass element-wise int8 primitives to the `i8` package: `Min`, `Max`, `Clamp`, `Abs`, `Neg` (the "Tier 2 element-wise" group from issue #132), with AVX2 + NEON kernels, pure-Go references, parity/bit-exact/zero-alloc tests, differential fuzzing, and asmcheck cross-checks.

**Architecture:** Mirror the existing `i8` package exactly. Each op gets a pure-Go reference in `i8_go.go` (the source of truth), a public validating wrapper in `i8.go` that clamps `n` and dispatches, an amd64 dispatch + AVX2 kernel, and an arm64 dispatch + NEON kernel. Saturating `Abs`/`Neg` use the saturating instructions (NEON `SQABS`/`SQNEG`; x86 `VPMAXSB`+`VPSUBSB` / `VPSUBSB` from zero) so `abs(-128) = neg(-128) = 127`. Non-arch builds route through the Go references in `i8_other.go`.

**Tech Stack:** Go 1.26, Plan9 assembly (AVX2 mnemonics on amd64; hand-encoded `WORD` NEON on arm64, cross-checked by `asmcheck`).

---

## Semantics (source of truth for the references)

- `Min(dst, a, b)`: `dst[i] = min(int8(a[i]), int8(b[i]))` signed, `n = min(len(dst),len(a),len(b))`.
- `Max(dst, a, b)`: `dst[i] = max(int8(a[i]), int8(b[i]))` signed.
- `Clamp(dst, src, lo, hi)`: `dst[i] = min(max(src[i], lo), hi)`, `n = min(len(dst),len(src))`. With `lo > hi` the result equals `hi` for every element (max-then-min ordering); documented, not validated against.
- `Abs(dst, a)`: `dst[i] = clamp(abs(int(a[i])), -128, 127)`, so `abs(-128) = 127`, otherwise the ordinary absolute value (always >= 0).
- `Neg(dst, a)`: `dst[i] = clamp(-int(a[i]), -128, 127)`, so `neg(-128) = 127`, `neg(127) = -127`.

All write exactly `n` elements and leave trailing dst capacity untouched. Zero allocation. Safe for concurrent use.

## Verified instruction encodings (authoritative)

x86 AVX2 (Go asm 3-operand order is dst-last; `VPSUBSB a,b,c` => `c = b - a` saturating signed, `VPMAXSB a,b,c` => `c = max(b,a)`, `VPMINSB a,b,c` => `c = min(b,a)`):
- `Min`: `VPMINSB`.  `Max`: `VPMAXSB`.
- `Clamp`: `VPBROADCASTB` the lo/hi bytes to YMM, then `VPMAXSB(src, loVec)` then `VPMINSB(., hiVec)`.
- `Neg`: `VPSUBSB(x, zero)` => `0 - x` saturating (`0-(-128)=127`).
- `Abs`: `VPMAXSB(x, VPSUBSB(x, zero))` => `max(x, -x)` with saturating neg (`abs(-128)=127`).
- `VPABSB` is NOT used: it is non-saturating (`abs(-128)` stays `-128`).

NEON .16B (verified with `aarch64-linux-gnu-as` + `objdump`, all base ARMv8.0 ASIMD, no feature gate beyond NEON):
- `SMIN  V2.16B, V0.16B, V1.16B` = `0x4E216C02`
- `SMAX  V2.16B, V0.16B, V1.16B` = `0x4E216402`
- `SQABS V2.16B, V0.16B`         = `0x4E207802`
- `SQNEG V2.16B, V0.16B`         = `0x6E207802`
- `DUP   V3.16B, W3`             = `0x4E010C63`
- `DUP   V4.16B, W4`             = `0x4E010C84`
(other register combos computed the same way during implementation; every `WORD` is decodable by `arm64asm`, so `TestArm64WordEncodings` checks them directly.)

---

## Task 1: Pure-Go references + public API

**Files:**
- Modify: `i8/i8_go.go` (add `minGo`, `maxGo`, `clampGo`, `absGo`, `negGo`)
- Modify: `i8/i8.go` (add public `Min`, `Max`, `Clamp`, `Abs`, `Neg` + doc)
- Modify: `i8/i8_other.go` (route the five ops to Go refs on non-amd64/arm64)

- [ ] **Step 1: Add references to `i8/i8_go.go`** (after `subSatGo`):

```go
func minGo(dst, a, b []int8) {
	for i := range dst {
		if a[i] < b[i] {
			dst[i] = a[i]
		} else {
			dst[i] = b[i]
		}
	}
}

func maxGo(dst, a, b []int8) {
	for i := range dst {
		if a[i] > b[i] {
			dst[i] = a[i]
		} else {
			dst[i] = b[i]
		}
	}
}

// clampGo clamps each element to [lo, hi]. With lo > hi every element maps to
// hi (the max-then-min ordering), matching the SIMD kernels.
func clampGo(dst, src []int8, lo, hi int8) {
	for i := range dst {
		v := src[i]
		if v < lo {
			v = lo
		}
		if v > hi {
			v = hi
		}
		dst[i] = v
	}
}

// absGo writes the saturating absolute value: abs(-128) clamps to 127.
func absGo(dst, a []int8) {
	for i := range dst {
		dst[i] = clampI8(absInt(int(a[i])))
	}
}

// negGo writes the saturating negation: -(-128) clamps to 127.
func negGo(dst, a []int8) {
	for i := range dst {
		dst[i] = clampI8(-int(a[i]))
	}
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
```

- [ ] **Step 2: Add public wrappers to `i8/i8.go`** (after `MinMax`):

```go
// Min writes dst[i] = min(a[i], b[i]) (signed) for i in [0, n),
// n = min(len(dst), len(a), len(b)). Any trailing capacity in dst is untouched.
func Min(dst, a, b []int8) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	minI8(dst[:n], a[:n], b[:n])
}

// Max writes dst[i] = max(a[i], b[i]) (signed) for i in [0, n),
// n = min(len(dst), len(a), len(b)). Any trailing capacity in dst is untouched.
func Max(dst, a, b []int8) {
	n := min(len(dst), len(a), len(b))
	if n == 0 {
		return
	}
	maxI8(dst[:n], a[:n], b[:n])
}

// Clamp writes dst[i] = min(max(src[i], lo), hi) (signed) for i in [0, n),
// n = min(len(dst), len(src)). If lo > hi every element maps to hi. Any
// trailing capacity in dst is untouched.
func Clamp(dst, src []int8, lo, hi int8) {
	n := min(len(dst), len(src))
	if n == 0 {
		return
	}
	clampI8(dst[:n], src[:n], lo, hi)
}

// Abs writes the saturating absolute value dst[i] = |a[i]| for i in [0, n),
// n = min(len(dst), len(a)). abs(-128) saturates to 127. Any trailing capacity
// in dst is untouched.
func Abs(dst, a []int8) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	absI8(dst[:n], a[:n])
}

// Neg writes the saturating negation dst[i] = -a[i] for i in [0, n),
// n = min(len(dst), len(a)). neg(-128) saturates to 127. Any trailing capacity
// in dst is untouched.
func Neg(dst, a []int8) {
	n := min(len(dst), len(a))
	if n == 0 {
		return
	}
	negI8(dst[:n], a[:n])
}
```

Note: the public `Clamp` and the reference `clampGo` differ in name, and the existing `clampI8` (scalar saturate helper in `i8_go.go`) is unrelated; the new arch dispatch func is `clampI8(dst, src, lo, hi)` which collides with the existing scalar `clampI8(v int) int8`. RESOLVE during Step 1: rename the dispatch entry points so there is no collision. Use dispatch names `minI8`, `maxI8`, `clampElemI8`, `absI8`, `negI8`. Update the public wrappers to call `clampElemI8`.

- [ ] **Step 3: Route non-arch builds in `i8/i8_other.go`:**

```go
func minI8(dst, a, b []int8)              { minGo(dst, a, b) }
func maxI8(dst, a, b []int8)              { maxGo(dst, a, b) }
func clampElemI8(dst, s []int8, lo, hi int8) { clampGo(dst, s, lo, hi) }
func absI8(dst, a []int8)                 { absGo(dst, a) }
func negI8(dst, a []int8)                 { negGo(dst, a) }
```

- [ ] **Step 4: Add unit tests to `i8/i8_test.go`** covering literal edge cases (esp. `abs(-128)=127`, `neg(-128)=127`, `clamp` with lo>hi), parity across `lengths`, zero-alloc additions, and trailing-capacity additions. (Full test code in Task 4.)

- [ ] **Step 5: Build + run Go-reference path** (kernels not written yet, so build will fail at the `//go:noescape` decls; this step is only `go build ./i8/ 2>&1` to confirm the Go files compile in isolation before adding asm). Defer running tests to Task 4.

- [ ] **Step 6: Commit** `feat(i8): add Min/Max/Clamp/Abs/Neg references and API`.

## Task 2: AMD64 AVX2 kernels

**Files:**
- Modify: `i8/i8_amd64.go` (dispatch + `//go:noescape` decls + block constants)
- Modify: `i8/i8_amd64.s` (five kernels)

- [ ] **Step 1: Dispatch in `i8_amd64.go`** following the existing `blockSat32`/`blockMinMax`=32 pattern. Min/Max/Clamp/Abs/Neg all process 32 bytes/iter, so gate on `len >= 32`; below that use the Go ref. Add:

```go
func minI8(dst, a, b []int8) {
	if hasAVX2 && len(dst) >= blockSat32 {
		minAVX2(dst, a, b)
		return
	}
	minGo(dst, a, b)
}
// ... maxI8, clampElemI8, absI8, negI8 likewise (clampElemI8 gates on len(dst))

//go:noescape
func minAVX2(dst, a, b []int8)
// ... etc
```

- [ ] **Step 2: Kernels in `i8_amd64.s`.** Model on `addSatAVX2` (32-byte main loop + scalar tail). Min/Max use `VPMINSB`/`VPMAXSB`. Neg loads `VPXOR Y3,Y3,Y3` then `VPSUBSB Y0,Y3,Y2` (Y2=0-a sat). Abs: `VPSUBSB Y0,Y3,Y1; VPMAXSB Y1,Y0,Y2`. Clamp: `MOVD lo,X3; VPBROADCASTB X3,Y3; MOVD hi,X4; VPBROADCASTB X4,Y4; VPMAXSB Y3,Y0,Y0; VPMINSB Y4,Y0,Y2`. Scalar tails reproduce the same math with `MOVBLSX` + compare/cmov or branch clamp.

- [ ] **Step 3: `go vet ./i8/`** to validate asm arg offsets (especially `clampAVX2` lo at +48, hi at +49). Expected: clean.

- [ ] **Step 4: `go test ./i8/ -run 'Min|Max|Clamp|Abs|Neg' -count=1 -v`.** Expected: PASS on amd64.

- [ ] **Step 5: Commit** `feat(i8): AVX2 kernels for Min/Max/Clamp/Abs/Neg`.

## Task 3: ARM64 NEON kernels

**Files:**
- Modify: `i8/i8_arm64.go` (dispatch + decls)
- Modify: `i8/i8_arm64.s` (five kernels)

- [ ] **Step 1: Dispatch in `i8_arm64.go`** gating on `len >= minNEON16` (16), Go ref below.

- [ ] **Step 2: Kernels in `i8_arm64.s`.** Model on `addSatNEON`/`minMaxNEON` (16-byte main loop + scalar tail). Min=`WORD 0x4E216C02` (SMIN V2,V0,V1), Max=`WORD 0x4E216402` (SMAX V2,V0,V1), Abs=`WORD 0x4E207802` (SQABS V2,V0), Neg=`WORD 0x6E207802` (SQNEG V2,V0). Clamp: load lo->R-reg, hi->R-reg, `DUP V3.16B, W..` / `DUP V4.16B, W..`, then per block `SMAX V0,V0,V3` then `SMIN V2,V0,V4`. Scalar tails reproduce via CMP/CSEL like `minMaxNEON`; saturating abs/neg tail uses the widened compute + clamp constants like `addSatNEON`.

- [ ] **Step 3: cross-compile + run on rpi5** (Task 5 covers the full run). Quick check here: `GOARCH=arm64 go build ./i8/` and `GOARCH=arm64 go vet ./i8/`. Expected: clean.

- [ ] **Step 4: `go test ./... -run TestArm64WordEncodings -count=1`** (runs on amd64 host; cross-checks the new `WORD`s via arm64asm). Expected: PASS.

- [ ] **Step 5: Commit** `feat(i8): NEON kernels for Min/Max/Clamp/Abs/Neg`.

## Task 4: Tests, fuzz, benchmarks

**Files:**
- Modify: `i8/i8_test.go` (unit + parity + zero-alloc + trailing-cap)
- Modify: `i8/fuzz_test.go` (`FuzzI8Elementwise`)
- Modify: `i8/benchmark_test.go` (five benchmarks)

- [ ] **Step 1: Unit/parity tests** for each op across `lengths`, with literal edge cases: `Abs([-128])==127`, `Neg([-128])==127`, `Neg([127])==-127`, `Clamp` normal and `lo>hi`, `Min`/`Max` mismatched-length clamp. Add the five ops to `TestZeroAllocations` and `TestTrailingCapacityUntouched`.

- [ ] **Step 2: Differential fuzz target `FuzzI8Elementwise`** reusing `lenSeeds`/`i8FromBytes`, asserting each op equals its Go ref (split buffer into a,b halves; derive lo/hi from two bytes).

- [ ] **Step 3: Benchmarks** mirroring `BenchmarkAddSaturate` (benchN=4096, SetBytes).

- [ ] **Step 4: Full amd64 suite** `go test ./i8/ -count=1` and a short fuzz `go test ./i8/ -run x -fuzz FuzzI8Elementwise -fuzztime 20s`. Expected: PASS, no new failures.

- [ ] **Step 5: Commit** `test(i8): parity, fuzz, benchmarks for Min/Max/Clamp/Abs/Neg`.

## Task 5: Cross-arch verification + docs

- [ ] **Step 1: Cross-compile the i8 test binary and run on rpi5** (native aarch64):
  `GOARCH=arm64 go test -c -o /tmp/i8.arm64.test ./i8 && scp /tmp/i8.arm64.test rpi5.local:/tmp/ && ssh rpi5.local '/tmp/i8.arm64.test -test.run "Min|Max|Clamp|Abs|Neg|ZeroAlloc|Trailing|Sum|Dot|MinMax|ToInt" -test.v'`. Expected: PASS (exercises both the SDOT-capable and base NEON paths).
- [ ] **Step 2: Benchmark on amd64 and rpi5**, capture `-benchmem` to confirm 0 allocs and record speedups for the PR body.
- [ ] **Step 3: `golangci-lint run ./i8/...`** clean.
- [ ] **Step 4: Update `doc.go` / README** only if the package surface is enumerated there (check; the i8 section may list primitives). Update package doc comment in `i8.go` to mention the new element-wise group.
- [ ] **Step 5: Full repo test** `go test ./... -count=1` on amd64. Expected: PASS.
- [ ] **Step 6: Commit** `docs(i8): note element-wise primitives` (if doc changes), then open PR referencing #132.

---

## Self-review notes

- Spec coverage: implements exactly the issue #132 "Tier 2 element-wise (Min/Max/Clamp/Abs/Neg)" group. `AbsDiff` and `AddScalarSaturate`/`SubScalarSaturate` are deliberately deferred to a follow-up PR to keep this one cohesive.
- Naming collision resolved: arch dispatch entry for clamp is `clampElemI8` (not `clampI8`, which is the existing scalar saturate helper). `absInt` added as a small helper; confirm no existing `absInt` in package before adding.
- Saturation correctness: `Abs`/`Neg` use saturating ops/refs so `-128` maps to `127`; this is the one behavior that must have explicit literal tests and a fuzz check.
</content>
</invoke>
