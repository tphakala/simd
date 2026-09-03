# simd - project notes for AI assistants

High-performance SIMD library in pure Go assembly (no CGO). AMD64 (SSE2/AVX/AVX-512)
and ARM64 (NEON), with a Go fallback. Cross-compiles cleanly because there is no C.

## Hand-written assembly: reserved registers

The single biggest footgun in this repo is clobbering a register the Go runtime
reserves. The Go internal ABI fixes the meaning of a few registers across calls;
hand-written `.s` files must respect that. Source of truth:
https://go.googlesource.com/go/+/refs/heads/master/src/cmd/compile/abi-internal.md

### AMD64

| Register | Reserved meaning | Safe to clobber in a leaf `.s` function? |
|----------|------------------|------------------------------------------|
| `R14`    | Current goroutine `g` | No - treat as off-limits |
| `R15`    | GOT temporary when dynamically linked | No |
| `BP`     | Frame pointer | No (unless you save/restore it) |
| `SP`     | Stack pointer | No |
| `R12`, `R13`, `R15`(static) | Permanent scratch | Yes |
| `RAX RBX RCX RDX RSI RDI R8 R9 R10 R11` | Args/results, general use | Yes |

Prefer the legacy registers `RAX RBX RCX RDX RSI RDI` for scratch: they avoid the
REX prefix that `R8`-`R15` require, so the encoding is one byte shorter.

**Do not use `R14` as a scratch register.** `R14` holds the current goroutine
pointer `g`. Use `BX` (or another general-purpose register) instead.

Nuance worth knowing (verified empirically against go1.26 on amd64): a *leaf*
`NOSPLIT` kernel that clobbers `R14` and returns does not actually crash today,
because (1) the ABIInternal->ABI0 call wrapper restores `g` into `R14` from TLS on
the way back, and (2) stackmap-less hand-written asm is never async-preempted, so
the preemption handler never reads the clobbered `R14` mid-loop. That safety is an
implementation detail, not a contract: the moment such a function gains a `CALL`
into the runtime, or a stack map, the clobbered `g` becomes a live, GC-time crash.
So follow the convention regardless and keep `R14` untouched. See PR #57 for the
ConvolveDecimate kernels that originally used `R14` for the strided `pos` and were
switched to `BX`.

### ARM64

| Register | Reserved meaning | Safe to clobber? |
|----------|------------------|------------------|
| `R28`    | Current goroutine `g` | No |
| `R27`    | Assembler scratch (instruction expansion) | No |
| `R18`    | Platform-reserved, never used | No |
| `R29`    | Frame pointer | No |
| `R30`    | Link register | Scratch in non-leaf bodies only |
| `R16`, `R17` | Linker/trampoline scratch | No |
| `R19`-`R25` | Permanent scratch | Yes |

NEON instructions in this repo are hand-encoded as `WORD $0x...` with a decoded
mnemonic in the trailing comment. The comment must match the encoding: the test
suite cross-checks every `WORD` against `golang.org/x/arch/arm64asm` (see
`asmcheck_test.go` / the public `asmcheck` package). When you add or change a `WORD`,
verify the encoding with `aarch64-linux-gnu-as` + objdump or `arm64asm`.

### AMD64: the dispatch guard is machine-checked, on two axes

An amd64 kernel's body is checked against the CPU features its dispatch actually
requires. A guard that is merely documented is not enough; both of these fail the
suite rather than SIGILLing on hardware CI does not own.

- **AVX2** (`asmcheck_amd64_isa_test.go`, `asmcheck_amd64_gate_test.go`, #200). The
  name suffix is a claim: a kernel called `...AVX` may not contain an AVX2-only
  encoding. The traps are register-source `VBROADCASTSS`/`VBROADCASTSD` and any
  256-bit integer op, since AVX1's YMM support is float-only. A kernel that must
  outrun its name goes in `avx2GatedAVXKernels` together with the gate that
  protects it; renaming it to `...AVX2` is the better fix.
- **FMA3** (`asmcheck_amd64_fma_test.go`, #201). `cpu.X86.FMA` is a CPUID bit
  separate from AVX, so a `VFMADD`-family instruction needs its own conjunct in
  the guard. A kernel using one may not be assigned by `initAVXNoFMA`, `initSSE`,
  `initSSE2` or `initGo`, and may not be dispatched from a branch testing only
  `cpu.X86.AVX`. f64 and c128 keep an `initAVXNoFMA` tier precisely for the parts
  that have AVX and no FMA3.

Both checks are pure source analysis, so they run on any host. Both resolve a
guard spelled through a cached package var, a function parameter or a local, so
write the dispatch in whatever shape reads best. Neither can see across a package
boundary: only `<pkg>/*_amd64.go` is parsed.

Separately, `TestNoFMAContract` forbids fusing in the kernels listed in
`singleRoundingKernels`. That is the #156 numerical contract about where rounding
happens, not about CPU features, and it applies to both architectures.

## Testing across architectures

- AMD64 tests run on an amd64 host.
- ARM64 is exercised on native ARM64 hardware (a Raspberry Pi 5, Cortex-A76):
  cross-compile, copy the test binary over, and run it there. No qemu emulation
  needed.
- Each SIMD primitive ships a Go reference, parity tests against the active SIMD
  path, a bit-exactness check vs the per-element scalar path where applicable, and
  a `testing.AllocsPerRun == 0` allocation-free assertion.

## Conventions

- Zero allocations: every operation writes into caller-provided slices.
- Public `XxxScale`/`Xxx` functions validate inputs and clamp `n := min(len(dst), ...)`;
  the `Unsafe` variants skip bounds reconciliation.
- `golangci-lint run ./...` runs with `new: false` (lints the whole tree), so
  dupl/goconst/mnd/modernize all apply to new code.
- Issues are tracked on GitHub (this repo), not elsewhere.
