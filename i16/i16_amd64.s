//go:build amd64

#include "textflag.h"

// int16 (de)interleave on AMD64.
//
// Interleaving is pure 16-bit-lane data movement: no arithmetic happens, so the
// bit pattern of each lane is irrelevant. Unlike the 32-bit lane case there is
// no float shuffle that moves 16-bit lanes, so these kernels use the integer
// word shuffles directly: PUNPCKLWD/PUNPCKHWD pack the lanes (SSE2 and the AVX2
// VEX form), the AVX2 deinterleave gathers even/odd words with a VPSHUFB byte
// mask, and lane-crossing fixups use VPERM2I128/VPERMQ (AVX2) or
// PUNPCKLQDQ/PUNPCKHQDQ (SSE2). The chain is load -> shuffle -> store with no
// integer ALU op in between. The scalar tails use the 16-bit MOVW.
//
// Two ISA tiers are provided; i16_amd64.go dispatches AVX2 > SSE2 > Go.

// deinterleave2Mask is the per-128-bit-lane VPSHUFB control that gathers the
// even words (channel a) into the low 8 bytes and the odd words (channel b)
// into the high 8 bytes of each lane. Each mask entry is a *source* byte offset
// within the lane; read in output order (output byte i takes source byte
// mask[i]):
//   a (even words 0,2,4,6): source bytes 0,1, 4,5, 8,9, 12,13
//   b (odd words 1,3,5,7):  source bytes 2,3, 6,7, 10,11, 14,15
// The same 16 bytes are repeated for the upper lane (VPSHUFB is lane-local).
DATA deinterleave2Mask<>+0(SB)/8, $0x0d0c090805040100
DATA deinterleave2Mask<>+8(SB)/8, $0x0f0e0b0a07060302
DATA deinterleave2Mask<>+16(SB)/8, $0x0d0c090805040100
DATA deinterleave2Mask<>+24(SB)/8, $0x0f0e0b0a07060302
GLOBL deinterleave2Mask<>(SB), RODATA|NOPTR, $32

// func interleave2AVX2(dst, a, b []int16)
// Interleaves two slices: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
//
// An 8-wide VEX.128 block runs BEFORE the 16-wide loop, so a remainder of 8-15
// pairs costs one vector block plus at most 7 scalar MOVW iterations rather than
// 8-15. Without it AVX2 lost to SSE2 wherever len(a) mod 16 was 8-15, which is
// exactly where interleave2I16 prefers AVX2; measured on the i7-1260P, n=24 went
// from 1.55x slower than SSE2 to faster. See #149.
//
// Unlike dotAVX2 there is no accumulator, so no upper-lane hazard and no ordering
// constraint: each block independently loads, shuffles and stores, and the
// VEX.128 writes carry nothing across the body. The block still uses VEX forms
// (not the SSE2 kernel's legacy MOVOU/PUNPCK) to avoid the SSE/AVX transition
// penalty against the 256-bit body.
TEXT ·interleave2AVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX    // dst pointer
    MOVQ a_base+24(FP), SI     // a pointer
    MOVQ a_len+32(FP), CX      // n = len(a)
    MOVQ b_base+48(FP), DI     // b pointer

    TESTQ $8, CX               // n % 16 >= 8? one XMM block absorbs the 8
    JZ   interleave2_avx2_blocks16
    VMOVDQU (SI), X0           // [a0..a7]
    VMOVDQU (DI), X1           // [b0..b7]
    VPUNPCKLWD X1, X0, X2      // [a0,b0,a1,b1,a2,b2,a3,b3]
    VPUNPCKHWD X1, X0, X3      // [a4,b4,a5,b5,a6,b6,a7,b7]
    VMOVDQU X2, (DX)
    VMOVDQU X3, 16(DX)
    ADDQ $16, SI               // a += 8 * 2
    ADDQ $16, DI               // b += 8 * 2
    ADDQ $32, DX               // dst += 16 * 2

    // Process 16 pairs at a time (32 output elements). The block count is taken
    // from the full n; the 8 the block consumed are exactly the ones n/16 drops.
interleave2_avx2_blocks16:
    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16
    JZ   interleave2_avx2_remainder

interleave2_avx2_loop16:
    VMOVDQU (SI), Y0           // Y0 = [a0..a15]
    VMOVDQU (DI), Y1           // Y1 = [b0..b15]

    // Interleave words within each 128-bit lane
    VPUNPCKLWD Y1, Y0, Y2      // [a0,b0,a1,b1,a2,b2,a3,b3 | a8,b8,a9,b9,a10,b10,a11,b11]
    VPUNPCKHWD Y1, Y0, Y3      // [a4,b4,a5,b5,a6,b6,a7,b7 | a12,b12,a13,b13,a14,b14,a15,b15]

    // Reorder the 128-bit lanes into the final stereo stream
    VPERM2I128 $0x20, Y3, Y2, Y4  // first 16 interleaved [a0,b0..a7,b7]
    VPERM2I128 $0x31, Y3, Y2, Y5  // next 16 interleaved  [a8,b8..a15,b15]

    VMOVDQU Y4, (DX)
    VMOVDQU Y5, 32(DX)

    ADDQ $32, SI               // a += 16 * 2
    ADDQ $32, DI               // b += 16 * 2
    ADDQ $64, DX               // dst += 32 * 2
    DECQ AX
    JNZ  interleave2_avx2_loop16

interleave2_avx2_remainder:
    ANDQ $7, CX                // $7 not $15: the 8-wide block above took that bit
    JZ   interleave2_avx2_done

interleave2_avx2_scalar:
    MOVW (SI), AX
    MOVW (DI), BX
    MOVW AX, (DX)
    MOVW BX, 2(DX)
    ADDQ $2, SI
    ADDQ $2, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  interleave2_avx2_scalar

interleave2_avx2_done:
    VZEROUPPER
    RET

// func deinterleave2AVX2(a, b, src []int16)
// Deinterleaves: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
// 1. VPSHUFB gathers evens (a's) to the low 8 bytes, odds (b's) to the high 8
//    bytes of each 128-bit lane.
// 2. VPUNPCKLQDQ/VPUNPCKHQDQ split the lanes into a-only and b-only registers.
// 3. VPERMQ fixes the lane interleave the two 128-bit halves introduced.
//
// An 8-wide VEX.128 block runs BEFORE the 16-wide loop, absorbing a remainder of
// 8-15 pairs in one vector block instead of 8-15 scalar MOVW iterations; without
// it AVX2 lost to SSE2 wherever len(a) mod 16 was 8-15, which is where the
// dispatcher prefers it (n=24 measured 1.5x slower on the i7-1260P). See #149.
// The block needs no VPERMQ: within one 128-bit lane VPSHUFB already lands a's in
// the low half and b's in the high, so a single VPUNPCKLQDQ/HQDQ pair splits them
// in order. It reuses X7 (the low lane of the gather mask) and VEX forms, so it
// carries nothing across the body and pays no SSE/AVX transition penalty.
TEXT ·deinterleave2AVX2(SB), NOSPLIT, $0-72
    MOVQ a_base+0(FP), DX      // a pointer
    MOVQ a_len+8(FP), CX       // n = len(a)
    MOVQ b_base+24(FP), R8     // b pointer
    MOVQ src_base+48(FP), SI   // src pointer

    VMOVDQU deinterleave2Mask<>(SB), Y7  // even/odd word gather mask

    TESTQ $8, CX               // n % 16 >= 8? one XMM block absorbs the 8
    JZ   deinterleave2_avx2_blocks16
    VMOVDQU (SI), X0           // [a0,b0,a1,b1,a2,b2,a3,b3]
    VMOVDQU 16(SI), X1         // [a4,b4,a5,b5,a6,b6,a7,b7]
    VPSHUFB X7, X0, X0         // [a0,a1,a2,a3,b0,b1,b2,b3]
    VPSHUFB X7, X1, X1         // [a4,a5,a6,a7,b4,b5,b6,b7]
    VPUNPCKLQDQ X1, X0, X2     // a = [a0..a3,a4..a7]
    VPUNPCKHQDQ X1, X0, X3     // b = [b0..b3,b4..b7]
    VMOVDQU X2, (DX)           // store a
    VMOVDQU X3, (R8)           // store b
    ADDQ $32, SI               // src += 16 * 2
    ADDQ $16, DX               // a += 8 * 2
    ADDQ $16, R8               // b += 8 * 2

    // Process 16 pairs at a time. The block count is taken from the full n; the 8
    // the block consumed are exactly the ones n/16 drops.
deinterleave2_avx2_blocks16:
    MOVQ CX, AX
    SHRQ $4, AX
    JZ   deinterleave2_avx2_remainder

deinterleave2_avx2_loop16:
    VMOVDQU (SI), Y0           // [a0,b0..a7,b7]
    VMOVDQU 32(SI), Y1         // [a8,b8..a15,b15]

    VPSHUFB Y7, Y0, Y0         // [a0,a1,a2,a3,b0,b1,b2,b3 | a4,a5,a6,a7,b4,b5,b6,b7]
    VPSHUFB Y7, Y1, Y1         // [a8..a11,b8..b11 | a12..a15,b12..b15]

    VPUNPCKLQDQ Y1, Y0, Y2     // a's, lane-interleaved: [a0..a3,a8..a11 | a4..a7,a12..a15]
    VPUNPCKHQDQ Y1, Y0, Y3     // b's, lane-interleaved: [b0..b3,b8..b11 | b4..b7,b12..b15]

    VPERMQ $0xD8, Y2, Y2       // a in order [a0..a15]
    VPERMQ $0xD8, Y3, Y3       // b in order [b0..b15]

    VMOVDQU Y2, (DX)           // store a
    VMOVDQU Y3, (R8)           // store b

    ADDQ $64, SI               // src += 32 * 2
    ADDQ $32, DX               // a += 16 * 2
    ADDQ $32, R8               // b += 16 * 2
    DECQ AX
    JNZ  deinterleave2_avx2_loop16

deinterleave2_avx2_remainder:
    ANDQ $7, CX                // $7 not $15: the 8-wide block above took that bit
    JZ   deinterleave2_avx2_done

deinterleave2_avx2_scalar:
    MOVW (SI), AX
    MOVW 2(SI), BX
    MOVW AX, (DX)
    MOVW BX, (R8)
    ADDQ $4, SI
    ADDQ $2, DX
    ADDQ $2, R8
    DECQ CX
    JNZ  deinterleave2_avx2_scalar

deinterleave2_avx2_done:
    VZEROUPPER
    RET

// func interleave2SSE2(dst, a, b []int16)
// 128-bit fallback: 8 pairs per iteration using PUNPCKLWD/PUNPCKHWD.
TEXT ·interleave2SSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX    // dst pointer
    MOVQ a_base+24(FP), SI     // a pointer
    MOVQ a_len+32(FP), CX      // n = len(a)
    MOVQ b_base+48(FP), DI     // b pointer

    MOVQ CX, AX
    SHRQ $3, AX                // AX = n / 8
    JZ   interleave2_sse2_remainder

interleave2_sse2_loop8:
    MOVOU (SI), X0             // [a0..a7]
    MOVOU (DI), X1             // [b0..b7]
    MOVOU X0, X2               // copy a (PUNPCK overwrites its destination)
    PUNPCKLWL X1, X0           // [a0,b0,a1,b1,a2,b2,a3,b3] (Intel PUNPCKLWD)
    PUNPCKHWL X1, X2           // [a4,b4,a5,b5,a6,b6,a7,b7] (Intel PUNPCKHWD)
    MOVOU X0, (DX)
    MOVOU X2, 16(DX)

    ADDQ $16, SI               // a += 8 * 2
    ADDQ $16, DI               // b += 8 * 2
    ADDQ $32, DX               // dst += 16 * 2
    DECQ AX
    JNZ  interleave2_sse2_loop8

interleave2_sse2_remainder:
    ANDQ $7, CX
    JZ   interleave2_sse2_done

interleave2_sse2_scalar:
    MOVW (SI), AX
    MOVW (DI), BX
    MOVW AX, (DX)
    MOVW BX, 2(DX)
    ADDQ $2, SI
    ADDQ $2, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  interleave2_sse2_scalar

interleave2_sse2_done:
    RET

// func deinterleave2SSE2(a, b, src []int16)
// 128-bit fallback: 8 pairs per iteration. PSHUFLW/PSHUFHW/PSHUFD move each
// lane's evens to its low 64 bits and odds to its high 64 bits without a memory
// mask; PUNPCKLQDQ/PUNPCKHQDQ then merge the two source halves into a and b.
TEXT ·deinterleave2SSE2(SB), NOSPLIT, $0-72
    MOVQ a_base+0(FP), DX      // a pointer
    MOVQ a_len+8(FP), CX       // n = len(a)
    MOVQ b_base+24(FP), R8     // b pointer
    MOVQ src_base+48(FP), SI   // src pointer

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   deinterleave2_sse2_remainder

deinterleave2_sse2_loop8:
    MOVOU (SI), X0             // [a0,b0,a1,b1,a2,b2,a3,b3]
    MOVOU 16(SI), X1           // [a4,b4,a5,b5,a6,b6,a7,b7]

    // X0 -> [a0,a1,a2,a3 | b0,b1,b2,b3]
    PSHUFLW $0xD8, X0, X0      // low words  -> [a0,a1,b0,b1]
    PSHUFHW $0xD8, X0, X0      // high words -> [a2,a3,b2,b3]
    PSHUFD  $0xD8, X0, X0      // dwords     -> [a0,a1,a2,a3,b0,b1,b2,b3]

    // X1 -> [a4,a5,a6,a7 | b4,b5,b6,b7]
    PSHUFLW $0xD8, X1, X1
    PSHUFHW $0xD8, X1, X1
    PSHUFD  $0xD8, X1, X1

    MOVOU X0, X2
    PUNPCKLQDQ X1, X2          // a = [a0..a3,a4..a7]
    PUNPCKHQDQ X1, X0          // b = [b0..b3,b4..b7]

    MOVOU X2, (DX)             // store a
    MOVOU X0, (R8)             // store b

    ADDQ $32, SI               // src += 16 * 2
    ADDQ $16, DX               // a += 8 * 2
    ADDQ $16, R8               // b += 8 * 2
    DECQ AX
    JNZ  deinterleave2_sse2_loop8

deinterleave2_sse2_remainder:
    ANDQ $7, CX
    JZ   deinterleave2_sse2_done

deinterleave2_sse2_scalar:
    MOVW (SI), AX
    MOVW 2(SI), BX
    MOVW AX, (DX)
    MOVW BX, (R8)
    ADDQ $4, SI
    ADDQ $2, DX
    ADDQ $2, R8
    DECQ CX
    JNZ  deinterleave2_sse2_scalar

deinterleave2_sse2_done:
    RET

// Widening int16 dot product.
//
// PMADDWD multiplies eight int16 pairs and adds them pairwise into four int32
// lanes; the AVX2 form does twice that. Note the Plan 9 spelling of the SSE2
// form is PMADDWL ("long" for the 32-bit result); the VEX form keeps the Intel
// name VPMADDWD.
//
// PMADDWD is SSE2, so it is present on the whole GOAMD64=v1 baseline: there is
// no availability cliff below the SSE2 tier, and the AVX2 tier is a width win
// rather than an availability one. The dispatcher still gates on hasSSE2, which
// is always true on amd64; that mirrors the package's other kernels and keeps
// the pure-Go path reachable as a non-amd64 safety net.
//
// Overflow: the only input that can exceed int32 within one instruction is a
// pair of (-32768 * -32768), summing to 2^31, which PMADDWD wraps to
// 0x80000000. The scalar reference wraps identically (two wrapping int32 adds
// of 2^30), and because wrapping addition is associative the pairwise pre-add
// inside PMADDWD is invisible. So no special-casing is needed and the kernels
// are bit-exact with dotGo for every input; the tests pin this with all-MinInt16
// sweeps.

// func dotSSE2(a, b []int16) int32
// Handles mismatched slice lengths: uses min(len(a), len(b)).
TEXT ·dotSSE2(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_len+32(FP), DX
    CMPQ DX, CX
    CMOVQLT DX, CX             // CX = n = min(len(a), len(b))
    MOVQ b_base+24(FP), DI

    PXOR X0, X0                // int32 accumulator

    MOVQ CX, BX
    SHRQ $3, BX                // BX = n / 8
    JZ   dot_sse2_reduce

dot_sse2_loop8:
    MOVOU (SI), X1
    MOVOU (DI), X2
    PMADDWL X2, X1             // X1 = pairwise (a*b) sums -> 4 int32
    PADDD X1, X0               // accumulate (wrapping)
    ADDQ $16, SI
    ADDQ $16, DI
    DECQ BX
    JNZ  dot_sse2_loop8

dot_sse2_reduce:
    PSHUFD $0x4E, X0, X1       // swap 64-bit halves
    PADDD X1, X0
    PSHUFD $0xB1, X0, X1       // swap 32-bit within pairs
    PADDD X1, X0
    MOVQ X0, AX                // low int32 = vector total (in EAX)

    ANDQ $7, CX
    JZ   dot_sse2_done

dot_sse2_scalar:
    MOVWLSX (SI), BX           // sign-extending 16-bit load
    MOVWLSX (DI), DX
    IMULL DX, BX
    ADDL BX, AX                // 32-bit add: wraps like dotGo
    ADDQ $2, SI
    ADDQ $2, DI
    DECQ CX
    JNZ  dot_sse2_scalar

dot_sse2_done:
    MOVL AX, ret+48(FP)
    RET

// func dotAVX2(a, b []int16) int32
// Handles mismatched slice lengths: uses min(len(a), len(b)).
//
// An 8-wide XMM block runs BEFORE the 16-wide loop, so a remainder of 8-15
// elements costs one vector block plus at most 7 scalar iterations rather than
// 8-15 iterations of a serial ADDL chain. Without it AVX2 lost to SSE2 wherever
// n mod 16 was 8-15, which is exactly where dotI16 prefers AVX2; see #149 for
// the measurements. dotNEON carries the same 8-wide tier (dot_neon_block8 in
// i16_arm64.s), though it sits after the body, in the one slot the hazard below
// closes off here; NEON has no upper-lane hazard to keep it out. dotSSE2 needs
// no such tier at all: its 8-wide loop already is that width.
//
// The placement is constrained, though not down to one slot as in xcorr4AVX2
// below. Accumulating into X0 with a VEX.128 write zeroes bits 255:128 of Y0,
// so a block between the 16-wide loop and the VEXTRACTI128 fold would silently
// discard the loop's upper-lane sums. Two slots survive that: before the body,
// while Y0 is still freshly VPXOR'd so the zeroing writes zero over zero, or
// after the fold, once the upper lane is dead (xcorr4AVX2 folds too late to
// have the second option). This kernel takes the first, so that this file
// carries one block shape rather than two.
//
// Summing the remainder before the body is bit-exact only because the
// accumulation wraps: wrapping int32 addition is associative and commutative,
// so regrouping the terms cannot change the result. That is the property the
// lane split already rests on and that DotProduct's doc guarantees to callers.
// It would not hold in a saturating kernel.
TEXT ·dotAVX2(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_len+32(FP), DX
    CMPQ DX, CX
    CMOVQLT DX, CX             // CX = n = min(len(a), len(b))
    MOVQ b_base+24(FP), DI

    VPXOR Y0, Y0, Y0           // int32 accumulator

    TESTQ $8, CX               // n % 16 >= 8? one XMM block absorbs the 8
    JZ   dot_avx2_blocks16

    VMOVDQU (SI), X1
    VMOVDQU (DI), X2
    VPMADDWD X2, X1, X1        // 8 int16 pairs -> 4 int32
    VPADDD X1, X0, X0          // VEX-128 zeroes Y0[255:128]; it is still zero
    ADDQ $16, SI
    ADDQ $16, DI

    // The block count is computed AFTER the 8-wide block, not before, so SHRQ's
    // own ZF still drives the branch below; computing it first would need a
    // separate TESTQ BX, BX. CX keeps the full n either way: the 8 elements the
    // block consumed are exactly the ones n/16 already excludes.
dot_avx2_blocks16:
    MOVQ CX, BX
    SHRQ $4, BX                // BX = n / 16
    JZ   dot_avx2_reduce

dot_avx2_loop16:
    VMOVDQU (SI), Y1
    VMOVDQU (DI), Y2
    VPMADDWD Y2, Y1, Y1        // 16 int16 pairs -> 8 int32
    VPADDD Y1, Y0, Y0          // accumulate (wrapping)
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ BX
    JNZ  dot_avx2_loop16

dot_avx2_reduce:
    VEXTRACTI128 $1, Y0, X1
    VPADDD X1, X0, X0          // fold 8 -> 4 int32
    VPSHUFD $0x4E, X0, X1      // swap 64-bit halves
    VPADDD X1, X0, X0
    VPSHUFD $0xB1, X0, X1      // swap 32-bit within pairs
    VPADDD X1, X0, X0
    MOVQ X0, AX                // low int32 = vector total (in EAX)

    ANDQ $7, CX                // the 8-wide block above took n % 16 down to n % 8
    JZ   dot_avx2_done

dot_avx2_scalar:
    MOVWLSX (SI), BX           // sign-extending 16-bit load
    MOVWLSX (DI), DX
    IMULL DX, BX
    ADDL BX, AX                // 32-bit add: wraps like dotGo
    ADDQ $2, SI
    ADDQ $2, DI
    DECQ CX
    JNZ  dot_avx2_scalar

dot_avx2_done:
    MOVL AX, ret+48(FP)
    VZEROUPPER
    RET

// Multi-lag cross-correlation: four lags per call.
//
// Load x once, then multiply-accumulate it against four overlapping y windows
// at element offsets 0/1/2/3. Reusing the x load across four lags is the point
// of the op.
//
// PMADDWD is per-lane commutative (it multiplies elementwise, then adds
// pairwise), so the destructive two-operand SSE2 form costs nothing here: put
// the freshly loaded y in the destination and x survives in its own register
// across all four lags. y is reloaded per lag regardless.
//
// The four y loads are deliberately unaligned relative to one another, being
// the same window stepped by 2 bytes, so MOVOU/VMOVDQU are load-bearing: at
// most one lag in four can be 16-byte aligned, so an aligned-load substitution
// would fault on at least three lags out of four, and on all four whenever y's
// own base is unaligned (which is the common case, since callers slide y).

// func xcorr4SSE2(dst []int32, x, y []int16)
// n = min(len(x), len(y)-3) is a safety net; the dispatcher passes a y window of
// exactly len(x)+3, so the clamp never fires. It turns a wrapper bug into a
// wrong number instead of an out-of-bounds read.
TEXT ·xcorr4SSE2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DI
    MOVQ x_base+24(FP), SI
    MOVQ x_len+32(FP), CX
    MOVQ y_base+48(FP), BX
    MOVQ y_len+56(FP), DX

    PXOR X4, X4                // lag 0 accumulator
    PXOR X5, X5                // lag 1
    PXOR X6, X6                // lag 2
    PXOR X7, X7                // lag 3

    SUBQ $3, DX                // DX = len(y) - 3
    JLE  xcorr4_sse2_empty
    CMPQ DX, CX
    CMOVQLT DX, CX             // CX = n = min(len(x), len(y)-3)
    JMP  xcorr4_sse2_blocks

xcorr4_sse2_empty:
    XORQ CX, CX                // n = 0: fold zeros, store zeros

xcorr4_sse2_blocks:
    MOVQ CX, AX
    SHRQ $3, AX                // AX = n / 8
    JZ   xcorr4_sse2_fold

xcorr4_sse2_loop8:
    MOVOU (SI), X0             // x[j..j+8), reused by all four lags
    MOVOU (BX), X1             // y[j+0 ..) lag 0
    PMADDWL X0, X1
    PADDD X1, X4
    MOVOU 2(BX), X1            // y[j+1 ..) lag 1
    PMADDWL X0, X1
    PADDD X1, X5
    MOVOU 4(BX), X1            // y[j+2 ..) lag 2
    PMADDWL X0, X1
    PADDD X1, X6
    MOVOU 6(BX), X1            // y[j+3 ..) lag 3
    PMADDWL X0, X1
    PADDD X1, X7
    ADDQ $16, SI
    ADDQ $16, BX
    DECQ AX
    JNZ  xcorr4_sse2_loop8

xcorr4_sse2_fold:
    PSHUFD $0x4E, X4, X0
    PADDD X0, X4
    PSHUFD $0xB1, X4, X0
    PADDD X0, X4
    MOVQ X4, R8                // lag 0 partial sum
    PSHUFD $0x4E, X5, X0
    PADDD X0, X5
    PSHUFD $0xB1, X5, X0
    PADDD X0, X5
    MOVQ X5, R9                // lag 1
    PSHUFD $0x4E, X6, X0
    PADDD X0, X6
    PSHUFD $0xB1, X6, X0
    PADDD X0, X6
    MOVQ X6, R10               // lag 2
    PSHUFD $0x4E, X7, X0
    PADDD X0, X7
    PSHUFD $0xB1, X7, X0
    PADDD X0, X7
    MOVQ X7, R11               // lag 3

    ANDQ $7, CX
    JZ   xcorr4_sse2_store

xcorr4_sse2_scalar:
    MOVWLSX (SI), AX           // x[j], sign-extended
    MOVWLSX (BX), DX
    IMULL AX, DX
    ADDL DX, R8
    MOVWLSX 2(BX), DX
    IMULL AX, DX
    ADDL DX, R9
    MOVWLSX 4(BX), DX
    IMULL AX, DX
    ADDL DX, R10
    MOVWLSX 6(BX), DX
    IMULL AX, DX
    ADDL DX, R11
    ADDQ $2, SI
    ADDQ $2, BX
    DECQ CX
    JNZ  xcorr4_sse2_scalar

xcorr4_sse2_store:
    MOVL R8, 0(DI)
    MOVL R9, 4(DI)
    MOVL R10, 8(DI)
    MOVL R11, 12(DI)
    RET

// func xcorr4AVX2(dst []int32, x, y []int16)
// As xcorr4SSE2 at twice the width, plus an 8-wide tier SSE2 has no need for;
// VPMADDWD's three-operand form also spares the reload-into-destination dance.
//
// An 8-wide XMM block, then a 4-wide one (#150), run BEFORE the 16-wide loop, so
// a remainder of 8-15 elements costs one or two vector blocks plus at most 3
// scalar iterations rather than 8-15 scalar iterations across four lags (the
// 4-wide block is detailed at its own site below). Without the 8-wide block AVX2
// lost to SSE2 across half the len(x) domain, which is exactly where the
// dispatcher prefers it; see
// #145 for the measurements. A narrow tier below the wide body is the usual
// shape here (f32/f32_amd64.s dot32_loop8_check, f64/f64_amd64.s var_avx_loop4),
// but both of those put it after the body, and get it for free from their
// unroll: at float32 a YMM holds 8, so the 8-wide tier IS one register. At int16
// a YMM holds 16, so the tier below the body has to be written by hand, and it
// goes before the body for the reason below.
//
// Accumulating into X4-X7 with VEX.128 writes is what forces that order, not
// taste. A VEX.128 write zeroes bits 255:128 of its destination (the hazard the
// f32 kernels flag as "VEX scalar ops zero upper YMM"), so a block running AFTER
// the 16-wide loop would silently discard that loop's upper-lane sums. Running
// it FIRST, while Y4-Y7 are still freshly VPXOR'd, means the zeroing writes zero
// over zero; the loop then accumulates on top with VEX.256, which preserves the
// low lane, and the fold needs no extra instructions. Mixing VEX.128 and VEX.256
// costs nothing: the transition penalty is a legacy-SSE-encoding problem, not a
// VEX one. (A block that LOADED VEX.128 but COMPUTED VEX.256 could sit after the
// body instead, since the zeroed upper lanes would contribute zero. That shape
// would not need the reordering, nor the argument below. This one does.)
//
// Summing the remainder before the body is bit-exact only because the
// accumulation wraps: wrapping int32 addition is associative and commutative, so
// regrouping the terms cannot change the result. The regrouping is real, not
// just a lane permutation: these 8 elements used to go through the scalar tail
// one product at a time, and now go through VPMADDWD, which adds each adjacent
// pair before accumulating. XCorr's doc comment guarantees to callers the
// wrapping this rests on. VPMADDWD itself is safe for the same reason: a pair of
// 0x8000*0x8000 products sums to 0x80000000 and it WRAPS there, it does not
// saturate. A saturating accumulator could not be reordered like this at all.
//
// At len(x)%16 == 0 the block is branched over and the kernel pays only the
// TESTQ and its predicted-taken JZ.
TEXT ·xcorr4AVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DI
    MOVQ x_base+24(FP), SI
    MOVQ x_len+32(FP), CX
    MOVQ y_base+48(FP), BX
    MOVQ y_len+56(FP), DX

    VPXOR Y4, Y4, Y4           // lag 0 accumulator
    VPXOR Y5, Y5, Y5           // lag 1
    VPXOR Y6, Y6, Y6           // lag 2
    VPXOR Y7, Y7, Y7           // lag 3

    SUBQ $3, DX                // DX = len(y) - 3
    JLE  xcorr4_avx2_empty
    CMPQ DX, CX
    CMOVQLT DX, CX             // CX = n = min(len(x), len(y)-3)
    JMP  xcorr4_avx2_blocks

xcorr4_avx2_empty:
    XORQ CX, CX

xcorr4_avx2_blocks:
    TESTQ $8, CX               // n % 16 >= 8? one XMM block absorbs the 8
    JZ   xcorr4_avx2_block4    // skip the 8-wide block but still try the 4-wide one

    VMOVDQU (SI), X0           // x[0..8), reused by all four lags
    VMOVDQU (BX), X1           // lag 0
    VPMADDWD X0, X1, X1
    VPADDD X1, X4, X4          // VEX-128 zeroes Y4[255:128]; it is still zero
    VMOVDQU 2(BX), X1          // lag 1
    VPMADDWD X0, X1, X1
    VPADDD X1, X5, X5
    VMOVDQU 4(BX), X1          // lag 2
    VPMADDWD X0, X1, X1
    VPADDD X1, X6, X6
    VMOVDQU 6(BX), X1          // lag 3
    VPMADDWD X0, X1, X1
    VPADDD X1, X7, X7
    ADDQ $16, SI               // 8 int16 consumed from x
    ADDQ $16, BX               // y slides by the same 8, lag offsets unchanged

    // A 4-wide XMM block below the 8-wide one, absorbing another 4 of the 0-7
    // scalar-tail elements. The scalar tail costs ~2.7 cyc/element versus ~0.19
    // for a vectorized one (#150), so a 4-wide block is worth its ~1.5 cycles.
    // It uses 64-bit VMOVQ loads: x[0..4) and each lag's y[0..4) land in the low
    // half of the register with the upper half zeroed, so VPMADDWD yields two
    // int32 partial sums (and two zero lanes) that VPADDD folds into the low lane
    // of each accumulator. Like the 8-wide block it must run BEFORE any 256-bit
    // accumulation: its VEX.128 VPADDD writes zero the upper lane, which is still
    // zero here, and the 16-wide loop then accumulates on top with VEX.256. The
    // reordering is bit-exact only because the wrapping int32 add is associative
    // (the property XCorr's doc guarantees), the same basis as the 8-wide block.
xcorr4_avx2_block4:
    TESTQ $4, CX               // n % 8 >= 4? one 4-wide XMM block absorbs the 4
    JZ   xcorr4_avx2_blocks16
    VMOVQ (SI), X0             // x[0..4) in the low 64 bits (upper zeroed)
    VMOVQ (BX), X1             // lag 0: y[0..4)
    VPMADDWD X0, X1, X1        // 2 int32 partial sums (+ 2 zero lanes)
    VPADDD X1, X4, X4          // VEX-128 zeroes Y4[255:128]; it is still zero
    VMOVQ 2(BX), X1            // lag 1
    VPMADDWD X0, X1, X1
    VPADDD X1, X5, X5
    VMOVQ 4(BX), X1            // lag 2
    VPMADDWD X0, X1, X1
    VPADDD X1, X6, X6
    VMOVQ 6(BX), X1            // lag 3
    VPMADDWD X0, X1, X1
    VPADDD X1, X7, X7
    ADDQ $8, SI                // 4 int16 consumed from x
    ADDQ $8, BX                // y slides by the same 4, lag offsets unchanged

    // The block count is computed AFTER the 8- and 4-wide blocks, not before, so
    // SHRQ's own ZF still drives the branch below; computing it first would need
    // a separate TESTQ AX, AX. CX keeps the full n: the 8 and 4 the blocks
    // consumed are exactly the ones n/16 already excludes.
xcorr4_avx2_blocks16:
    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16, the 16-wide block count
    JZ   xcorr4_avx2_fold

xcorr4_avx2_loop16:
    VMOVDQU (SI), Y0           // x[j..j+16), reused by all four lags
    VMOVDQU (BX), Y1           // lag 0
    VPMADDWD Y0, Y1, Y1
    VPADDD Y1, Y4, Y4
    VMOVDQU 2(BX), Y1          // lag 1
    VPMADDWD Y0, Y1, Y1
    VPADDD Y1, Y5, Y5
    VMOVDQU 4(BX), Y1          // lag 2
    VPMADDWD Y0, Y1, Y1
    VPADDD Y1, Y6, Y6
    VMOVDQU 6(BX), Y1          // lag 3
    VPMADDWD Y0, Y1, Y1
    VPADDD Y1, Y7, Y7
    ADDQ $32, SI
    ADDQ $32, BX
    DECQ AX
    JNZ  xcorr4_avx2_loop16

// Horizontal fold (#150 item 3). Each Yn holds eight int32 partials for lag n
// across two 128-bit lanes. The two exit paths need the four lag sums in
// different places, so each folds the cheapest way for its need.
//
// No scalar tail (n%4 == 0, the aligned lengths the fixed-point callers use):
// VEXTRACTI128+VPADDD collapses each lag's high lane into its low, then a
// three-node VPHADDD tree reduces all four lags at once. VPHADDD Xb, Xa, Xd sets
// Xd = [a0+a1, a2+a3, b0+b1, b2+b3] with plain wrapping 32-bit adds (it never
// saturates, unlike VPHADDSW), so it preserves the wrapping-add contract XCorr
// documents to callers. The final VPHADDD lands the four sums as contiguous
// int32 in lane order 0,1,2,3, stored with one 16-byte VMOVDQU. Bit-exact only
// because wrapping int32 addition is associative and commutative: this regroups
// each lag's four partials as (n0+n1)+(n2+n3). Kernel-direct measurement shows
// -2% to -7% cycles versus the per-lag fold here, backed by a matching drop in
// retired instructions (the raw aligned-length cycle delta alone is muddied by
// the code-layout lottery of #159, so the instruction count is the load-bearing
// evidence).
//
// Scalar tail present (n%4 != 0): the tail loop accumulates into R8-R11, so the
// four sums have to reach GP registers. Extracting them out of a VPHADDD result
// costs a VPEXTRD per lag whose latency exceeds the shuffles the tree saved
// (measured +5% at small n), so this path keeps the original per-lag fold that
// lands each sum straight in a register with MOVQ. It is byte-for-byte the
// pre-#150 fold; only the no-tail path above is new. ANDQ runs first so the
// aligned path pays no VPEXTRD and the tail path pays no VPHADDD.
xcorr4_avx2_fold:
    ANDQ $3, CX                // $3 not $7: the 8- and 4-wide blocks took those bits
    JNZ  xcorr4_avx2_fold_tail
    VEXTRACTI128 $1, Y4, X0
    VPADDD X0, X4, X4          // X4 = four int32 partials for lag 0
    VEXTRACTI128 $1, Y5, X0
    VPADDD X0, X5, X5          // X5 = lag 1 partials
    VEXTRACTI128 $1, Y6, X0
    VPADDD X0, X6, X6          // X6 = lag 2 partials
    VEXTRACTI128 $1, Y7, X0
    VPADDD X0, X7, X7          // X7 = lag 3 partials
    VPHADDD X5, X4, X4         // X4 = [lag0(0+1), lag0(2+3), lag1(0+1), lag1(2+3)]
    VPHADDD X7, X6, X6         // X6 = [lag2(0+1), lag2(2+3), lag3(0+1), lag3(2+3)]
    VPHADDD X6, X4, X4         // X4 = [sumLag0, sumLag1, sumLag2, sumLag3]
    VMOVDQU X4, (DI)           // no scalar tail: store all four lag sums at once
    VZEROUPPER
    RET

xcorr4_avx2_fold_tail:
    VEXTRACTI128 $1, Y4, X0
    VPADDD X0, X4, X4
    VPSHUFD $0x4E, X4, X0
    VPADDD X0, X4, X4
    VPSHUFD $0xB1, X4, X0
    VPADDD X0, X4, X4
    MOVQ X4, R8                // lag 0 partial sum
    VEXTRACTI128 $1, Y5, X0
    VPADDD X0, X5, X5
    VPSHUFD $0x4E, X5, X0
    VPADDD X0, X5, X5
    VPSHUFD $0xB1, X5, X0
    VPADDD X0, X5, X5
    MOVQ X5, R9                // lag 1
    VEXTRACTI128 $1, Y6, X0
    VPADDD X0, X6, X6
    VPSHUFD $0x4E, X6, X0
    VPADDD X0, X6, X6
    VPSHUFD $0xB1, X6, X0
    VPADDD X0, X6, X6
    MOVQ X6, R10               // lag 2
    VEXTRACTI128 $1, Y7, X0
    VPADDD X0, X7, X7
    VPSHUFD $0x4E, X7, X0
    VPADDD X0, X7, X7
    VPSHUFD $0xB1, X7, X0
    VPADDD X0, X7, X7
    MOVQ X7, R11               // lag 3

xcorr4_avx2_scalar:
    MOVWLSX (SI), AX           // x[j], sign-extended
    MOVWLSX (BX), DX
    IMULL AX, DX
    ADDL DX, R8
    MOVWLSX 2(BX), DX
    IMULL AX, DX
    ADDL DX, R9
    MOVWLSX 4(BX), DX
    IMULL AX, DX
    ADDL DX, R10
    MOVWLSX 6(BX), DX
    IMULL AX, DX
    ADDL DX, R11
    ADDQ $2, SI
    ADDQ $2, BX
    DECQ CX
    JNZ  xcorr4_avx2_scalar

xcorr4_avx2_store:
    MOVL R8, 0(DI)
    MOVL R9, 4(DI)
    MOVL R10, 8(DI)
    MOVL R11, 12(DI)
    VZEROUPPER
    RET

// func xcorr4AVXVNNI(dst []int32, x, y []int16)
// xcorr4AVX2 with the 16-wide loop's four VPMADDWD+VPADDD pairs (madd the 16-bit
// lane pairs to int32, then accumulate) each fused into one VPDPWSSD (#150 item
// 1). VPDPWSSD acc, a, b computes acc += VPMADDWD(a, b) with the same WRAPPING
// dword accumulation, so it is bit-identical to the pair it replaces and
// preserves the wrapping contract XCorr documents to callers. (VPDPWSSDS, the
// saturating sibling, must never be used here.) Fusing the loop's 8 vector-ALU
// ops to 4, all on p0/p1, drops the 16-wide body's port floor from 3 to 2
// cyc/block. That floor is not realized: fusion moves the whole
// multiply-accumulate onto each lag's loop-carried accumulator chain (in
// xcorr4AVX2 the VPMADDWD fed VPADDD from loads, off the chain), so with one
// accumulator per lag the 4-way ILP keeps the ports busy but cannot shorten any
// single chain, and the realized loop rate is set by VPDPWSSD's accumulator
// latency, ~2.5 cyc/block measured on the i7-1260P (Alder Lake), not the 2-cyc
// floor. The net XCorr win is a modest ~2-7% per length (~2.3% geomean), with
// the aligned-length tails muddied by the code-layout lottery of #159; see #150
// for the port model. Requires AVX-VNNI in VEX form (cpu.X86.AVXVNNI), which the
// dispatcher checks above AVX2.
//
// Only the 16-wide loop is fused. The 8- and 4-wide remainder blocks run at most
// once per call, so VNNI would save them nothing measurable; they are the exact
// VPMADDWD+VPADDD blocks xcorr4AVX2 carries, including the reason they run BEFORE
// the 256-bit loop (their VEX.128 writes zero Y4-Y7's upper lanes while those are
// still the freshly-VPXOR'd zero, so the VEX.256 loop then accumulates on top
// preserving the low lane). The reordering is bit-exact only because the int32
// accumulation wraps, hence is associative; the fold and scalar tail are
// unchanged. See xcorr4AVX2 above for the full argument.
//
// VPDPWSSD is HAND-ENCODED below. The Go assembler (go1.26) knows only the EVEX
// form of this mnemonic: avx_optabs.go lists AVPDPWSSD with evex128/256/512 and
// no vex form, so `VPDPWSSD Y0, Y1, Y4` assembles to EVEX.256 (62-prefixed), the
// AVX-512-VNNI encoding, which #UDs on AVX-VNNI-only parts such as Alder Lake
// where AVX-512 is fused off (confirmed: it SIGILLs on the i7-1260P). The BYTE
// directives emit the VEX form instead, VEX.256.66.0F38.W0 52 /r:
//   C4 E2 75 52 modrm   x in Y0 (ModRM.rm=0), y in Y1 (VEX.vvvv=~1=1110), L=1
//                       (256), pp=01 (66); modrm = 0xC0 | reg<<3, reg = dst YMM.
// So Y4->0xE0, Y5->0xE8, Y6->0xF0, Y7->0xF8. This is the same hand-encoding
// discipline the arm64 NEON kernels use (WORD there, BYTE here); asmcheck only
// cross-checks arm64 words, so the encoding's runtime gate is the host
// ParityWithGo test, which executes the kernel (a wrong byte SIGILLs or missums).
TEXT ·xcorr4AVXVNNI(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DI
    MOVQ x_base+24(FP), SI
    MOVQ x_len+32(FP), CX
    MOVQ y_base+48(FP), BX
    MOVQ y_len+56(FP), DX

    VPXOR Y4, Y4, Y4           // lag 0 accumulator
    VPXOR Y5, Y5, Y5           // lag 1
    VPXOR Y6, Y6, Y6           // lag 2
    VPXOR Y7, Y7, Y7           // lag 3

    SUBQ $3, DX                // DX = len(y) - 3
    JLE  xcorrvnni_empty
    CMPQ DX, CX
    CMOVQLT DX, CX             // CX = n = min(len(x), len(y)-3)
    JMP  xcorrvnni_blocks

xcorrvnni_empty:
    XORQ CX, CX

xcorrvnni_blocks:
    TESTQ $8, CX               // n % 16 >= 8? one XMM block absorbs the 8
    JZ   xcorrvnni_block4

    VMOVDQU (SI), X0           // x[0..8), reused by all four lags
    VMOVDQU (BX), X1           // lag 0
    VPMADDWD X0, X1, X1
    VPADDD X1, X4, X4          // VEX-128 zeroes Y4[255:128]; it is still zero
    VMOVDQU 2(BX), X1          // lag 1
    VPMADDWD X0, X1, X1
    VPADDD X1, X5, X5
    VMOVDQU 4(BX), X1          // lag 2
    VPMADDWD X0, X1, X1
    VPADDD X1, X6, X6
    VMOVDQU 6(BX), X1          // lag 3
    VPMADDWD X0, X1, X1
    VPADDD X1, X7, X7
    ADDQ $16, SI               // 8 int16 consumed from x
    ADDQ $16, BX               // y slides by the same 8, lag offsets unchanged

    // A 4-wide XMM block below the 8-wide one, identical to xcorr4AVX2's. VMOVQ
    // loads put x[0..4) and each lag's y[0..4) in the low 64 bits with the upper
    // zeroed, so VPMADDWD yields two int32 partial sums (and two zero lanes) that
    // VPADDD folds into the accumulator's low lane. Same pre-body ordering and
    // wrapping-associativity basis as the 8-wide block. Left as plain AVX2: it
    // runs at most once, so fusing it to VPDPWSSD would save nothing measurable.
xcorrvnni_block4:
    TESTQ $4, CX               // n % 8 >= 4? one 4-wide XMM block absorbs the 4
    JZ   xcorrvnni_blocks16
    VMOVQ (SI), X0             // x[0..4) in the low 64 bits (upper zeroed)
    VMOVQ (BX), X1             // lag 0: y[0..4)
    VPMADDWD X0, X1, X1        // 2 int32 partial sums (+ 2 zero lanes)
    VPADDD X1, X4, X4          // VEX-128 zeroes Y4[255:128]; it is still zero
    VMOVQ 2(BX), X1            // lag 1
    VPMADDWD X0, X1, X1
    VPADDD X1, X5, X5
    VMOVQ 4(BX), X1            // lag 2
    VPMADDWD X0, X1, X1
    VPADDD X1, X6, X6
    VMOVQ 6(BX), X1            // lag 3
    VPMADDWD X0, X1, X1
    VPADDD X1, X7, X7
    ADDQ $8, SI                // 4 int16 consumed from x
    ADDQ $8, BX                // y slides by the same 4, lag offsets unchanged

xcorrvnni_blocks16:
    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16, the 16-wide block count
    JZ   xcorrvnni_fold

xcorrvnni_loop16:
    VMOVDQU (SI), Y0           // x[j..j+16), reused by all four lags
    VMOVDQU (BX), Y1           // lag 0
    BYTE $0xC4; BYTE $0xE2; BYTE $0x75; BYTE $0x52; BYTE $0xE0  // VPDPWSSD Y0, Y1, Y4 (Y4 += madd(Y1, Y0))
    VMOVDQU 2(BX), Y1          // lag 1
    BYTE $0xC4; BYTE $0xE2; BYTE $0x75; BYTE $0x52; BYTE $0xE8  // VPDPWSSD Y0, Y1, Y5
    VMOVDQU 4(BX), Y1          // lag 2
    BYTE $0xC4; BYTE $0xE2; BYTE $0x75; BYTE $0x52; BYTE $0xF0  // VPDPWSSD Y0, Y1, Y6
    VMOVDQU 6(BX), Y1          // lag 3
    BYTE $0xC4; BYTE $0xE2; BYTE $0x75; BYTE $0x52; BYTE $0xF8  // VPDPWSSD Y0, Y1, Y7
    ADDQ $32, SI
    ADDQ $32, BX
    DECQ AX
    JNZ  xcorrvnni_loop16

// Horizontal fold (#150 item 3), identical to xcorr4AVX2's dual-path fold above:
// the no-tail path (n%4 == 0) uses the three-node VPHADDD tree plus a single
// 16-byte VMOVDQU store, and the scalar-tail path (n%4 != 0) keeps the original
// per-lag VEXTRACTI128/VPSHUFD fold that lands each sum straight in R8-R11. See
// the xcorr4_avx2_fold comment for the full rationale: VPHADDD operand order and
// the [sumLag0..3] lane layout, the wrapping-add bit-exactness of regrouping the
// partials, and why the tail path avoids VPEXTRD (its latency exceeds the
// shuffles the tree saves, measured +5% at small n). ANDQ runs first so the
// aligned path pays no VPEXTRD and the tail path pays no VPHADDD.
xcorrvnni_fold:
    ANDQ $3, CX                // $3 not $7: the 8- and 4-wide blocks took those bits
    JNZ  xcorrvnni_fold_tail
    VEXTRACTI128 $1, Y4, X0
    VPADDD X0, X4, X4          // X4 = four int32 partials for lag 0
    VEXTRACTI128 $1, Y5, X0
    VPADDD X0, X5, X5          // X5 = lag 1 partials
    VEXTRACTI128 $1, Y6, X0
    VPADDD X0, X6, X6          // X6 = lag 2 partials
    VEXTRACTI128 $1, Y7, X0
    VPADDD X0, X7, X7          // X7 = lag 3 partials
    VPHADDD X5, X4, X4         // X4 = [lag0(0+1), lag0(2+3), lag1(0+1), lag1(2+3)]
    VPHADDD X7, X6, X6         // X6 = [lag2(0+1), lag2(2+3), lag3(0+1), lag3(2+3)]
    VPHADDD X6, X4, X4         // X4 = [sumLag0, sumLag1, sumLag2, sumLag3]
    VMOVDQU X4, (DI)           // no scalar tail: store all four lag sums at once
    VZEROUPPER
    RET

xcorrvnni_fold_tail:
    VEXTRACTI128 $1, Y4, X0
    VPADDD X0, X4, X4
    VPSHUFD $0x4E, X4, X0
    VPADDD X0, X4, X4
    VPSHUFD $0xB1, X4, X0
    VPADDD X0, X4, X4
    MOVQ X4, R8                // lag 0 partial sum
    VEXTRACTI128 $1, Y5, X0
    VPADDD X0, X5, X5
    VPSHUFD $0x4E, X5, X0
    VPADDD X0, X5, X5
    VPSHUFD $0xB1, X5, X0
    VPADDD X0, X5, X5
    MOVQ X5, R9                // lag 1
    VEXTRACTI128 $1, Y6, X0
    VPADDD X0, X6, X6
    VPSHUFD $0x4E, X6, X0
    VPADDD X0, X6, X6
    VPSHUFD $0xB1, X6, X0
    VPADDD X0, X6, X6
    MOVQ X6, R10               // lag 2
    VEXTRACTI128 $1, Y7, X0
    VPADDD X0, X7, X7
    VPSHUFD $0x4E, X7, X0
    VPADDD X0, X7, X7
    VPSHUFD $0xB1, X7, X0
    VPADDD X0, X7, X7
    MOVQ X7, R11               // lag 3

xcorrvnni_scalar:
    MOVWLSX (SI), AX           // x[j], sign-extended
    MOVWLSX (BX), DX
    IMULL AX, DX
    ADDL DX, R8
    MOVWLSX 2(BX), DX
    IMULL AX, DX
    ADDL DX, R9
    MOVWLSX 4(BX), DX
    IMULL AX, DX
    ADDL DX, R10
    MOVWLSX 6(BX), DX
    IMULL AX, DX
    ADDL DX, R11
    ADDQ $2, SI
    ADDQ $2, BX
    DECQ CX
    JNZ  xcorrvnni_scalar

xcorrvnni_store:
    MOVL R8, 0(DI)
    MOVL R9, 4(DI)
    MOVL R10, 8(DI)
    MOVL R11, 12(DI)
    VZEROUPPER
    RET

// Tier-3 element-wise and reduction kernels (AVX2).
//
// All three receive pre-clamped slices from the public API, so unlike the dot
// kernels there is no in-assembly length clamp anywhere below: dst_len (or
// a_len) is the trusted element count. These ops are AVX2-or-Go by dispatch
// (see i16_amd64.go); no SSE2 tier exists for them.

// func mulQ15AVX2(dst, a, b []int16)
// Rounding Q15 multiply, 16 lanes per iteration. VPMULHRSW computes
// ((a*b >> 14) + 1) >> 1, bit-identical to the rounding form
// (a*b + 2^14) >> 15 for every int16 pair, and it WRAPS the one product
// outside int16 range: (-32768)^2 -> 0x8000 = -32768, matching mulQ15Go.
// NEON's SQRDMULH saturates that pair and can never be used for this op; see
// mulq15.go. The scalar tail does the widen/round/narrow in 32-bit GPR math;
// the MOVW store keeps the low 16 bits, wrapping identically.
TEXT ·mulQ15AVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16
    JZ   mulq15_avx2_tail

mulq15_avx2_loop16:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPMULHRSW Y1, Y0, Y2       // 16 rounded Q15 products
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  mulq15_avx2_loop16

mulq15_avx2_tail:
    ANDQ $15, CX
    JZ   mulq15_avx2_done

mulq15_avx2_scalar:
    MOVWLSX (SI), AX           // a[i], sign-extended
    MOVWLSX (DI), BX           // b[i], sign-extended
    IMULL BX, AX               // product, |p| <= 2^30
    ADDL $16384, AX            // + q15Round
    SARL $15, AX               // rounding shift
    MOVW AX, (DX)              // low 16 bits: 32768 wraps to -32768
    ADDQ $2, SI
    ADDQ $2, DI
    ADDQ $2, DX
    DECQ CX
    JNZ  mulq15_avx2_scalar

mulq15_avx2_done:
    VZEROUPPER
    RET

// func absAVX2(dst, a []int16)
// Wrapping absolute value, 16 lanes per iteration: VPABSW wraps at the type
// minimum (abs(-32768) = -32768 in a 16-bit lane), which is absGo's contract.
// The scalar tail negates only the negative elements in 32-bit GPR math,
// where -(-32768) is +32768, and the MOVW store keeps the low 16 bits,
// wrapping it back to -32768 identically.
TEXT ·absAVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16
    JZ   abs_avx2_tail

abs_avx2_loop16:
    VPABSW (SI), Y0
    VMOVDQU Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  abs_avx2_loop16

abs_avx2_tail:
    ANDQ $15, CX
    JZ   abs_avx2_done

abs_avx2_scalar:
    MOVWLSX (SI), AX           // v, sign-extended
    TESTL AX, AX
    JGE  abs_avx2_store
    NEGL AX                    // |v|; -(-32768) = 32768
abs_avx2_store:
    MOVW AX, (DX)              // low 16 bits: 32768 wraps to -32768
    ADDQ $2, SI
    ADDQ $2, DX
    DECQ CX
    JNZ  abs_avx2_scalar

abs_avx2_done:
    VZEROUPPER
    RET

// func maxAbsAVX2(a []int16) int
// Per-frame abs-max (the headroom probe): VPABSW maps each lane to its
// magnitude (abs(-32768) -> 0x8000, i.e. 32768 read unsigned), VPMAXUW folds
// 16-lane blocks into an unsigned-max accumulator, a VPMAXUW/VPSRLDQ cascade
// reduces a 128-bit lane to one word, and a scalar tail folds the (n mod 16)
// remainder. The word is read zero-extended, so the result lands in
// [0, 32768], and the tail compare can stay signed because both sides fit
// that range.
TEXT ·maxAbsAVX2(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VPXOR Y0, Y0, Y0           // unsigned-max accumulator = 0

    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16
    JZ   maxabs_avx2_reduce

maxabs_avx2_loop16:
    VPABSW (SI), Y1            // |a| as unsigned words
    VPMAXUW Y1, Y0, Y0         // unsigned max accumulate
    ADDQ $32, SI
    DECQ AX
    JNZ  maxabs_avx2_loop16

maxabs_avx2_reduce:
    VEXTRACTI128 $1, Y0, X1
    VPMAXUW X1, X0, X0         // fold 16 -> 8 words
    VPSRLDQ $8, X0, X1
    VPMAXUW X1, X0, X0         // 4 words
    VPSRLDQ $4, X0, X1
    VPMAXUW X1, X0, X0         // 2 words
    VPSRLDQ $2, X0, X1
    VPMAXUW X1, X0, X0         // 1 word
    MOVQ X0, AX
    ANDQ $0xFFFF, AX           // running abs-max (unsigned word) in [0, 32768]

    ANDQ $15, CX
    JZ   maxabs_avx2_done

maxabs_avx2_scalar:
    MOVWLSX (SI), BX           // v, sign-extended
    TESTL BX, BX
    JGE  maxabs_avx2_cmp
    NEGL BX                    // |v|; -(-32768) = 32768
maxabs_avx2_cmp:
    CMPL BX, AX                // both in [0, 32768], signed compare is safe
    JLE  maxabs_avx2_next
    MOVL BX, AX                // new running max
maxabs_avx2_next:
    ADDQ $2, SI
    DECQ CX
    JNZ  maxabs_avx2_scalar

maxabs_avx2_done:
    MOVQ AX, ret+24(FP)
    VZEROUPPER
    RET
