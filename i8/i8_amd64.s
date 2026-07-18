//go:build amd64

#include "textflag.h"

// int8 SIMD kernels on AMD64 (AVX2).
//
// All kernels gate on AVX2 in i8_amd64.go and run at least one full vector
// block (the dispatch guards the minimum length), with a scalar tail for the
// (n mod block) remainder. The Go assembler's 3-operand AVX order is dst-last:
// VPSUBSB a, b, c is c = b - a, and VPMADDWD a, b, c is c = madd(b, a). No
// hand-encoded directives are used; every mnemonic is one the Go assembler
// emits directly, so TestNoUncheckedAmd64Encodings stays clean.
//
// Saturating arithmetic (VPADDSB/VPSUBSB) clamps each byte lane to [-128, 127];
// the scalar tail reproduces that with a widened add/sub and an explicit clamp.
// The reductions (Sum, DotProduct) widen bytes to int16 with VPMOVSXBW and pair-
// reduce to int32 with VPMADDWD, accumulating in int32 lanes; since int32
// wrapping addition is associative, the lane-parallel total matches the scalar
// reference modulo 2^32. Intermediate products never overflow an int32 lane
// (|int8 * int8| <= 16384).

// func addSatAVX2(dst, a, b []int8)
TEXT ·addSatAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    // A 16-wide then an 8-wide XMM block absorb up to 24 of the 0-31 remainder
    // bytes before the 32-wide loop, shrinking the branchy scalar tail from up to
    // 31 elements to at most 7. FORWARD blocks (not overlapping): each input byte
    // is read then its output written exactly once, so in-place dst==a (or dst==b)
    // stays correct for this non-idempotent op. Both input pointers advance.
    TESTQ $16, CX
    JZ   addsat_block8
    VMOVDQU (SI), X0
    VMOVDQU (DI), X1
    VPADDSB X1, X0, X2         // saturating(a + b)
    VMOVDQU X2, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
addsat_block8:
    TESTQ $8, CX
    JZ   addsat_blocks32
    VMOVQ (SI), X0             // 8 bytes (upper zeroed)
    VMOVQ (DI), X1
    VPADDSB X1, X0, X2         // low 8 = a+b; upper 8 = 0+0 = 0, never stored
    VMOVQ X2, (DX)             // store 8 bytes
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
addsat_blocks32:
    MOVQ CX, AX
    SHRQ $5, AX                // AX = n / 32
    JZ   addsat_remainder

addsat_loop32:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPADDSB Y1, Y0, Y2         // Y2 = saturating(a + b)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  addsat_loop32

addsat_remainder:
    ANDQ $7, CX                // the 16- and 8-wide blocks took n % 32 to n % 8
    JZ   addsat_done

addsat_scalar:
    MOVBLSX (SI), AX           // a (sign-extended to int32)
    MOVBLSX (DI), BX           // b
    ADDL BX, AX                // a + b in int32 (no overflow: |sum| <= 254)
    CMPL AX, $127
    JLE  addsat_chklo
    MOVL $127, AX
addsat_chklo:
    CMPL AX, $-128
    JGE  addsat_store
    MOVL $-128, AX
addsat_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DI
    INCQ DX
    DECQ CX
    JNZ  addsat_scalar

addsat_done:
    VZEROUPPER
    RET

// func subSatAVX2(dst, a, b []int8)
TEXT ·subSatAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    // 16-wide then 8-wide forward XMM pre-block; shrinks the branchy scalar tail
    // from up to 31 elements to at most 7. Each input byte is read then written
    // once, so in-place dst==a/dst==b stays correct. Both input pointers advance.
    TESTQ $16, CX
    JZ   subsat_block8
    VMOVDQU (SI), X0
    VMOVDQU (DI), X1
    VPSUBSB X1, X0, X2         // saturating(a - b)
    VMOVDQU X2, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
subsat_block8:
    TESTQ $8, CX
    JZ   subsat_blocks32
    VMOVQ (SI), X0             // 8 bytes (upper zeroed)
    VMOVQ (DI), X1
    VPSUBSB X1, X0, X2         // low 8 = a-b; upper 8 = 0-0 = 0, never stored
    VMOVQ X2, (DX)             // store 8 bytes
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
subsat_blocks32:
    MOVQ CX, AX
    SHRQ $5, AX
    JZ   subsat_remainder

subsat_loop32:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPSUBSB Y1, Y0, Y2         // Y2 = saturating(a - b)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  subsat_loop32

subsat_remainder:
    ANDQ $7, CX                // the 16- and 8-wide blocks took n % 32 to n % 8
    JZ   subsat_done

subsat_scalar:
    MOVBLSX (SI), AX
    MOVBLSX (DI), BX
    SUBL BX, AX                // a - b in int32 (|diff| <= 255)
    CMPL AX, $127
    JLE  subsat_chklo
    MOVL $127, AX
subsat_chklo:
    CMPL AX, $-128
    JGE  subsat_store
    MOVL $-128, AX
subsat_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DI
    INCQ DX
    DECQ CX
    JNZ  subsat_scalar

subsat_done:
    VZEROUPPER
    RET

// func toI16AVX2(dst []int16, src []int8)
// Sign-extends 16 int8 -> 16 int16 (256-bit) per iteration with VPMOVSXBW.
TEXT ·toI16AVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ src_base+24(FP), SI
    MOVQ src_len+32(FP), CX

    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16
    JZ   toi16_remainder

toi16_loop16:
    VPMOVSXBW (SI), Y0         // 16 bytes -> 16 int16
    VMOVDQU Y0, (DX)
    ADDQ $16, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  toi16_loop16

toi16_remainder:
    ANDQ $15, CX
    JZ   toi16_done

toi16_scalar:
    MOVBLSX (SI), AX
    MOVW AX, (DX)
    INCQ SI
    ADDQ $2, DX
    DECQ CX
    JNZ  toi16_scalar

toi16_done:
    VZEROUPPER
    RET

// func toI32AVX2(dst []int32, src []int8)
// Sign-extends 8 int8 -> 8 int32 (256-bit) per iteration with VPMOVSXBD.
TEXT ·toI32AVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ src_base+24(FP), SI
    MOVQ src_len+32(FP), CX

    MOVQ CX, AX
    SHRQ $3, AX                // AX = n / 8
    JZ   toi32_remainder

toi32_loop8:
    VPMOVSXBD (SI), Y0         // 8 bytes -> 8 int32
    VMOVDQU Y0, (DX)
    ADDQ $8, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  toi32_loop8

toi32_remainder:
    ANDQ $7, CX
    JZ   toi32_done

toi32_scalar:
    MOVBLSX (SI), AX
    MOVL AX, (DX)
    INCQ SI
    ADDQ $4, DX
    DECQ CX
    JNZ  toi32_scalar

toi32_done:
    VZEROUPPER
    RET

// func sumAVX2(a []int8) int32
// Widens 16 bytes/iter to int16 (VPMOVSXBW) and pair-reduces to int32 with
// VPMADDWD against an all-ones int16 vector, accumulating in Y2; a horizontal
// add then folds the 8 int32 lanes and a scalar tail adds the remainder.
TEXT ·sumAVX2(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VPXOR Y2, Y2, Y2           // int32 accumulator = 0
    VPCMPEQW Y4, Y4, Y4        // all ones (each int16 lane = -1)
    VPXOR Y5, Y5, Y5
    VPSUBW Y4, Y5, Y3          // Y3 = 0 - (-1) = +1 per int16 lane

    // An 8-wide XMM block absorbs 8 of the 0-15 remainder bytes before the
    // 16-wide loop, into the still-zero accumulator, so the VEX.128 write that
    // zeroes Y2[255:128] is harmless. Legal to reorder because the int32 sum
    // wraps (associative). Without it a residue of 8-15 falls entirely to the
    // serial scalar tail. Same shape as i16 dotAVX2 (#160).
    TESTQ $8, CX               // n % 16 >= 8?
    JZ   sum_blocks16
    VPMOVSXBW (SI), X0         // 8 int16
    VPMADDWD X3, X0, X1        // pairwise (x*1 + x*1) -> 4 int32
    VPADDD X1, X2, X2          // Y2[255:128] still zero after this
    ADDQ $8, SI

sum_blocks16:
    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16
    JZ   sum_reduce

sum_loop16:
    VPMOVSXBW (SI), Y0         // 16 int16
    VPMADDWD Y3, Y0, Y1        // Y1 = pairwise (x*1 + x*1) -> 8 int32
    VPADDD Y1, Y2, Y2          // accumulate
    ADDQ $16, SI
    DECQ AX
    JNZ  sum_loop16

sum_reduce:
    VEXTRACTI128 $1, Y2, X3
    VPADDD X3, X2, X2          // fold 8 -> 4 int32
    VPSHUFD $0x4E, X2, X3      // swap 64-bit halves
    VPADDD X3, X2, X2
    VPSHUFD $0xB1, X2, X3      // swap 32-bit within pairs
    VPADDD X3, X2, X2
    MOVQ X2, AX                // low int32 = vector total (in EAX)

    ANDQ $7, CX                // the 8-wide block took n % 16 down to n % 8
    JZ   sum_done

sum_scalar:
    MOVBLSX (SI), BX
    ADDL BX, AX
    INCQ SI
    DECQ CX
    JNZ  sum_scalar

sum_done:
    MOVL AX, ret+24(FP)
    VZEROUPPER
    RET

// func dotAVX2(a, b []int8) int32
// Widens 16 bytes/iter of each operand to int16 (VPMOVSXBW) and reduces the
// products to int32 with VPMADDWD, accumulating in Y2; a horizontal add folds
// the lanes and a scalar tail adds the remaining products. An 8-wide XMM block
// before the loop absorbs 8 of the 0-15 remainder bytes so a residue of 8-15
// does not fall entirely to the serial scalar tail (see i16 dotAVX2, #160).
TEXT ·dotAVX2(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    VPXOR Y2, Y2, Y2           // int32 accumulator = 0

    // 8-wide XMM block into the still-zero accumulator (the VEX.128 write that
    // zeroes Y2[255:128] is harmless). Legal to reorder because the int32
    // accumulation wraps (associative).
    TESTQ $8, CX               // n % 16 >= 8?
    JZ   dot_blocks16
    VPMOVSXBW (SI), X0         // a -> 8 int16
    VPMOVSXBW (DI), X1         // b -> 8 int16
    VPMADDWD X1, X0, X4        // 4 int32 products
    VPADDD X4, X2, X2          // Y2[255:128] still zero after this
    ADDQ $8, SI
    ADDQ $8, DI

dot_blocks16:
    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16
    JZ   dot_reduce

dot_loop16:
    VPMOVSXBW (SI), Y0         // a -> 16 int16
    VPMOVSXBW (DI), Y1         // b -> 16 int16
    VPMADDWD Y1, Y0, Y4        // Y4 = pairwise (a*b) sums -> 8 int32
    VPADDD Y4, Y2, Y2          // accumulate
    ADDQ $16, SI
    ADDQ $16, DI
    DECQ AX
    JNZ  dot_loop16

dot_reduce:
    VEXTRACTI128 $1, Y2, X3
    VPADDD X3, X2, X2
    VPSHUFD $0x4E, X2, X3
    VPADDD X3, X2, X2
    VPSHUFD $0xB1, X2, X3
    VPADDD X3, X2, X2
    MOVQ X2, AX                // EAX = vector total

    ANDQ $7, CX                // the 8-wide block took n % 16 down to n % 8
    JZ   dot_done

dot_scalar:
    MOVBLSX (SI), BX
    MOVBLSX (DI), DX
    IMULL DX, BX               // a*b (signed, fits int32)
    ADDL BX, AX
    INCQ SI
    INCQ DI
    DECQ CX
    JNZ  dot_scalar

dot_done:
    MOVL AX, ret+48(FP)
    VZEROUPPER
    RET

// func minMaxAVX2(a []int8) (minVal, maxVal int8)
// Signed byte min and max in one pass: VPMINSB/VPMAXSB fold 32-byte blocks into
// running accumulators, a per-lane cascade reduces a 128-bit lane to a single
// byte, and an overlapping final 32-byte block folds the (n mod 32) remainder
// (idempotent, so reprocessing the overlap is exact). The dispatch gates
// n >= 32, so at least one full block exists.
TEXT ·minMaxAVX2(SB), NOSPLIT, $0-26
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VMOVDQU (SI), Y0           // min acc = block 0
    VMOVDQU (SI), Y1           // max acc = block 0
    MOVQ CX, AX
    SHRQ $5, AX                // AX = full 32-byte blocks (>=1)
    DECQ AX                    // blocks remaining after block 0
    JZ   mm_overlap
    LEAQ 32(SI), DI            // working ptr at block 1

mm_loop:
    VMOVDQU (DI), Y2
    VPMINSB Y2, Y0, Y0
    VPMAXSB Y2, Y1, Y1
    ADDQ $32, DI
    DECQ AX
    JNZ  mm_loop

mm_overlap:
    // Absorb the (n mod 32) tail with an overlapping final 32-wide block instead
    // of a serial scalar tail. minMaxAVX2 is dispatched only for n >= 32, so
    // a+n-32 is in bounds; signed min/max are idempotent, so reprocessing the
    // overlap with the last full block is bit-exact. Guarded on a nonzero residue
    // so aligned n pays nothing.
    TESTQ $31, CX
    JZ   mm_reduce
    MOVQ SI, DI                // SI is still a_base (never advanced)
    ADDQ CX, DI                // DI = a + n
    VMOVDQU -32(DI), Y2        // a[n-32 .. n)
    VPMINSB Y2, Y0, Y0
    VPMAXSB Y2, Y1, Y1

mm_reduce:
    // Fold the 256-bit min accumulator to one byte.
    VEXTRACTI128 $1, Y0, X3
    VPMINSB X3, X0, X0         // 16 bytes
    VPSRLDQ $8, X0, X3
    VPMINSB X3, X0, X0         // 8 bytes
    VPSRLDQ $4, X0, X3
    VPMINSB X3, X0, X0         // 4 bytes
    VPSRLDQ $2, X0, X3
    VPMINSB X3, X0, X0         // 2 bytes
    VPSRLDQ $1, X0, X3
    VPMINSB X3, X0, X0         // 1 byte
    MOVD X0, AX                // AL = running min

    // Fold the 256-bit max accumulator to one byte.
    VEXTRACTI128 $1, Y1, X3
    VPMAXSB X3, X1, X1
    VPSRLDQ $8, X1, X3
    VPMAXSB X3, X1, X1
    VPSRLDQ $4, X1, X3
    VPMAXSB X3, X1, X1
    VPSRLDQ $2, X1, X3
    VPMAXSB X3, X1, X1
    VPSRLDQ $1, X1, X3
    VPMAXSB X3, X1, X1
    MOVD X1, DX                // DL = running max

mm_done:
    MOVB AX, minVal+24(FP)
    MOVB DX, maxVal+25(FP)
    VZEROUPPER
    RET

// func minAVX2(dst, a, b []int8)
// Element-wise signed min: VPMINSB folds 32-byte blocks; a signed scalar tail
// handles the (n mod 32) remainder.
TEXT ·minAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    // 16-wide then 8-wide forward XMM pre-block; shrinks the branchy scalar tail
    // from up to 31 elements to at most 7. Each input byte is read then written
    // once, so in-place dst==a/dst==b stays correct. Both input pointers advance.
    TESTQ $16, CX
    JZ   min_block8
    VMOVDQU (SI), X0
    VMOVDQU (DI), X1
    VPMINSB X1, X0, X2         // min(a, b)
    VMOVDQU X2, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
min_block8:
    TESTQ $8, CX
    JZ   min_blocks32
    VMOVQ (SI), X0             // 8 bytes (upper zeroed)
    VMOVQ (DI), X1
    VPMINSB X1, X0, X2         // low 8 = min(a,b); upper 8 = min(0,0) = 0, unstored
    VMOVQ X2, (DX)             // store 8 bytes
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
min_blocks32:
    MOVQ CX, AX
    SHRQ $5, AX                // AX = n / 32
    JZ   min_remainder

min_loop32:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPMINSB Y1, Y0, Y2         // Y2 = min(a, b)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  min_loop32

min_remainder:
    ANDQ $7, CX                // the 16- and 8-wide blocks took n % 32 to n % 8
    JZ   min_done

min_scalar:
    MOVBLSX (SI), AX           // a (sign-extended)
    MOVBLSX (DI), BX           // b
    CMPL AX, BX
    JLE  min_store             // a <= b -> keep a
    MOVL BX, AX
min_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DI
    INCQ DX
    DECQ CX
    JNZ  min_scalar

min_done:
    VZEROUPPER
    RET

// func maxAVX2(dst, a, b []int8)
// Element-wise signed max: VPMAXSB folds 32-byte blocks; a signed scalar tail
// handles the (n mod 32) remainder.
TEXT ·maxAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    // 16-wide then 8-wide forward XMM pre-block; shrinks the branchy scalar tail
    // from up to 31 elements to at most 7. Each input byte is read then written
    // once, so in-place dst==a/dst==b stays correct. Both input pointers advance.
    TESTQ $16, CX
    JZ   max_block8
    VMOVDQU (SI), X0
    VMOVDQU (DI), X1
    VPMAXSB X1, X0, X2         // max(a, b)
    VMOVDQU X2, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
max_block8:
    TESTQ $8, CX
    JZ   max_blocks32
    VMOVQ (SI), X0             // 8 bytes (upper zeroed)
    VMOVQ (DI), X1
    VPMAXSB X1, X0, X2         // low 8 = max(a,b); upper 8 = max(0,0) = 0, unstored
    VMOVQ X2, (DX)             // store 8 bytes
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
max_blocks32:
    MOVQ CX, AX
    SHRQ $5, AX
    JZ   max_remainder

max_loop32:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPMAXSB Y1, Y0, Y2         // Y2 = max(a, b)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  max_loop32

max_remainder:
    ANDQ $7, CX                // the 16- and 8-wide blocks took n % 32 to n % 8
    JZ   max_done

max_scalar:
    MOVBLSX (SI), AX
    MOVBLSX (DI), BX
    CMPL AX, BX
    JGE  max_store             // a >= b -> keep a
    MOVL BX, AX
max_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DI
    INCQ DX
    DECQ CX
    JNZ  max_scalar

max_done:
    VZEROUPPER
    RET

// func clampAVX2(dst, src []int8, lo, hi int8)
// Activation clip: broadcast lo/hi to all 32 lanes, then VPMAXSB(src, lo) and
// VPMINSB(., hi) per block. With lo > hi every element maps to hi. A scalar tail
// reproduces the max-then-min clamp on the (n mod 32) remainder.
TEXT ·clampAVX2(SB), NOSPLIT, $0-50
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI
    VPBROADCASTB lo+48(FP), Y3 // loVec: lo in all 32 lanes
    VPBROADCASTB hi+49(FP), Y4 // hiVec

    // 16-wide then 8-wide forward XMM pre-block; shrinks the branchy scalar tail
    // from up to 31 elements to at most 7. Each src byte is read then written
    // once, so in-place dst==src stays correct. loVec/hiVec are read-only, so the
    // pre-block reuses their X-halves (X3, X4).
    TESTQ $16, CX
    JZ   clamp_block8
    VMOVDQU (SI), X0
    VPMAXSB X3, X0, X0         // max(src, lo)
    VPMINSB X4, X0, X2         // min(., hi)
    VMOVDQU X2, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
clamp_block8:
    TESTQ $8, CX
    JZ   clamp_blocks32
    VMOVQ (SI), X0             // 8 bytes (upper zeroed)
    VPMAXSB X3, X0, X0         // upper 8 = clamp(0) garbage, never stored
    VPMINSB X4, X0, X2
    VMOVQ X2, (DX)             // store 8 bytes
    ADDQ $8, SI
    ADDQ $8, DX
clamp_blocks32:
    MOVQ CX, AX
    SHRQ $5, AX
    JZ   clamp_remainder

clamp_loop32:
    VMOVDQU (SI), Y0
    VPMAXSB Y3, Y0, Y0         // max(src, lo)
    VPMINSB Y4, Y0, Y2         // min(., hi)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  clamp_loop32

clamp_remainder:
    ANDQ $7, CX                // the 16- and 8-wide blocks took n % 32 to n % 8
    JZ   clamp_done
    MOVBLSX lo+48(FP), DI      // lo (sign-extended)
    MOVBLSX hi+49(FP), R8      // hi

clamp_scalar:
    MOVBLSX (SI), AX
    CMPL AX, DI
    JGE  clamp_chkhi
    MOVL DI, AX                // v < lo -> lo
clamp_chkhi:
    CMPL AX, R8
    JLE  clamp_store
    MOVL R8, AX                // v > hi -> hi
clamp_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DX
    DECQ CX
    JNZ  clamp_scalar

clamp_done:
    VZEROUPPER
    RET

// func absAVX2(dst, a []int8)
// Saturating absolute value: |a| = max(a, saturating(0 - a)), so abs(-128)
// clamps to 127 (VPSUBSB saturates 0-(-128)=128 to 127, and the VPMAXSB picks
// it). A scalar tail negates-if-negative then clamps the (n mod 32) remainder.
TEXT ·absAVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VPXOR Y3, Y3, Y3           // zero

    // A 16-wide then an 8-wide XMM block absorb up to 24 of the 0-31 remainder
    // bytes before the 32-wide loop, shrinking the branchy scalar tail from up to
    // 31 elements to at most 7. FORWARD blocks (not overlapping): each byte is
    // read then written exactly once, so in-place dst==a stays correct. The
    // scalar tail's sign branch mispredicts on random data (~3.6 cyc/element), so
    // moving 24 of those onto branchless SIMD is the win.
    TESTQ $16, CX
    JZ   abs_block8
    VMOVDQU (SI), X0
    VPSUBSB X0, X3, X1
    VPMAXSB X1, X0, X2
    VMOVDQU X2, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
abs_block8:
    TESTQ $8, CX
    JZ   abs_blocks32
    VMOVQ (SI), X0             // 8 bytes (upper zeroed)
    VPSUBSB X0, X3, X1
    VPMAXSB X1, X0, X2         // |a| for the low 8; upper 8 are |0|=0
    VMOVQ X2, (DX)             // store 8 bytes
    ADDQ $8, SI
    ADDQ $8, DX
abs_blocks32:
    MOVQ CX, AX
    SHRQ $5, AX
    JZ   abs_remainder

abs_loop32:
    VMOVDQU (SI), Y0
    VPSUBSB Y0, Y3, Y1         // Y1 = saturating(0 - a)
    VPMAXSB Y1, Y0, Y2         // Y2 = max(a, -a) = |a|
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  abs_loop32

abs_remainder:
    ANDQ $7, CX                // the 16- and 8-wide blocks took n % 32 to n % 8
    JZ   abs_done

abs_scalar:
    MOVBLSX (SI), AX
    TESTL AX, AX
    JGE  abs_clamp
    NEGL AX                    // |a|; -(-128) = 128
abs_clamp:
    CMPL AX, $127
    JLE  abs_store
    MOVL $127, AX              // saturate 128 -> 127
abs_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DX
    DECQ CX
    JNZ  abs_scalar

abs_done:
    VZEROUPPER
    RET

// func negAVX2(dst, a []int8)
// Saturating negation: VPSUBSB(a, 0) = saturating(0 - a), so neg(-128) clamps to
// 127. A scalar tail negates then clamps high (-a is always >= -127, so the low
// bound never binds) on the (n mod 32) remainder.
TEXT ·negAVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI

    VPXOR Y3, Y3, Y3           // zero

    // A 16-wide then an 8-wide XMM block absorb up to 24 of the 0-31 remainder
    // bytes before the 32-wide loop, shrinking the branchy scalar tail from up to
    // 31 elements to at most 7. FORWARD blocks (not overlapping): each byte is
    // read then written exactly once, so in-place dst==a stays correct even for
    // this non-idempotent op. Same template as absAVX2.
    TESTQ $16, CX
    JZ   neg_block8
    VMOVDQU (SI), X0
    VPSUBSB X0, X3, X2         // saturating(0 - a)
    VMOVDQU X2, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
neg_block8:
    TESTQ $8, CX
    JZ   neg_blocks32
    VMOVQ (SI), X0             // 8 bytes (upper zeroed)
    VPSUBSB X0, X3, X2         // low 8 = -a; upper 8 = -0 = 0, never stored
    VMOVQ X2, (DX)             // store 8 bytes
    ADDQ $8, SI
    ADDQ $8, DX
neg_blocks32:
    MOVQ CX, AX
    SHRQ $5, AX
    JZ   neg_remainder

neg_loop32:
    VMOVDQU (SI), Y0
    VPSUBSB Y0, Y3, Y2         // Y2 = saturating(0 - a)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  neg_loop32

neg_remainder:
    ANDQ $7, CX                // the 16- and 8-wide blocks took n % 32 to n % 8
    JZ   neg_done

neg_scalar:
    MOVBLSX (SI), AX
    NEGL AX                    // -a; -(-128) = 128
    CMPL AX, $127
    JLE  neg_store
    MOVL $127, AX              // saturate 128 -> 127
neg_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DX
    DECQ CX
    JNZ  neg_scalar

neg_done:
    VZEROUPPER
    RET

// func maxAbsAVX2(a []int8) int
// Per-tensor abs-max for dynamic quantization: VPABSB maps each byte to its
// magnitude (abs(-128) -> 0x80, i.e. 128 read unsigned), VPMAXUB folds 32-byte
// blocks into an unsigned-max accumulator, a VPMAXUB/VPSRLDQ cascade reduces a
// 128-bit lane to one byte, and an overlapping final 32-byte block folds the
// (n mod 32) remainder (idempotent, so reprocessing the overlap is exact).
// The result is read zero-extended, so it lands in [0, 128].
TEXT ·maxAbsAVX2(SB), NOSPLIT, $0-32
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VPXOR Y0, Y0, Y0           // unsigned-max accumulator = 0

    MOVQ CX, AX
    SHRQ $5, AX                // AX = n / 32
    JZ   maxabs_reduce

maxabs_loop32:
    VPABSB (SI), Y1            // |a| as unsigned bytes
    VPMAXUB Y1, Y0, Y0         // unsigned max accumulate
    ADDQ $32, SI
    DECQ AX
    JNZ  maxabs_loop32

    // Absorb the (n mod 32) tail with an overlapping final 32-wide block instead
    // of a serial scalar tail. maxAbsAVX2 is dispatched only for n >= 32, so
    // a+n-32 is in bounds; unsigned max is idempotent, so reprocessing the
    // overlap with the last full block is bit-exact. Guarded on a nonzero residue
    // so aligned n pays nothing.
    TESTQ $31, CX
    JZ   maxabs_reduce
    MOVQ a_base+0(FP), DI      // reload base (SI has advanced past the last block)
    ADDQ CX, DI                // DI = a + n
    VPABSB -32(DI), Y1         // |a[n-32 .. n)|
    VPMAXUB Y1, Y0, Y0

maxabs_reduce:
    VEXTRACTI128 $1, Y0, X1
    VPMAXUB X1, X0, X0         // fold 32 -> 16 bytes
    VPSRLDQ $8, X0, X1
    VPMAXUB X1, X0, X0         // 8 bytes
    VPSRLDQ $4, X0, X1
    VPMAXUB X1, X0, X0         // 4 bytes
    VPSRLDQ $2, X0, X1
    VPMAXUB X1, X0, X0         // 2 bytes
    VPSRLDQ $1, X0, X1
    VPMAXUB X1, X0, X0         // 1 byte
    MOVD X0, AX
    ANDQ $0xFF, AX             // running abs-max (unsigned byte) in [0, 128]

maxabs_done:
    MOVQ AX, ret+24(FP)
    VZEROUPPER
    RET

// func absDiffAVX2(dst, a, b []int8)
// Saturating absolute difference clamped to [0, 127]: |a-b| = max(saturating
// (a-b), saturating(b-a)); the VPMAXSB picks the non-negative capped difference,
// so |127 - (-128)| = 255 saturates to 127. A scalar tail subtracts, negates if
// negative, and clamps high on the (n mod 32) remainder.
TEXT ·absDiffAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    // 16-wide then 8-wide forward XMM pre-block; shrinks the branchy scalar tail
    // from up to 31 elements to at most 7. Each input byte is read then written
    // once, so in-place dst==a/dst==b stays correct. Both input pointers advance.
    TESTQ $16, CX
    JZ   absdiff_block8
    VMOVDQU (SI), X0           // a
    VMOVDQU (DI), X1           // b
    VPSUBSB X1, X0, X2         // saturating(a - b)
    VPSUBSB X0, X1, X3         // saturating(b - a)
    VPMAXSB X3, X2, X4         // |a - b| clamped to [0, 127]
    VMOVDQU X4, (DX)
    ADDQ $16, SI
    ADDQ $16, DI
    ADDQ $16, DX
absdiff_block8:
    TESTQ $8, CX
    JZ   absdiff_blocks32
    VMOVQ (SI), X0             // 8 bytes (upper zeroed)
    VMOVQ (DI), X1
    VPSUBSB X1, X0, X2         // low 8 = a-b; upper 8 = 0-0 = 0
    VPSUBSB X0, X1, X3         // upper 8 = 0
    VPMAXSB X3, X2, X4         // upper 8 = |0| = 0, never stored
    VMOVQ X4, (DX)             // store 8 bytes
    ADDQ $8, SI
    ADDQ $8, DI
    ADDQ $8, DX
absdiff_blocks32:
    MOVQ CX, AX
    SHRQ $5, AX
    JZ   absdiff_remainder

absdiff_loop32:
    VMOVDQU (SI), Y0           // a
    VMOVDQU (DI), Y1           // b
    VPSUBSB Y1, Y0, Y2         // Y2 = saturating(a - b)
    VPSUBSB Y0, Y1, Y3         // Y3 = saturating(b - a)
    VPMAXSB Y3, Y2, Y4         // Y4 = |a - b| clamped to [0, 127]
    VMOVDQU Y4, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  absdiff_loop32

absdiff_remainder:
    ANDQ $7, CX                // the 16- and 8-wide blocks took n % 32 to n % 8
    JZ   absdiff_done

absdiff_scalar:
    MOVBLSX (SI), AX
    MOVBLSX (DI), BX
    SUBL BX, AX                // a - b in int32 (|.| <= 255)
    TESTL AX, AX
    JGE  absdiff_clamp
    NEGL AX                    // |a - b|
absdiff_clamp:
    CMPL AX, $127
    JLE  absdiff_store
    MOVL $127, AX              // saturate to 127
absdiff_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DI
    INCQ DX
    DECQ CX
    JNZ  absdiff_scalar

absdiff_done:
    VZEROUPPER
    RET

// func addScalarSatAVX2(dst, a []int8, s int8)
// Broadcast s to all 32 lanes and add with signed saturation (VPADDSB). A scalar
// tail reproduces the widened add + clamp on the (n mod 32) remainder.
TEXT ·addScalarSatAVX2(SB), NOSPLIT, $0-49
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VPBROADCASTB s+48(FP), Y1  // s in all 32 lanes

    // 16-wide then 8-wide forward XMM pre-block; shrinks the branchy scalar tail
    // from up to 31 elements to at most 7. Each input byte is read then written
    // once, so in-place dst==a stays correct. sVec is read-only, so the pre-block
    // reuses its X-half (X1).
    TESTQ $16, CX
    JZ   addscalar_block8
    VMOVDQU (SI), X0
    VPADDSB X1, X0, X2         // saturating(a + s)
    VMOVDQU X2, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
addscalar_block8:
    TESTQ $8, CX
    JZ   addscalar_blocks32
    VMOVQ (SI), X0             // 8 bytes (upper zeroed)
    VPADDSB X1, X0, X2         // upper 8 = 0+s garbage, never stored
    VMOVQ X2, (DX)             // store 8 bytes
    ADDQ $8, SI
    ADDQ $8, DX
addscalar_blocks32:
    MOVQ CX, AX
    SHRQ $5, AX
    JZ   addscalar_remainder

addscalar_loop32:
    VMOVDQU (SI), Y0
    VPADDSB Y1, Y0, Y2         // saturating(a + s)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  addscalar_loop32

addscalar_remainder:
    ANDQ $7, CX                // the 16- and 8-wide blocks took n % 32 to n % 8
    JZ   addscalar_done
    MOVBLSX s+48(FP), DI       // s (sign-extended)

addscalar_scalar:
    MOVBLSX (SI), AX
    ADDL DI, AX                // a + s (|.| <= 255)
    CMPL AX, $127
    JLE  addscalar_chklo
    MOVL $127, AX
addscalar_chklo:
    CMPL AX, $-128
    JGE  addscalar_store
    MOVL $-128, AX
addscalar_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DX
    DECQ CX
    JNZ  addscalar_scalar

addscalar_done:
    VZEROUPPER
    RET

// func subScalarSatAVX2(dst, a []int8, s int8)
// Broadcast s to all 32 lanes and subtract with signed saturation (VPSUBSB). A
// scalar tail reproduces the widened subtract + clamp on the (n mod 32) tail.
TEXT ·subScalarSatAVX2(SB), NOSPLIT, $0-49
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    VPBROADCASTB s+48(FP), Y1  // s in all 32 lanes

    // 16-wide then 8-wide forward XMM pre-block; shrinks the branchy scalar tail
    // from up to 31 elements to at most 7. Each input byte is read then written
    // once, so in-place dst==a stays correct. sVec is read-only, so the pre-block
    // reuses its X-half (X1).
    TESTQ $16, CX
    JZ   subscalar_block8
    VMOVDQU (SI), X0
    VPSUBSB X1, X0, X2         // saturating(a - s)
    VMOVDQU X2, (DX)
    ADDQ $16, SI
    ADDQ $16, DX
subscalar_block8:
    TESTQ $8, CX
    JZ   subscalar_blocks32
    VMOVQ (SI), X0             // 8 bytes (upper zeroed)
    VPSUBSB X1, X0, X2         // upper 8 = 0-s garbage, never stored
    VMOVQ X2, (DX)             // store 8 bytes
    ADDQ $8, SI
    ADDQ $8, DX
subscalar_blocks32:
    MOVQ CX, AX
    SHRQ $5, AX
    JZ   subscalar_remainder

subscalar_loop32:
    VMOVDQU (SI), Y0
    VPSUBSB Y1, Y0, Y2         // saturating(a - s)
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  subscalar_loop32

subscalar_remainder:
    ANDQ $7, CX                // the 16- and 8-wide blocks took n % 32 to n % 8
    JZ   subscalar_done
    MOVBLSX s+48(FP), DI       // s (sign-extended)

subscalar_scalar:
    MOVBLSX (SI), AX
    SUBL DI, AX                // a - s (|.| <= 255)
    CMPL AX, $127
    JLE  subscalar_chklo
    MOVL $127, AX
subscalar_chklo:
    CMPL AX, $-128
    JGE  subscalar_store
    MOVL $-128, AX
subscalar_store:
    MOVB AX, (DX)
    INCQ SI
    INCQ DX
    DECQ CX
    JNZ  subscalar_scalar

subscalar_done:
    VZEROUPPER
    RET

// func sumAbsAVX2(a []int8) int32
// L1 norm: VPABSB maps each byte to its magnitude, VPSADBW against zero sums
// each group of 8 unsigned bytes into a u64 lane, accumulated in Y2; a fold and
// a scalar tail produce the int32 total (low 32 bits of the u64 sum, which equals
// the int32 two's-complement-wraparound sum of the non-negative terms).
TEXT ·sumAbsAVX2(SB), NOSPLIT, $0-28
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX

    VPXOR Y0, Y0, Y0           // zero (PSADBW reference)
    VPXOR Y2, Y2, Y2           // u64 accumulator

    // A 16-wide then an 8-wide XMM block absorb up to 24 of the 0-31 remainder
    // bytes before the 32-wide loop, into the still-zero u64 accumulator, so the
    // VEX.128 writes that zero Y2[255:128] are harmless (it is already zero).
    // Legal to reorder because the sum wraps and PSADBW's u64 partials add
    // associatively. Same shape as sumAVX2's 8-wide block (#173), scaled to the
    // 32-wide body. Reduces the serial scalar tail from up to 31 to at most 7.
    TESTQ $16, CX              // n % 32 >= 16?
    JZ   sumabs_block8
    VPABSB (SI), X1            // 16 |bytes|
    VPSADBW X0, X1, X3         // -> 2 u64 lanes (low 128)
    VPADDQ X3, X2, X2          // Y2[255:128] still zero after this
    ADDQ $16, SI

sumabs_block8:
    TESTQ $8, CX               // n % 16 >= 8?
    JZ   sumabs_blocks32
    VMOVQ (SI), X1             // 8 bytes in low qword (upper zeroed)
    VPABSB X1, X1
    VPSADBW X0, X1, X3         // upper 8 bytes are 0, |0|=0 -> 1 u64 lane
    VPADDQ X3, X2, X2
    ADDQ $8, SI

sumabs_blocks32:
    MOVQ CX, AX
    SHRQ $5, AX                // AX = n / 32
    JZ   sumabs_reduce

sumabs_loop32:
    VPABSB (SI), Y1            // |a| as unsigned bytes
    VPSADBW Y0, Y1, Y3         // sum|Y1 - 0| -> 4 u64 lanes
    VPADDQ Y3, Y2, Y2          // accumulate
    ADDQ $32, SI
    DECQ AX
    JNZ  sumabs_loop32

sumabs_reduce:
    VEXTRACTI128 $1, Y2, X3
    VPADDQ X3, X2, X2          // fold 4 -> 2 u64
    VPSHUFD $0x4E, X2, X3      // swap 64-bit halves
    VPADDQ X3, X2, X2          // -> total in low qword
    MOVQ X2, AX                // u64 running total

    ANDQ $7, CX                // the 16- and 8-wide blocks took n % 32 to n % 8
    JZ   sumabs_done

sumabs_scalar:
    MOVBLSX (SI), BX           // v (sign-extended)
    TESTL BX, BX
    JGE  sumabs_add
    NEGL BX                    // |v|
sumabs_add:
    ADDQ BX, AX
    INCQ SI
    DECQ CX
    JNZ  sumabs_scalar

sumabs_done:
    MOVL AX, ret+24(FP)        // low 32 bits = int32 wraparound sum
    VZEROUPPER
    RET

// func sadAVX2(a, b []int8) int32
// Sum of absolute differences. PSADBW is unsigned, so both operands are offset
// by 128 (XOR 0x80) - (a+128)-(b+128) = a-b - making the unsigned PSADBW compute
// the true signed sum|a-b| (per element in [0,255], not saturated). The 0x80
// mask is built in-register (VPCMPEQB -> VPABSB -> VPSLLW $7 = 0x80 per byte).
TEXT ·sadAVX2(SB), NOSPLIT, $0-52
    MOVQ a_base+0(FP), SI
    MOVQ a_len+8(FP), CX
    MOVQ b_base+24(FP), DI

    VPCMPEQB Y4, Y4, Y4        // 0xFF bytes
    VPABSB Y4, Y4             // 0x01 bytes
    VPSLLW $7, Y4, Y4          // 0x0101<<7 = 0x8080 per word -> 0x80 per byte
    VPXOR Y2, Y2, Y2           // u64 accumulator

    // A 16-wide then an 8-wide XMM block absorb up to 24 of the 0-31 remainder
    // bytes before the 32-wide loop, into the still-zero u64 accumulator, so the
    // VEX.128 writes that zero Y2[255:128] are harmless (it is already zero).
    // Legal to reorder because the sum wraps and PSADBW's u64 partials add
    // associatively. Reduces the serial scalar tail from up to 31 to at most 7.
    TESTQ $16, CX
    JZ   sad_block8
    VMOVDQU (SI), X0
    VMOVDQU (DI), X1
    VPXOR X4, X0, X0           // a+128 (X4 low 128 = 0x80 bytes)
    VPXOR X4, X1, X1           // b+128
    VPSADBW X1, X0, X3         // sum|a-b| over 16 bytes -> 2 u64 lanes
    VPADDQ X3, X2, X2          // Y2[255:128] still zero after this
    ADDQ $16, SI
    ADDQ $16, DI

sad_block8:
    TESTQ $8, CX
    JZ   sad_blocks32
    VMOVQ (SI), X0             // 8 bytes, upper zeroed
    VMOVQ (DI), X1
    VPXOR X4, X0, X0           // upper 8 bytes become 0x80 for BOTH a and b
    VPXOR X4, X1, X1
    VPSADBW X1, X0, X3         // upper 8: |0x80-0x80| = 0 -> contributes 0
    VPADDQ X3, X2, X2
    ADDQ $8, SI
    ADDQ $8, DI

sad_blocks32:
    MOVQ CX, AX
    SHRQ $5, AX
    JZ   sad_reduce

sad_loop32:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPXOR Y4, Y0, Y0           // a + 128
    VPXOR Y4, Y1, Y1           // b + 128
    VPSADBW Y1, Y0, Y3         // sum|(a+128)-(b+128)| = sum|a-b| -> 4 u64 lanes
    VPADDQ Y3, Y2, Y2
    ADDQ $32, SI
    ADDQ $32, DI
    DECQ AX
    JNZ  sad_loop32

sad_reduce:
    VEXTRACTI128 $1, Y2, X3
    VPADDQ X3, X2, X2
    VPSHUFD $0x4E, X2, X3
    VPADDQ X3, X2, X2
    MOVQ X2, AX                // u64 running total

    ANDQ $7, CX                // the 16- and 8-wide blocks took n % 32 to n % 8
    JZ   sad_done

sad_scalar:
    MOVBLSX (SI), BX
    MOVBLSX (DI), DX
    SUBL DX, BX                // a - b
    TESTL BX, BX
    JGE  sad_add
    NEGL BX                    // |a - b|
sad_add:
    ADDQ BX, AX
    INCQ SI
    INCQ DI
    DECQ CX
    JNZ  sad_scalar

sad_done:
    MOVL AX, ret+48(FP)
    VZEROUPPER
    RET
