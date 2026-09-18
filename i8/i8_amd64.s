//go:build amd64

#include "textflag.h"

// int8 SIMD kernels on AMD64 (AVX2).
//
// All kernels gate on AVX2 in i8_amd64.go and run at least one full vector
// block (the dispatch guards the minimum length), with a scalar tail for the
// (n mod block) remainder. The Go assembler's 3-operand AVX order is dst-last:
// VPSUBSB a, b, c is c = b - a, and VPMADDWD a, b, c is c = madd(b, a). Every
// mnemonic is one the Go assembler emits directly, except the AVX-VNNI kernel
// dotProduct4AVXVNNI, which spells VPDPBUSD out as VEX-form BYTE directives (the
// assembler knows only the EVEX form, which faults on AVX-VNNI-only parts; see
// that kernel and #169). Those bytes are the only lines this file contributes to
// TestNoUncheckedAmd64Encodings.
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

    // Overlapping final 16-wide block absorbs the (n mod 16) tail instead of a
    // scalar loop. toI16AVX2 is dispatched only for n >= 16, so src[n-16 .. n) is
    // in bounds; re-writing the overlap with the same widened int16 values is
    // idempotent, and src (int8) and dst (int16) are distinct slices that cannot
    // alias. Guarded on a nonzero residue so aligned n pays nothing.
toi16_remainder:
    TESTQ $15, CX
    JZ   toi16_done
    MOVQ src_base+24(FP), SI   // reload src base (SI advanced past the last block)
    ADDQ CX, SI
    SUBQ $16, SI               // SI = &src[n-16]
    VPMOVSXBW (SI), Y0         // src[n-16 .. n) -> 16 int16
    MOVQ dst_base+0(FP), DX    // reload dst base
    LEAQ (DX)(CX*2), DX        // DX = dst + 2*n
    SUBQ $32, DX               // DX = &dst[n-16]
    VMOVDQU Y0, (DX)

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

    // Overlapping final 8-wide block absorbs the (n mod 8) tail (same idempotent
    // widening-overlap argument as toI16; dispatched only for n >= 8). A short
    // non-branchy widening tail, so the win is modest, but it drops the scalar
    // loop and keeps the kernel uniform with toI16.
toi32_remainder:
    TESTQ $7, CX
    JZ   toi32_done
    MOVQ src_base+24(FP), SI   // reload src base
    ADDQ CX, SI
    SUBQ $8, SI                // SI = &src[n-8]
    VPMOVSXBD (SI), Y0         // src[n-8 .. n) -> 8 int32
    MOVQ dst_base+0(FP), DX    // reload dst base
    LEAQ (DX)(CX*4), DX        // DX = dst + 4*n
    SUBQ $32, DX               // DX = &dst[n-8]
    VMOVDQU Y0, (DX)

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

// -----------------------------------------------------------------------------
// Quantization kernels (Part of #132). All three are AVX2 and gate on hasAVX2
// in i8_amd64.go; the dispatch guards the minimum length, so each runs at least
// one full vector block and absorbs the (n mod block) tail with an overlapping
// final block (dst and src have distinct element types and cannot alias, so the
// re-store of identical values is safe). The Go assembler's 3-operand AVX order
// is dst-last. No hand-encoded directives; every mnemonic is assembler-native.
// -----------------------------------------------------------------------------

// func quantizeAVX2(dst []int8, src []float32, scale float32, zeroPoint int8)
// dst[i] = clamp(rne(src[i]/scale) + zeroPoint, -128, 127), 16 lanes/iteration.
// The clamp bounds lo=-128-zp and hi=127-zp are built as int32 then converted to
// float32 (exact, both in [-255,255]); clamping the float before VCVTPS2DQ is
// equivalent to rounding then clamping because RNE is monotone and the bounds are
// exactly representable integers. NaN is scrubbed to 0.0 (self-compare + AND) so
// it lands on zeroPoint; +Inf clamps to hi (=127 after +zp), -Inf to lo (=-128).
// The int8 pack is written fresh (not reused from f32's int16 packer): VPACKSSDW
// then VPERMQ linearizes the int16, VPACKSSWB then VPERMQ linearizes the int8, so
// the 16 output bytes are in source order (proven by the caution-B ramp test).
TEXT ·quantizeAVX2(SB), NOSPLIT, $0-53
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    VBROADCASTSS scale+48(FP), Y3   // scale x8
    MOVBLSX zeroPoint+52(FP), BX    // zp (sign-extended int32)
    MOVL $-128, AX
    SUBL BX, AX                     // AX = -128 - zp = lo (int32)
    VMOVD AX, X4
    VPBROADCASTD X4, Y4
    VCVTDQ2PS Y4, Y4                // loVec (float32)
    MOVL $127, AX
    SUBL BX, AX                     // AX = 127 - zp = hi (int32)
    VMOVD AX, X5
    VPBROADCASTD X5, Y5
    VCVTDQ2PS Y5, Y5                // hiVec (float32)
    VMOVD BX, X6
    VPBROADCASTD X6, Y6             // zpVec (int32)

    MOVQ CX, AX
    SHRQ $4, AX                     // AX = n / 16
    JZ   quant_tail

quant_loop16:
    VMOVUPS (SI), Y0                // src[0..7]
    VMOVUPS 32(SI), Y1             // src[8..15]
    VDIVPS Y3, Y0, Y0              // src / scale
    VDIVPS Y3, Y1, Y1
    VCMPPS $0, Y0, Y0, Y7         // EQ_OQ self: 0 for NaN lanes
    VANDPS Y7, Y0, Y0             // NaN -> 0.0
    VCMPPS $0, Y1, Y1, Y7
    VANDPS Y7, Y1, Y1
    VMAXPS Y4, Y0, Y0             // max(x, lo)
    VMINPS Y5, Y0, Y0             // min(., hi)
    VMAXPS Y4, Y1, Y1
    VMINPS Y5, Y1, Y1
    VCVTPS2DQ Y0, Y0              // round nearest-even -> int32
    VCVTPS2DQ Y1, Y1
    VPADDD Y6, Y0, Y0            // + zeroPoint
    VPADDD Y6, Y1, Y1
    VPACKSSDW Y1, Y0, Y2         // int32 -> int16 (lane-interleaved)
    VPERMQ $0xD8, Y2, Y2         // linearize int16
    VPACKSSWB Y2, Y2, Y2         // int16 -> int8
    VPERMQ $0xD8, Y2, Y2         // linearize: low 128 = 16 int8 in order
    VMOVDQU X2, (DX)
    ADDQ $64, SI
    ADDQ $16, DX
    DECQ AX
    JNZ  quant_loop16

quant_tail:
    ANDQ $15, CX                  // n % 16
    JZ   quant_done
    MOVQ src_base+24(FP), SI      // reload bases
    MOVQ dst_len+8(FP), AX        // n
    LEAQ -16(AX), BX              // n - 16
    LEAQ (SI)(BX*4), SI          // &src[n-16]
    MOVQ dst_base+0(FP), DX
    ADDQ BX, DX                  // &dst[n-16]
    VMOVUPS (SI), Y0
    VMOVUPS 32(SI), Y1
    VDIVPS Y3, Y0, Y0
    VDIVPS Y3, Y1, Y1
    VCMPPS $0, Y0, Y0, Y7
    VANDPS Y7, Y0, Y0
    VCMPPS $0, Y1, Y1, Y7
    VANDPS Y7, Y1, Y1
    VMAXPS Y4, Y0, Y0
    VMINPS Y5, Y0, Y0
    VMAXPS Y4, Y1, Y1
    VMINPS Y5, Y1, Y1
    VCVTPS2DQ Y0, Y0
    VCVTPS2DQ Y1, Y1
    VPADDD Y6, Y0, Y0
    VPADDD Y6, Y1, Y1
    VPACKSSDW Y1, Y0, Y2
    VPERMQ $0xD8, Y2, Y2
    VPACKSSWB Y2, Y2, Y2
    VPERMQ $0xD8, Y2, Y2
    VMOVDQU X2, (DX)

quant_done:
    VZEROUPPER
    RET

// func dequantizeAVX2(dst []float32, src []int8, scale float32, zeroPoint int8)
// dst[i] = float32(int32(src[i]) - zeroPoint) * scale, 8 lanes/iteration. The
// subtraction is exact and the int->float convert is exact (|x-zp| <= 255), so
// the single VMULPS is the only rounding.
TEXT ·dequantizeAVX2(SB), NOSPLIT, $0-53
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ src_base+24(FP), SI

    VBROADCASTSS scale+48(FP), Y3   // scale x8
    MOVBLSX zeroPoint+52(FP), BX
    VMOVD BX, X4
    VPBROADCASTD X4, Y4             // zpVec (int32)

    MOVQ CX, AX
    SHRQ $3, AX                     // AX = n / 8
    JZ   dequant_tail

dequant_loop8:
    VPMOVSXBD (SI), Y0             // 8 int8 -> 8 int32
    VPSUBD Y4, Y0, Y0             // - zeroPoint
    VCVTDQ2PS Y0, Y0             // -> float32
    VMULPS Y3, Y0, Y0           // * scale
    VMOVUPS Y0, (DX)
    ADDQ $8, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  dequant_loop8

dequant_tail:
    ANDQ $7, CX                   // n % 8
    JZ   dequant_done
    MOVQ src_base+24(FP), SI
    MOVQ dst_len+8(FP), AX        // n
    LEAQ -8(AX), BX              // n - 8
    ADDQ BX, SI                 // &src[n-8]
    MOVQ dst_base+0(FP), DX
    LEAQ (DX)(BX*4), DX         // &dst[n-8]
    VPMOVSXBD (SI), Y0
    VPSUBD Y4, Y0, Y0
    VCVTDQ2PS Y0, Y0
    VMULPS Y3, Y0, Y0
    VMOVUPS Y0, (DX)

dequant_done:
    VZEROUPPER
    RET

// func requantizeAVX2(dst []int8, acc []int32, multiplier int32, shift int, zeroPoint int8)
// The gemmlowp double-rounding epilogue, 8 lanes/iteration. left = max(shift,0)
// and right = max(-shift,0); the out-of-contract reroute in i8_amd64.go keeps
// multiplier != MinInt32 and shift in [-31,30], so the SRDHM saturation guard is
// never needed and the products cannot wrap. SRDHM reuses i32 scaleQ31AVX2's
// VPMULDQ even/odd recipe with a (1<<30) nudge added before VPSRLQ $31 (the low
// 32 bits of the logical shift equal the arithmetic shift at shift 31).
// RoundingDivideByPOT is done branch-free: q = y>>right (VPSRAVD), and the
// round-half-away increment is q - ((y&mask) > (mask>>1) - sign(y)). At right==0
// the mask is 0 so the remainder is 0 and no increment fires, giving the exact
// identity. The clamp is applied BEFORE the +zp (VPADDD wraps, so z+zp could
// overflow int32); the bounds are -128-zp and 127-zp.
TEXT ·requantizeAVX2(SB), NOSPLIT, $0-65
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), DI          // n
    MOVQ acc_base+24(FP), SI

    MOVL multiplier+48(FP), AX
    VMOVD AX, X0
    VPBROADCASTD X0, Y6            // multiplier x8 (int32)

    MOVQ shift+56(FP), BX          // shift (int64)
    XORQ CX, CX
    MOVQ BX, R9
    CMPQ R9, CX
    CMOVQLT CX, R9                // R9 = max(shift, 0) = left
    MOVQ CX, R10
    SUBQ BX, R10                  // -shift
    CMPQ R10, CX
    CMOVQLT CX, R10               // R10 = max(-shift, 0) = right

    VMOVD R9, X1
    VPBROADCASTD X1, Y7           // leftVec (int32, all lanes = left)
    VMOVD R10, X2
    VPBROADCASTD X2, Y8           // rightVec (int32, all lanes = right)

    MOVQ $0x40000000, AX          // 1<<30 nudge
    VMOVQ AX, X3
    VPBROADCASTQ X3, Y9           // nudgeVec (int64 x4)

    MOVL $1, AX
    MOVQ R10, CX                  // CX = right (shift count in CL)
    SHLL CX, AX                   // 1 << right
    DECL AX                       // mask = (1<<right) - 1
    MOVL AX, BX
    SARL $1, BX                   // halfmask = mask >> 1
    VMOVD AX, X4
    VPBROADCASTD X4, Y10          // maskVec (int32)
    VMOVD BX, X5
    VPBROADCASTD X5, Y11          // halfVec (int32)

    MOVBLSX zeroPoint+64(FP), BX  // zp (int32)
    MOVL $-128, AX
    SUBL BX, AX
    VMOVD AX, X0
    VPBROADCASTD X0, Y12          // loVec = -128 - zp
    MOVL $127, AX
    SUBL BX, AX
    VMOVD AX, X1
    VPBROADCASTD X1, Y13          // hiVec = 127 - zp
    VMOVD BX, X2
    VPBROADCASTD X2, Y14          // zpVec

    MOVQ DI, AX
    SHRQ $3, AX                   // AX = n / 8
    JZ   requant_tail

requant_loop8:
    VMOVDQU (SI), Y0              // acc[0..7]
    VPSLLVD Y7, Y0, Y0           // x = acc << left
    VPMULDQ Y6, Y0, Y1          // even-lane products (int64)
    VPSRLQ $32, Y0, Y2          // slide odd lanes into even positions
    VPMULDQ Y6, Y2, Y2          // odd-lane products (int64)
    VPADDQ Y9, Y1, Y1           // + nudge (even)
    VPADDQ Y9, Y2, Y2           // + nudge (odd)
    VPSRLQ $31, Y1, Y1          // low 32 = even SRDHM results
    VPSRLQ $31, Y2, Y2          // low 32 = odd SRDHM results
    VPSLLQ $32, Y2, Y2          // lift odd results to positions 1,3,5,7
    VPBLENDD $0xAA, Y2, Y1, Y0  // y = merge even/odd
    VPSRAVD Y8, Y0, Y1          // q = y >> right (arithmetic)
    VPAND Y10, Y0, Y2           // rem = y & mask
    VPSRAD $31, Y0, Y3          // sign = y >> 31 (all ones if negative)
    VPSUBD Y3, Y11, Y3          // thr = halfmask - sign  (+1 when y<0)
    VPCMPGTD Y3, Y2, Y2         // cmp = (rem > thr) ? -1 : 0
    VPSUBD Y2, Y1, Y0           // z = q - cmp  (+1 when rem>thr)
    VPMAXSD Y12, Y0, Y0         // clamp low  (>= -128-zp)
    VPMINSD Y13, Y0, Y0         // clamp high (<= 127-zp)
    VPADDD Y14, Y0, Y0          // + zeroPoint -> [-128,127]
    VPACKSSDW Y0, Y0, Y0        // int32 -> int16
    VPERMQ $0x08, Y0, Y0        // low 128 = [z0..z7] int16
    VPACKSSWB Y0, Y0, Y0        // low 64 = [z0..z7] int8
    VMOVQ X0, (DX)
    ADDQ $32, SI
    ADDQ $8, DX
    DECQ AX
    JNZ  requant_loop8

requant_tail:
    MOVQ DI, AX
    ANDQ $7, AX                  // n % 8
    JZ   requant_done
    MOVQ acc_base+24(FP), SI
    MOVQ dst_base+0(FP), DX
    LEAQ -8(DI), BX             // n - 8
    LEAQ (SI)(BX*4), SI        // &acc[n-8]
    ADDQ BX, DX                // &dst[n-8]
    VMOVDQU (SI), Y0
    VPSLLVD Y7, Y0, Y0
    VPMULDQ Y6, Y0, Y1
    VPSRLQ $32, Y0, Y2
    VPMULDQ Y6, Y2, Y2
    VPADDQ Y9, Y1, Y1
    VPADDQ Y9, Y2, Y2
    VPSRLQ $31, Y1, Y1
    VPSRLQ $31, Y2, Y2
    VPSLLQ $32, Y2, Y2
    VPBLENDD $0xAA, Y2, Y1, Y0
    VPSRAVD Y8, Y0, Y1
    VPAND Y10, Y0, Y2
    VPSRAD $31, Y0, Y3
    VPSUBD Y3, Y11, Y3
    VPCMPGTD Y3, Y2, Y2
    VPSUBD Y2, Y1, Y0
    VPMAXSD Y12, Y0, Y0
    VPMINSD Y13, Y0, Y0
    VPADDD Y14, Y0, Y0
    VPACKSSDW Y0, Y0, Y0
    VPERMQ $0x08, Y0, Y0
    VPACKSSWB Y0, Y0, Y0
    VMOVQ X0, (DX)

requant_done:
    VZEROUPPER
    RET

// func dotProduct4AVX2(res []int32, r0, r1, r2, r3, vec []int8)
// Four-row quantized matrix-vector kernel: res[k] = sum_j r_k[j]*vec[j] (int32,
// two's-complement wraparound), for k in 0..3. vec is sign-extended to int16
// once per block and reused across the four rows, so the query stays in
// registers instead of being re-streamed per row (the DotProductBatch win). Each
// row is at least len(vec) long (the caller's 4-row group gate), so vec_len
// drives the element count. Structure mirrors dotAVX2: an 8-wide XMM prelude
// folds n%16>=8 into the still-zero accumulators, a 16-wide YMM loop does the
// bulk with VPMOVSXBW+VPMADDWD, each row's YMM accumulator reduces to a dword,
// and an n%8 scalar tail finishes. int32 addition is associative, so any block
// order is bit-identical to the scalar dotGo reference.
TEXT ·dotProduct4AVX2(SB), NOSPLIT, $0-144
    MOVQ res_base+0(FP), DI
    MOVQ r0_base+24(FP), R8
    MOVQ r1_base+48(FP), R9
    MOVQ r2_base+72(FP), R10
    MOVQ r3_base+96(FP), R11
    MOVQ vec_base+120(FP), SI
    MOVQ vec_len+128(FP), CX

    VPXOR Y8, Y8, Y8           // acc r0 = 0
    VPXOR Y9, Y9, Y9           // acc r1 = 0
    VPXOR Y10, Y10, Y10        // acc r2 = 0
    VPXOR Y11, Y11, Y11        // acc r3 = 0

    // 8-wide XMM block into the still-zero accumulators (the VEX.128 writes that
    // zero the upper lanes are harmless while the accumulators are zero).
    TESTQ $8, CX               // n % 16 >= 8?
    JZ    b4_blocks16
    VPMOVSXBW (SI), X0         // vec -> 8 int16
    VPMOVSXBW (R8), X1
    VPMADDWD X1, X0, X2
    VPADDD X2, X8, X8
    VPMOVSXBW (R9), X1
    VPMADDWD X1, X0, X2
    VPADDD X2, X9, X9
    VPMOVSXBW (R10), X1
    VPMADDWD X1, X0, X2
    VPADDD X2, X10, X10
    VPMOVSXBW (R11), X1
    VPMADDWD X1, X0, X2
    VPADDD X2, X11, X11
    ADDQ $8, SI
    ADDQ $8, R8
    ADDQ $8, R9
    ADDQ $8, R10
    ADDQ $8, R11

b4_blocks16:
    MOVQ CX, AX
    SHRQ $4, AX                // AX = n / 16
    JZ   b4_reduce

b4_loop16:
    VPMOVSXBW (SI), Y0         // vec -> 16 int16 (loaded once, reused x4)
    VPMOVSXBW (R8), Y1
    VPMADDWD Y1, Y0, Y2
    VPADDD Y2, Y8, Y8
    VPMOVSXBW (R9), Y1
    VPMADDWD Y1, Y0, Y2
    VPADDD Y2, Y9, Y9
    VPMOVSXBW (R10), Y1
    VPMADDWD Y1, Y0, Y2
    VPADDD Y2, Y10, Y10
    VPMOVSXBW (R11), Y1
    VPMADDWD Y1, Y0, Y2
    VPADDD Y2, Y11, Y11
    ADDQ $16, SI
    ADDQ $16, R8
    ADDQ $16, R9
    ADDQ $16, R10
    ADDQ $16, R11
    DECQ AX
    JNZ  b4_loop16

b4_reduce:
    // Reduce each row accumulator to a single int32 and store it to res[k].
    VEXTRACTI128 $1, Y8, X3
    VPADDD X3, X8, X8
    VPSHUFD $0x4E, X8, X3
    VPADDD X3, X8, X8
    VPSHUFD $0xB1, X8, X3
    VPADDD X3, X8, X8
    MOVQ X8, AX
    MOVL AX, 0(DI)

    VEXTRACTI128 $1, Y9, X3
    VPADDD X3, X9, X9
    VPSHUFD $0x4E, X9, X3
    VPADDD X3, X9, X9
    VPSHUFD $0xB1, X9, X3
    VPADDD X3, X9, X9
    MOVQ X9, AX
    MOVL AX, 4(DI)

    VEXTRACTI128 $1, Y10, X3
    VPADDD X3, X10, X10
    VPSHUFD $0x4E, X10, X3
    VPADDD X3, X10, X10
    VPSHUFD $0xB1, X10, X3
    VPADDD X3, X10, X10
    MOVQ X10, AX
    MOVL AX, 8(DI)

    VEXTRACTI128 $1, Y11, X3
    VPADDD X3, X11, X11
    VPSHUFD $0x4E, X11, X3
    VPADDD X3, X11, X11
    VPSHUFD $0xB1, X11, X3
    VPADDD X3, X11, X11
    MOVQ X11, AX
    MOVL AX, 12(DI)

    ANDQ $7, CX                // the 8-wide block took n % 16 down to n % 8
    JZ   b4_done

b4_scalar:
    MOVBLSX (SI), DX           // vec[j], shared across the four rows
    MOVBLSX (R8), AX
    IMULL DX, AX
    ADDL AX, 0(DI)
    MOVBLSX (R9), AX
    IMULL DX, AX
    ADDL AX, 4(DI)
    MOVBLSX (R10), AX
    IMULL DX, AX
    ADDL AX, 8(DI)
    MOVBLSX (R11), AX
    IMULL DX, AX
    ADDL AX, 12(DI)
    INCQ SI
    INCQ R8
    INCQ R9
    INCQ R10
    INCQ R11
    DECQ CX
    JNZ  b4_scalar

b4_done:
    VZEROUPPER
    RET

// dotProduct4AVXVNNI is dotProduct4AVX2's 4-row register-blocked matrix-vector
// dot product, fused with AVX-VNNI's VPDPBUSD. VPDPBUSD dst, u, s computes
// dst += madd over 4-byte groups of (unsigned u) * (signed s), the one
// instruction the AVX2 kernel spells as VPMOVSXBW + VPMADDWD + VPADDD.
//
// VPDPBUSD is unsigned x signed, so each signed row byte is biased to unsigned:
// ur = row XOR 0x80 = row + 128 (0..255). Then, per element,
//   ur*vec = (row+128)*vec = row*vec + 128*vec,
// so sum(ur_i*vec_i) = dot(row,vec) + 128*sum(vec_i), and the true signed dot is
// that VPDPBUSD accumulator minus 128*sum(vec). vec is the signed operand and is
// shared across the four rows, so the correction 128*sum(vec) is computed once,
// via a fifth accumulator VPDPBUSD(ones=0x01.., vec) that sums vec over exactly
// the VNNI-processed prefix. The scalar tail (n % 8) runs directly as
// signed*signed, so no correction applies there. Every add is int32 modulo 2^32;
// the bias identity is exact over that ring and wrapping adds are associative,
// so results are bit-identical to dotGo for all inputs, including forced overflow.
//
// VPDPBUSD is HAND-ENCODED as VEX.256/128.66.0F38.W0 50 /r BYTE directives, for
// the same reason as i16's xcorr4AVXVNNI (see #169): the Go assembler knows only
// the EVEX form of the mnemonic, which #UDs on AVX-VNNI-only parts such as Alder
// Lake where AVX-512 is fused off. C4 E2 is the 3-byte VEX prefix with RXB=111,
// so the destination (ModRM.reg) and the vec operand (ModRM.rm) must be Y0-Y7;
// VEX.vvvv (the unsigned operand) may use Y8-Y15. The trailing comment on each
// BYTE line is the {vex} vpdpbusd form objdump decodes it to (dst, unsigned,
// signed); a wrong byte SIGILLs or missums under the host ParityWithGo test.
//
// Registers: Y0 vec (rm/signed), Y1-Y4 row accumulators, Y5 vec-sum accumulator
// (all dst, hence Y0-Y7), Y6 = 0x80 bias, Y7 = 0x01 ones, Y8/Y9 row load + bias.
TEXT ·dotProduct4AVXVNNI(SB), NOSPLIT, $0-144
    MOVQ res_base+0(FP), DI
    MOVQ r0_base+24(FP), R8
    MOVQ r1_base+48(FP), R9
    MOVQ r2_base+72(FP), R10
    MOVQ r3_base+96(FP), R11
    MOVQ vec_base+120(FP), SI
    MOVQ vec_len+128(FP), CX

    VPCMPEQB Y7, Y7, Y7        // 0xFF bytes
    VPABSB Y7, Y7              // 0x01 bytes: ones, the unsigned operand summing vec
    VPSLLW $7, Y7, Y6          // 0x0101<<7 = 0x8080 per word -> 0x80 per byte (bias)

    VPXOR Y1, Y1, Y1           // acc r0 = 0
    VPXOR Y2, Y2, Y2           // acc r1 = 0
    VPXOR Y3, Y3, Y3           // acc r2 = 0
    VPXOR Y4, Y4, Y4           // acc r3 = 0
    VPXOR Y5, Y5, Y5           // acc sum(vec) = 0

    // Two XMM peels (16-wide then 8-wide) run BEFORE the 32-wide YMM loop, into
    // the still-zero accumulators, so their VEX.128 writes (which zero the upper
    // YMM lanes) are harmless: the lanes are zero until the YMM loop fills them,
    // and it accumulates on top afterward. Together they take n % 32 down to
    // n % 8, matching the AVX2 kernel's 8-wide prelude so the scalar tail is at
    // most 7 elements rather than 15. Legal to reorder because the int32 sums wrap
    // and add associatively.
    TESTQ $16, CX              // n % 32 >= 16?
    JZ    b4vnni_block8
    VMOVDQU (SI), X0           // vec[0..16)
    BYTE $0xC4; BYTE $0xE2; BYTE $0x41; BYTE $0x50; BYTE $0xE8  // vpdpbusd X5, X7, X0 (sum vec)
    VMOVDQU (R8), X8
    VPXOR X6, X8, X8           // biased r0
    BYTE $0xC4; BYTE $0xE2; BYTE $0x39; BYTE $0x50; BYTE $0xC8  // vpdpbusd X1, X8, X0
    VMOVDQU (R9), X9
    VPXOR X6, X9, X9
    BYTE $0xC4; BYTE $0xE2; BYTE $0x31; BYTE $0x50; BYTE $0xD0  // vpdpbusd X2, X9, X0
    VMOVDQU (R10), X8
    VPXOR X6, X8, X8
    BYTE $0xC4; BYTE $0xE2; BYTE $0x39; BYTE $0x50; BYTE $0xD8  // vpdpbusd X3, X8, X0
    VMOVDQU (R11), X9
    VPXOR X6, X9, X9
    BYTE $0xC4; BYTE $0xE2; BYTE $0x31; BYTE $0x50; BYTE $0xE0  // vpdpbusd X4, X9, X0
    ADDQ $16, SI
    ADDQ $16, R8
    ADDQ $16, R9
    ADDQ $16, R10
    ADDQ $16, R11

    // 8-wide block via 8-byte VMOVQ loads: the upper 8 bytes of each register are
    // zeroed by the move, so after the bias XOR the padding row lanes are 0x80
    // (128) but the padding vec lanes are 0, hence every padding product 128*0 = 0
    // contributes nothing to a row accumulator or to the vec sum. The same VEX.128
    // XMM VPDPBUSD encodings as the 16-wide block above.
b4vnni_block8:
    TESTQ $8, CX               // (n % 16) >= 8?
    JZ    b4vnni_loop32_setup
    VMOVQ (SI), X0             // vec[0..8) in the low 64 bits (upper zeroed)
    BYTE $0xC4; BYTE $0xE2; BYTE $0x41; BYTE $0x50; BYTE $0xE8  // vpdpbusd X5, X7, X0 (sum vec)
    VMOVQ (R8), X8
    VPXOR X6, X8, X8           // biased r0 (padding lanes -> 0x80)
    BYTE $0xC4; BYTE $0xE2; BYTE $0x39; BYTE $0x50; BYTE $0xC8  // vpdpbusd X1, X8, X0
    VMOVQ (R9), X9
    VPXOR X6, X9, X9
    BYTE $0xC4; BYTE $0xE2; BYTE $0x31; BYTE $0x50; BYTE $0xD0  // vpdpbusd X2, X9, X0
    VMOVQ (R10), X8
    VPXOR X6, X8, X8
    BYTE $0xC4; BYTE $0xE2; BYTE $0x39; BYTE $0x50; BYTE $0xD8  // vpdpbusd X3, X8, X0
    VMOVQ (R11), X9
    VPXOR X6, X9, X9
    BYTE $0xC4; BYTE $0xE2; BYTE $0x31; BYTE $0x50; BYTE $0xE0  // vpdpbusd X4, X9, X0
    ADDQ $8, SI
    ADDQ $8, R8
    ADDQ $8, R9
    ADDQ $8, R10
    ADDQ $8, R11

b4vnni_loop32_setup:
    MOVQ CX, AX
    SHRQ $5, AX                // AX = n / 32
    JZ   b4vnni_reduce

b4vnni_loop32:
    VMOVDQU (SI), Y0           // vec[j..j+32), reused by the four rows and the sum
    BYTE $0xC4; BYTE $0xE2; BYTE $0x45; BYTE $0x50; BYTE $0xE8  // vpdpbusd Y5, Y7, Y0 (sum vec)
    VMOVDQU (R8), Y8
    VPXOR Y6, Y8, Y8           // biased r0 = r0 XOR 0x80
    BYTE $0xC4; BYTE $0xE2; BYTE $0x3D; BYTE $0x50; BYTE $0xC8  // vpdpbusd Y1, Y8, Y0
    VMOVDQU (R9), Y9
    VPXOR Y6, Y9, Y9
    BYTE $0xC4; BYTE $0xE2; BYTE $0x35; BYTE $0x50; BYTE $0xD0  // vpdpbusd Y2, Y9, Y0
    VMOVDQU (R10), Y8
    VPXOR Y6, Y8, Y8
    BYTE $0xC4; BYTE $0xE2; BYTE $0x3D; BYTE $0x50; BYTE $0xD8  // vpdpbusd Y3, Y8, Y0
    VMOVDQU (R11), Y9
    VPXOR Y6, Y9, Y9
    BYTE $0xC4; BYTE $0xE2; BYTE $0x35; BYTE $0x50; BYTE $0xE0  // vpdpbusd Y4, Y9, Y0
    ADDQ $32, SI
    ADDQ $32, R8
    ADDQ $32, R9
    ADDQ $32, R10
    ADDQ $32, R11
    DECQ AX
    JNZ  b4vnni_loop32

b4vnni_reduce:
    // Reduce the vec-sum accumulator to a scalar, then corr = 128*sum(vec) (int32
    // wrap = left shift 7). X0 is the fold temporary (vec is dead now).
    VEXTRACTI128 $1, Y5, X0
    VPADDD X0, X5, X5
    VPSHUFD $0x4E, X5, X0
    VPADDD X0, X5, X5
    VPSHUFD $0xB1, X5, X0
    VPADDD X0, X5, X5
    MOVQ X5, BX
    SHLL $7, BX                // BX = 128 * sum(vec) mod 2^32

    // Reduce each row accumulator, subtract corr, store res[k].
    VEXTRACTI128 $1, Y1, X0
    VPADDD X0, X1, X1
    VPSHUFD $0x4E, X1, X0
    VPADDD X0, X1, X1
    VPSHUFD $0xB1, X1, X0
    VPADDD X0, X1, X1
    MOVQ X1, AX
    SUBL BX, AX
    MOVL AX, 0(DI)

    VEXTRACTI128 $1, Y2, X0
    VPADDD X0, X2, X2
    VPSHUFD $0x4E, X2, X0
    VPADDD X0, X2, X2
    VPSHUFD $0xB1, X2, X0
    VPADDD X0, X2, X2
    MOVQ X2, AX
    SUBL BX, AX
    MOVL AX, 4(DI)

    VEXTRACTI128 $1, Y3, X0
    VPADDD X0, X3, X3
    VPSHUFD $0x4E, X3, X0
    VPADDD X0, X3, X3
    VPSHUFD $0xB1, X3, X0
    VPADDD X0, X3, X3
    MOVQ X3, AX
    SUBL BX, AX
    MOVL AX, 8(DI)

    VEXTRACTI128 $1, Y4, X0
    VPADDD X0, X4, X4
    VPSHUFD $0x4E, X4, X0
    VPADDD X0, X4, X4
    VPSHUFD $0xB1, X4, X0
    VPADDD X0, X4, X4
    MOVQ X4, AX
    SUBL BX, AX
    MOVL AX, 12(DI)

    ANDQ $7, CX                // the 16- and 8-wide blocks took n % 32 down to n % 8
    JZ   b4vnni_done

b4vnni_scalar:
    MOVBLSX (SI), DX           // vec[j], shared across the four rows (signed)
    MOVBLSX (R8), AX
    IMULL DX, AX
    ADDL AX, 0(DI)
    MOVBLSX (R9), AX
    IMULL DX, AX
    ADDL AX, 4(DI)
    MOVBLSX (R10), AX
    IMULL DX, AX
    ADDL AX, 8(DI)
    MOVBLSX (R11), AX
    IMULL DX, AX
    ADDL AX, 12(DI)
    INCQ SI
    INCQ R8
    INCQ R9
    INCQ R10
    INCQ R11
    DECQ CX
    JNZ  b4vnni_scalar

b4vnni_done:
    VZEROUPPER
    RET
