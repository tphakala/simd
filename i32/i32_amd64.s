//go:build amd64

#include "textflag.h"

// int32 (de)interleave on AMD64.
//
// Interleaving is pure 32-bit-lane data movement: no arithmetic happens, so the
// bit pattern of each lane is irrelevant and the AVX1 floating-point shuffles
// (VUNPCKLPS / VUNPCKHPS / VPERM2F128 / VSHUFPS / VUNPCKLPD / VUNPCKHPD) move
// int32 lanes exactly as they move float32 lanes. The chain is load -> shuffle
// -> store with no integer ALU op in between, so there is no FP/integer
// domain-crossing penalty. The scalar tails use the integer MOVL instead of the
// f32 VMOVSS. These kernels are the int32 counterparts of interleave2AVX /
// deinterleave2AVX in ../f32/f32_amd64.s and require only AVX (not AVX2).

// func interleave2AVX(dst, a, b []int32)
// Interleaves two slices: dst[0]=a[0], dst[1]=b[0], dst[2]=a[1], dst[3]=b[1], ...
TEXT ·interleave2AVX(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX    // dst pointer
    MOVQ a_base+24(FP), SI     // a pointer
    MOVQ a_len+32(FP), CX      // n = len(a)
    MOVQ b_base+48(FP), DI     // b pointer

    // Process 8 pairs at a time (16 output elements)
    MOVQ CX, AX
    SHRQ $3, AX                // AX = n / 8
    JZ   interleave2_avx_remainder

interleave2_avx_loop8:
    VMOVUPS (SI), Y0           // Y0 = [a0..a7]
    VMOVUPS (DI), Y1           // Y1 = [b0..b7]

    // Unpack within 128-bit lanes
    VUNPCKLPS Y1, Y0, Y2       // [a0,b0,a1,b1 | a4,b4,a5,b5]
    VUNPCKHPS Y1, Y0, Y3       // [a2,b2,a3,b3 | a6,b6,a7,b7]

    // Permute to get final order
    VPERM2F128 $0x20, Y3, Y2, Y4  // [a0,b0,a1,b1,a2,b2,a3,b3]
    VPERM2F128 $0x31, Y3, Y2, Y5  // [a4,b4,a5,b5,a6,b6,a7,b7]

    VMOVUPS Y4, (DX)
    VMOVUPS Y5, 32(DX)

    ADDQ $32, SI               // a += 8 * 4
    ADDQ $32, DI               // b += 8 * 4
    ADDQ $64, DX               // dst += 16 * 4
    DECQ AX
    JNZ  interleave2_avx_loop8

interleave2_avx_remainder:
    ANDQ $7, CX
    JZ   interleave2_avx_done

interleave2_avx_scalar:
    MOVL (SI), AX
    MOVL (DI), BX
    MOVL AX, (DX)
    MOVL BX, 4(DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $8, DX
    DECQ CX
    JNZ  interleave2_avx_scalar

interleave2_avx_done:
    VZEROUPPER
    RET

// func deinterleave2AVX(a, b, src []int32)
// Deinterleaves: a[0]=src[0], b[0]=src[1], a[1]=src[2], b[1]=src[3], ...
// 1. VSHUFPS $0x88 gathers the evens (a's), $0xDD gathers the odds (b's)
// 2. VPERM2F128 + VUNPCKLPD/VUNPCKHPD reorder the 64-bit chunks across lanes
TEXT ·deinterleave2AVX(SB), NOSPLIT, $0-72
    MOVQ a_base+0(FP), DX      // a pointer
    MOVQ a_len+8(FP), CX       // n = len(a)
    MOVQ b_base+24(FP), R8     // b pointer
    MOVQ src_base+48(FP), SI   // src pointer

    // Process 8 pairs at a time
    MOVQ CX, AX
    SHRQ $3, AX
    JZ   deinterleave2_avx_remainder

deinterleave2_avx_loop8:
    VMOVUPS (SI), Y0           // [a0,b0,a1,b1 | a2,b2,a3,b3]
    VMOVUPS 32(SI), Y1         // [a4,b4,a5,b5 | a6,b6,a7,b7]

    // Gather evens / odds within each 128-bit lane
    VSHUFPS $0x88, Y1, Y0, Y2  // [a0,a1,a4,a5 | a2,a3,a6,a7]
    VSHUFPS $0xDD, Y1, Y0, Y3  // [b0,b1,b4,b5 | b2,b3,b6,b7]

    // Reorder Y2 -> [a0,a1,a2,a3 | a4,a5,a6,a7]
    VPERM2F128 $0x01, Y2, Y2, Y4  // [a2,a3,a6,a7 | a0,a1,a4,a5]
    VUNPCKLPD Y4, Y2, Y5
    VUNPCKHPD Y4, Y2, Y6
    VPERM2F128 $0x20, Y6, Y5, Y7  // a's in order

    // Reorder Y3 -> [b0,b1,b2,b3 | b4,b5,b6,b7]
    VPERM2F128 $0x01, Y3, Y3, Y4
    VUNPCKLPD Y4, Y3, Y5
    VUNPCKHPD Y4, Y3, Y6
    VPERM2F128 $0x20, Y6, Y5, Y4  // b's in order

    VMOVUPS Y7, (DX)           // store a
    VMOVUPS Y4, (R8)           // store b

    ADDQ $64, SI               // src += 16 * 4
    ADDQ $32, DX               // a += 8 * 4
    ADDQ $32, R8               // b += 8 * 4
    DECQ AX
    JNZ  deinterleave2_avx_loop8

deinterleave2_avx_remainder:
    ANDQ $7, CX
    JZ   deinterleave2_avx_done

deinterleave2_avx_scalar:
    MOVL (SI), AX
    MOVL 4(SI), BX
    MOVL AX, (DX)
    MOVL BX, (R8)
    ADDQ $8, SI
    ADDQ $4, DX
    ADDQ $4, R8
    DECQ CX
    JNZ  deinterleave2_avx_scalar

deinterleave2_avx_done:
    VZEROUPPER
    RET

// Arithmetic, decorrelation and fixed-predictor kernels.
//
// Unlike the (de)interleave kernels above (pure 32-bit-lane data movement on
// AVX1 float shuffles), these do integer ALU work on 256-bit lanes, which
// requires AVX2 (VPADDD/VPSUBD/VPSRAD/VPSLLD/VPAND/VPOR). They are dispatched
// from i32_amd64.go gated on the AVX2 CPU feature, with the pure-Go reference as
// the fallback. Each processes 8 int32 (one YMM) per iteration with a scalar
// tail. The Go assembler's 3-operand order is dst-last: VPSUBD a, b, c is
// c = b - a.

// func addAVX2(dst, a, b []int32)
TEXT ·addAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   add_remainder

add_loop8:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPADDD  Y1, Y0, Y2         // Y2 = a + b
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  add_loop8

add_remainder:
    ANDQ $7, CX
    JZ   add_done

add_scalar:
    MOVL (SI), AX
    ADDL (DI), AX
    MOVL AX, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  add_scalar

add_done:
    VZEROUPPER
    RET

// func subAVX2(dst, a, b []int32)
TEXT ·subAVX2(SB), NOSPLIT, $0-72
    MOVQ dst_base+0(FP), DX
    MOVQ dst_len+8(FP), CX
    MOVQ a_base+24(FP), SI
    MOVQ b_base+48(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   sub_remainder

sub_loop8:
    VMOVDQU (SI), Y0
    VMOVDQU (DI), Y1
    VPSUBD  Y1, Y0, Y2         // Y2 = a - b
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    DECQ AX
    JNZ  sub_loop8

sub_remainder:
    ANDQ $7, CX
    JZ   sub_done

sub_scalar:
    MOVL (SI), AX
    SUBL (DI), AX
    MOVL AX, (DX)
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    DECQ CX
    JNZ  sub_scalar

sub_done:
    VZEROUPPER
    RET

// func midSideEncodeAVX2(mid, side, left, right []int32)
// mid = (left + right) >> 1 (arithmetic), side = left - right
TEXT ·midSideEncodeAVX2(SB), NOSPLIT, $0-96
    MOVQ mid_base+0(FP), DX
    MOVQ mid_len+8(FP), CX
    MOVQ side_base+24(FP), R8
    MOVQ left_base+48(FP), SI
    MOVQ right_base+72(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   mse_remainder

mse_loop8:
    VMOVDQU (SI), Y0           // left
    VMOVDQU (DI), Y1           // right
    VPADDD  Y1, Y0, Y2         // left + right
    VPSRAD  $1, Y2, Y2         // (left + right) >> 1
    VMOVDQU Y2, (DX)           // mid
    VPSUBD  Y1, Y0, Y3         // left - right
    VMOVDQU Y3, (R8)           // side
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    ADDQ $32, R8
    DECQ AX
    JNZ  mse_loop8

mse_remainder:
    ANDQ $7, CX
    JZ   mse_done

mse_scalar:
    MOVL (SI), AX              // left
    MOVL (DI), BX              // right
    MOVL AX, R9
    ADDL BX, R9                // left + right
    SARL $1, R9                // >> 1 arithmetic
    MOVL R9, (DX)              // mid
    SUBL BX, AX                // left - right
    MOVL AX, (R8)              // side
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    ADDQ $4, R8
    DECQ CX
    JNZ  mse_scalar

mse_done:
    VZEROUPPER
    RET

// func midSideDecodeAVX2(left, right, mid, side []int32)
// sum = (mid<<1)|(side&1); left = (sum+side)>>1; right = (sum-side)>>1
TEXT ·midSideDecodeAVX2(SB), NOSPLIT, $0-96
    MOVQ left_base+0(FP), DX
    MOVQ left_len+8(FP), CX
    MOVQ right_base+24(FP), R8
    MOVQ mid_base+48(FP), SI
    MOVQ side_base+72(FP), DI

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   msd_remainder

    VPCMPEQD Y4, Y4, Y4        // all ones
    VPSRLD   $31, Y4, Y4       // each lane = 1 (the parity mask)

msd_loop8:
    VMOVDQU (SI), Y0           // mid
    VMOVDQU (DI), Y1           // side
    VPSLLD  $1, Y0, Y2         // mid << 1
    VPAND   Y4, Y1, Y3         // side & 1
    VPOR    Y3, Y2, Y2         // sum = (mid<<1)|(side&1)
    VPADDD  Y1, Y2, Y5         // sum + side
    VPSRAD  $1, Y5, Y5         // left
    VMOVDQU Y5, (DX)
    VPSUBD  Y1, Y2, Y6         // sum - side
    VPSRAD  $1, Y6, Y6         // right
    VMOVDQU Y6, (R8)
    ADDQ $32, SI
    ADDQ $32, DI
    ADDQ $32, DX
    ADDQ $32, R8
    DECQ AX
    JNZ  msd_loop8

msd_remainder:
    ANDQ $7, CX
    JZ   msd_done

msd_scalar:
    MOVL (SI), AX              // mid
    MOVL (DI), BX              // side
    MOVL AX, R9
    SHLL $1, R9                // mid << 1
    MOVL BX, R10
    ANDL $1, R10               // side & 1
    ORL  R10, R9               // sum
    MOVL R9, AX
    ADDL BX, AX
    SARL $1, AX                // (sum+side)>>1
    MOVL AX, (DX)              // left
    MOVL R9, AX
    SUBL BX, AX
    SARL $1, AX                // (sum-side)>>1
    MOVL AX, (R8)              // right
    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, DX
    ADDQ $4, R8
    DECQ CX
    JNZ  msd_scalar

msd_done:
    VZEROUPPER
    RET

// func diff1AVX2(dst, src []int32)
// dst[0]=src[0]; dst[n]=src[n]-src[n-1]
TEXT ·diff1AVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ src_base+24(FP), SI
    MOVQ src_len+32(FP), CX

    MOVL (SI), AX
    MOVL AX, (DX)              // warm-up dst[0]=src[0]
    DECQ CX                    // residual count = len-1
    JZ   diff1_done
    ADDQ $4, DX                // &dst[1]

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   diff1_remainder

diff1_loop8:
    VMOVDQU 4(SI), Y0          // src[n]
    VMOVDQU (SI), Y1           // src[n-1]
    VPSUBD  Y1, Y0, Y2         // src[n] - src[n-1]
    VMOVDQU Y2, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  diff1_loop8

diff1_remainder:
    ANDQ $7, CX
    JZ   diff1_done

diff1_scalar:
    MOVL 4(SI), AX
    SUBL (SI), AX
    MOVL AX, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  diff1_scalar

diff1_done:
    VZEROUPPER
    RET

// func diff2AVX2(dst, src []int32)
// dst[0:2]=src[0:2]; dst[n]=src[n]-2*src[n-1]+src[n-2] = (v0+v2)-2*v1
TEXT ·diff2AVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ src_base+24(FP), SI
    MOVQ src_len+32(FP), CX

    MOVL (SI), AX
    MOVL AX, (DX)
    MOVL 4(SI), AX
    MOVL AX, 4(DX)             // warm-up dst[0:2]=src[0:2]
    SUBQ $2, CX                // residual count = len-2
    ADDQ $8, DX                // &dst[2]

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   diff2_remainder

diff2_loop8:
    VMOVDQU 8(SI), Y0          // v0 = src[n]
    VMOVDQU (SI), Y2           // v2 = src[n-2]
    VPADDD  Y2, Y0, Y0         // v0 + v2
    VMOVDQU 4(SI), Y1          // v1 = src[n-1]
    VPSLLD  $1, Y1, Y1         // 2*v1
    VPSUBD  Y1, Y0, Y0         // (v0+v2) - 2*v1
    VMOVDQU Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  diff2_loop8

diff2_remainder:
    ANDQ $7, CX
    JZ   diff2_done

diff2_scalar:
    MOVL 8(SI), AX             // v0
    MOVL 4(SI), BX             // v1
    SHLL $1, BX                // 2*v1
    SUBL BX, AX                // v0 - 2*v1
    ADDL (SI), AX              // + v2
    MOVL AX, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  diff2_scalar

diff2_done:
    VZEROUPPER
    RET

// func diff3AVX2(dst, src []int32)
// dst[0:3]=src[0:3]; dst[n]=src[n]-3src[n-1]+3src[n-2]-src[n-3] = (v0-v3)-3*(v1-v2)
TEXT ·diff3AVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ src_base+24(FP), SI
    MOVQ src_len+32(FP), CX

    MOVL (SI), AX
    MOVL AX, (DX)
    MOVL 4(SI), AX
    MOVL AX, 4(DX)
    MOVL 8(SI), AX
    MOVL AX, 8(DX)             // warm-up dst[0:3]=src[0:3]
    SUBQ $3, CX                // residual count = len-3
    ADDQ $12, DX               // &dst[3]

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   diff3_remainder

diff3_loop8:
    VMOVDQU 12(SI), Y0         // v0 = src[n]
    VMOVDQU (SI), Y3           // v3 = src[n-3]
    VPSUBD  Y3, Y0, Y0         // a = v0 - v3
    VMOVDQU 8(SI), Y1          // v1 = src[n-1]
    VMOVDQU 4(SI), Y2          // v2 = src[n-2]
    VPSUBD  Y2, Y1, Y1         // b = v1 - v2
    VPSLLD  $1, Y1, Y4         // 2*b
    VPADDD  Y1, Y4, Y4         // 3*b
    VPSUBD  Y4, Y0, Y0         // a - 3*b
    VMOVDQU Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  diff3_loop8

diff3_remainder:
    ANDQ $7, CX
    JZ   diff3_done

diff3_scalar:
    MOVL 12(SI), AX            // v0
    SUBL (SI), AX              // v0 - v3
    MOVL 8(SI), BX             // v1
    SUBL 4(SI), BX             // b = v1 - v2
    MOVL BX, R9
    SHLL $1, R9                // 2*b
    ADDL R9, BX                // 3*b
    SUBL BX, AX                // (v0-v3) - 3*b
    MOVL AX, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  diff3_scalar

diff3_done:
    VZEROUPPER
    RET

// func diff4AVX2(dst, src []int32)
// dst[0:4]=src[0:4]; dst[n]=src[n]-4src[n-1]+6src[n-2]-4src[n-3]+src[n-4]
//                          = (v0+v4) - 4*(v1+v3) + 6*v2
TEXT ·diff4AVX2(SB), NOSPLIT, $0-48
    MOVQ dst_base+0(FP), DX
    MOVQ src_base+24(FP), SI
    MOVQ src_len+32(FP), CX

    MOVL (SI), AX
    MOVL AX, (DX)
    MOVL 4(SI), AX
    MOVL AX, 4(DX)
    MOVL 8(SI), AX
    MOVL AX, 8(DX)
    MOVL 12(SI), AX
    MOVL AX, 12(DX)            // warm-up dst[0:4]=src[0:4]
    SUBQ $4, CX                // residual count = len-4
    ADDQ $16, DX               // &dst[4]

    MOVQ CX, AX
    SHRQ $3, AX
    JZ   diff4_remainder

diff4_loop8:
    VMOVDQU 16(SI), Y0         // v0 = src[n]
    VMOVDQU (SI), Y1           // v4 = src[n-4]
    VPADDD  Y1, Y0, Y0         // s04 = v0 + v4
    VMOVDQU 12(SI), Y1         // v1 = src[n-1]
    VMOVDQU 4(SI), Y2          // v3 = src[n-3]
    VPADDD  Y2, Y1, Y1         // s13 = v1 + v3
    VPSLLD  $2, Y1, Y1         // 4*s13
    VPSUBD  Y1, Y0, Y0         // s04 - 4*s13
    VMOVDQU 8(SI), Y2          // v2 = src[n-2]
    VPSLLD  $2, Y2, Y3         // 4*v2
    VPSLLD  $1, Y2, Y2         // 2*v2
    VPADDD  Y2, Y3, Y3         // 6*v2
    VPADDD  Y3, Y0, Y0         // + 6*v2
    VMOVDQU Y0, (DX)
    ADDQ $32, SI
    ADDQ $32, DX
    DECQ AX
    JNZ  diff4_loop8

diff4_remainder:
    ANDQ $7, CX
    JZ   diff4_done

diff4_scalar:
    MOVL 16(SI), AX            // v0
    ADDL (SI), AX              // v0 + v4
    MOVL 12(SI), BX            // v1
    ADDL 4(SI), BX             // v1 + v3
    SHLL $2, BX                // 4*(v1+v3)
    SUBL BX, AX                // s04 - 4*s13
    MOVL 8(SI), BX             // v2
    MOVL BX, R9
    SHLL $2, R9                // 4*v2
    SHLL $1, BX                // 2*v2
    ADDL R9, BX                // 6*v2
    ADDL BX, AX                // + 6*v2
    MOVL AX, (DX)
    ADDQ $4, SI
    ADDQ $4, DX
    DECQ CX
    JNZ  diff4_scalar

diff4_done:
    VZEROUPPER
    RET

// func cumsumAVX2(a []int32)
// In-place inclusive prefix sum: a[i] += a[i-1] for i in 1..n-1 (a[0] kept).
// This is the order-1 fixed-predictor restore and the building block the
// Restore1..Restore4 wrappers compose. Each 256-bit block computes a standalone
// 8-element inclusive prefix sum (two within-128 shift-adds plus one cross-128
// carry), adds the running total of all earlier blocks, then broadcasts its own
// last lane as the running total for the next block. The scalar tail reads the
// previous cumulative value straight from memory, so it needs no vector carry.
TEXT ·cumsumAVX2(SB), NOSPLIT, $0-24
    MOVQ a_base+0(FP), DX
    MOVQ a_len+8(FP), CX

    MOVQ CX, AX
    SHRQ $3, AX                // AX = n / 8 (block count; >=1, dispatch gates n>=8)
    JZ   cumsum_remainder

    VPXOR Y3, Y3, Y3           // Y3 = running carry (total of earlier blocks) = 0

cumsum_loop8:
    VMOVDQU (DX), Y0           // block [x0..x7]
    VPSLLDQ $4, Y0, Y1         // shift each 128-bit half left 1 lane (zero fill)
    VPADDD  Y1, Y0, Y0         // partial within-128 prefix
    VPSLLDQ $8, Y0, Y1         // shift each 128-bit half left 2 lanes
    VPADDD  Y1, Y0, Y0         // low/high halves each hold a 4-elem prefix sum
    VPSHUFD $0xFF, Y0, Y2      // broadcast lane 3 of each half: low=[L3..], high=[H3..]
    VPERM2I128 $0x08, Y2, Y2, Y2 // low half -> 0, high half -> [L3,L3,L3,L3]
    VPADDD  Y2, Y0, Y0         // fold low-half total into high half: full 8-elem prefix
    VPADDD  Y3, Y0, Y0         // add carry accumulated from earlier blocks
    VMOVDQU Y0, (DX)
    VPSHUFD $0xFF, Y0, Y2      // high half = [lane7 x4]
    VPERM2I128 $0x11, Y2, Y2, Y3 // Y3 = [lane7 x8] = new running total
    ADDQ $32, DX
    DECQ AX
    JNZ  cumsum_loop8

cumsum_remainder:
    ANDQ $7, CX
    JZ   cumsum_done

    MOVL -4(DX), BX            // carry = previous cumulative value (a[i-1] in memory)

cumsum_scalar:
    MOVL (DX), AX
    ADDL BX, AX
    MOVL AX, (DX)
    MOVL AX, BX
    ADDQ $4, DX
    DECQ CX
    JNZ  cumsum_scalar

cumsum_done:
    VZEROUPPER
    RET

// func lpcResidualEncodeAVX2(res, samples, coeffs []int32, shift uint)
// Quantized-LPC encode FIR, vectorized across 8 output samples per iteration:
//
//	res[i] = samples[i] - int32((Σ_j coeffs[j]*samples[i-1-j]) >> shift)
//
// For each tap j the 8-sample window samples[i-1-j..i+6-j] is widened to int64
// (VPMULDQ on the even int32 lanes, then again on the odd lanes shifted into
// place) and multiply-accumulated into two int64x4 accumulators: Y0 holds output
// lanes {0,2,4,6}, Y1 holds {1,3,5,7}. After the tap loop each accumulator is
// arithmetic-shifted right by shift, then the low 32 bits are recombined into the
// int32x8 prediction and subtracted from samples[i..i+7].
//
// AVX2 has no 64-bit arithmetic shift, so it is emulated as
// asr(x,s) = ((x XOR 2^63) >>u s) - 2^(63-s), with VPSRLVQ supplying the logical
// shift by the broadcast count. The first 'order' samples are the verbatim
// warm-up; the (n-order) mod 8 trailing outputs use a scalar int64 recurrence.
// The dispatch guarantees order >= 1 and n-order >= 8 (at least one full block).
TEXT ·lpcResidualEncodeAVX2(SB), NOSPLIT, $0-80
    MOVQ res_base+0(FP), DI
    MOVQ res_len+8(FP), DX          // n
    MOVQ samples_base+24(FP), SI
    MOVQ coeffs_base+48(FP), R9
    MOVQ coeffs_len+56(FP), R10      // order

    // warm-up: res[0:order] = samples[0:order]
    XORQ AX, AX
lpcenc_warmup:
    MOVL (SI)(AX*4), CX
    MOVL CX, (DI)(AX*4)
    INCQ AX
    CMPQ AX, R10
    JL   lpcenc_warmup

    LEAQ -4(SI)(R10*4), BX           // winBase = &samples[order-1]
    LEAQ (DI)(R10*4), DI             // DI = &res[order]
    SUBQ R10, DX                     // residual count = n - order
    MOVQ DX, R12
    SHRQ $3, R12                     // R12 = full 8-blocks (>=1)
    ANDQ $7, DX
    MOVQ DX, R13                     // R13 = remaining tail outputs

    // Build the shift constants once. Y8 = broadcast count; Y9 = 2^(63-s)
    // offset; Y7 = 2^63 bias; Y6 = low-32 mask 0x00000000FFFFFFFF.
    VPCMPEQD Y10, Y10, Y10           // all ones
    VPSLLQ   $63, Y10, Y7            // bias = 0x8000000000000000 per qword
    VPSRLQ   $32, Y10, Y6            // mask = 0x00000000FFFFFFFF per qword
    MOVQ shift+72(FP), AX
    MOVQ AX, X8
    VPBROADCASTQ X8, Y8              // count vector (shift per qword)
    MOVQ $63, CX
    SUBQ AX, CX                      // CX = 63 - shift
    MOVQ $1, AX
    SHLQ CX, AX                      // AX = 1 << (63 - shift)
    MOVQ AX, X9
    VPBROADCASTQ X9, Y9              // offset vector

lpcenc_block:
    VPXOR Y0, Y0, Y0                 // acc_even = 0
    VPXOR Y1, Y1, Y1                 // acc_odd  = 0
    XORQ R11, R11                    // j = 0
    MOVQ BX, AX                      // window ptr = winBase - j*4
lpcenc_tap:
    VMOVDQU (AX), Y2                 // S = samples[i-1-j .. i+6-j]
    VPBROADCASTD (R9)(R11*4), Y3     // C = coeffs[j] in all lanes
    VPMULDQ Y3, Y2, Y5               // Peven = S_even * C  (int64x4, lanes 0,2,4,6)
    VPSRLQ  $32, Y2, Y4              // bring odd int32 lanes into the low halves
    VPMULDQ Y3, Y4, Y4               // Podd  = S_odd  * C  (int64x4, lanes 1,3,5,7)
    VPADDQ  Y5, Y0, Y0               // acc_even += Peven
    VPADDQ  Y4, Y1, Y1               // acc_odd  += Podd
    SUBQ $4, AX                      // window for tap j+1 starts one sample earlier
    INCQ R11
    CMPQ R11, R10
    JL   lpcenc_tap

    // acc_even: arithmetic shift right by shift, then keep low 32 of each qword.
    VPXOR   Y7, Y0, Y0               // x XOR bias
    VPSRLVQ Y8, Y0, Y0               // >>u shift (per lane)
    VPSUBQ  Y9, Y0, Y0               // - 2^(63-shift)  => arithmetic shift result
    VPAND   Y6, Y0, Y0               // keep low 32 (even predictions at lanes 0,2,4,6)
    // acc_odd: same shift, then move low 32 into the high half of each qword.
    VPXOR   Y7, Y1, Y1
    VPSRLVQ Y8, Y1, Y1
    VPSUBQ  Y9, Y1, Y1
    VPSLLQ  $32, Y1, Y1              // odd predictions at lanes 1,3,5,7
    VPOR    Y1, Y0, Y0               // pred = int32x8 [p0..p7]
    VMOVDQU 4(BX), Y2                // samples[i..i+7]
    VPSUBD  Y0, Y2, Y2               // res = samples - pred
    VMOVDQU Y2, (DI)

    ADDQ $32, BX
    ADDQ $32, DI
    DECQ R12
    JNZ  lpcenc_block

    TESTQ R13, R13
    JZ    lpcenc_done
    MOVQ shift+72(FP), CX            // CL = shift for the scalar SARQ
lpcenc_tail_out:
    XORQ R8, R8                      // acc = 0 (int64)
    XORQ R11, R11                    // j = 0
    MOVQ BX, AX                      // window ptr
lpcenc_tail_tap:
    MOVLQSX (AX), DX                 // sign-extend samples[i-1-j]
    MOVLQSX (R9)(R11*4), SI          // sign-extend coeffs[j] (SI is free after warm-up)
    IMULQ   SI, DX
    ADDQ    DX, R8                   // acc += coeffs[j]*samples[i-1-j]
    SUBQ    $4, AX
    INCQ    R11
    CMPQ    R11, R10
    JL      lpcenc_tail_tap
    MOVQ R8, AX
    SARQ CX, AX                      // pred = acc >> shift (arithmetic)
    MOVL 4(BX), DX                   // samples[i]
    SUBL AX, DX                      // - pred (low 32)
    MOVL DX, (DI)
    ADDQ $4, BX
    ADDQ $4, DI
    DECQ R13
    JNZ  lpcenc_tail_out

lpcenc_done:
    VZEROUPPER
    RET

// func lpcRestoreAVX2(out, residual, rcoeffs []int32, shift uint)
// Quantized-LPC decode recurrence:
//
//	out[i] = residual[i] + int32((Σ_j coeffs[j]*out[i-1-j]) >> shift)
//
// Each out[i] feeds the next prediction, so this cannot vectorize across i; only
// the per-output tap dot product is vectorized. The caller passes rcoeffs, the
// coefficients reversed (rcoeffs[k] = coeffs[order-1-k]), so the window
// out[i-order..i-1] and rcoeffs line up as ascending contiguous loads:
// Σ_k rcoeffs[k]*out[i-order+k] = Σ_j coeffs[j]*out[i-1-j]. The oldest vecTaps
// taps are widened (VPMULDQ even/odd) into int64 accumulators and horizontally
// summed; the newest scalarTaps are added scalar.
//
// The split is not order mod 8: the newest samples out[i-1], out[i-2], ... were
// just written by this loop's narrow 4-byte stores, and a wide 32-byte vector
// load that overlaps a store still in the store buffer cannot forward and stalls
// for ~15 cycles every output. So scalarTaps = ((order+6) & 7) + 2 always keeps
// the newest 2..9 taps (which include out[i-1]) on forwardable scalar loads and
// leaves vecTaps = order - scalarTaps (a multiple of 8) for the vector loop,
// whose oldest samples have long since drained to L1. This is ~3x faster on the
// order-is-a-multiple-of-8 cases than a naive order/8 split.
// The dispatch gates order in [minLPCRestoreOrder, 32] and n-order >= 1.
TEXT ·lpcRestoreAVX2(SB), NOSPLIT, $0-80
    MOVQ out_base+0(FP), R8
    MOVQ out_len+8(FP), R12          // n
    MOVQ residual_base+24(FP), SI
    MOVQ rcoeffs_base+48(FP), R9
    MOVQ rcoeffs_len+56(FP), R10      // order
    MOVQ shift+72(FP), CX

    // warm-up: out[0:order] = residual[0:order]
    XORQ AX, AX
lpcdec_warmup:
    MOVL (SI)(AX*4), DX
    MOVL DX, (R8)(AX*4)
    INCQ AX
    CMPQ AX, R10
    JL   lpcdec_warmup

    LEAQ (R8)(R10*4), DI             // DI = &out[i],          i = order
    MOVQ R8, BX                      // BX = &out[i-order],    = &out[0]
    LEAQ (SI)(R10*4), SI             // SI = &residual[i]
    LEAQ (R8)(R12*4), R8             // R8 = &out[n] (end sentinel; R12 still = n)

    // scalarTaps = ((order+6) & 7) + 2 ; vecTaps = order - scalarTaps ; numVec = vecTaps/8.
    // vecTaps = numVec*8, so the scalar leftover starts at tap numVec*8 (R13<<3).
    MOVQ R10, AX
    ADDQ $6, AX
    ANDQ $7, AX
    ADDQ $2, AX                      // AX = scalarTaps (2..9)
    MOVQ R10, R13
    SUBQ AX, R13                     // R13 = vecTaps (>=0, multiple of 8)
    SHRQ $3, R13                     // R13 = numVec

lpcdec_out:
    CMPQ DI, R8
    JGE  lpcdec_done
    VPXOR Y0, Y0, Y0                 // acc_even = 0
    VPXOR Y1, Y1, Y1                 // acc_odd  = 0
    XORQ R11, R11                    // byte offset into window/coeffs
    MOVQ R13, AX                     // groups remaining
    TESTQ AX, AX
    JZ   lpcdec_reduce
lpcdec_vec:
    VMOVDQU (BX)(R11*1), Y2          // W = out[i-order+gi*8 ..]
    VMOVDQU (R9)(R11*1), Y3          // C = rcoeffs[gi*8 ..]
    VPMULDQ Y3, Y2, Y5              // even lanes products
    VPADDQ  Y5, Y0, Y0
    VPSRLQ  $32, Y2, Y4             // odd int32 lanes into low halves
    VPSRLQ  $32, Y3, Y6
    VPMULDQ Y6, Y4, Y4             // odd lanes products
    VPADDQ  Y4, Y1, Y1
    ADDQ $32, R11
    DECQ AX
    JNZ  lpcdec_vec
lpcdec_reduce:
    VPADDQ Y1, Y0, Y0               // S = acc_even + acc_odd (int64x4)
    VEXTRACTI128 $1, Y0, X2         // high 128
    VPADDQ X2, X0, X0              // [s0,s1] = low2 + high2
    VPSHUFD $0xEE, X0, X2          // [s1,s1,..]
    VPADDQ X2, X0, X0             // low qword = s0 + s1
    MOVQ X0, AX                    // AX = vector partial sum (int64)
    // scalar leftover taps k = numVec*8 .. order-1
    MOVQ R13, DX
    SHLQ $3, DX                    // DX = k
lpcdec_rem:
    CMPQ DX, R10
    JGE  lpcdec_pred
    MOVLQSX (R9)(DX*4), R11        // rcoeffs[k]
    MOVLQSX (BX)(DX*4), CX         // out[i-order+k] (CX reloaded as shift below)
    IMULQ   CX, R11
    ADDQ    R11, AX
    INCQ    DX
    JMP     lpcdec_rem
lpcdec_pred:
    MOVQ shift+72(FP), CX          // CL = shift
    SARQ CX, AX                    // pred = sum >> shift (arithmetic)
    MOVL (SI), DX                  // residual[i]
    ADDL AX, DX                    // + pred (low 32)
    MOVL DX, (DI)                  // out[i]
    ADDQ $4, DI
    ADDQ $4, BX
    ADDQ $4, SI
    JMP  lpcdec_out

lpcdec_done:
    VZEROUPPER
    RET
