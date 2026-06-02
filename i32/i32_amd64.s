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
