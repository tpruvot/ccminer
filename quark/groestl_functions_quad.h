#include "cuda_helper.h"

__device__ __forceinline__
void G256_Mul2(uint32_t *regs)
{
    uint32_t tmp = regs[7];
    regs[7] = regs[6];
    regs[6] = regs[5];
    regs[5] = regs[4];
    regs[4] = regs[3] ^ tmp;
    regs[3] = regs[2] ^ tmp;
    regs[2] = regs[1];
    regs[1] = regs[0] ^ tmp;
    regs[0] = tmp;
}

__device__ __forceinline__
void G256_AddRoundConstantQ_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0, int rnd)
{
    x0 = ~x0;
    x1 = ~x1;
    x2 = ~x2;
    x3 = ~x3;
    x4 = ~x4;
    x5 = ~x5;
    x6 = ~x6;
    x7 = ~x7;

#if 0
    if ((threadIdx.x & 3) != 3)
        return;

    int andmask = 0xFFFF0000;
#else
    /* from sp: faster (branching problem with if ?) */
    uint32_t andmask = -((threadIdx.x & 3) == 3) & 0xFFFF0000U;
#endif

    x0 ^= ((- (rnd & 0x01)    ) & andmask);
    x1 ^= ((-((rnd & 0x02)>>1)) & andmask);
    x2 ^= ((-((rnd & 0x04)>>2)) & andmask);
    x3 ^= ((-((rnd & 0x08)>>3)) & andmask);

    x4 ^= (0xAAAA0000 & andmask);
    x5 ^= (0xCCCC0000 & andmask);
    x6 ^= (0xF0F00000 & andmask);
    x7 ^= (0xFF000000 & andmask);
}

__device__ __forceinline__
void G256_AddRoundConstantP_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0, int rnd)
{
    if (threadIdx.x & 3)
        return;

    int andmask = 0xFFFF;

    x0 ^= ((- (rnd & 0x01)    ) & andmask);
    x1 ^= ((-((rnd & 0x02)>>1)) & andmask);
    x2 ^= ((-((rnd & 0x04)>>2)) & andmask);
    x3 ^= ((-((rnd & 0x08)>>3)) & andmask);

    x4 ^= 0xAAAAU;
    x5 ^= 0xCCCCU;
    x6 ^= 0xF0F0U;
    x7 ^= 0xFF00U;
}

__device__ __forceinline__
void G16mul_quad(uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0,
                                       uint32_t &y3, uint32_t &y2, uint32_t &y1, uint32_t &y0)
{
    uint32_t t0,t1,t2;
    
    t0 = ((x2 ^ x0) ^ (x3 ^ x1)) & ((y2 ^ y0) ^ (y3 ^ y1));
    t1 = ((x2 ^ x0) & (y2 ^ y0)) ^ t0;
    t2 = ((x3 ^ x1) & (y3 ^ y1)) ^ t0 ^ t1;

    t0 = (x2^x3) & (y2^y3);
    x3 = (x3 & y3) ^ t0 ^ t1;
    x2 = (x2 & y2) ^ t0 ^ t2;

    t0 = (x0^x1) & (y0^y1);
    x1 = (x1 & y1) ^ t0 ^ t1;
    x0 = (x0 & y0) ^ t0 ^ t2;
}

__device__ __forceinline__
void G256_inv_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0)
{
    uint32_t t0,t1,t2,t3,t4,t5,t6,a,b;

    t3 = x7;
    t2 = x6;
    t1 = x5;
    t0 = x4;

    G16mul_quad(t3, t2, t1, t0, x3, x2, x1, x0);

    a = (x4 ^ x0);
    t0 ^= a;
    t2 ^= (x7 ^ x3) ^ (x5 ^ x1); 
    t1 ^= (x5 ^ x1) ^ a;
    t3 ^= (x6 ^ x2) ^ a;

    b = t0 ^ t1;
    t4 = (t2 ^ t3) & b;
    a = t4 ^ t3 ^ t1;
    t5 = (t3 & t1) ^ a;
    t6 = (t2 & t0) ^ a ^ (t2 ^ t0);

    t4 = (t5 ^ t6) & b;
    t1 = (t6 & t1) ^ t4;
    t0 = (t5 & t0) ^ t4;

    t4 = (t5 ^ t6) & (t2^t3);
    t3 = (t6 & t3) ^ t4;
    t2 = (t5 & t2) ^ t4;

    G16mul_quad(x3, x2, x1, x0, t1, t0, t3, t2);

    G16mul_quad(x7, x6, x5, x4, t1, t0, t3, t2);
}

__device__ __forceinline__
void transAtoX_quad(uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3, uint32_t &x4, uint32_t &x5, uint32_t &x6, uint32_t &x7)
{
    uint32_t t0, t1;
    t0 = x0 ^ x1 ^ x2;
    t1 = x5 ^ x6;
    x2 = t0 ^ t1 ^ x7;
    x6 = t0 ^ x3 ^ x6;
    x3 = x0 ^ x1 ^ x3 ^ x4 ^ x7;    
    x4 = x0 ^ x4 ^ t1;
    x2 = t0 ^ t1 ^ x7;
    x1 = x0 ^ x1 ^ t1;
    x7 = x0 ^ t1 ^ x7;
    x5 = x0 ^ t1;
}

__device__ __forceinline__
void transXtoA_quad(uint32_t &x0, uint32_t &x1, uint32_t &x2, uint32_t &x3, uint32_t &x4, uint32_t &x5, uint32_t &x6, uint32_t &x7)
{
    uint32_t t0,t2,t3,t5;

    x1 ^= x4;
    t0 = x1 ^ x6;
    x1 ^= x5;

    t2 = x0 ^ x2;
    x2 = x3 ^ x5;
    t2 ^= x2 ^ x6;
    x2 ^= x7;
    t3 = x4 ^ x2 ^ x6;

    t5 = x0 ^ x6;
    x4 = x3 ^ x7;
    x0 = x3 ^ x5;

    x6 = t0;    
    x3 = t2;
    x7 = t3;    
    x5 = t5;    
}

__device__ __forceinline__
void sbox_quad(uint32_t *r)
{
    transAtoX_quad(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);

    G256_inv_quad(r[2], r[4], r[1], r[7], r[3], r[0], r[5], r[6]);

    transXtoA_quad(r[7], r[1], r[4], r[2], r[6], r[5], r[0], r[3]);
    
    r[0] = ~r[0];
    r[1] = ~r[1];
    r[5] = ~r[5];
    r[6] = ~r[6];
}

__device__ __forceinline__
void G256_ShiftBytesP_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0)
{
    uint32_t t0,t1;

    int tpos = threadIdx.x & 0x03;
    int shift1 = tpos << 1;
    int shift2 = shift1+1 + ((tpos == 3)<<2);

    t0 = __byte_perm(x0, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x0, 0, 0x3232)>>shift2;
    x0 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x1, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x1, 0, 0x3232)>>shift2;
    x1 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x2, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x2, 0, 0x3232)>>shift2;
    x2 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x3, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x3, 0, 0x3232)>>shift2;
    x3 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x4, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x4, 0, 0x3232)>>shift2;
    x4 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x5, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x5, 0, 0x3232)>>shift2;
    x5 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x6, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x6, 0, 0x3232)>>shift2;
    x6 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x7, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x7, 0, 0x3232)>>shift2;
    x7 = __byte_perm(t0, t1, 0x5410);
}

__device__ __forceinline__
void G256_ShiftBytesQ_quad(uint32_t &x7, uint32_t &x6, uint32_t &x5, uint32_t &x4, uint32_t &x3, uint32_t &x2, uint32_t &x1, uint32_t &x0)
{
    uint32_t t0,t1;

    int tpos = threadIdx.x & 0x03;
    int shift1 = (1-(tpos>>1)) + ((tpos & 0x01)<<2);
    int shift2 = shift1+2 + ((tpos == 1)<<2);

    t0 = __byte_perm(x0, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x0, 0, 0x3232)>>shift2;
    x0 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x1, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x1, 0, 0x3232)>>shift2;
    x1 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x2, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x2, 0, 0x3232)>>shift2;
    x2 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x3, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x3, 0, 0x3232)>>shift2;
    x3 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x4, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x4, 0, 0x3232)>>shift2;
    x4 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x5, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x5, 0, 0x3232)>>shift2;
    x5 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x6, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x6, 0, 0x3232)>>shift2;
    x6 = __byte_perm(t0, t1, 0x5410);

    t0 = __byte_perm(x7, 0, 0x1010)>>shift1;
    t1 = __byte_perm(x7, 0, 0x3232)>>shift2;
    x7 = __byte_perm(t0, t1, 0x5410);
}

#if __CUDA_ARCH__ < 300
/**
 * __shfl() returns the value of var held by the thread whose ID is given by srcLane.
 * If srcLane is outside the range 0..width-1, the thread’s own value of var is returned.
 */
#undef __shfl
#define __shfl(var, srcLane, width) (uint32_t)(var)
#endif

#if CUDA_VERSION >= 9000
#undef __shfl
#define __shfl(var, srcLane, width)  __shfl_sync(0xFFFFFFFF, var, srcLane, width)
#endif 

__device__ __forceinline__
void G256_MixFunction_quad(uint32_t *r)
{
#define SHIFT64_16(hi, lo)    __byte_perm(lo, hi, 0x5432)
#define A(v, u)             __shfl((int)r[v], ((threadIdx.x+u)&0x03), 4)
#define S(idx, l)            SHIFT64_16( A(idx, (l+1)), A(idx, l) )

#define DOUBLE_ODD(i, bc)        ( S(i, (bc)) ^ A(i, (bc) + 1) )
#define DOUBLE_EVEN(i, bc)        ( S(i, (bc)) ^ A(i, (bc)    ) )

#define SINGLE_ODD(i, bc)        ( S(i, (bc)) )
#define SINGLE_EVEN(i, bc)        ( A(i, (bc)) )
    uint32_t b[8];

#pragma unroll 8
    for(int i=0;i<8;i++)
        b[i] = DOUBLE_ODD(i, 1) ^ DOUBLE_EVEN(i, 3);

    G256_Mul2(b);
#pragma unroll 8
    for(int i=0;i<8;i++)
        b[i] = b[i] ^ DOUBLE_ODD(i, 3) ^ DOUBLE_ODD(i, 4) ^ SINGLE_ODD(i, 6);

    G256_Mul2(b);
#pragma unroll 8
    for(int i=0;i<8;i++)
        r[i] = b[i] ^ DOUBLE_EVEN(i, 2) ^ DOUBLE_EVEN(i, 3) ^ SINGLE_EVEN(i, 5);

#undef S
#undef A
#undef SHIFT64_16
#undef t
#undef X
}

__device__ __forceinline__
void groestl512_perm_P_quad(uint32_t *r)
{
    for(int round=0;round<14;round++)
    {
        G256_AddRoundConstantP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0], round);
        sbox_quad(r);
        G256_ShiftBytesP_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
        G256_MixFunction_quad(r);
    }
}

__device__ __forceinline__
void groestl512_perm_Q_quad(uint32_t *r)
{    
    for(int round=0;round<14;round++)
    {
        G256_AddRoundConstantQ_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0], round);
        sbox_quad(r);
        G256_ShiftBytesQ_quad(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
        G256_MixFunction_quad(r);
    }
}

__device__ __forceinline__
void groestl512_progressMessage_quad(uint32_t *state, uint32_t *message)
{
#pragma unroll 8
    for(int u=0;u<8;u++) state[u] = message[u];

    if ((threadIdx.x & 0x03) == 3) state[ 1] ^= 0x00008000;
    groestl512_perm_P_quad(state);
    if ((threadIdx.x & 0x03) == 3) state[ 1] ^= 0x00008000;
    groestl512_perm_Q_quad(message);
#pragma unroll 8
    for(int u=0;u<8;u++) state[u] ^= message[u];
#pragma unroll 8
    for(int u=0;u<8;u++) message[u] = state[u];
    groestl512_perm_P_quad(message);
#pragma unroll 8
    for(int u=0;u<8;u++) state[u] ^= message[u];
}
