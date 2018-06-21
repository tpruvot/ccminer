#pragma once
#include <cuda_runtime.h>
#include <miner.h>

#ifdef __INTELLISENSE__
/* avoid red underlining */
#define __CUDA_ARCH__ 520
struct uint3 {
	unsigned int x, y, z;
};
struct uint3  threadIdx;
struct uint3  blockIdx;
struct uint3  blockDim;
#define atomicExch(p,y) (*p) = y
#define __funnelshift_r(a,b,c) 1
#define __syncthreads()
#define __threadfence_block()
#define asm(x)
#define __shfl(a,b,c) 1
#define __umul64hi(a,b) a*b
#endif

#define MEMORY         (1UL << 20) /* 1 MiB - 1048576 */
#define ITER           (1UL << 19) /* 512k */
#define E2I_MASK1       0xFFFF0    /* MEMORY / AES_BLOCK_SIZE - 1 = 0xFFFF */
#define E2I_MASK2       0x3FFFC    /* 0xFFFF0 >> 2 */

#define AES_BLOCK_SIZE  16
#define AES_KEY_SIZE    32
#define INIT_SIZE_BLK   8
#define INIT_SIZE_BYTE (INIT_SIZE_BLK * AES_BLOCK_SIZE) // 128 B

#define AES_RKEY_LEN 4
#define AES_COL_LEN 4
#define AES_ROUND_BASE 7

#ifndef HASH_SIZE
#define HASH_SIZE 32
#endif

#ifndef HASH_DATA_AREA
#define HASH_DATA_AREA 136
#endif

#define hi_dword(x) (x >> 32)
#define lo_dword(x) (x & 0xFFFFFFFF)

#define C32(x)    ((uint32_t)(x ## U))
#define T32(x) ((x) & C32(0xFFFFFFFF))

#ifndef ROTL64
    #if __CUDA_ARCH__ >= 350
        __forceinline__ __device__ uint64_t cuda_ROTL64(const uint64_t value, const int offset) {
            uint2 result;
            if(offset >= 32) {
                asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
                asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
            } else {
                asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
                asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
            }
            return  __double_as_longlong(__hiloint2double(result.y, result.x));
        }
        #define ROTL64(x, n) (cuda_ROTL64(x, n))
    #else
        #define ROTL64(x, n)        (((x) << (n)) | ((x) >> (64 - (n))))
    #endif
#endif

#ifndef ROTL32
    #if __CUDA_ARCH__ < 350
        #define ROTL32(x, n) T32(((x) << (n)) | ((x) >> (32 - (n))))
    #else
        #define ROTL32(x, n) __funnelshift_l( (x), (x), (n) )
    #endif
#endif

#ifndef ROTR32
    #if __CUDA_ARCH__ < 350
        #define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
    #else
        #define ROTR32(x, n) __funnelshift_r( (x), (x), (n) )
    #endif
#endif

#define MEMSET8(dst,what,cnt) { \
    int i_memset8; \
    uint64_t *out_memset8 = (uint64_t *)(dst); \
    for( i_memset8 = 0; i_memset8 < cnt; i_memset8++ ) \
        out_memset8[i_memset8] = (what); }

#define MEMSET4(dst,what,cnt) { \
    int i_memset4; \
    uint32_t *out_memset4 = (uint32_t *)(dst); \
    for( i_memset4 = 0; i_memset4 < cnt; i_memset4++ ) \
        out_memset4[i_memset4] = (what); }

#define MEMCPY8(dst,src,cnt) { \
    int i_memcpy8; \
    uint64_t *in_memcpy8 = (uint64_t *)(src); \
    uint64_t *out_memcpy8 = (uint64_t *)(dst); \
    for( i_memcpy8 = 0; i_memcpy8 < cnt; i_memcpy8++ ) \
        out_memcpy8[i_memcpy8] = in_memcpy8[i_memcpy8]; }

#define MEMCPY4(dst,src,cnt) { \
    int i_memcpy4; \
    uint32_t *in_memcpy4 = (uint32_t *)(src); \
    uint32_t *out_memcpy4 = (uint32_t *)(dst); \
    for( i_memcpy4 = 0; i_memcpy4 < cnt; i_memcpy4++ ) \
        out_memcpy4[i_memcpy4] = in_memcpy4[i_memcpy4]; }

#define XOR_BLOCKS_DST(x,y,z) { \
    ((uint64_t *)z)[0] = ((uint64_t *)(x))[0] ^ ((uint64_t *)(y))[0]; \
    ((uint64_t *)z)[1] = ((uint64_t *)(x))[1] ^ ((uint64_t *)(y))[1]; }

union hash_state {
  uint8_t b[200];
  uint64_t w[25];
};

union cn_slow_hash_state {
    union hash_state hs;
    struct {
        uint8_t k[64];
        uint8_t init[INIT_SIZE_BYTE];
    };
};

static inline void exit_if_cudaerror(int thr_id, const char *src, int line)
{
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess) {
		gpulog(LOG_ERR, thr_id, "%s %s line %d", cudaGetErrorString(err), src, line);
		exit(1);
	}
}

void cryptolight_core_cpu_hash(int thr_id, int blocks, int threads, uint32_t *d_long_state, uint32_t *d_ctx_state, uint32_t *d_ctx_a, uint32_t *d_ctx_b, uint32_t *d_ctx_key1, uint32_t *d_ctx_key2);

void cryptonight_extra_cpu_setData(int thr_id, const void *data, const void *pTargetIn);
void cryptonight_extra_cpu_init(int thr_id/*, uint32_t threads*/);
void cryptonight_extra_cpu_free(int thr_id);
void cryptonight_extra_cpu_prepare(int thr_id, int threads, uint32_t startNonce, uint32_t *d_ctx_state, uint32_t *d_ctx_a, uint32_t *d_ctx_b, uint32_t *d_ctx_key1, uint32_t *d_ctx_key2, int variant, uint32_t *d_ctx_tweak1_2);
void cryptonight_extra_cpu_final(int thr_id, int threads, uint32_t startNonce, uint32_t *resnonce, uint32_t *d_ctx_state);
