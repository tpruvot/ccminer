#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cryptonight.h"

#ifndef _WIN32
#include <unistd.h>
#endif

extern int device_arch[MAX_GPU][2];
extern int device_bfactor[MAX_GPU];
extern int device_bsleep[MAX_GPU];

#include "cn_aes.cuh"

#define VARIANT1_1(p) \
  do if (variant > 0 && sub == 2) \
  { \
    uint32_t tmp32 = loadGlobal32<uint32_t>(p); \
    uint8_t tmp = tmp32 >> 24; \
    uint8_t tmp1 = (tmp>>4)&1, tmp2 = (tmp>>5)&1, tmp3 = tmp1^tmp2; \
    uint8_t tmp0 = nonce_flag ? tmp3 : tmp1 + 1; \
    tmp32 &= 0x00ffffff; tmp32 |= ((tmp & 0xef) | (tmp0<<4)) << 24; \
    storeGlobal32<uint32_t>(p, tmp32); \
  } while(0)

#define VARIANT1_2(p) VARIANT1_1(p)
#define VARIANT1_INIT() \
  nonce_flag ^= (thread & 1)

__device__ __forceinline__ uint64_t cuda_mul128(uint64_t multiplier, uint64_t multiplicand, uint64_t* product_hi)
{
	*product_hi = __umul64hi(multiplier, multiplicand);
	return(multiplier * multiplicand);
}

template< typename T >
__device__ __forceinline__ T loadGlobal64(T * const addr)
{
	T x;
	asm volatile(
		"ld.global.cg.u64 %0, [%1];" : "=l"(x) : "l"(addr)
		);
	return x;
}

template< typename T >
__device__ __forceinline__ T loadGlobal32(T * const addr)
{
	T x;
	asm volatile(
		"ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(addr)
		);
	return x;
}

template< typename T >
__device__ __forceinline__ void storeGlobal32(T* addr, T const & val)
{
	asm volatile(
		"st.global.cg.u32 [%0], %1;" : : "l"(addr), "r"(val)
		);

}

__global__ void cryptonight_core_gpu_phase1(int threads, uint32_t * __restrict__ long_state, uint32_t * __restrict__ ctx_state, uint32_t * __restrict__ ctx_key1)
{
	__shared__ uint32_t sharedMemory[1024];

	const int thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;
	if(thread < threads)
	{
		cn_aes_gpu_init(sharedMemory);
		__syncthreads();

		const int sub = (threadIdx.x & 7) << 2;
		uint32_t *longstate = &long_state[(thread << 19) + sub];

		uint32_t key[40], text[4];

		MEMCPY8(key, ctx_key1 + thread * 40, 20);
		MEMCPY8(text, ctx_state + thread * 50 + sub + 16, 2);

		for(int i = 0; i < 0x80000; i += 32)
		{
			cn_aes_pseudo_round_mut(sharedMemory, text, key);
			MEMCPY8(&longstate[i], text, 2);
		}
	}
}

__device__ __forceinline__ void MUL_SUM_XOR_DST(uint64_t a, uint64_t *__restrict__ c, uint64_t *__restrict__ dst)
{
	uint64_t hi = __umul64hi(a, dst[0]) + c[0];
	uint64_t lo = a * dst[0] + c[1];
	c[0] = dst[0] ^ hi;
	c[1] = dst[1] ^ lo;
	dst[0] = hi;
	dst[1] = lo;
}

__global__ void cryptonight_core_gpu_phase2(uint32_t threads, int bfactor, int partidx, uint32_t * __restrict__ d_long_state, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b, int variant, uint8_t nonce_flag)
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const int thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
	if (thread >= threads)
		return;

	VARIANT1_INIT();

	const int sub = threadIdx.x & 3;
	const int sub2 = threadIdx.x & 2;

	int i, k;
	uint32_t j;
	const int batchsize = ITER >> (2 + bfactor);
	const int start = partidx * batchsize;
	const int end = start + batchsize;
	uint32_t * long_state = &d_long_state[thread << 19];
	uint32_t * ctx_a = d_ctx_a + thread * 4;
	uint32_t * ctx_b = d_ctx_b + thread * 4;
	uint32_t a, d[2];
	uint32_t t1[2], t2[2], res;

	a = ctx_a[sub];
	d[1] = ctx_b[sub];
#pragma unroll 2
	for (i = start; i < end; ++i)
	{
#pragma unroll 2
		for (int x = 0; x < 2; ++x)
		{
			j = ((__shfl(a, 0, 4) & 0x1FFFF0) >> 2) + sub;

			const uint32_t x_0 = loadGlobal32<uint32_t>(long_state + j);
			const uint32_t x_1 = __shfl(x_0, sub + 1, 4);
			const uint32_t x_2 = __shfl(x_0, sub + 2, 4);
			const uint32_t x_3 = __shfl(x_0, sub + 3, 4);
			d[x] = a ^
				t_fn0(x_0 & 0xff) ^
				t_fn1((x_1 >> 8) & 0xff) ^
				t_fn2((x_2 >> 16) & 0xff) ^
				t_fn3((x_3 >> 24));


			//XOR_BLOCKS_DST(c, b, &long_state[j]);
			t1[0] = __shfl(d[x], 0, 4);
			//long_state[j] = d[0] ^ d[1];
			storeGlobal32(long_state + j, d[0] ^ d[1]);
			VARIANT1_1(long_state + j);

			//MUL_SUM_XOR_DST(c, a, &long_state[((uint32_t *)c)[0] & 0x1FFFF0]);
			j = ((*t1 & 0x1FFFF0) >> 2) + sub;

			uint32_t yy[2];
			*((uint64_t*)yy) = loadGlobal64<uint64_t>(((uint64_t *)long_state) + (j >> 1));
			uint32_t zz[2];
			zz[0] = __shfl(yy[0], 0, 4);
			zz[1] = __shfl(yy[1], 0, 4);

			t1[1] = __shfl(d[x], 1, 4);
#pragma unroll
			for (k = 0; k < 2; k++)
				t2[k] = __shfl(a, k + sub2, 4);

			*((uint64_t *)t2) += sub2 ? (*((uint64_t *)t1) * *((uint64_t*)zz)) : __umul64hi(*((uint64_t *)t1), *((uint64_t*)zz));

			res = *((uint64_t *)t2) >> (sub & 1 ? 32 : 0);

			storeGlobal32(long_state + j, res);
			VARIANT1_2(long_state + j);
			a = (sub & 1 ? yy[1] : yy[0]) ^ res;
		}
	}

	if (bfactor > 0)
	{
		ctx_a[sub] = a;
		ctx_b[sub] = d[1];
	}
}

__global__ void cryptonight_core_gpu_phase3(int threads, const uint32_t * __restrict__ long_state, uint32_t * __restrict__ d_ctx_state, const uint32_t * __restrict__ d_ctx_key2)
{
	__shared__ uint32_t sharedMemory[1024];

	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const int thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;

	if(thread < threads)
	{
		const int sub = (threadIdx.x & 7) << 2;
		const uint32_t *longstate = &long_state[(thread << 19) + sub];
		uint32_t key[40], text[4], i, j;
		MEMCPY8(key, d_ctx_key2 + thread * 40, 20);
		MEMCPY8(text, d_ctx_state + thread * 50 + sub + 16, 2);

		for(i = 0; i < 0x80000; i += 32)
		{
#pragma unroll
			for(j = 0; j < 4; ++j)
				text[j] ^= longstate[i + j];

			cn_aes_pseudo_round_mut(sharedMemory, text, key);
		}

		MEMCPY8(d_ctx_state + thread * 50 + sub + 16, text, 2);
	}
}

__host__ void cryptonight_core_cpu_hash(int thr_id, int blocks, int threads, uint32_t *d_long_state, uint32_t *d_ctx_state, uint32_t *d_ctx_a, uint32_t *d_ctx_b, uint32_t *d_ctx_key1, uint32_t *d_ctx_key2, int variant, uint8_t nonce_flag)
{
	dim3 grid(blocks);
	dim3 block(threads);
	dim3 block4(threads << 2);
	dim3 block8(threads << 3);

	int i, partcount = 1 << device_bfactor[thr_id];

	cryptonight_core_gpu_phase1 <<< grid, block8 >>>(blocks*threads, d_long_state, d_ctx_state, d_ctx_key1);
	exit_if_cudaerror(thr_id, __FILE__, __LINE__);
	if(partcount > 1) usleep(device_bsleep[thr_id]);

	for(i = 0; i < partcount; i++)
	{
		cryptonight_core_gpu_phase2 <<< grid, block4 >>>(blocks*threads, device_bfactor[thr_id], i, d_long_state, d_ctx_a, d_ctx_b, variant, nonce_flag);
		exit_if_cudaerror(thr_id, __FILE__, __LINE__);
		if(partcount > 1) usleep(device_bsleep[thr_id]);
	}
	cudaDeviceSynchronize();
	exit_if_cudaerror(thr_id, __FILE__, __LINE__);
	cryptonight_core_gpu_phase3 <<< grid, block8 >>>(blocks*threads, d_long_state, d_ctx_state, d_ctx_key2);
	exit_if_cudaerror(thr_id, __FILE__, __LINE__);
}