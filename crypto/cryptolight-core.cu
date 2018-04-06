#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include "cryptolight.h"
#define LONG_SHL_IDX 18
#define LONG_LOOPS32 0x40000

#include "cl_aes.cuh"

#define MUL_SUM_XOR_DST(a,c,dst) { \
	uint64_t hi, lo = cuda_mul128(((uint64_t *)a)[0], ((uint64_t *)dst)[0], &hi) + ((uint64_t *)c)[1]; \
	hi += ((uint64_t *)c)[0]; \
	((uint64_t *)c)[0] = ((uint64_t *)dst)[0] ^ hi; \
	((uint64_t *)c)[1] = ((uint64_t *)dst)[1] ^ lo; \
	((uint64_t *)dst)[0] = hi; \
	((uint64_t *)dst)[1] = lo; }

__device__ __forceinline__ uint64_t cuda_mul128(uint64_t multiplier, uint64_t multiplicand, uint64_t* product_hi)
{
	*product_hi = __umul64hi(multiplier, multiplicand);
	return(multiplier * multiplicand);
}

__global__
void cryptolight_core_gpu_phase1(int threads, uint32_t * long_state, uint32_t * ctx_state, uint32_t * ctx_key1)
{
	__shared__ uint32_t __align__(16) sharedMemory[1024];

	cn_aes_gpu_init(sharedMemory);

	const int thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;
	const int sub = (threadIdx.x & 7) << 2;

	if(thread < threads)
	{
		const int oft = thread * 52 + sub + 16; // not aligned 16!
		const int long_oft = (thread << LONG_SHL_IDX) + sub;
		uint32_t __align__(16) key[40];
		uint32_t __align__(16) text[4];

		// copy 160 bytes
		#pragma unroll
		for (int i = 0; i < 40; i += 4)
			AS_UINT4(&key[i]) = AS_UINT4(ctx_key1 + thread * 40 + i);

		AS_UINT2(&text[0]) = AS_UINT2(&ctx_state[oft]);
		AS_UINT2(&text[2]) = AS_UINT2(&ctx_state[oft + 2]);

		__syncthreads();
		for(int i = 0; i < LONG_LOOPS32; i += 32) {
			cn_aes_pseudo_round_mut(sharedMemory, text, key);
			AS_UINT4(&long_state[long_oft + i]) = AS_UINT4(text);
		}
	}
}

__global__
void cryptolight_core_gpu_phase2(const int threads, const int bfactor, const int partidx, uint32_t * d_long_state, uint32_t * d_ctx_a, uint32_t * d_ctx_b)
{
	__shared__ uint32_t __align__(16) sharedMemory[1024];

	cn_aes_gpu_init(sharedMemory);

	__syncthreads();

#if 0 && __CUDA_ARCH__ >= 300

	const int thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 2;
	const int sub = threadIdx.x & 3;

	if(thread < threads)
	{
		const int batchsize = ITER >> (2 + bfactor);
		const int start = partidx * batchsize;
		const int end = start + batchsize;
		uint32_t * __restrict__ long_state = &d_long_state[thread << LONG_SHL_IDX];
		uint32_t * __restrict__ ctx_a = d_ctx_a + thread * 4;
		uint32_t * __restrict__ ctx_b = d_ctx_b + thread * 4;
		uint32_t a, b, c, x[4];
		uint32_t t1[4], t2[4], res;
		uint64_t reshi, reslo;
		int j;

		a = ctx_a[sub];
		b = ctx_b[sub];

		#pragma unroll 8
		for(int i = start; i < end; ++i)
		{
			//j = ((uint32_t *)a)[0] & 0xFFFF0;
			j = (__shfl((int)a, 0, 4) & E2I_MASK1) >> 2;

			//cn_aes_single_round(sharedMemory, &long_state[j], c, a);
			x[0] = long_state[j + sub];
			x[1] = __shfl((int)x[0], sub + 1, 4);
			x[2] = __shfl((int)x[0], sub + 2, 4);
			x[3] = __shfl((int)x[0], sub + 3, 4);
			c = a ^
				t_fn0(x[0] & 0xff) ^
				t_fn1((x[1] >> 8) & 0xff) ^
				t_fn2((x[2] >> 16) & 0xff) ^
				t_fn3((x[3] >> 24) & 0xff);

			//XOR_BLOCKS_DST(c, b, &long_state[j]);
			long_state[j + sub] = c ^ b;

			//MUL_SUM_XOR_DST(c, a, &long_state[((uint32_t *)c)[0] & 0xFFFF0]);
			j = (__shfl((int)c, 0, 4) & E2I_MASK1) >> 2;
			#pragma unroll
			for(int k = 0; k < 2; k++)
				t1[k] = __shfl((int)c, k, 4);
			#pragma unroll
			for(int k = 0; k < 4; k++)
				t2[k] = __shfl((int)a, k, 4);
			asm(
				"mad.lo.u64 %0, %2, %3, %4;\n\t"
				"mad.hi.u64 %1, %2, %3, %5;\n\t"
				: "=l"(reslo), "=l"(reshi)
				: "l"(((uint64_t *)t1)[0]), "l"(((uint64_t *)long_state)[j >> 1]), "l"(((uint64_t *)t2)[1]), "l"(((uint64_t *)t2)[0]));
			res = (sub & 2 ? reslo : reshi) >> (sub & 1 ? 32 : 0);
			a = long_state[j + sub] ^ res;
			long_state[j + sub] = res;

			//j = ((uint32_t *)a)[0] & 0xFFFF0;
			j = (__shfl((int)a, 0, 4) & E2I_MASK1) >> 2;

			//cn_aes_single_round(sharedMemory, &long_state[j], b, a);
			x[0] = long_state[j + sub];
			x[1] = __shfl((int)x[0], sub + 1, 4);
			x[2] = __shfl((int)x[0], sub + 2, 4);
			x[3] = __shfl((int)x[0], sub + 3, 4);
			b = a ^
				t_fn0(x[0] & 0xff) ^
				t_fn1((x[1] >> 8) & 0xff) ^
				t_fn2((x[2] >> 16) & 0xff) ^
				t_fn3((x[3] >> 24) & 0xff);

			//XOR_BLOCKS_DST(b, c, &long_state[j]);
			long_state[j + sub] = c ^ b;

			//MUL_SUM_XOR_DST(b, a, &long_state[((uint32_t *)b)[0] & 0xFFFF0]);
			j = (__shfl((int)b, 0, 4) & E2I_MASK1) >> 2;

			#pragma unroll
			for(int k = 0; k < 2; k++)
				t1[k] = __shfl((int)b, k, 4);

			#pragma unroll
			for(int k = 0; k < 4; k++)
				t2[k] = __shfl((int)a, k, 4);
			asm(
				"mad.lo.u64 %0, %2, %3, %4;\n\t"
				"mad.hi.u64 %1, %2, %3, %5;\n\t"
				: "=l"(reslo), "=l"(reshi)
				: "l"(((uint64_t *)t1)[0]), "l"(((uint64_t *)long_state)[j >> 1]), "l"(((uint64_t *)t2)[1]), "l"(((uint64_t *)t2)[0]));
			res = (sub & 2 ? reslo : reshi) >> (sub & 1 ? 32 : 0);
			a = long_state[j + sub] ^ res;
			long_state[j + sub] = res;
		}

		if(bfactor > 0)
		{
			ctx_a[sub] = a;
			ctx_b[sub] = b;
		}
	}

#else // __CUDA_ARCH__ < 300

	const int thread = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread < threads)
	{
		const int batchsize = ITER >> (2 + bfactor);
		const int start = partidx * batchsize;
		const int end = start + batchsize;
		const int longptr = thread << LONG_SHL_IDX;
		uint32_t * long_state = &d_long_state[longptr];

		uint64_t * ctx_a = (uint64_t*)(&d_ctx_a[thread * 4]);
		uint64_t * ctx_b = (uint64_t*)(&d_ctx_b[thread * 4]);
		uint4 A = AS_UINT4(ctx_a);
		uint4 B = AS_UINT4(ctx_b);
		uint32_t* a = (uint32_t*)&A;
		uint32_t* b = (uint32_t*)&B;

		for (int i = start; i < end; i++) // end = 262144
		{
			uint32_t c[4];
			uint32_t j = (a[0] >> 2) & E2I_MASK2;
			cn_aes_single_round(sharedMemory, &long_state[j], c, a);
			XOR_BLOCKS_DST(c, b, &long_state[j]);
			MUL_SUM_XOR_DST(c, a, &long_state[(c[0] >> 2) & E2I_MASK2]);

			j = (a[0] >> 2) & E2I_MASK2;
			cn_aes_single_round(sharedMemory, &long_state[j], b, a);
			XOR_BLOCKS_DST(b, c, &long_state[j]);
			MUL_SUM_XOR_DST(b, a, &long_state[(b[0] >> 2) & E2I_MASK2]);
		}

		if (bfactor > 0) {
			AS_UINT4(ctx_a) = A;
			AS_UINT4(ctx_b) = B;
		}
	}
#endif // __CUDA_ARCH__ >= 300
}

__global__
void cryptolight_core_gpu_phase3(int threads, const uint32_t * long_state, uint32_t * ctx_state, uint32_t * ctx_key2)
{
	__shared__ uint32_t __align__(16) sharedMemory[1024];

	cn_aes_gpu_init(sharedMemory);

	const int thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;
	const int sub = (threadIdx.x & 7) << 2;

	if(thread < threads)
	{
		const int long_oft = (thread << LONG_SHL_IDX) + sub;
		const int oft = thread * 52 + sub + 16;
		uint32_t __align__(16) key[40];
		uint32_t __align__(16) text[4];

		#pragma unroll
		for (int i = 0; i < 40; i += 4)
			AS_UINT4(&key[i]) = AS_UINT4(ctx_key2 + thread * 40 + i);

		AS_UINT2(&text[0]) = AS_UINT2(&ctx_state[oft + 0]);
		AS_UINT2(&text[2]) = AS_UINT2(&ctx_state[oft + 2]);

		__syncthreads();
		for(int i = 0; i < LONG_LOOPS32; i += 32)
		{
			#pragma unroll
			for(int j = 0; j < 4; j++)
				text[j] ^= long_state[long_oft + i + j];

			cn_aes_pseudo_round_mut(sharedMemory, text, key);
		}

		AS_UINT2(&ctx_state[oft + 0]) = AS_UINT2(&text[0]);
		AS_UINT2(&ctx_state[oft + 2]) = AS_UINT2(&text[2]);
	}
}

extern int device_bfactor[MAX_GPUS];

__host__
void cryptolight_core_cpu_hash(int thr_id, int blocks, int threads, uint32_t *d_long_state, uint32_t *d_ctx_state,
	uint32_t *d_ctx_a, uint32_t *d_ctx_b, uint32_t *d_ctx_key1, uint32_t *d_ctx_key2)
{
	dim3 grid(blocks);
	dim3 block(threads);
	dim3 block4(threads << 2);
	dim3 block8(threads << 3);

	const int bfactor = device_bfactor[thr_id];
	const int bsleep = bfactor ? 100 : 0;

	int i, partcount = 1 << bfactor;
	int dev_id = device_map[thr_id];

	cryptolight_core_gpu_phase1 <<<grid, block8 >>>(blocks*threads, d_long_state, d_ctx_state, d_ctx_key1);
	exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
	if(partcount > 1) usleep(bsleep);

	for(i = 0; i < partcount; i++)
	{
		cryptolight_core_gpu_phase2 <<<grid, (device_sm[dev_id] >= 300 ? block4 : block)>>>(blocks*threads, bfactor, i, d_long_state, d_ctx_a, d_ctx_b);
		exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
		if(partcount > 1) usleep(bsleep);
	}

	cryptolight_core_gpu_phase3 <<<grid, block8 >>>(blocks*threads, d_long_state, d_ctx_state, d_ctx_key2);
	exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
}
