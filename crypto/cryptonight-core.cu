#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 300
#undef __shfl
#define __shfl(var, srcLane, width) __shfl_sync(0xFFFFFFFFu, var, srcLane, width)
#endif

#include "cryptonight.h"

#define LONG_SHL32 19 // 1<<19 (uint32_t* index)
#define LONG_SHL64 18 // 1<<18 (uint64_t* index)
#define LONG_SHL128 17 // 1<<17 (ulonglong2* index)
#define LONG_LOOPS32 0x80000U

#include "cn_aes.cuh"

__global__
void cryptonight_gpu_phase1(const uint32_t threads, ulonglong2 * __restrict__ d_long_state,
	uint32_t * __restrict__ ctx_state, uint32_t * __restrict__ ctx_key1)
{
	__shared__ uint32_t sharedMemory[1024];
	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;
	if(thread < threads)
	{
		const uint32_t sub = (threadIdx.x & 0x7U) << 2;
		ulonglong2 *longstate = &d_long_state[(thread << LONG_SHL128) + (sub >> 2)];
		uint32_t __align__(8) key[40];
		LDG_MEMCPY8(key, &ctx_key1[thread * 40U], 20);
		uint32_t __align__(16) text[4];
		LDG_MEMCPY8(text, &ctx_state[thread * 50U + sub + 16U], 2);
		ulonglong2 *txt = (ulonglong2*)text;

		for(int i = 0; i < LONG_LOOPS32; i += 32)
		{
			cn_aes_pseudo_round_mut(sharedMemory, text, key);
			longstate[i >> 2] = *txt;
		}
	}
}

// --------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ ulonglong2 cuda_mul128(const uint64_t multiplier, const uint64_t multiplicand)
{
	ulonglong2 product;
	product.x = __umul64hi(multiplier, multiplicand);
	product.y = multiplier * multiplicand;
	return product;
}

static __forceinline__ __device__ void operator += (ulonglong2 &a, const ulonglong2 b) {
	a.x += b.x; a.y += b.y;
}

static __forceinline__ __device__ ulonglong2 operator ^ (const ulonglong2 &a, const ulonglong2 &b) {
	return make_ulonglong2(a.x ^ b.x, a.y ^ b.y);
}

__device__ __forceinline__ void MUL_SUM_XOR_DST_0(const uint64_t m, uint4 &a, void* far_dst)
{
	ulonglong2 d = AS_UL2(far_dst);
	ulonglong2 p = cuda_mul128(m, d.x);
	p += AS_UL2(&a);
	AS_UL2(&a) = p ^ d;
	AS_UL2(far_dst) = p;
}

__global__
#if __CUDA_ARCH__ >= 500
//__launch_bounds__(128,12) /* force 40 regs to allow -l ...x32 */
#endif
void cryptonight_gpu_phase2(const uint32_t threads, const uint16_t bfactor, const uint32_t partidx,
	uint64_t * __restrict__ d_long_state, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b)
{
	__shared__ __align__(16) uint32_t sharedMemory[1024];
	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread < threads)
	{
		const uint32_t batchsize = ITER >> (2 + bfactor);
		const uint32_t start = partidx * batchsize;
		const uint32_t end = start + batchsize;

		void * ctx_a = (void*)(&d_ctx_a[thread << 2U]);
		void * ctx_b = (void*)(&d_ctx_b[thread << 2U]);
		uint4 A = AS_UINT4(ctx_a); // ld.global.u32.v4
		uint4 B = AS_UINT4(ctx_b);

		uint64_t * long_state = &d_long_state[thread << LONG_SHL64];
		for (int i = start; i < end; i++) // end = 262144
		{
			uint4 C;

			uint32_t j = (A.x & E2I_MASK) >> 3;
			cn_aes_single_round_b((uint8_t*)sharedMemory, &long_state[j], A, &C);
			AS_UINT4(&long_state[j]) = C ^ B; // st.global.u32.v4
			MUL_SUM_XOR_DST_0((AS_UL2(&C)).x, A, &long_state[(C.x & E2I_MASK) >> 3]);

			j = (A.x & E2I_MASK) >> 3;
			cn_aes_single_round_b((uint8_t*)sharedMemory, &long_state[j], A, &B);
			AS_UINT4(&long_state[j]) = C ^ B;
			MUL_SUM_XOR_DST_0((AS_UL2(&B)).x, A, &long_state[(B.x & E2I_MASK) >> 3]);
		}

		if (bfactor) {
			AS_UINT4(ctx_a) = A;
			AS_UINT4(ctx_b) = B;
		}
	}
}

// --------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ void store_variant1(uint64_t* long_state, uint4 Z)
{
	const uint32_t tmp = (Z.z >> 24); // __byte_perm(src, 0, 0x7773);
	const uint32_t index = (((tmp >> 3) & 6u) | (tmp & 1u)) << 1;
	Z.z = (Z.z & 0x00ffffffu) | ((tmp ^ ((0x75310u >> index) & 0x30u)) << 24);
	AS_UINT4(long_state) = Z;
}

__device__ __forceinline__ void store_variant2(uint64_t* long_state, uint4 Z)
{
	const uint32_t tmp = (Z.z >> 24); // __byte_perm(src, 0, 0x7773);
	const uint32_t index = (((tmp >> 4) & 6u) | (tmp & 1u)) << 1;
	Z.z = (Z.z & 0x00ffffffu) | ((tmp ^ ((0x75312u >> index) & 0x30u)) << 24);
	AS_UINT4(long_state) = Z;
}

__device__ __forceinline__ void MUL_SUM_XOR_DST_1(const uint64_t m, uint4 &a, void* far_dst, uint64_t tweak)
{
	ulonglong2 d = AS_UL2(far_dst);
	ulonglong2 p = cuda_mul128(m, d.x);
	p += AS_UL2(&a);
	AS_UL2(&a) = p ^ d;
	p.y = p.y ^ tweak;
	AS_UL2(far_dst) = p;
}

__device__ __forceinline__ void AES_SHUFFLE_ADD(const uint4 a, const uint4 b, const ulonglong2 b1,
	uint4 __restrict__ &c, const uint32_t * __restrict__ sm1, ulonglong2 * __restrict__ chunk)
{
	uchar4 *in = (uchar4 *)&chunk[0];

	cn_aes_single_round_c(sm1, a, &c, in);

	const ulonglong2 chunk1_old = chunk[1];

	chunk[1].x = chunk[3].x + (AS_UL2(&b1)).x;
	chunk[1].y = chunk[3].y + (AS_UL2(&b1)).y;
	chunk[3].x = chunk[2].x + (AS_UL2(&a)).x;
	chunk[3].y = chunk[2].y + (AS_UL2(&a)).y;
	chunk[2].x = chunk1_old.x + (AS_UL2(&b)).x;
	chunk[2].y = chunk1_old.y + (AS_UL2(&b)).y;

	chunk[0] = AS_UL2(&c) ^ AS_UL2(&b);
}
/* SChernykh@github */
__device__ __forceinline__ uint32_t get_reciprocal(const uint32_t a)
{
	const float a_hi = __uint_as_float((a >> 8) + ((126U + 31U) << 23));
	const float a_lo = __uint2float_rn(a & 0xFF);

	float r;
	asm("rcp.f32.rn %0, %1;" : "=f"(r) : "f"(a_hi));

	const float r_scaled = __uint_as_float(__float_as_uint(r) + (64U << 23));

	const float h = __fmaf_rn(a_lo, r, __fmaf_rn(a_hi, r, -1.0f));
	return (__float_as_uint(r) << 9) - __float2int_rn(h * r_scaled);
}
/* SChernykh@github */
__device__ __forceinline__ uint64_t fast_div_v2(const uint64_t a, const uint32_t b)
{
	const uint32_t r = get_reciprocal(b);
	const uint64_t k = __umulhi(((uint32_t*)&a)[0], r) + ((uint64_t)(r) * ((uint32_t*)&a)[1]) + a;

	uint32_t q[2];
	q[0] = ((uint32_t*)&k)[1];

	int64_t tmp = a - (uint64_t)(q[0]) * b;
	((int32_t*)(&tmp))[1] -= (k < a) ? b : 0;

	const int8_t overshoot = ((int32_t*)(&tmp))[1] >> 31;
	const int8_t undershoot = (tmp - b) >> 63;

	q[0] += 1 + undershoot + overshoot;
	q[1] = ((uint32_t*)(&tmp))[0] + (b & overshoot) + ((b & undershoot) - b);

	return *((uint64_t*)(q));
}
/* SChernykh@github */
__device__ __forceinline__ uint32_t fast_sqrt_v2(const uint64_t n1)
{
	float x = __uint_as_float((((uint32_t*)&n1)[1] >> 9) + ((64U + 127U) << 23));
	float x1;
	asm("rsqrt.approx.f32 %0, %1;" : "=f"(x1) : "f"(x));
	asm("rcp.approx.f32 %0, %1;" : "=f"(x) : "f"(x1));

	// The following line does x1 *= 4294967296.0f;
	x1 = __uint_as_float(__float_as_uint(x1) + (32U << 23));

	const uint32_t x0 = __float_as_uint(x) - (158U << 23);
	const int64_t delta0 = n1 - (((int64_t)(x0) * x0) << 18);
	const float delta = __int2float_rn(((int32_t*)&delta0)[1]) * x1;

	uint32_t result = (x0 << 10) + __float2int_rn(delta);
	const uint32_t s = result >> 1;
	const uint32_t b = result & 1;

	const uint64_t x2 = (uint64_t)(s) * (s + b) + ((uint64_t)(result) << 32) - n1;
	result -= (((int64_t)(x2 + b) >> 63) + 1) + ((int64_t)(x2 + 0x100000000UL + s) >> 63);
	return result;
}

__device__ __forceinline__ void DIVSQ_MUL_SHF_SUM_XOR_DST(const ulonglong2 c,
	ulonglong2 __restrict__ &ds, const ulonglong2 b1, uint64_t __restrict__ &bs,
	uint4 __restrict__ &a, const uint4 bb, ulonglong2 * __restrict__ chunk)
{
	chunk[0].x ^= ds.x ^ (ds.y << 32);

	const uint64_t dividend = c.y;
	const uint32_t divisor = (c.x + (uint32_t)(ds.y << 1)) | 0x80000001UL;

	ds.x = fast_div_v2(dividend, divisor);

	ulonglong2 p = cuda_mul128(c.x, chunk[0].x);

	const uint64_t sqrt_input = c.x + ds.x;
	ds.y = fast_sqrt_v2(sqrt_input);

	const ulonglong2 chunk1_old = chunk[1] ^ p;
	const ulonglong2 chunk0_old = chunk[0];
	p = p ^ chunk[2];
	p += AS_UL2(&a);
	chunk[0] = p;

	chunk[1].x = chunk[3].x + b1.x;
	chunk[1].y = chunk[3].y + b1.y;

	chunk[3].x = chunk[2].x + (AS_UL2(&a)).x;
	chunk[3].y = chunk[2].y + (AS_UL2(&a)).y;

	chunk[2].x = chunk1_old.x + (AS_UL2(&bb)).x;
	chunk[2].y = chunk1_old.y + (AS_UL2(&bb)).y;

	bs = chunk0_old.x;

	(AS_UL2(&a)).x = chunk0_old.x ^ p.x;
	(AS_UL2(&a)).y = chunk0_old.y ^ p.y;
}

#if __CUDA_ARCH__ >= 500
__launch_bounds__(64) /* avoid register spill, launch limited to -l ..x64 */
#endif
__global__
void monero_gpu_phase2(const uint32_t threads, const uint16_t bfactor, const uint32_t partidx,
	ulonglong2 * __restrict__ d_long_state, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b,
	uint64_t * __restrict__ d_tweak)
{
	__shared__ __align__(16) uint32_t sm1[256]; /* 1st 256 table */
//	__shared__ __align__(16) uint32_t sm2[256];
//	__shared__ __align__(16) uint32_t sm3[256];
//	__shared__ __align__(16) uint32_t sm4[256];
	cn_aes_gpu_init4(sm1); /* One 256 table */
//	cn_aes_gpu_init4(sm1, sm2, sm3, sm4); /* Four 256 table */
	__syncthreads();

	extern __shared__ ulonglong2 sm_chunk[];
	ulonglong2 *chunk = &sm_chunk[threadIdx.x << 2];

	for (int thread = blockIdx.x * blockDim.x + threadIdx.x; thread < threads; thread += blockDim.x * gridDim.x)
	{
		const uint32_t batchsize = ITER >> (2 + bfactor);
		const uint32_t start = partidx * batchsize;
		const uint32_t end = start + batchsize;
		const int sub = threadIdx.x & 3;

		ulonglong2 div_sq = __ldg((ulonglong2 *)&d_tweak[thread << 2]);
		ulonglong2 B1 = __ldg((ulonglong2 *)&d_tweak[(thread << 2) + 2]);

		uint4 A = __ldg((uint4 *)&d_ctx_a[thread << 2]);
		uint4 B = __ldg((uint4 *)&d_ctx_b[thread << 2]);

		for (int i = start; i < end; i++) // end = 262144
		{
			uint4 C;
			uint64_t bs;

			int j = (A.x & E2I_MASK) >> 4;

			#pragma unroll
			for (int k = 0; k < 4; k++) //Copy from global mem using 4 threads
			{
				int master_tid = __shfl(thread, k, 4);
				ulonglong2 * master_ls = &d_long_state[master_tid << LONG_SHL128];
				int master_j = __shfl(j, k, 4);

				chunk[sub + ((k - sub) * 4)] = master_ls[master_j ^ sub];
			}
			__syncthreads();

			AES_SHUFFLE_ADD(A, B, B1, C, sm1, chunk);

			#pragma unroll
			for (int k = 0; k < 4; k++) //Copy to global mem using 4 threads
			{
				int master_tid = __shfl(thread, k, 4);
				ulonglong2 * master_ls = &d_long_state[master_tid << LONG_SHL128];
				int master_j = __shfl(j, k, 4);

				master_ls[master_j ^ sub] = chunk[sub + ((k - sub) * 4)];
			}
			__syncthreads();

			j = (C.x & E2I_MASK) >> 4;

			#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				int master_tid = __shfl(thread, k, 4);
				ulonglong2 * master_ls = &d_long_state[master_tid << LONG_SHL128];
				int master_j = __shfl(j, k, 4);

				chunk[sub + ((k - sub) * 4)] = master_ls[master_j ^ sub];
			}
			__syncthreads();

			DIVSQ_MUL_SHF_SUM_XOR_DST(AS_UL2(&C), div_sq, B1, bs, A, B, chunk);

			#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				int master_tid = __shfl(thread, k, 4);
				ulonglong2 * master_ls = &d_long_state[master_tid << LONG_SHL128];
				int master_j = __shfl(j, k, 4);

				master_ls[master_j ^ sub] = chunk[sub + ((k - sub) * 4)];
			}
			__syncthreads();

			AS_UINT4(&B1) = B;

			j = (A.x & E2I_MASK) >> 4;

			#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				int master_tid = __shfl(thread, k, 4);
				ulonglong2 * master_ls = &d_long_state[master_tid << LONG_SHL128];
				int master_j = __shfl(j, k, 4);

				chunk[sub + ((k - sub) * 4)] = master_ls[master_j ^ sub];
			}
			__syncthreads();

			AES_SHUFFLE_ADD(A, C, B1, B, sm1, chunk);

			#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				int master_tid = __shfl(thread, k, 4);
				ulonglong2 * master_ls = &d_long_state[master_tid << LONG_SHL128];
				int master_j = __shfl(j, k, 4);

				master_ls[master_j ^ sub] = chunk[sub + ((k - sub) * 4)];
			}
			__syncthreads();

			j = (B.x & E2I_MASK) >> 4;

			#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				int master_tid = __shfl(thread, k, 4);
				ulonglong2 * master_ls = &d_long_state[master_tid << LONG_SHL128];
				int master_j = __shfl(j, k, 4);

				chunk[sub + ((k - sub) * 4)] = master_ls[master_j ^ sub];
			}
			__syncthreads();

			DIVSQ_MUL_SHF_SUM_XOR_DST(AS_UL2(&B), div_sq, B1, bs, A, C, chunk);

			#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				int master_tid = __shfl(thread, k, 4);
				ulonglong2 * master_ls = &d_long_state[master_tid << LONG_SHL128];
				int master_j = __shfl(j, k, 4);

				master_ls[master_j ^ sub] = chunk[sub + ((k - sub) * 4)];
			}
			__syncthreads();

			AS_UINT4(&B1) = C;
		}
		if (bfactor) {
			ulonglong2 *dsrb = (ulonglong2 *)&d_tweak[thread << 2];
			void * ctx_a = (void*)(&d_ctx_a[thread << 2]);
			void * ctx_b = (void*)(&d_ctx_b[thread << 2]);
			AS_UINT4(ctx_a) = A;
			AS_UINT4(ctx_b) = B;
			dsrb[0] = div_sq;
			dsrb[1] = B1;
		}
	}
}

// --------------------------------------------------------------------------------------------------------------

__global__
void stellite_gpu_phase2(const uint32_t threads, const uint16_t bfactor, const uint32_t partidx,
	uint64_t * __restrict__ d_long_state, uint32_t * __restrict__ d_ctx_a, uint32_t * __restrict__ d_ctx_b,
	uint64_t * __restrict__ d_tweak)
{
	__shared__ __align__(16) uint32_t sharedMemory[1024];
	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	if (thread < threads)
	{
		const uint32_t batchsize = ITER >> (2 + bfactor);
		const uint32_t start = partidx * batchsize;
		const uint32_t end = start + batchsize;
		uint64_t tweak = d_tweak[thread];

		void * ctx_a = (void*)(&d_ctx_a[thread << 2]);
		void * ctx_b = (void*)(&d_ctx_b[thread << 2]);
		uint4 A = AS_UINT4(ctx_a); // ld.global.u32.v4
		uint4 B = AS_UINT4(ctx_b);

		uint64_t * long_state = &d_long_state[thread << LONG_SHL64];
		for (int i = start; i < end; i++) // end = 262144
		{
			uint4 C;
			uint32_t j = (A.x & E2I_MASK) >> 3;
			cn_aes_single_round_b((uint8_t*)sharedMemory, &long_state[j], A, &C);
			store_variant2(&long_state[j], C ^ B); // st.global
			MUL_SUM_XOR_DST_1((AS_UL2(&C)).x, A, &long_state[(C.x & E2I_MASK) >> 3], tweak);

			j = (A.x & E2I_MASK) >> 3;
			cn_aes_single_round_b((uint8_t*)sharedMemory, &long_state[j], A, &B);
			store_variant2(&long_state[j], C ^ B);
			MUL_SUM_XOR_DST_1((AS_UL2(&B)).x, A, &long_state[(B.x & E2I_MASK) >> 3], tweak);
		}
		if (bfactor) {
			AS_UINT4(ctx_a) = A;
			AS_UINT4(ctx_b) = B;
		}
	}
}

// --------------------------------------------------------------------------------------------------------------

__global__
void cryptonight_gpu_phase3(const uint32_t threads, const ulonglong2 * __restrict__ d_long_state,
	uint32_t * __restrict__ d_ctx_state, const uint32_t * __restrict__ d_ctx_key2)
{
	__shared__ uint32_t sharedMemory[1024];
	cn_aes_gpu_init(sharedMemory);
	__syncthreads();

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) >> 3;

	if(thread < threads)
	{
		const int sub = (threadIdx.x & 7) << 2;
		const ulonglong2 *longstate = &d_long_state[(thread << LONG_SHL128) + (sub >> 2)];
		uint32_t key[40], __align__(16) text[4];
		LDG_MEMCPY8(key, d_ctx_key2 + thread * 40, 20);
		LDG_MEMCPY8(text, d_ctx_state + thread * 50 + sub + 16, 2);
		ulonglong2 *txt = (ulonglong2*)text;

		for(int i = 0; i < LONG_LOOPS32; i += 32)
		{
			*txt = *txt ^ __ldg(&longstate[i >> 2]);

			cn_aes_pseudo_round_mut(sharedMemory, text, key);
		}

		MEMCPY8(d_ctx_state + thread * 50 + sub + 16, text, 2);
	}
}

// --------------------------------------------------------------------------------------------------------------

extern int device_bfactor[MAX_GPUS];

__host__
void cryptonight_core_cuda(int thr_id, uint32_t blocks, uint32_t threads, ulonglong2 *d_long_state, uint32_t *d_ctx_state,
	uint32_t *d_ctx_a, uint32_t *d_ctx_b, uint32_t *d_ctx_key1, uint32_t *d_ctx_key2, int variant, uint64_t *d_ctx_tweak)
{
	dim3 grid(blocks);
	dim3 block(threads);
	dim3 block4(threads << 2);
	dim3 block8(threads << 3);

	const uint16_t bfactor = (uint16_t) device_bfactor[thr_id];
	const uint32_t partcount = 1U << bfactor;
	const uint32_t throughput = (uint32_t) (blocks*threads);

	const int bsleep = bfactor ? 100 : 0;
	const int dev_id = device_map[thr_id];

	cryptonight_gpu_phase1 <<<grid, block8>>> (throughput, d_long_state, d_ctx_state, d_ctx_key1);
	exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
	if(partcount > 1) usleep(bsleep);

	for (uint32_t i = 0; i < partcount; i++)
	{
		dim3 b = device_sm[dev_id] >= 300 ? block4 : block;
		dim3 b1 = block; /* monero */
		b1.x -= b1.x & 3; /* round down to nearest multiple of 4 for correct __shfl operation */
		const uint32_t sm_sz = sizeof(ulonglong2) * b1.x * 4; /* shared mem for global mem copy */

		if (variant == 0)
			cryptonight_gpu_phase2 <<<grid, b>>> (throughput, bfactor, i, (uint64_t*) d_long_state, d_ctx_a, d_ctx_b);
		else if (variant == 1 || cryptonight_fork == 7 || cryptonight_fork == 8) {
			monero_gpu_phase2 <<<grid, b1, sm_sz>>> (throughput, bfactor, i, d_long_state, d_ctx_a, d_ctx_b, d_ctx_tweak);
		}
		else if (variant == 2 && cryptonight_fork == 3)
			stellite_gpu_phase2 <<<grid, b>>> (throughput, bfactor, i, (uint64_t*) d_long_state, d_ctx_a, d_ctx_b, d_ctx_tweak);
		exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);

		if(partcount > 1) usleep(bsleep);
	}
	//cudaDeviceSynchronize();
	//exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
	cryptonight_gpu_phase3 <<<grid, block8>>> (throughput, d_long_state, d_ctx_state, d_ctx_key2);
	exit_if_cudaerror(thr_id, __FUNCTION__, __LINE__);
}
