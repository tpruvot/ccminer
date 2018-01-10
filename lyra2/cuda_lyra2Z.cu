/**
 * Lyra2 (v1) cuda implementation based on djm34 work
 * tpruvot@github 2015, Nanashi 08/2016 (from 1.8-r2)
 * Lyra2Z implentation for Zcoin based on all the previous
 * djm34 2017
 **/

#include <stdio.h>
#include <memory.h>

#define TPB52 32
#define TPB30 160
#define TPB20 160

#include "cuda_lyra2Z_sm5.cuh"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
__device__ uint32_t __shfl(uint32_t a, uint32_t b, uint32_t c);
#define atomicMin()
#define __CUDA_ARCH__ 520
#endif

static uint32_t *h_GNonces[16]; // this need to get fixed as the rest of that routine
static uint32_t *d_GNonces[16];

#define reduceDuplexRow(rowIn, rowInOut, rowOut) { \
	for (int i = 0; i < 8; i++) { \
		for (int j = 0; j < 12; j++) \
			state[j] ^= Matrix[12 * i + j][rowIn] + Matrix[12 * i + j][rowInOut]; \
		round_lyra_sm2(state); \
		for (int j = 0; j < 12; j++) \
			Matrix[j + 12 * i][rowOut] ^= state[j]; \
		Matrix[0 + 12 * i][rowInOut] ^= state[11]; \
		Matrix[1 + 12 * i][rowInOut] ^= state[0]; \
		Matrix[2 + 12 * i][rowInOut] ^= state[1]; \
		Matrix[3 + 12 * i][rowInOut] ^= state[2]; \
		Matrix[4 + 12 * i][rowInOut] ^= state[3]; \
		Matrix[5 + 12 * i][rowInOut] ^= state[4]; \
		Matrix[6 + 12 * i][rowInOut] ^= state[5]; \
		Matrix[7 + 12 * i][rowInOut] ^= state[6]; \
		Matrix[8 + 12 * i][rowInOut] ^= state[7]; \
		Matrix[9 + 12 * i][rowInOut] ^= state[8]; \
		Matrix[10+ 12 * i][rowInOut] ^= state[9]; \
		Matrix[11+ 12 * i][rowInOut] ^= state[10]; \
	} \
  }

#define absorbblock(in)  { \
	state[0] ^= Matrix[0][in]; \
	state[1] ^= Matrix[1][in]; \
	state[2] ^= Matrix[2][in]; \
	state[3] ^= Matrix[3][in]; \
	state[4] ^= Matrix[4][in]; \
	state[5] ^= Matrix[5][in]; \
	state[6] ^= Matrix[6][in]; \
	state[7] ^= Matrix[7][in]; \
	state[8] ^= Matrix[8][in]; \
	state[9] ^= Matrix[9][in]; \
	state[10] ^= Matrix[10][in]; \
	state[11] ^= Matrix[11][in]; \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
	round_lyra_sm2(state); \
  }

__device__ __forceinline__
static void round_lyra_sm2(uint2 *s)
{
	Gfunc(s[0], s[4], s[8], s[12]);
	Gfunc(s[1], s[5], s[9], s[13]);
	Gfunc(s[2], s[6], s[10], s[14]);
	Gfunc(s[3], s[7], s[11], s[15]);
	Gfunc(s[0], s[5], s[10], s[15]);
	Gfunc(s[1], s[6], s[11], s[12]);
	Gfunc(s[2], s[7], s[8], s[13]);
	Gfunc(s[3], s[4], s[9], s[14]);
}

__device__ __forceinline__
void reduceDuplexRowSetup(const int rowIn, const int rowInOut, const int rowOut, uint2 state[16], uint2 Matrix[96][8])
{
#if __CUDA_ARCH__ > 500
#pragma unroll
#endif
	for (int i = 0; i < 8; i++)
	{
		#pragma unroll
		for (int j = 0; j < 12; j++)
			state[j] ^= Matrix[12 * i + j][rowIn] + Matrix[12 * i + j][rowInOut];

		round_lyra_sm2(state);

		#pragma unroll
		for (int j = 0; j < 12; j++)
			Matrix[j + 84 - 12 * i][rowOut] = Matrix[12 * i + j][rowIn] ^ state[j];

		Matrix[0 +  12 * i][rowInOut] ^= state[11];
		Matrix[1 +  12 * i][rowInOut] ^= state[0];
		Matrix[2 +  12 * i][rowInOut] ^= state[1];
		Matrix[3 +  12 * i][rowInOut] ^= state[2];
		Matrix[4 +  12 * i][rowInOut] ^= state[3];
		Matrix[5 +  12 * i][rowInOut] ^= state[4];
		Matrix[6 +  12 * i][rowInOut] ^= state[5];
		Matrix[7 +  12 * i][rowInOut] ^= state[6];
		Matrix[8 +  12 * i][rowInOut] ^= state[7];
		Matrix[9 +  12 * i][rowInOut] ^= state[8];
		Matrix[10 + 12 * i][rowInOut] ^= state[9];
		Matrix[11 + 12 * i][rowInOut] ^= state[10];
	}
}

#if __CUDA_ARCH__ < 350

__constant__ static uint2 blake2b_IV_sm2[8] = {
	{ 0xf3bcc908, 0x6a09e667 }, { 0x84caa73b, 0xbb67ae85 },
	{ 0xfe94f82b, 0x3c6ef372 }, { 0x5f1d36f1, 0xa54ff53a },
	{ 0xade682d1, 0x510e527f }, { 0x2b3e6c1f, 0x9b05688c },
	{ 0xfb41bd6b, 0x1f83d9ab }, { 0x137e2179, 0x5be0cd19 }
};

__global__ __launch_bounds__(TPB30, 1)
void lyra2Z_gpu_hash_32_sm2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *resNonces)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint2 Mask[8] = {
		{ 0x00000020, 0x00000000 },{ 0x00000020, 0x00000000 },
		{ 0x00000020, 0x00000000 },{ 0x00000008, 0x00000000 },
		{ 0x00000008, 0x00000000 },{ 0x00000008, 0x00000000 },
		{ 0x00000080, 0x00000000 },{ 0x00000000, 0x01000000 }
	};
	if (thread < threads)
	{
		uint2 state[16];

		#pragma unroll
		for (int i = 0; i<4; i++) {
			LOHI(state[i].x, state[i].y, g_hash[threads*i + thread]);
		} //password

		#pragma unroll
		for (int i = 0; i<4; i++) {
			state[i + 4] = state[i];
		} //salt

		#pragma unroll
		for (int i = 0; i<8; i++) {
			state[i + 8] = blake2b_IV_sm2[i];
		}

		// blake2blyra x2
		//#pragma unroll 24
		for (int i = 0; i<12; i++) {
			round_lyra_sm2(state);
		}

		for (int i = 0; i<8; i++)
			state[i] ^= Mask[i];

		for (int i = 0; i<12; i++) {
			round_lyra_sm2(state);
		}

		uint2 Matrix[96][8]; // not cool

		// reducedSqueezeRow0
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
		{
			#pragma unroll 12
			for (int j = 0; j<12; j++) {
				Matrix[j + 84 - 12 * i][0] = state[j];
			}
			round_lyra_sm2(state);
		}

		// reducedSqueezeRow1
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
		{
			#pragma unroll 12
			for (int j = 0; j<12; j++) {
				state[j] ^= Matrix[j + 12 * i][0];
			}
			round_lyra_sm2(state);
			#pragma unroll 12
			for (int j = 0; j<12; j++) {
				Matrix[j + 84 - 12 * i][1] = Matrix[j + 12 * i][0] ^ state[j];
			}
		}

		reduceDuplexRowSetup(1, 0, 2, state, Matrix);
		reduceDuplexRowSetup(2, 1, 3, state, Matrix);
		reduceDuplexRowSetup(3, 0, 4, state, Matrix);
		reduceDuplexRowSetup(4, 3, 5, state, Matrix);
		reduceDuplexRowSetup(5, 2, 6, state, Matrix);
		reduceDuplexRowSetup(6, 1, 7, state, Matrix);

		uint32_t rowa;
		uint32_t prev = 7;
		uint32_t iterator = 0;
		for (uint32_t i = 0; i<8; i++) {
			rowa = state[0].x & 7;
			reduceDuplexRow(prev, rowa, iterator);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}
		for (uint32_t i = 0; i<8; i++) {
			rowa = state[0].x & 7;
			reduceDuplexRow(prev, rowa, iterator);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = state[0].x & 7;
			reduceDuplexRow(prev, rowa, iterator);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}
		for (uint32_t i = 0; i<8; i++) {
			rowa = state[0].x & 7;
			reduceDuplexRow(prev, rowa, iterator);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = state[0].x & 7;
			reduceDuplexRow(prev, rowa, iterator);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}
		for (uint32_t i = 0; i<8; i++) {
			rowa = state[0].x & 7;
			reduceDuplexRow(prev, rowa, iterator);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = state[0].x & 7;
			reduceDuplexRow(prev, rowa, iterator);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}
		for (uint32_t i = 0; i<8; i++) {
			rowa = state[0].x & 7;
			reduceDuplexRow(prev, rowa, iterator);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}

		absorbblock(rowa);
		uint32_t nonce = startNounce + thread;
		if (((uint64_t*)state)[3] <= ((uint64_t*)pTarget)[3]) {
			atomicMin(&resNonces[1], resNonces[0]);
			atomicMin(&resNonces[0], nonce);
		}
	} //thread
}
#else
__global__ void lyra2Z_gpu_hash_32_sm2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *resNonces) {}
#endif

#if __CUDA_ARCH__ > 500

#include "cuda_lyra2_vectors.h"
//#include "cuda_vector_uint2x4.h"

#define Nrow 8
#define Ncol 8
#define memshift 3

#define BUF_COUNT 0

__device__ uint2 *DMatrix;

__device__ __forceinline__
void LD4S(uint2 res[3], const int row, const int col, const int thread, const int threads)
{
#if BUF_COUNT != 8
	extern __shared__ uint2 shared_mem[];
	const int s0 = (Ncol * (row - BUF_COUNT) + col) * memshift;
#endif
#if BUF_COUNT != 0
	const int d0 = (memshift *(Ncol * row + col) * threads + thread)*blockDim.x + threadIdx.x;
#endif

#if BUF_COUNT == 8
	#pragma unroll
	for (int j = 0; j < 3; j++)
		res[j] = *(DMatrix + d0 + j * threads * blockDim.x);
#elif BUF_COUNT == 0
	#pragma unroll
	for (int j = 0; j < 3; j++)
		res[j] = shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
#else
	if (row < BUF_COUNT) {
		#pragma unroll
		for (int j = 0; j < 3; j++)
			res[j] = *(DMatrix + d0 + j * threads * blockDim.x);
	} else {
		#pragma unroll
		for (int j = 0; j < 3; j++)
			res[j] = shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
	}
#endif
}

__device__ __forceinline__
void ST4S(const int row, const int col, const uint2 data[3], const int thread, const int threads)
{
#if BUF_COUNT != 8
	extern __shared__ uint2 shared_mem[];
	const int s0 = (Ncol * (row - BUF_COUNT) + col) * memshift;
#endif
#if BUF_COUNT != 0
	const int d0 = (memshift *(Ncol * row + col) * threads + thread)*blockDim.x + threadIdx.x;
#endif

#if BUF_COUNT == 8
	#pragma unroll
	for (int j = 0; j < 3; j++)
		*(DMatrix + d0 + j * threads * blockDim.x) = data[j];

#elif BUF_COUNT == 0
	#pragma unroll
	for (int j = 0; j < 3; j++)
		shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data[j];
#else
	if (row < BUF_COUNT) {
	#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + d0 + j * threads * blockDim.x) = data[j];
	} else {
	#pragma unroll
		for (int j = 0; j < 3; j++)
			shared_mem[((s0 + j) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data[j];
	}
#endif
}

#if __CUDA_ARCH__ >= 300
__device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
	return __shfl(a, b, c);
}

__device__ __forceinline__ uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c)
{
	return make_uint2(__shfl(a.x, b, c), __shfl(a.y, b, c));
}

__device__ __forceinline__
void WarpShuffle3(uint2 &a1, uint2 &a2, uint2 &a3, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
	a1 = WarpShuffle(a1, b1, c);
	a2 = WarpShuffle(a2, b2, c);
	a3 = WarpShuffle(a3, b3, c);
}

#else
__device__ __forceinline__ uint32_t WarpShuffle(uint32_t a, uint32_t b, uint32_t c)
{
	extern __shared__ uint2 shared_mem[];

	const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;
	uint32_t *_ptr = (uint32_t*)shared_mem;

	__threadfence_block();
	uint32_t buf = _ptr[thread];

	_ptr[thread] = a;
	__threadfence_block();
	uint32_t result = _ptr[(thread&~(c - 1)) + (b&(c - 1))];

	__threadfence_block();
	_ptr[thread] = buf;

	__threadfence_block();
	return result;
}

__device__ __forceinline__ uint2 WarpShuffle(uint2 a, uint32_t b, uint32_t c)
{
	extern __shared__ uint2 shared_mem[];

	const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;

	__threadfence_block();
	uint2 buf = shared_mem[thread];

	shared_mem[thread] = a;
	__threadfence_block();
	uint2 result = shared_mem[(thread&~(c - 1)) + (b&(c - 1))];

	__threadfence_block();
	shared_mem[thread] = buf;

	__threadfence_block();
	return result;
}

__device__ __forceinline__ void WarpShuffle3(uint2 &a1, uint2 &a2, uint2 &a3, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
{
	extern __shared__ uint2 shared_mem[];

	const uint32_t thread = blockDim.x * threadIdx.y + threadIdx.x;

	__threadfence_block();
	uint2 buf = shared_mem[thread];

	shared_mem[thread] = a1;
	__threadfence_block();
	a1 = shared_mem[(thread&~(c - 1)) + (b1&(c - 1))];
	__threadfence_block();
	shared_mem[thread] = a2;
	__threadfence_block();
	a2 = shared_mem[(thread&~(c - 1)) + (b2&(c - 1))];
	__threadfence_block();
	shared_mem[thread] = a3;
	__threadfence_block();
	a3 = shared_mem[(thread&~(c - 1)) + (b3&(c - 1))];

	__threadfence_block();
	shared_mem[thread] = buf;
	__threadfence_block();
}
#endif

__device__ __forceinline__ void round_lyra(uint2 s[4])
{
	Gfunc(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4);
	Gfunc(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4);
}

static __device__ __forceinline__
void round_lyra(uint2x4* s)
{
	Gfunc(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc(s[0].w, s[1].w, s[2].w, s[3].w);
	Gfunc(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc(s[0].w, s[1].x, s[2].y, s[3].z);
}

static __device__ __forceinline__
void reduceDuplex(uint2 state[4], uint32_t thread, const uint32_t threads)
{
	uint2 state1[3];

#if __CUDA_ARCH__ > 500
#pragma unroll
#endif
	for (int i = 0; i < Nrow; i++)
	{
		ST4S(0, Ncol - i - 1, state, thread, threads);

		round_lyra(state);
	}

	#pragma unroll 4
	for (int i = 0; i < Nrow; i++)
	{
		LD4S(state1, 0, i, thread, threads);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra(state);

		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];
		ST4S(1, Ncol - i - 1, state1, thread, threads);
	}
}

static __device__ __forceinline__
void reduceDuplexRowSetup(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3];

	#pragma unroll 1
	for (int i = 0; i < Nrow; i++)
	{
		LD4S(state1, rowIn, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4S(rowOut, Ncol - i - 1, state1, thread, threads);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		} else {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		ST4S(rowInOut, i, state2, thread, threads);
	}
}

static __device__ __forceinline__
void reduceDuplexRowt(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	for (int i = 0; i < Nrow; i++)
	{
		uint2 state1[3], state2[3];

		LD4S(state1, rowIn, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		ST4S(rowInOut, i, state2, thread, threads);

		LD4S(state1, rowOut, i, thread, threads);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] ^= state[j];

		ST4S(rowOut, i, state1, thread, threads);
	}
}

#if 0
static __device__ __forceinline__
void reduceDuplexRowt_8(const int rowInOut, uint2* state, const uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3], last[3];

	LD4S(state1, 2, 0, thread, threads);
	LD4S(last, rowInOut, 0, thread, threads);

	#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= state1[j] + last[j];

	round_lyra(state);

	//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
	uint2 Data0 = state[0];
	uint2 Data1 = state[1];
	uint2 Data2 = state[2];
	WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

	if (threadIdx.x == 0)
	{
		last[0] ^= Data2;
		last[1] ^= Data0;
		last[2] ^= Data1;
	} else {
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == 5)
	{
		#pragma unroll
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	for (int i = 1; i < Nrow; i++)
	{
		LD4S(state1, 2, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);
	}

	#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}
#endif

static __device__ __forceinline__
void reduceDuplexRowt_8_v2(const int rowIn, const int rowOut, const int rowInOut, uint2* state, const uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3], last[3];

	LD4S(state1, rowIn, 0, thread, threads);
	LD4S(last, rowInOut, 0, thread, threads);

	#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= state1[j] + last[j];

	round_lyra(state);

	//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
	uint2 Data0 = state[0];
	uint2 Data1 = state[1];
	uint2 Data2 = state[2];
	WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

	if (threadIdx.x == 0)
	{
		last[0] ^= Data2;
		last[1] ^= Data0;
		last[2] ^= Data1;
	}
	else {
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == rowOut) {
		#pragma unroll
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	for (int i = 1; i < Nrow; i++)
	{
		LD4S(state1, rowIn, i, thread, threads);
		LD4S(state2, rowInOut, i, thread, threads);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);
	}

	#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}

__global__
__launch_bounds__(64, 1)
void lyra2Z_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	const uint2x4 Mask[2] = {
		0x00000020UL, 0x00000000UL, 0x00000020UL, 0x00000000UL,
		0x00000020UL, 0x00000000UL, 0x00000008UL, 0x00000000UL,
		0x00000008UL, 0x00000000UL, 0x00000008UL, 0x00000000UL,
		0x00000080UL, 0x00000000UL, 0x00000000UL, 0x01000000UL
	};
	const uint2x4 blake2b_IV[2] = {
		0xf3bcc908lu, 0x6a09e667lu,
		0x84caa73blu, 0xbb67ae85lu,
		0xfe94f82blu, 0x3c6ef372lu,
		0x5f1d36f1lu, 0xa54ff53alu,
		0xade682d1lu, 0x510e527flu,
		0x2b3e6c1flu, 0x9b05688clu,
		0xfb41bd6blu, 0x1f83d9ablu,
		0x137e2179lu, 0x5be0cd19lu
	};
	if (thread < threads)
	{
		uint2x4 state[4];

		state[0].x = state[1].x = __ldg(&g_hash[thread + threads * 0]);
		state[0].y = state[1].y = __ldg(&g_hash[thread + threads * 1]);
		state[0].z = state[1].z = __ldg(&g_hash[thread + threads * 2]);
		state[0].w = state[1].w = __ldg(&g_hash[thread + threads * 3]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i<12; i++)
			round_lyra(state);

		state[0] ^= Mask[0];
		state[1] ^= Mask[1];

		for (int i = 0; i<12; i++)
			round_lyra(state); //because 12 is not enough

		((uint2x4*)DMatrix)[threads * 0 + thread] = state[0];
		((uint2x4*)DMatrix)[threads * 1 + thread] = state[1];
		((uint2x4*)DMatrix)[threads * 2 + thread] = state[2];
		((uint2x4*)DMatrix)[threads * 3 + thread] = state[3];
	}
}

__global__
__launch_bounds__(TPB52, 1)
void lyra2Z_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash)
{
	const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;

	if (thread < threads)
	{
		uint2 state[4];
		state[0] = __ldg(&DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x]);
		state[1] = __ldg(&DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x]);
		state[2] = __ldg(&DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x]);
		state[3] = __ldg(&DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x]);

		reduceDuplex(state, thread, threads);
		reduceDuplexRowSetup(1, 0, 2, state, thread, threads);
		reduceDuplexRowSetup(2, 1, 3, state, thread, threads);
		reduceDuplexRowSetup(3, 0, 4, state, thread, threads);
		reduceDuplexRowSetup(4, 3, 5, state, thread, threads);
		reduceDuplexRowSetup(5, 2, 6, state, thread, threads);
		reduceDuplexRowSetup(6, 1, 7, state, thread, threads);

		uint32_t rowa; // = WarpShuffle(state[0].x, 0, 4) & 7;
		uint32_t prev = 7;
		uint32_t iterator = 0;

	//for (uint32_t j=0;j<4;j++) {

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}

		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}

		for (uint32_t i = 0; i<7; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowt(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}

	//}
		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowt_8_v2(prev,iterator,rowa, state, thread, threads);

		DMatrix[(0 * threads + thread) * blockDim.x + threadIdx.x] = state[0];
		DMatrix[(1 * threads + thread) * blockDim.x + threadIdx.x] = state[1];
		DMatrix[(2 * threads + thread) * blockDim.x + threadIdx.x] = state[2];
		DMatrix[(3 * threads + thread) * blockDim.x + threadIdx.x] = state[3];
	}
}

__global__
__launch_bounds__(64, 1)
void lyra2Z_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *g_hash, uint32_t *resNonces)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint28 state[4];

	if (thread < threads)
	{
		state[0] = __ldg4(&((uint2x4*)DMatrix)[threads * 0 + thread]);
		state[1] = __ldg4(&((uint2x4*)DMatrix)[threads * 1 + thread]);
		state[2] = __ldg4(&((uint2x4*)DMatrix)[threads * 2 + thread]);
		state[3] = __ldg4(&((uint2x4*)DMatrix)[threads * 3 + thread]);

		for (int i = 0; i < 12; i++)
			round_lyra(state);

		uint32_t nonce = startNounce + thread;
		if (((uint64_t*)state)[3] <= ((uint64_t*)pTarget)[3]) {
			atomicMin(&resNonces[1], resNonces[0]);
			atomicMin(&resNonces[0], nonce);
		}
/*
		g_hash[thread + threads * 0] = state[0].x;
		g_hash[thread + threads * 1] = state[0].y;
		g_hash[thread + threads * 2] = state[0].z;
		g_hash[thread + threads * 3] = state[0].w;
*/
	}
}
#else
#if __CUDA_ARCH__ < 350
__device__ void* DMatrix;
#endif
__global__ void lyra2Z_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *g_hash) {}
__global__ void lyra2Z_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint64_t *g_hash) {}
__global__ void lyra2Z_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *g_hash, uint32_t *resNonces) {}
#endif

__host__
void lyra2Z_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix)
{
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DMatrix, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
	cudaMalloc(&d_GNonces[thr_id], 2 * sizeof(uint32_t));
	cudaMallocHost(&h_GNonces[thr_id], 2 * sizeof(uint32_t));
}

__host__
void lyra2Z_cpu_init_sm2(int thr_id, uint32_t threads)
{
	// just assign the device pointer allocated in main loop
	cudaMalloc(&d_GNonces[thr_id], 2 * sizeof(uint32_t));
	cudaMallocHost(&h_GNonces[thr_id], 2 * sizeof(uint32_t));
}

__host__
void lyra2Z_cpu_free(int thr_id)
{
	cudaFree(d_GNonces[thr_id]);
	cudaFreeHost(h_GNonces[thr_id]);
}

__host__
uint32_t lyra2Z_getSecNonce(int thr_id, int num)
{
	uint32_t results[2];
	memset(results, 0xFF, sizeof(results));
	cudaMemcpy(results, d_GNonces[thr_id], sizeof(results), cudaMemcpyDeviceToHost);
	if (results[1] == results[0])
		return UINT32_MAX;
	return results[num];
}

__host__
void lyra2Z_setTarget(const void *pTargetIn)
{
	cudaMemcpyToSymbol(pTarget, pTargetIn, 32, 0, cudaMemcpyHostToDevice);
}

__host__
uint32_t lyra2Z_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, bool gtx750ti)
{
	uint32_t result = UINT32_MAX;
	cudaMemset(d_GNonces[thr_id], 0xff, 2 * sizeof(uint32_t));
	int dev_id = device_map[thr_id % MAX_GPUS];

	uint32_t tpb = TPB52;

	if (device_sm[dev_id] == 500)
		tpb = TPB50;
	if (device_sm[dev_id] == 200)
		tpb = TPB20;

	dim3 grid1((threads * 4 + tpb - 1) / tpb);
	dim3 block1(4, tpb >> 2);

	dim3 grid2((threads + 64 - 1) / 64);
	dim3 block2(64);

	dim3 grid3((threads + tpb - 1) / tpb);
	dim3 block3(tpb);

	if (device_sm[dev_id] >= 520)
	{
		lyra2Z_gpu_hash_32_1 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash);

		lyra2Z_gpu_hash_32_2 <<< grid1, block1, 24 * (8 - 0) * sizeof(uint2) * tpb >>> (threads, startNounce, d_hash);

		lyra2Z_gpu_hash_32_3 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash, d_GNonces[thr_id]);
	}
	else if (device_sm[dev_id] == 500 || device_sm[dev_id] == 350)
	{
		size_t shared_mem = 0;

		if (gtx750ti)
			// 8Warpに調整のため、8192バイト確保する
			shared_mem = 8192;
		else
			// 10Warpに調整のため、6144バイト確保する
			shared_mem = 6144;

		lyra2Z_gpu_hash_32_1_sm5 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash);

		lyra2Z_gpu_hash_32_2_sm5 <<< grid1, block1, shared_mem >>> (threads, startNounce, (uint2*)d_hash);

		lyra2Z_gpu_hash_32_3_sm5 <<< grid2, block2 >>> (threads, startNounce, (uint2*)d_hash, d_GNonces[thr_id]);
	}
	else
		lyra2Z_gpu_hash_32_sm2 <<< grid3, block3 >>> (threads, startNounce, d_hash, d_GNonces[thr_id]);

	// get first found nonce
	cudaMemcpy(h_GNonces[thr_id], d_GNonces[thr_id], 1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	result = *h_GNonces[thr_id];

	return result;
}
