#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#define TPB52 32
#define TPB50 32
#define TPB30 32
#define TPB20 32

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 520
#endif

#include "cuda_lyra2_vectors.h"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
__device__ void __threadfence_block();
#if __CUDA_ARCH__ >= 300
__device__ uint32_t __shfl(uint32_t a, uint32_t b, uint32_t c);
#endif
#endif

#define Nrow 4
#define Ncol 4
#define memshift 3

__device__ uint2x4 *DState;

__device__ __forceinline__ uint2 LD4S(const int index)
{
	extern __shared__ uint2 shared_mem[];

	return shared_mem[(index * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
}

__device__ __forceinline__ void ST4S(const int index, const uint2 data)
{
	extern __shared__ uint2 shared_mem[];

	shared_mem[(index * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data;
}

__device__ __forceinline__
void Gfunc_v5(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
{
	a += b; uint2 tmp = d; d.y = a.x ^ tmp.x; d.x = a.y ^ tmp.y;
	c += d; b ^= c; b = ROR24(b);
	a += b; d ^= a; d = ROR16(d);
	c += d; b ^= c; b = ROR2(b, 63);
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

__device__ __forceinline__ void WarpShuffle3(uint2 &a1, uint2 &a2, uint2 &a3, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t c)
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
	Gfunc_v5(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 1, threadIdx.x + 2, threadIdx.x + 3, 4);
	Gfunc_v5(s[0], s[1], s[2], s[3]);
	WarpShuffle3(s[1], s[2], s[3], threadIdx.x + 3, threadIdx.x + 2, threadIdx.x + 1, 4);
}

__device__ __forceinline__
void round_lyra(uint2x4* s)
{
	Gfunc_v5(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc_v5(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc_v5(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc_v5(s[0].w, s[1].w, s[2].w, s[3].w);

	Gfunc_v5(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc_v5(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc_v5(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc_v5(s[0].w, s[1].x, s[2].y, s[3].z);
}


__device__ __forceinline__ void reduceDuplexRowSetupV2(uint2 state[4])
{
	int i, j;
	uint2 state1[Ncol][3], state0[Ncol][3], state2[3];

#if __CUDA_ARCH__ > 500
#pragma unroll
#endif
	for (int i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (j = 0; j < 3; j++)
			state0[Ncol - i - 1][j] = state[j];
		round_lyra(state);
	}

	//#pragma unroll 4
	for (i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (j = 0; j < 3; j++)
			state[j] ^= state0[i][j];

		round_lyra(state);

#pragma unroll
		for (j = 0; j < 3; j++)
			state1[Ncol - i - 1][j] = state0[i][j];

#pragma unroll
		for (j = 0; j < 3; j++)
			state1[Ncol - i - 1][j] ^= state[j];
	}

	uint32_t s0 = 0;
	uint32_t s2 = 33;
	for (i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (j = 0; j < 3; j++)
			state[j] ^= state1[i][j] + state0[i][j];

		round_lyra(state);

#pragma unroll
		for (j = 0; j < 3; j++)
			state2[j] = state1[i][j];

#pragma unroll
		for (j = 0; j < 3; j++)
			state2[j] ^= state[j];

#pragma unroll
		for (j = 0; j < 3; j++)
			ST4S(s2 + j, state2[j]);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state0[i][0] ^= Data2;
			state0[i][1] ^= Data0;
			state0[i][2] ^= Data1;
		}
		else
		{
			state0[i][0] ^= Data0;
			state0[i][1] ^= Data1;
			state0[i][2] ^= Data2;
		}

#pragma unroll
		for (j = 0; j < 3; j++)
			ST4S(s0 + j, state0[i][j]);

#pragma unroll
		for (j = 0; j < 3; j++)
			state0[i][j] = state2[j];

		s0 += memshift;
		s2 -= memshift;
	}

	s2 += 24;
	for (i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (j = 0; j < 3; j++)
			state[j] ^= state1[i][j] + state0[Ncol - i - 1][j];

		round_lyra(state);

#pragma unroll
		for (j = 0; j < 3; j++)
			state0[Ncol - i - 1][j] ^= state[j];
#pragma unroll
		for (j = 0; j < 3; j++)
			ST4S(s2 + j, state0[Ncol - i - 1][j]);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state1[i][0] ^= Data2;
			state1[i][1] ^= Data0;
			state1[i][2] ^= Data1;
		}
		else
		{
			state1[i][0] ^= Data0;
			state1[i][1] ^= Data1;
			state1[i][2] ^= Data2;
		}

#pragma unroll
		for (j = 0; j < 3; j++)
			ST4S(s0 + j, state1[i][j]);

		s0 += memshift;
		s2 -= memshift;
	}
}

__device__ void reduceDuplexRowtV2(uint2 state[4])
{
	uint32_t rowInOut = WarpShuffle(state[0].x, 0, 4) & 3;

	uint2 state2[3], state1[3], last[3];
	uint32_t s1 = 36;
	uint32_t s2 = 12 * rowInOut;
	uint32_t s3 = 0;

	for (int i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = LD4S(s2 + j);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= LD4S(s1 + j) + state2[j];

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

#pragma unroll
		for (int j = 0; j < 3; j++)
		{
			ST4S(s2 + j, state2[j]);
			ST4S(s3 + j, LD4S(s3 + j) ^ state[j]);
		}

		s1 += memshift;
		s2 += memshift;
		s3 += memshift;
	}
	s1 = 0;
	rowInOut = WarpShuffle(state[0].x, 0, 4) & 3;
	s2 = 12 * rowInOut;

	for (int i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = LD4S(s2 + j);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= LD4S(s1 + j) + state2[j];

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

#pragma unroll
		for (int j = 0; j < 3; j++)
		{
			ST4S(s2 + j, state2[j]);
			ST4S(s3 + j, LD4S(s3 + j) ^ state[j]);
		}

		s1 += memshift;
		s2 += memshift;
		s3 += memshift;
	}

	rowInOut = WarpShuffle(state[0].x, 0, 4) & 3;
	s2 = 12 * rowInOut;

	for (int i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = LD4S(s2 + j);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= LD4S(s1 + j) + state2[j];

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

#pragma unroll
		for (int j = 0; j < 3; j++)
		{
			ST4S(s2 + j, state2[j]);
			ST4S(s3 + j, LD4S(s3 + j) ^ state[j]);
		}

		s1 += memshift;
		s2 += memshift;
		s3 += memshift;
	}

	rowInOut = WarpShuffle(state[0].x, 0, 4) & 3;
	s2 = 12 * rowInOut;

#pragma unroll
	for (int j = 0; j < 3; j++)
		last[j] = LD4S(s2 + j);

#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= LD4S(s1 + j) + last[j];

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
	else
	{
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == 3)
	{
#pragma unroll 
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}
	s1 += memshift;
	s2 += memshift;

	for (int i = 1; i < Ncol; i++)
	{
#pragma unroll 
		for (int j = 0; j < 3; j++)
			state[j] ^= LD4S(s1 + j) + LD4S(s2 + j);

		round_lyra(state);

		s1 += memshift;
		s2 += memshift;
	}

#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}

__constant__ uint28 blake2b_IV[2] = {
	0xf3bcc908lu, 0x6a09e667lu,
	0x84caa73blu, 0xbb67ae85lu,
	0xfe94f82blu, 0x3c6ef372lu,
	0x5f1d36f1lu, 0xa54ff53alu,
	0xade682d1lu, 0x510e527flu,
	0x2b3e6c1flu, 0x9b05688clu,
	0xfb41bd6blu, 0x1f83d9ablu,
	0x137e2179lu, 0x5be0cd19lu
};

__constant__ uint28 Mask[2] = {
	0x00000020lu, 0x00000000lu,
	0x00000020lu, 0x00000000lu,
	0x00000020lu, 0x00000000lu,
	0x00000001lu, 0x00000000lu,
	0x00000004lu, 0x00000000lu,
	0x00000004lu, 0x00000000lu,
	0x00000080lu, 0x00000000lu,
	0x00000000lu, 0x01000000lu
};

__global__ __launch_bounds__(64, 1)
void lyra2v2_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint28 state[4];

	if (thread < threads)
	{
		state[0].x = state[1].x = __ldg(&outputHash[thread + threads * 0]);
		state[0].y = state[1].y = __ldg(&outputHash[thread + threads * 1]);
		state[0].z = state[1].z = __ldg(&outputHash[thread + threads * 2]);
		state[0].w = state[1].w = __ldg(&outputHash[thread + threads * 3]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

#pragma unroll 2
		for (int i = 0; i<12; i++)
			round_lyra(state);

		state[0] ^= Mask[0];
		state[1] ^= Mask[1];

#pragma unroll 2
		for (int i = 0; i<12; i++)
			round_lyra(state);

		DState[blockDim.x * gridDim.x * 0 + blockDim.x * blockIdx.x + threadIdx.x] = state[0];
		DState[blockDim.x * gridDim.x * 1 + blockDim.x * blockIdx.x + threadIdx.x] = state[1];
		DState[blockDim.x * gridDim.x * 2 + blockDim.x * blockIdx.x + threadIdx.x] = state[2];
		DState[blockDim.x * gridDim.x * 3 + blockDim.x * blockIdx.x + threadIdx.x] = state[3];

	} //thread
}

#if __CUDA_ARCH__ < 300
__global__ __launch_bounds__(TPB20, 1)
#elif __CUDA_ARCH__ < 500
__global__ __launch_bounds__(TPB30, 1)
#elif __CUDA_ARCH__ == 500
__global__ __launch_bounds__(TPB50, 1)
#else
__global__ __launch_bounds__(TPB52, 1)
#endif
void lyra2v2_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint64_t *outputHash)
{
	const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;

	if (thread < threads)
	{
		uint2 state[4];
		state[0] = ((uint2*)DState)[(0 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[1] = ((uint2*)DState)[(1 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[2] = ((uint2*)DState)[(2 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[3] = ((uint2*)DState)[(3 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];

		reduceDuplexRowSetupV2(state);

		reduceDuplexRowtV2(state);

		((uint2*)DState)[(0 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[0];
		((uint2*)DState)[(1 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[1];
		((uint2*)DState)[(2 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[2];
		((uint2*)DState)[(3 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[3];
	} //thread
}

__global__ __launch_bounds__(64, 1)
void lyra2v2_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint28 state[4];

	if (thread < threads)
	{
		state[0] = __ldg4(&DState[blockDim.x * gridDim.x * 0 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[1] = __ldg4(&DState[blockDim.x * gridDim.x * 1 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[2] = __ldg4(&DState[blockDim.x * gridDim.x * 2 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[3] = __ldg4(&DState[blockDim.x * gridDim.x * 3 + blockDim.x * blockIdx.x + threadIdx.x]);

#pragma unroll 2
		for (int i = 0; i < 12; i++)
			round_lyra(state);

		outputHash[thread + threads * 0] = state[0].x;
		outputHash[thread + threads * 1] = state[0].y;
		outputHash[thread + threads * 2] = state[0].z;
		outputHash[thread + threads * 3] = state[0].w;

	} //thread
}

__host__
void lyra2v2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix)
{
	int dev_id = device_map[thr_id % MAX_GPUS];
	// just assign the device pointer allocated in main loop
	cudaMemcpyToSymbol(DState, &d_matrix, sizeof(uint64_t*), 0, cudaMemcpyHostToDevice);
}

__host__
void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, int order)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	uint32_t tpb = TPB52;

	if (cuda_arch[dev_id] > 500) tpb = TPB52;
	else if (cuda_arch[dev_id] == 500) tpb = TPB50;
	else if (cuda_arch[dev_id] >= 300) tpb = TPB30;
	else if (cuda_arch[dev_id] >= 200) tpb = TPB20;

	dim3 grid1((threads * 4 + tpb - 1) / tpb);
	dim3 block1(4, tpb >> 2);

	dim3 grid2((threads + 64 - 1) / 64);
	dim3 block2(64);

	if (cuda_arch[dev_id] < 500)
		cudaFuncSetCacheConfig(lyra2v2_gpu_hash_32_2, cudaFuncCachePreferShared);

	lyra2v2_gpu_hash_32_1 << <grid2, block2 >> > (threads, startNounce, (uint2*)g_hash);

	lyra2v2_gpu_hash_32_2 << <grid1, block1, 48 * sizeof(uint2) * tpb >> > (threads, startNounce, g_hash);

	lyra2v2_gpu_hash_32_3 << <grid2, block2 >> > (threads, startNounce, (uint2*)g_hash);
	//MyStreamSynchronize(NULL, order, thr_id);
}
