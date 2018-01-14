#include <memory.h>

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
//#define __CUDA_ARCH__ 500
#define __threadfence_block()
#define __ldg(x) *(x)
#define atomicMin(p,y) y
#endif

#include "cuda_helper.h"

#define TPB50 32

__constant__ uint32_t pTarget[8];

static __device__ __forceinline__
void Gfunc(uint2 & a, uint2 &b, uint2 &c, uint2 &d)
{
#if __CUDA_ARCH__ > 500
	a += b; uint2 tmp = d; d.y = a.x ^ tmp.x; d.x = a.y ^ tmp.y;
	c += d; b ^= c; b = ROR24(b);
	a += b; d ^= a; d = ROR16(d);
	c += d; b ^= c; b = ROR2(b, 63);
#else
	a += b; d ^= a; d = SWAPUINT2(d);
	c += d; b ^= c; b = ROR2(b, 24);
	a += b; d ^= a; d = ROR2(d, 16);
	c += d; b ^= c; b = ROR2(b, 63);
#endif
}

#if __CUDA_ARCH__ == 500 || __CUDA_ARCH__ == 350
#include "cuda_lyra2_vectors.h"

#define Nrow 8
#define Ncol 8
#define memshift 3

__device__ uint2 *DMatrix;

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

#if __CUDA_ARCH__ == 300
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
#else // != 300

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

#endif // != 300

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
void reduceDuplexV5(uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	uint2 state1[3], state2[3];

	const uint32_t ps0 = (memshift * Ncol * 0 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps1 = (memshift * Ncol * 1 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps2 = (memshift * Ncol * 2 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps3 = (memshift * Ncol * 3 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps4 = (memshift * Ncol * 4 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps5 = (memshift * Ncol * 5 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps6 = (memshift * Ncol * 6 * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps7 = (memshift * Ncol * 7 * threads + thread)*blockDim.x + threadIdx.x;

	for (int i = 0; i < 8; i++)
	{
		const uint32_t s0 = memshift * Ncol * 0 + (Ncol - 1 - i) * memshift;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			ST4S(s0 + j, state[j]);
		round_lyra(state);
	}

	for (int i = 0; i < 8; i++)
	{
		const uint32_t s0 = memshift * Ncol * 0 + i * memshift;
		const uint32_t s1 = ps1 + (7 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = LD4S(s0 + j);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s1 + j*threads*blockDim.x) = state1[j] ^ state[j];
	}

	// 1, 0, 2
	for (int i = 0; i < 8; i++)
	{
		const uint32_t s0 = memshift * Ncol * 0 + i * memshift;
		const uint32_t s1 = ps1 + i * memshift* threads*blockDim.x;
		const uint32_t s2 = ps2 + (7 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s1 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = LD4S(s0 + j);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s2 + j*threads*blockDim.x) = state1[j] ^ state[j];

		// simultaneously receive data from preceding thread and send data to following thread
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
			ST4S(s0 + j, state2[j]);
	}

	// 2, 1, 3
	for (int i = 0; i < 8; i++)
	{
		const uint32_t s1 = ps1 + i * memshift* threads*blockDim.x;
		const uint32_t s2 = ps2 + i * memshift* threads*blockDim.x;
		const uint32_t s3 = ps3 + (7 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s2 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s1 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s3 + j*threads*blockDim.x) = state1[j] ^ state[j];

		// simultaneously receive data from preceding thread and send data to following thread
		uint2 Data0 = state[0];
		uint2 Data1 = state[1];
		uint2 Data2 = state[2];
		WarpShuffle3(Data0, Data1, Data2, threadIdx.x - 1, threadIdx.x - 1, threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		} else  {
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s1 + j*threads*blockDim.x) = state2[j];
	}

	// 3, 0, 4
	for (int i = 0; i < 8; i++)
	{
		const uint32_t ls0 = memshift * Ncol * 0 + i * memshift;
		const uint32_t s0 = ps0 + i * memshift* threads*blockDim.x;
		const uint32_t s3 = ps3 + i * memshift* threads*blockDim.x;
		const uint32_t s4 = ps4 + (7 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s3 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = LD4S(ls0 + j);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s4 + j*threads*blockDim.x) = state1[j] ^ state[j];

		// simultaneously receive data from preceding thread and send data to following thread
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

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s0 + j*threads*blockDim.x) = state2[j];
	}

	// 4, 3, 5
	for (int i = 0; i < 8; i++)
	{
		const uint32_t s3 = ps3 + i * memshift* threads*blockDim.x;
		const uint32_t s4 = ps4 + i * memshift* threads*blockDim.x;
		const uint32_t s5 = ps5 + (7 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s4 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s3 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s5 + j*threads*blockDim.x) = state1[j] ^ state[j];

		// simultaneously receive data from preceding thread and send data to following thread
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
			*(DMatrix + s3 + j*threads*blockDim.x) = state2[j];
	}

	// 5, 2, 6
	for (int i = 0; i < 8; i++)
	{
		const uint32_t s2 = ps2 + i * memshift* threads*blockDim.x;
		const uint32_t s5 = ps5 + i * memshift* threads*blockDim.x;
		const uint32_t s6 = ps6 + (7 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s5 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s2 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s6 + j*threads*blockDim.x) = state1[j] ^ state[j];

		// simultaneously receive data from preceding thread and send data to following thread
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
			*(DMatrix + s2 + j*threads*blockDim.x) = state2[j];
	}

	// 6, 1, 7
	for (int i = 0; i < 8; i++)
	{
		const uint32_t s1 = ps1 + i * memshift* threads*blockDim.x;
		const uint32_t s6 = ps6 + i * memshift* threads*blockDim.x;
		const uint32_t s7 = ps7 + (7 - i)*memshift* threads*blockDim.x;
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = *(DMatrix + s6 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = *(DMatrix + s1 + j*threads*blockDim.x);
		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra(state);

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s7 + j*threads*blockDim.x) = state1[j] ^ state[j];

		// simultaneously receive data from preceding thread and send data to following thread
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

		#pragma unroll
		for (int j = 0; j < 3; j++)
			*(DMatrix + s1 + j*threads*blockDim.x) = state2[j];
	}
}

static __device__ __forceinline__
void reduceDuplexRowV50(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	const uint32_t ps1 = (memshift * Ncol * rowIn*threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps2 = (memshift * Ncol * rowInOut *threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps3 = (memshift * Ncol * rowOut*threads + thread)*blockDim.x + threadIdx.x;

	#pragma unroll 1
	for (int i = 0; i < 8; i++)
	{
		uint2 state1[3], state2[3];

		const uint32_t s1 = ps1 + i*memshift*threads *blockDim.x;
		const uint32_t s2 = ps2 + i*memshift*threads *blockDim.x;
		const uint32_t s3 = ps3 + i*memshift*threads *blockDim.x;

		#pragma unroll
		for (int j = 0; j < 3; j++) {
			state1[j] = *(DMatrix + s1 + j*threads*blockDim.x);
			state2[j] = *(DMatrix + s2 + j*threads*blockDim.x);
		}

		#pragma unroll
		for (int j = 0; j < 3; j++) {
			state1[j] += state2[j];
			state[j] ^= state1[j];
		}

		round_lyra(state);

		// simultaneously receive data from preceding thread and send data to following thread
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

		#pragma unroll
		for (int j = 0; j < 3; j++)
		{
			*(DMatrix + s2 + j*threads*blockDim.x) = state2[j];
			*(DMatrix + s3 + j*threads*blockDim.x) ^= state[j];
		}
	}
}

static __device__ __forceinline__
void reduceDuplexRowV50_8(const int rowInOut, uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	const uint32_t ps1 = (memshift * Ncol * 2*threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps2 = (memshift * Ncol * rowInOut *threads + thread)*blockDim.x + threadIdx.x;
	// const uint32_t ps3 = (memshift * Ncol * 5*threads + thread)*blockDim.x + threadIdx.x;

	uint2 state1[3], last[3];

	#pragma unroll
	for (int j = 0; j < 3; j++) {
		state1[j] = *(DMatrix + ps1 + j*threads*blockDim.x);
		last[j] = *(DMatrix + ps2 + j*threads*blockDim.x);
	}

	#pragma unroll
	for (int j = 0; j < 3; j++) {
		state1[j] += last[j];
		state[j] ^= state1[j];
	}

	round_lyra(state);

	// simultaneously receive data from preceding thread and send data to following thread
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

	for (int i = 1; i < 8; i++)
	{
		const uint32_t s1 = ps1 + i*memshift*threads *blockDim.x;
		const uint32_t s2 = ps2 + i*memshift*threads *blockDim.x;

		#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= *(DMatrix + s1 + j*threads*blockDim.x) + *(DMatrix + s2 + j*threads*blockDim.x);

		round_lyra(state);
	}

	#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}

static __device__ __forceinline__
void reduceDuplexRowV50_8_v2(const int rowIn, const int rowOut,const int rowInOut, uint2 state[4], const uint32_t thread, const uint32_t threads)
{
	const uint32_t ps1 = (memshift * Ncol * rowIn * threads + thread)*blockDim.x + threadIdx.x;
	const uint32_t ps2 = (memshift * Ncol * rowInOut *threads + thread)*blockDim.x + threadIdx.x;
	// const uint32_t ps3 = (memshift * Ncol * 5*threads + thread)*blockDim.x + threadIdx.x;

	uint2 state1[3], last[3];

	#pragma unroll
	for (int j = 0; j < 3; j++) {
		state1[j] = *(DMatrix + ps1 + j*threads*blockDim.x);
		last[j] = *(DMatrix + ps2 + j*threads*blockDim.x);
	}

	#pragma unroll
	for (int j = 0; j < 3; j++) {
		state1[j] += last[j];
		state[j] ^= state1[j];
	}

	round_lyra(state);

	// simultaneously receive data from preceding thread and send data to following thread
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

	if (rowInOut == rowOut)
	{
#pragma unroll
		for (int j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	for (int i = 1; i < 8; i++)
	{
		const uint32_t s1 = ps1 + i*memshift*threads *blockDim.x;
		const uint32_t s2 = ps2 + i*memshift*threads *blockDim.x;

#pragma unroll
		for (int j = 0; j < 3; j++)
			state[j] ^= *(DMatrix + s1 + j*threads*blockDim.x) + *(DMatrix + s2 + j*threads*blockDim.x);

		round_lyra(state);
	}


#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];

}


__global__ __launch_bounds__(64, 1)
void lyra2Z_gpu_hash_32_1_sm5(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	const uint2x4 blake2b_IV[2] = {
		{ { 0xf3bcc908, 0x6a09e667 }, { 0x84caa73b, 0xbb67ae85 }, { 0xfe94f82b, 0x3c6ef372 }, { 0x5f1d36f1, 0xa54ff53a } },
		{ { 0xade682d1, 0x510e527f }, { 0x2b3e6c1f, 0x9b05688c }, { 0xfb41bd6b, 0x1f83d9ab }, { 0x137e2179, 0x5be0cd19 } }
	};
	const uint2x4 Mask[2] = {
		0x00000020UL, 0x00000000UL, 0x00000020UL, 0x00000000UL,
		0x00000020UL, 0x00000000UL, 0x00000008UL, 0x00000000UL,
		0x00000008UL, 0x00000000UL, 0x00000008UL, 0x00000000UL,
		0x00000080UL, 0x00000000UL, 0x00000000UL, 0x01000000UL
	};
	if (thread < threads)
	{
		uint2x4 state[4];

		((uint2*)state)[0] = __ldg(&g_hash[thread]);
		((uint2*)state)[1] = __ldg(&g_hash[thread + threads]);
		((uint2*)state)[2] = __ldg(&g_hash[thread + threads * 2]);
		((uint2*)state)[3] = __ldg(&g_hash[thread + threads * 3]);

		state[1] = state[0];
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i < 12; i++)
			round_lyra(state); //because 12 is not enough

		state[0] ^= Mask[0];
		state[1] ^= Mask[1];

		for (int i = 0; i < 12; i++)
			round_lyra(state); //because 12 is not enough


		((uint2x4*)DMatrix)[0 * threads + thread] = state[0];
		((uint2x4*)DMatrix)[1 * threads + thread] = state[1];
		((uint2x4*)DMatrix)[2 * threads + thread] = state[2];
		((uint2x4*)DMatrix)[3 * threads + thread] = state[3];
	}
}

__global__ __launch_bounds__(TPB50, 1)
void lyra2Z_gpu_hash_32_2_sm5(uint32_t threads, uint32_t startNounce, uint2 *g_hash)
{
	const uint32_t thread = (blockDim.y * blockIdx.x + threadIdx.y);

	if (thread < threads)
	{
		uint2 state[4];

		state[0] = __ldg(&DMatrix[(0 * threads + thread)*blockDim.x + threadIdx.x]);
		state[1] = __ldg(&DMatrix[(1 * threads + thread)*blockDim.x + threadIdx.x]);
		state[2] = __ldg(&DMatrix[(2 * threads + thread)*blockDim.x + threadIdx.x]);
		state[3] = __ldg(&DMatrix[(3 * threads + thread)*blockDim.x + threadIdx.x]);

		reduceDuplexV5(state, thread, threads);

		uint32_t rowa; // = WarpShuffle(state[0].x, 0, 4) & 7;
		uint32_t prev = 7;
		uint32_t iterator = 0;
		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowV50(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}
		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowV50(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}
		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowV50(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}
		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowV50(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}
		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowV50(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}
		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowV50(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}
		for (uint32_t i = 0; i<8; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowV50(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator + 3) & 7;
		}
		for (uint32_t i = 0; i<7; i++) {
			rowa = WarpShuffle(state[0].x, 0, 4) & 7;
			reduceDuplexRowV50(prev, rowa, iterator, state, thread, threads);
			prev = iterator;
			iterator = (iterator - 1) & 7;
		}

		rowa = WarpShuffle(state[0].x, 0, 4) & 7;
		reduceDuplexRowV50_8_v2(prev,iterator,rowa, state, thread, threads);

		DMatrix[(0 * threads + thread)*blockDim.x + threadIdx.x] = state[0];
		DMatrix[(1 * threads + thread)*blockDim.x + threadIdx.x] = state[1];
		DMatrix[(2 * threads + thread)*blockDim.x + threadIdx.x] = state[2];
		DMatrix[(3 * threads + thread)*blockDim.x + threadIdx.x] = state[3];
	}
}

__global__ __launch_bounds__(64, 1)
void lyra2Z_gpu_hash_32_3_sm5(uint32_t threads, uint32_t startNounce, uint2 *g_hash, uint32_t *resNonces)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		uint2x4 state[4];

		state[0] = __ldg4(&((uint2x4*)DMatrix)[0 * threads + thread]);
		state[1] = __ldg4(&((uint2x4*)DMatrix)[1 * threads + thread]);
		state[2] = __ldg4(&((uint2x4*)DMatrix)[2 * threads + thread]);
		state[3] = __ldg4(&((uint2x4*)DMatrix)[3 * threads + thread]);

		for (int i = 0; i < 12; i++)
			round_lyra(state);

		uint32_t nonce = startNounce + thread;
		if (((uint64_t*)state)[3] <= ((uint64_t*)pTarget)[3]) {
			atomicMin(&resNonces[1], resNonces[0]);
			atomicMin(&resNonces[0], nonce);
		}
	}
}

#else
/* if __CUDA_ARCH__ != 500 .. host */
__global__ void lyra2Z_gpu_hash_32_1_sm5(uint32_t threads, uint32_t startNounce, uint2 *g_hash) {}
__global__ void lyra2Z_gpu_hash_32_2_sm5(uint32_t threads, uint32_t startNounce, uint2 *g_hash) {}
__global__ void lyra2Z_gpu_hash_32_3_sm5(uint32_t threads, uint32_t startNounce, uint2 *g_hash, uint32_t *resNonces) {}
#endif
