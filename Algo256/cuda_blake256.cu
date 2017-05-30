/**
 * Blake-256 Cuda Kernel (Tested on SM 5.0)
 *
 * Tanguy Pruvot - Nov. 2014
 */
extern "C" {
#include "sph/sph_blake.h"
}

#include "cuda_helper.h"

#include <memory.h>

static __device__ uint64_t cuda_swab32ll(uint64_t x) {
	return MAKE_ULONGLONG(cuda_swab32(_LODWORD(x)), cuda_swab32(_HIDWORD(x)));
}

__constant__ static uint32_t c_data[3 + 1];

__constant__ static uint32_t sigma[16][16];
static uint32_t  c_sigma[16][16] = {
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
	{ 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
	{ 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
	{ 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
	{ 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
	{ 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
	{ 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
	{ 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
	{ 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
};

static const uint32_t  c_IV256[8] = {
	0x6A09E667, 0xBB67AE85,
	0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C,
	0x1F83D9AB, 0x5BE0CD19
};

__device__ __constant__ static uint32_t cpu_h[8];

__device__ __constant__ static  uint32_t  u256[16];
static const uint32_t  c_u256[16] = {
	0x243F6A88, 0x85A308D3,
	0x13198A2E, 0x03707344,
	0xA4093822, 0x299F31D0,
	0x082EFA98, 0xEC4E6C89,
	0x452821E6, 0x38D01377,
	0xBE5466CF, 0x34E90C6C,
	0xC0AC29B7, 0xC97C50DD,
	0x3F84D5B5, 0xB5470917
};

#define GS2(a,b,c,d,x) { \
	const uint32_t idx1 = sigma[r][x]; \
	const uint32_t idx2 = sigma[r][x + 1]; \
	v[a] += (m[idx1] ^ u256[idx2]) + v[b]; \
	v[d] = SPH_ROTL32(v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
\
	v[a] += (m[idx2] ^ u256[idx1]) + v[b]; \
	v[d] = SPH_ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
}

//#define ROTL32(x, n) ((x) << (n)) | ((x) >> (32 - (n)))
//#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define hostGS(a,b,c,d,x) { \
	const uint32_t idx1 = c_sigma[r][x]; \
	const uint32_t idx2 = c_sigma[r][x + 1]; \
	v[a] += (m[idx1] ^ c_u256[idx2]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 16); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 12); \
\
	v[a] += (m[idx2] ^ c_u256[idx1]) + v[b]; \
	v[d] = ROTR32(v[d] ^ v[a], 8); \
	v[c] += v[d]; \
	v[b] = ROTR32(v[b] ^ v[c], 7); \
	}

/* Second part (64-80) msg never change, store it */
__device__ __constant__ static const uint32_t  c_Padding[16] = {
	0, 0, 0, 0,
	0x80000000, 0, 0, 0,
	0, 0, 0, 0,
	0, 1, 0, 640,
};

__host__ __forceinline__
static void blake256_compress1st(uint32_t *h, const uint32_t *block, const uint32_t T0)
{
	uint32_t m[16];
	uint32_t v[16];

	for (int i = 0; i < 16; i++) {
		m[i] = block[i];
	}

	for (int i = 0; i < 8; i++)
		v[i] = h[i];

	v[8] = c_u256[0];
	v[9] = c_u256[1];
	v[10] = c_u256[2];
	v[11] = c_u256[3];

	v[12] = c_u256[4] ^ T0;
	v[13] = c_u256[5] ^ T0;
	v[14] = c_u256[6];
	v[15] = c_u256[7];

	for (int r = 0; r < 14; r++) {
		/* column step */
		hostGS(0, 4, 0x8, 0xC, 0x0);
		hostGS(1, 5, 0x9, 0xD, 0x2);
		hostGS(2, 6, 0xA, 0xE, 0x4);
		hostGS(3, 7, 0xB, 0xF, 0x6);
		/* diagonal step */
		hostGS(0, 5, 0xA, 0xF, 0x8);
		hostGS(1, 6, 0xB, 0xC, 0xA);
		hostGS(2, 7, 0x8, 0xD, 0xC);
		hostGS(3, 4, 0x9, 0xE, 0xE);
	}

	for (int i = 0; i < 16; i++) {
		int j = i & 7;
		h[j] ^= v[i];
	}
}

__device__ __forceinline__
static void blake256_compress2nd(uint32_t *h, const uint32_t *block, const uint32_t T0)
{
	uint32_t m[16];
	uint32_t v[16];

	m[0] = block[0];
	m[1] = block[1];
	m[2] = block[2];
	m[3] = block[3];

	#pragma unroll
	for (int i = 4; i < 16; i++) {
		m[i] = c_Padding[i];
	}

	#pragma unroll 8
	for (int i = 0; i < 8; i++)
		v[i] = h[i];

	v[8] =  u256[0];
	v[9] =  u256[1];
	v[10] = u256[2];
	v[11] = u256[3];

	v[12] = u256[4] ^ T0;
	v[13] = u256[5] ^ T0;
	v[14] = u256[6];
	v[15] = u256[7];

	#pragma unroll 14
	for (int r = 0; r < 14; r++) {
		/* column step */
		GS2(0, 4, 0x8, 0xC, 0x0);
		GS2(1, 5, 0x9, 0xD, 0x2);
		GS2(2, 6, 0xA, 0xE, 0x4);
		GS2(3, 7, 0xB, 0xF, 0x6);
		/* diagonal step */
		GS2(0, 5, 0xA, 0xF, 0x8);
		GS2(1, 6, 0xB, 0xC, 0xA);
		GS2(2, 7, 0x8, 0xD, 0xC);
		GS2(3, 4, 0x9, 0xE, 0xE);
	}

	#pragma unroll 16
	for (int i = 0; i < 16; i++) {
		int j = i & 7;
		h[j] ^= v[i];
	}
}

__global__ __launch_bounds__(256,3)
void blake256_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint64_t * Hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t h[8];
		uint32_t input[4];

		#pragma unroll
		for (int i = 0; i < 8; i++) h[i] = cpu_h[i];

		#pragma unroll
		for (int i = 0; i < 3; ++i) input[i] = c_data[i];

		input[3] = startNonce + thread;
		blake256_compress2nd(h, input, 640);

		#pragma unroll
		for (int i = 0; i<4; i++) {
			Hash[i*threads + thread] = cuda_swab32ll(MAKE_ULONGLONG(h[2 * i], h[2*i+1]));
		}
	}
}

__host__
void blake256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash, int order)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	blake256_gpu_hash_80 <<<grid, block>>> (threads, startNonce, Hash);
	MyStreamSynchronize(NULL, order, thr_id);
}

__host__
void blake256_cpu_setBlock_80(uint32_t *pdata)
{
	uint32_t h[8], data[20];

	memcpy(data, pdata, 80);
	memcpy(h, c_IV256, sizeof(c_IV256));
	blake256_compress1st(h, pdata, 512);

	cudaMemcpyToSymbol(cpu_h, h, sizeof(h), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_data, &data[16], sizeof(c_data), 0, cudaMemcpyHostToDevice);
}

__host__
void blake256_cpu_init(int thr_id, uint32_t threads)
{
	cuda_get_arch(thr_id);
	cudaMemcpyToSymbol(u256, c_u256, sizeof(c_u256), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(sigma, c_sigma, sizeof(c_sigma), 0, cudaMemcpyHostToDevice);
}
