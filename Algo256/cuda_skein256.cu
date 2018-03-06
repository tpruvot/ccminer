#include <memory.h>

#include "cuda_helper.h"

static __device__ __forceinline__
void Round512(uint64_t &p0, uint64_t &p1, uint64_t &p2, uint64_t &p3,
	uint64_t &p4, uint64_t &p5, uint64_t &p6, uint64_t &p7,
	int ROT0, int ROT1, int ROT2, int ROT3)
{
	p0 += p1; p1 = ROTL64(p1, ROT0) ^ p0;
	p2 += p3; p3 = ROTL64(p3, ROT1) ^ p2;
	p4 += p5; p5 = ROTL64(p5, ROT2) ^ p4;
	p6 += p7; p7 = ROTL64(p7, ROT3) ^ p6;
}

static __device__ __forceinline__
void Round_8_512(const uint64_t *__restrict__ ks, const uint64_t *__restrict__ ts,
	uint64_t &p0, uint64_t &p1, uint64_t &p2, uint64_t &p3,
	uint64_t &p4, uint64_t &p5, uint64_t &p6, uint64_t &p7, int R)
{
	Round512(p0, p1, p2, p3, p4, p5, p6, p7, 46, 36, 19, 37);
	Round512(p2, p1, p4, p7, p6, p5, p0, p3, 33, 27, 14, 42);
	Round512(p4, p1, p6, p3, p0, p5, p2, p7, 17, 49, 36, 39);
	Round512(p6, p1, p0, p7, p2, p5, p4, p3, 44,  9, 54, 56);

	p0 += ks[(R+0) % 9];
	p1 += ks[(R+1) % 9];
	p2 += ks[(R+2) % 9];
	p3 += ks[(R+3) % 9];
	p4 += ks[(R+4) % 9];
	p5 += ks[(R+5) % 9] + ts[(R+0) % 3];
	p6 += ks[(R+6) % 9] + ts[(R+1) % 3];
	p7 += ks[(R+7) % 9] + R;

	Round512(p0, p1, p2, p3, p4, p5, p6, p7, 39, 30, 34, 24);
	Round512(p2, p1, p4, p7, p6, p5, p0, p3, 13, 50, 10, 17);
	Round512(p4, p1, p6, p3, p0, p5, p2, p7, 25, 29, 39, 43);
	Round512(p6, p1, p0, p7, p2, p5, p4, p3,  8, 35, 56, 22);

	p0 += ks[(R+1) % 9];
	p1 += ks[(R+2) % 9];
	p2 += ks[(R+3) % 9];
	p3 += ks[(R+4) % 9];
	p4 += ks[(R+5) % 9];
	p5 += ks[(R+6) % 9] + ts[(R+1) % 3];
	p6 += ks[(R+7) % 9] + ts[(R+2) % 3];
	p7 += ks[(R+8) % 9] + R+1;
}

__global__ __launch_bounds__(256,3)
void skein256_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint64_t *outputHash)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads)
	{
		const uint64_t skein_ks_parity64 = 0x1BD11BDAA9FC1A22ull;

		const uint64_t t12[6] = {
			0x20, 0xf000000000000000ull, 0xf000000000000020ull,
			0x08, 0xff00000000000000ull, 0xff00000000000008ull
		};

		uint64_t h[9] = {
			0xCCD044A12FDB3E13ull, 0xE83590301A79A9EBull,
			0x55AEA0614F816E6Full, 0x2A2767A4AE9B94DBull,
			0xEC06025E74DD7683ull, 0xE7A436CDC4746251ull,
			0xC36FBAF9393AD185ull, 0x3EEDBA1833EDFC13ull,
			0xB69D3CFCC73A4E2Aull // skein_ks_parity64 ^ h[0..7]
		};
		uint64_t dt0, dt1, dt2, dt3;
		uint64_t p0, p1, p2, p3, p4, p5, p6, p7;

		dt0 = __ldg(&outputHash[0 * threads + thread]);
		dt1 = __ldg(&outputHash[1 * threads + thread]);
		dt2 = __ldg(&outputHash[2 * threads + thread]);
		dt3 = __ldg(&outputHash[3 * threads + thread]);

		p0 = h[0] + dt0;
		p1 = h[1] + dt1;
		p2 = h[2] + dt2;
		p3 = h[3] + dt3;
		p4 = h[4];
		p5 = h[5] + t12[0];
		p6 = h[6] + t12[1];
		p7 = h[7];

		// forced unroll required
		Round_8_512(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 1);
		Round_8_512(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 3);
		Round_8_512(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 5);
		Round_8_512(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 7);
		Round_8_512(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 9);
		Round_8_512(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 11);
		Round_8_512(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 13);
		Round_8_512(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 15);
		Round_8_512(h, t12, p0, p1, p2, p3, p4, p5, p6, p7, 17);

		p0 ^= dt0;
		p1 ^= dt1;
		p2 ^= dt2;
		p3 ^= dt3;

		h[0] = p0;
		h[1] = p1;
		h[2] = p2;
		h[3] = p3;
		h[4] = p4;
		h[5] = p5;
		h[6] = p6;
		h[7] = p7;
		h[8] = skein_ks_parity64 ^ h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[4] ^ h[5] ^ h[6] ^ h[7];

		const uint64_t *t = t12+3;
		p5 += t12[3];  //p5 already equal h[5]
		p6 += t12[4];

		// forced unroll
		Round_8_512(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 1);
		Round_8_512(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 3);
		Round_8_512(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 5);
		Round_8_512(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 7);
		Round_8_512(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 9);
		Round_8_512(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 11);
		Round_8_512(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 13);
		Round_8_512(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 15);
		Round_8_512(h, t, p0, p1, p2, p3, p4, p5, p6, p7, 17);

		outputHash[0 * threads + thread] = p0;
		outputHash[1 * threads + thread] = p1;
		outputHash[2 * threads + thread] = p2;
		outputHash[3 * threads + thread] = p3;
	}
}

__host__
void skein256_cpu_init(int thr_id, uint32_t threads)
{
	cuda_get_arch(thr_id);
}

__host__
void skein256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, int order)
{
	const uint32_t threadsperblock = 256;
	int dev_id = device_map[thr_id];

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	skein256_gpu_hash_32<<<grid, block>>>(threads, startNounce, d_outputHash);

	MyStreamSynchronize(NULL, order, thr_id);
}

