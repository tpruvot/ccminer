/**
 * Blake-256 Decred 180-Bytes input Cuda Kernel (Tested on SM 5/5.2)
 *
 * Tanguy Pruvot - Feb 2016
 */

#include <stdint.h>
#include <memory.h>

#include <miner.h>

extern "C" {
#include <sph/sph_blake.h>
}

/* threads per block */
#define TPB 256

/* hash by cpu with blake 256 */
extern "C" void decred_hash(void *output, const void *input)
{
	sph_blake256_context ctx;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 180);
	sph_blake256_close(&ctx, output);
}

#include <cuda_helper.h>

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, b) x
#endif

__constant__ uint32_t _ALIGN(4) d_data[24];

/* 16 adapters max */
static uint32_t *d_resNonce[MAX_GPUS];
static uint32_t *h_resNonce[MAX_GPUS];

/* max count of found nonces in one call */
#define NBN 2
#if NBN > 1
static uint32_t extra_results[NBN] = { UINT32_MAX };
#endif

/* ############################################################################################################################### */

#define GSPREC(a,b,c,d,x,y) { \
	v[a] += (m[x] ^ c_u256[y]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a], 0, 0x1032); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 12); \
	v[a] += (m[y] ^ c_u256[x]) + v[b]; \
	v[d] = __byte_perm(v[d] ^ v[a], 0, 0x0321); \
	v[c] += v[d]; \
	v[b] = SPH_ROTR32(v[b] ^ v[c], 7); \
}

__device__ __forceinline__
void blake256_compress_14(uint32_t *h, const uint32_t nonce, const uint32_t T0)
{
	uint32_t v[16];

	#pragma unroll 8
	for(uint32_t i = 0; i < 8; i++)
		v[i] = h[i];

	const uint32_t c_u256[16] = {
		0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344,
		0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
		0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C,
		0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917
	};

	v[ 8] = c_u256[0];
	v[ 9] = c_u256[1];
	v[10] = c_u256[2];
	v[11] = c_u256[3];

	v[12] = c_u256[4] ^ T0;
	v[13] = c_u256[5] ^ T0;
	v[14] = c_u256[6];
	v[15] = c_u256[7];

	uint32_t m[16];

	m[0] = d_data[8];
	m[1] = d_data[9];
	m[2] = d_data[10];
	m[3] = nonce;

	#pragma unroll
	for (uint32_t i = 4; i < 16; i++) {
		m[i] = d_data[i+8U];
	}

	// round 1
	GSPREC(0, 4, 0x8, 0xC, 0,  1);
	GSPREC(1, 5, 0x9, 0xD, 2,  3);
	GSPREC(2, 6, 0xA, 0xE, 4,  5);
	GSPREC(3, 7, 0xB, 0xF, 6,  7);
	GSPREC(0, 5, 0xA, 0xF, 8,  9);
	GSPREC(1, 6, 0xB, 0xC, 10, 11);
	GSPREC(2, 7, 0x8, 0xD, 12, 13);
	GSPREC(3, 4, 0x9, 0xE, 14, 15);
	// round 2
	GSPREC(0, 4, 0x8, 0xC, 14, 10);
	GSPREC(1, 5, 0x9, 0xD, 4,  8);
	GSPREC(2, 6, 0xA, 0xE, 9,  15);
	GSPREC(3, 7, 0xB, 0xF, 13, 6);
	GSPREC(0, 5, 0xA, 0xF, 1,  12);
	GSPREC(1, 6, 0xB, 0xC, 0,  2);
	GSPREC(2, 7, 0x8, 0xD, 11, 7);
	GSPREC(3, 4, 0x9, 0xE, 5,  3);
	// round 3
	GSPREC(0, 4, 0x8, 0xC, 11, 8);
	GSPREC(1, 5, 0x9, 0xD, 12, 0);
	GSPREC(2, 6, 0xA, 0xE, 5,  2);
	GSPREC(3, 7, 0xB, 0xF, 15, 13);
	GSPREC(0, 5, 0xA, 0xF, 10, 14);
	GSPREC(1, 6, 0xB, 0xC, 3,  6);
	GSPREC(2, 7, 0x8, 0xD, 7,  1);
	GSPREC(3, 4, 0x9, 0xE, 9,  4);
	// round 4
	GSPREC(0, 4, 0x8, 0xC, 7,  9);
	GSPREC(1, 5, 0x9, 0xD, 3,  1);
	GSPREC(2, 6, 0xA, 0xE, 13, 12);
	GSPREC(3, 7, 0xB, 0xF, 11, 14);
	GSPREC(0, 5, 0xA, 0xF, 2,  6);
	GSPREC(1, 6, 0xB, 0xC, 5,  10);
	GSPREC(2, 7, 0x8, 0xD, 4,  0);
	GSPREC(3, 4, 0x9, 0xE, 15, 8);
	// round 5
	GSPREC(0, 4, 0x8, 0xC, 9,  0);
	GSPREC(1, 5, 0x9, 0xD, 5,  7);
	GSPREC(2, 6, 0xA, 0xE, 2,  4);
	GSPREC(3, 7, 0xB, 0xF, 10, 15);
	GSPREC(0, 5, 0xA, 0xF, 14, 1);
	GSPREC(1, 6, 0xB, 0xC, 11, 12);
	GSPREC(2, 7, 0x8, 0xD, 6,  8);
	GSPREC(3, 4, 0x9, 0xE, 3,  13);
	// round 6
	GSPREC(0, 4, 0x8, 0xC, 2, 12);
	GSPREC(1, 5, 0x9, 0xD, 6, 10);
	GSPREC(2, 6, 0xA, 0xE, 0, 11);
	GSPREC(3, 7, 0xB, 0xF, 8, 3);
	GSPREC(0, 5, 0xA, 0xF, 4, 13);
	GSPREC(1, 6, 0xB, 0xC, 7, 5);
	GSPREC(2, 7, 0x8, 0xD, 15,14);
	GSPREC(3, 4, 0x9, 0xE, 1, 9);
	// round 7
	GSPREC(0, 4, 0x8, 0xC, 12, 5);
	GSPREC(1, 5, 0x9, 0xD, 1, 15);
	GSPREC(2, 6, 0xA, 0xE, 14,13);
	GSPREC(3, 7, 0xB, 0xF, 4, 10);
	GSPREC(0, 5, 0xA, 0xF, 0,  7);
	GSPREC(1, 6, 0xB, 0xC, 6,  3);
	GSPREC(2, 7, 0x8, 0xD, 9,  2);
	GSPREC(3, 4, 0x9, 0xE, 8, 11);
	// round 8
	GSPREC(0, 4, 0x8, 0xC, 13,11);
	GSPREC(1, 5, 0x9, 0xD, 7, 14);
	GSPREC(2, 6, 0xA, 0xE, 12, 1);
	GSPREC(3, 7, 0xB, 0xF, 3,  9);
	GSPREC(0, 5, 0xA, 0xF, 5,  0);
	GSPREC(1, 6, 0xB, 0xC, 15, 4);
	GSPREC(2, 7, 0x8, 0xD, 8,  6);
	GSPREC(3, 4, 0x9, 0xE, 2, 10);
	// round 9
	GSPREC(0, 4, 0x8, 0xC, 6, 15);
	GSPREC(1, 5, 0x9, 0xD, 14, 9);
	GSPREC(2, 6, 0xA, 0xE, 11, 3);
	GSPREC(3, 7, 0xB, 0xF, 0,  8);
	GSPREC(0, 5, 0xA, 0xF, 12, 2);
	GSPREC(1, 6, 0xB, 0xC, 13, 7);
	GSPREC(2, 7, 0x8, 0xD, 1,  4);
	GSPREC(3, 4, 0x9, 0xE, 10, 5);
	// round 10
	GSPREC(0, 4, 0x8, 0xC, 10, 2);
	GSPREC(1, 5, 0x9, 0xD, 8,  4);
	GSPREC(2, 6, 0xA, 0xE, 7,  6);
	GSPREC(3, 7, 0xB, 0xF, 1,  5);
	GSPREC(0, 5, 0xA, 0xF, 15,11);
	GSPREC(1, 6, 0xB, 0xC, 9, 14);
	GSPREC(2, 7, 0x8, 0xD, 3, 12);
	GSPREC(3, 4, 0x9, 0xE, 13, 0);
	// round 11
	GSPREC(0, 4, 0x8, 0xC, 0,  1);
	GSPREC(1, 5, 0x9, 0xD, 2,  3);
	GSPREC(2, 6, 0xA, 0xE, 4,  5);
	GSPREC(3, 7, 0xB, 0xF, 6,  7);
	GSPREC(0, 5, 0xA, 0xF, 8,  9);
	GSPREC(1, 6, 0xB, 0xC, 10,11);
	GSPREC(2, 7, 0x8, 0xD, 12,13);
	GSPREC(3, 4, 0x9, 0xE, 14,15);
	// round 12
	GSPREC(0, 4, 0x8, 0xC, 14,10);
	GSPREC(1, 5, 0x9, 0xD, 4,  8);
	GSPREC(2, 6, 0xA, 0xE, 9, 15);
	GSPREC(3, 7, 0xB, 0xF, 13, 6);
	GSPREC(0, 5, 0xA, 0xF, 1, 12);
	GSPREC(1, 6, 0xB, 0xC, 0,  2);
	GSPREC(2, 7, 0x8, 0xD, 11, 7);
	GSPREC(3, 4, 0x9, 0xE, 5,  3);
	// round 13
	GSPREC(0, 4, 0x8, 0xC, 11, 8);
	GSPREC(1, 5, 0x9, 0xD, 12, 0);
	GSPREC(2, 6, 0xA, 0xE, 5,  2);
	GSPREC(3, 7, 0xB, 0xF, 15,13);
	GSPREC(0, 5, 0xA, 0xF, 10,14);
	GSPREC(1, 6, 0xB, 0xC, 3,  6);
	GSPREC(2, 7, 0x8, 0xD, 7,  1);
	GSPREC(3, 4, 0x9, 0xE, 9,  4);
	// round 14
	GSPREC(0, 4, 0x8, 0xC, 7,  9);
	GSPREC(1, 5, 0x9, 0xD, 3,  1);
	GSPREC(2, 6, 0xA, 0xE, 13,12);
	GSPREC(3, 7, 0xB, 0xF, 11,14);
	GSPREC(0, 5, 0xA, 0xF, 2,  6);
	GSPREC(2, 7, 0x8, 0xD, 4,  0);

	h[7] ^= v[7] ^ v[15];

	if (h[7] == 0) {
		GSPREC(1, 6, 0xB, 0xC, 5, 10);
		GSPREC(3, 4, 0x9, 0xE, 15, 8);
		h[6] ^= v[6] ^ v[14];
	}
}

/* ############################################################################################################################### */

__global__
void blake256_gpu_hash_nonce(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint64_t highTarget)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
//	if (thread < threads)
	{
		const uint32_t nonce = startNonce + thread;
		uint32_t h[8];

		#pragma unroll
		for(int i=0; i < 8; i++) {
			h[i] = d_data[i];
		}

		// ------ Close: Last 52/64 bytes ------

		blake256_compress_14(h, nonce, (180U*8U));

		if (h[7] == 0 && cuda_swab32(h[6]) <= highTarget) {
#if NBN == 2
			if (resNonce[0] != UINT32_MAX)
				resNonce[1] = nonce;
			else
				resNonce[0] = nonce;
#else
			resNonce[0] = nonce;
#endif
		}
	}
}

__host__
static uint32_t decred_cpu_hash_nonce(const int thr_id, const uint32_t threads, const uint32_t startNonce, const uint64_t highTarget)
{
	uint32_t result = UINT32_MAX;

	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);

	/* Check error on Ctrl+C or kill to prevent segfaults on exit */
	if (cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
		return result;

	blake256_gpu_hash_nonce <<<grid, block>>> (threads, startNonce, d_resNonce[thr_id], highTarget);
	cudaThreadSynchronize();

	if (cudaSuccess == cudaMemcpy(h_resNonce[thr_id], d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		result = h_resNonce[thr_id][0];
#if NBN > 1
		for (int n=0; n < (NBN-1); n++)
			extra_results[n] = h_resNonce[thr_id][n+1];
#endif
	}
	return result;
}

__host__
static void decred_midstate_128(uint32_t *output, const uint32_t *input)
{
	sph_blake256_context ctx;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, 128);

	memcpy(output, (void*)ctx.H, 32);
}

__host__
void decred_cpu_setBlock_52(uint32_t *penddata, const uint32_t *midstate, const uint32_t *ptarget)
{
	uint32_t _ALIGN(64) data[24];
	memcpy(data, midstate, 32);
	// pre swab32
	for (int i=0; i<13; i++)
		data[8+i] = swab32(penddata[i]);
	data[21] = 0x80000001;
	data[22] = 0;
	data[23] = 0x000005a0;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_data, data, 32 + 64, 0, cudaMemcpyHostToDevice));
}

/* ############################################################################################################################### */

static bool init[MAX_GPUS] = { 0 };

// nonce position is different in decred
#define DCR_NONCE_OFT32 35

extern "C" int scanhash_decred(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[48];
	uint32_t _ALIGN(64) midstate[8];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t *pnonce = &pdata[DCR_NONCE_OFT32];

	const uint32_t first_nonce = *pnonce;
	uint64_t targetHigh = ((uint64_t*)ptarget)[3];

	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 29 : 25;
	if (device_sm[dev_id] < 350) intensity = 22;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	int rc = 0;

	if (opt_benchmark) {
		targetHigh = 0x1ULL << 32;
		ptarget[6] = swab32(0xff);
	}

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)), -1);
		CUDA_CALL_OR_RET_X(cudaMallocHost(&h_resNonce[thr_id], NBN * sizeof(uint32_t)), -1);
		init[thr_id] = true;
	}

	memcpy(endiandata, pdata, 180);
	decred_midstate_128(midstate, endiandata);
	decred_cpu_setBlock_52(&pdata[32], midstate, ptarget);

	do {
		// GPU HASH
		uint32_t foundNonce = decred_cpu_hash_nonce(thr_id, throughput, (*pnonce), targetHigh);

		if (foundNonce != UINT32_MAX)
		{
			uint32_t vhashcpu[8];
			uint32_t Htarg = ptarget[6];

			be32enc(&endiandata[DCR_NONCE_OFT32], foundNonce);
			decred_hash(vhashcpu, endiandata);

			if (vhashcpu[6] <= Htarg && fulltest(vhashcpu, ptarget))
			{
				rc = 1;
				work_set_target_ratio(work, vhashcpu);
				*hashes_done = (*pnonce) - first_nonce + throughput;
				work->nonces[0] = swab32(foundNonce);
#if NBN > 1
				if (extra_results[0] != UINT32_MAX) {
					be32enc(&endiandata[DCR_NONCE_OFT32], extra_results[0]);
					decred_hash(vhashcpu, endiandata);
					if (vhashcpu[6] <= Htarg && fulltest(vhashcpu, ptarget)) {
						work->nonces[1] = swab32(extra_results[0]);
						if (bn_hash_target_ratio(vhashcpu, ptarget) > work->shareratio) {
							work_set_target_ratio(work, vhashcpu);
							xchg(work->nonces[1], work->nonces[0]);
						}
						rc = 2;
					}
					extra_results[0] = UINT32_MAX;
				}
#endif
				*pnonce = work->nonces[0];
				return rc;
			}
			else if (opt_debug) {
				applog_hash(ptarget);
				applog_compare_hash(vhashcpu, ptarget);
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", foundNonce);
			}
		}

		*pnonce += throughput;

	} while (!work_restart[thr_id].restart && max_nonce > (uint64_t)throughput + (*pnonce));

	*hashes_done = (*pnonce) - first_nonce;
	return rc;
}

// cleanup
extern "C" void free_decred(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFreeHost(h_resNonce[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}

