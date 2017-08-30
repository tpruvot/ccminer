/**
 * Skunk Algo for Signatum
 * (skein, cube, fugue, gost streebog)
 *
 * tpruvot@github 08 2017 - GPLv3
 */
extern "C" {
#include "sph/sph_skein.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_fugue.h"
#include "sph/sph_streebog.h"
}

#include "miner.h"
#include "cuda_helper.h"

//#define WANT_COMPAT_KERNEL

// compatibility kernels
extern void skein512_cpu_setBlock_80(void *pdata);
extern void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int swap);
extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x13_fugue512_cpu_free(int thr_id);
extern void streebog_sm3_set_target(uint32_t* ptarget);
extern void streebog_sm3_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t* d_resNonce);

// krnlx merged kernel (for high-end cards only)
extern void skunk_cpu_init(int thr_id, uint32_t threads);
extern void skunk_streebog_set_target(uint32_t* ptarget);
extern void skunk_setBlock_80(int thr_id, void *pdata);
extern void skunk_cuda_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void skunk_cuda_streebog(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t* d_resNonce);

#include <stdio.h>
#include <memory.h>

#define NBN 2
static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];

// CPU Hash
extern "C" void skunk_hash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[128] = { 0 };

	sph_skein512_context ctx_skein;
	sph_cubehash512_context ctx_cubehash;
	sph_fugue512_context ctx_fugue;
	sph_gost512_context ctx_gost;

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, (void*) hash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, (const void*) hash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*) hash);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, (const void*) hash, 64);
	sph_fugue512_close(&ctx_fugue, (void*) hash);

	sph_gost512_init(&ctx_gost);
	sph_gost512(&ctx_gost, (const void*) hash, 64);
	sph_gost512_close(&ctx_gost, (void*) hash);

	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_skunk(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	int dev_id = device_map[thr_id];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int intensity = (device_sm[device_map[thr_id]] > 500) ? 18 : 17;
	if (strstr(device_name[dev_id], "GTX 10")) intensity = 20;
	if (strstr(device_name[dev_id], "GTX 1080")) intensity = 21;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	//if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0xf;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		skunk_cpu_init(thr_id, throughput);
		use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);
		if (use_compat_kernels[thr_id]) x13_fugue512_cpu_init(thr_id, throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)), -1);

		init[thr_id] = true;
	}

	uint32_t _ALIGN(64) h_resNonce[NBN];
	uint32_t _ALIGN(64) endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
	if (use_compat_kernels[thr_id]) {
		skein512_cpu_setBlock_80(endiandata);
		streebog_sm3_set_target(ptarget);
	} else {
		skunk_setBlock_80(thr_id, endiandata);
		skunk_streebog_set_target(ptarget);
	}

	do {
		int order = 0;
		if (use_compat_kernels[thr_id]) {
			skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
			x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			streebog_sm3_hash_64_final(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id]);
		} else {
			skunk_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]);
			skunk_cuda_streebog(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id]);
		}
		cudaMemcpy(h_resNonce, d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (h_resNonce[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];

			be32enc(&endiandata[19], startNounce + h_resNonce[0]);
			skunk_hash(vhash, endiandata);
			if (vhash[7] <= Htarg && fulltest(vhash, ptarget))
			{
				work->nonces[0] = startNounce + h_resNonce[0];
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (h_resNonce[1] != UINT32_MAX)
				{
					uint32_t secNonce = work->nonces[1] = startNounce + h_resNonce[1];
					be32enc(&endiandata[19], secNonce);
					skunk_hash(vhash, endiandata);
					work->nonces[1] = secNonce;
					if (bn_hash_target_ratio(vhash, ptarget) > work->shareratio[0]) {
						work_set_target_ratio(work, vhash);
						xchg(work->nonces[1], work->nonces[0]);
					} else {
						bn_set_target_ratio(work, vhash, work->valid_nonces);
					}
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
				gpulog(LOG_WARNING, thr_id, "result does not validate on CPU!");
				pdata[19] = startNounce + h_resNonce[0] + 1;
				continue;
			}
		}
		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;

	return 0;
}

// cleanup
extern "C" void free_skunk(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	if (use_compat_kernels[thr_id])
		x13_fugue512_cpu_free(thr_id);

	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
