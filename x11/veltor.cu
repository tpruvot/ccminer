extern "C" {
#include "sph/sph_skein.h"
#include "sph/sph_shavite.h"
#include "sph/sph_shabal.h"
#include "sph/sph_streebog.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x11.h"

extern void skein512_cpu_setBlock_80(void *pdata);
extern void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int swap);
extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

// for SM3.x
extern void streebog_sm3_set_target(uint32_t* ptarget);
extern void streebog_sm3_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t* d_resNonce);

// for latest cards only
extern void skunk_cpu_init(int thr_id, uint32_t threads);
extern void skunk_streebog_set_target(uint32_t* ptarget);
extern void skunk_cuda_streebog(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t* d_resNonce);

#include <stdio.h>
#include <memory.h>

#define NBN 2
static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];

// veltor CPU Hash
extern "C" void veltorhash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[128] = { 0 };

	sph_skein512_context ctx_skein;
	sph_gost512_context ctx_gost;
	sph_shabal512_context ctx_shabal;
	sph_shavite512_context ctx_shavite;

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, (void*) hash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512(&ctx_shavite, (const void*) hash, 64);
	sph_shavite512_close(&ctx_shavite, (void*) hash);

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, (const void*) hash, 64);
	sph_shabal512_close(&ctx_shabal, (void*) hash);

	sph_gost512_init(&ctx_gost);
	sph_gost512(&ctx_gost, (const void*) hash, 64);
	sph_gost512_close(&ctx_gost, (void*) hash);

	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_veltor(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	int dev_id = device_map[thr_id];

	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int intensity = (device_sm[device_map[thr_id]] > 500) ? 20 : 18;
	if (strstr(device_name[dev_id], "GTX 10")) intensity = 21;
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

		x11_shavite512_cpu_init(thr_id, throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], NBN * sizeof(uint32_t)), -1);

		init[thr_id] = true;
	}

	uint32_t _ALIGN(64) h_resNonce[NBN];
	uint32_t _ALIGN(64) endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	skein512_cpu_setBlock_80(endiandata);

	cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
	if(use_compat_kernels[thr_id])
		streebog_sm3_set_target(ptarget);
	else
		skunk_streebog_set_target(ptarget);

	do {
		int order = 0;
		skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], 1); order++;
		x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		if(use_compat_kernels[thr_id])
			streebog_sm3_hash_64_final(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id]);
		else
			skunk_cuda_streebog(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id]);

		cudaMemcpy(h_resNonce, d_resNonce[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (h_resNonce[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];

			be32enc(&endiandata[19], startNounce + h_resNonce[0]);
			veltorhash(vhash, endiandata);
			if (vhash[7] <= Htarg && fulltest(vhash, ptarget))
			{
				work->nonces[0] = startNounce + h_resNonce[0];
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (h_resNonce[1] != UINT32_MAX)
				{
					uint32_t secNonce = work->nonces[1] = startNounce + h_resNonce[1];
					be32enc(&endiandata[19], secNonce);
					veltorhash(vhash, endiandata);
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
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", h_resNonce[0]);
				cudaMemset(d_resNonce[thr_id], 0xff, NBN*sizeof(uint32_t));
				pdata[19] = startNounce + h_resNonce[0] + 1;
				continue;
			}
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && (uint64_t) max_nonce > (uint64_t) pdata[19] + throughput);

	*hashes_done = pdata[19] - first_nonce;

	return 0;
}

// cleanup
extern "C" void free_veltor(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_resNonce[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
