/*
 * Keccak 256
 *
 */

extern "C"
{
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_keccak.h"

#include "miner.h"
}

#include "cuda_helper.h"

static uint32_t *d_hash[MAX_GPUS];

extern void keccak256_cpu_init(int thr_id, uint32_t threads);
extern void keccak256_cpu_free(int thr_id);
extern void keccak256_setBlock_80(void *pdata,const void *ptarget);
extern uint32_t keccak256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order);

// CPU Hash
extern "C" void keccak256_hash(void *state, const void *input)
{
	uint32_t _ALIGN(64) hash[16];
	sph_keccak_context ctx_keccak;

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256 (&ctx_keccak, input, 80);
	sph_keccak256_close(&ctx_keccak, (void*) hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_keccak256(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 21); // 256*256*8*4
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x000f;

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

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], throughput * 64));
		keccak256_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	for (int k=0; k < 19; k++) {
		be32enc(&endiandata[k], pdata[k]);
	}

	keccak256_setBlock_80((void*)endiandata, ptarget);
	do {
		int order = 0;

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = keccak256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		if (work->nonces[0] != UINT32_MAX && bench_algo < 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];

			be32enc(&endiandata[19], work->nonces[0]);
			keccak256_hash(vhash, endiandata);

			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				pdata[19] = work->nonces[0] + 1;
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && (uint64_t) max_nonce > (uint64_t) pdata[19] + throughput);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_keccak256(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	keccak256_cpu_free(thr_id);

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
