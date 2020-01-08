#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>

#include "sph/sph_groestl.h"
#include "cuda_groestlcoin.h"

#include "miner.h"

// CPU hash
void groestlhash(void *state, const void *input)
{
	uint32_t _ALIGN(64) hash[16];
	sph_groestl512_context ctx_groestl;

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, input, 80);
	sph_groestl512_close(&ctx_groestl, hash);

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, hash, 64);
	sph_groestl512_close(&ctx_groestl, hash);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_groestlcoin(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[32];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t start_nonce = pdata[19];
	uint32_t throughput = cuda_default_throughput(thr_id, 1 << 19); // 256*256*8
	if (init[thr_id]) throughput = min(throughput, max_nonce - start_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x001f;

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

		CUDA_LOG_ERROR();
		groestlcoin_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	groestlcoin_cpu_setBlock(thr_id, endiandata, (void*)ptarget);

	do {
		memset(work->nonces, 0xff, sizeof(work->nonces));

		*hashes_done = pdata[19] - start_nonce + throughput;

		// GPU hash
		groestlcoin_cpu_hash(thr_id, throughput, pdata[19], &work->nonces[0]);

		if (work->nonces[0] < UINT32_MAX && bench_algo < 0)
		{
			uint32_t _ALIGN(64) vhash[8];
			endiandata[19] = swab32(work->nonces[0]);
			groestlhash(vhash, endiandata);

			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				pdata[19] = work->nonces[0] + 1; // cursor
				return work->valid_nonces;
			} else if (vhash[7] > ptarget[7]) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
			}
		}

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - start_nonce;
	return 0;
}

// cleanup
void free_groestlcoin(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	groestlcoin_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
