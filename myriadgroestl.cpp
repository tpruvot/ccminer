#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <openssl/sha.h>

#include "sph/sph_groestl.h"

#include "miner.h"

void myriadgroestl_cpu_init(int thr_id, uint32_t threads);
void myriadgroestl_cpu_free(int thr_id);
void myriadgroestl_cpu_setBlock(int thr_id, void *data, uint32_t *target);
void myriadgroestl_cpu_hash(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resNonces);

void myriadhash(void *state, const void *input)
{
	uint32_t _ALIGN(64) hash[16];
	sph_groestl512_context ctx_groestl;
	SHA256_CTX sha256;

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512(&ctx_groestl, input, 80);
	sph_groestl512_close(&ctx_groestl, hash);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256,(unsigned char *)hash, 64);
	SHA256_Final((unsigned char *)hash, &sha256);

	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_myriad(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[32];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	uint32_t start_nonce = pdata[19];
	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] >= 600) ? 20 : 18;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - start_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x0000ff;

	// init
	if(!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		myriadgroestl_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	myriadgroestl_cpu_setBlock(thr_id, endiandata, ptarget);

	do {
		memset(work->nonces, 0xff, sizeof(work->nonces));

		// GPU
		myriadgroestl_cpu_hash(thr_id, throughput, pdata[19], work->nonces);

		*hashes_done = pdata[19] - start_nonce + throughput;

		if (work->nonces[0] < UINT32_MAX && bench_algo < 0)
		{
			uint32_t _ALIGN(64) vhash[8];
			endiandata[19] = swab32(work->nonces[0]);
			myriadhash(vhash, endiandata);
			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != UINT32_MAX) {
					endiandata[19] = swab32(work->nonces[1]);
					myriadhash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces = 2;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > ptarget[7]) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && (uint64_t) max_nonce > (uint64_t) pdata[19] + throughput);

	*hashes_done = pdata[19] - start_nonce;

	return 0;
}

// cleanup
void free_myriad(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	myriadgroestl_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
