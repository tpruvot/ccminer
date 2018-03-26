#include <cuda_runtime.h>
#include <string.h>
#include <miner.h>

#include "neoscrypt.h"

extern void neoscrypt_setBlockTarget(uint32_t* const data, uint32_t* const ptarget);

extern void neoscrypt_init(int thr_id, uint32_t threads);
extern void neoscrypt_free(int thr_id);
extern void neoscrypt_hash_k4(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNonces, bool stratum);

static bool init[MAX_GPUS] = { 0 };

int scanhash_neoscrypt(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	int dev_id = device_map[thr_id];
	int intensity = is_windows() ? 18 : 19;
	if (strstr(device_name[dev_id], "GTX 10")) intensity = 21; // >= 20 need more than 2GB
	if (strstr(device_name[dev_id], "TITAN")) intensity = 21;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	throughput = throughput / 32; /* set for max intensity ~= 20 */
	api_set_throughput(thr_id, throughput);

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

	if (!init[thr_id])
	{
		cudaDeviceSynchronize();
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaGetLastError(); // reset errors if device is not "reset"
		}
		if (device_sm[dev_id] <= 300) {
			gpulog(LOG_ERR, thr_id, "Sorry neoscrypt is not supported on SM 3.0 devices");
			proper_exit(EXIT_CODE_CUDA_ERROR);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g (+5), %u cuda threads", throughput2intensity(throughput), throughput);

		neoscrypt_init(thr_id, throughput);

		init[thr_id] = true;
	}

	if (have_stratum) {
		for (int k = 0; k < 20; k++)
			be32enc(&endiandata[k], pdata[k]);
	} else {
		for (int k = 0; k < 20; k++)
			endiandata[k] = pdata[k];
	}

	neoscrypt_setBlockTarget(endiandata,ptarget);

	do {
		memset(work->nonces, 0xff, sizeof(work->nonces));
		neoscrypt_hash_k4(thr_id, throughput, pdata[19], work->nonces, have_stratum);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];

			if (have_stratum) {
				be32enc(&endiandata[19], work->nonces[0]);
			} else {
				endiandata[19] = work->nonces[0];
			}
			neoscrypt((uchar*)vhash, (uchar*) endiandata, 0x80000620U);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				pdata[19] = work->nonces[0] + 1; // cursor
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "nonce %08x does not validate on CPU!", work->nonces[0]);
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
void free_neoscrypt(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	neoscrypt_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
