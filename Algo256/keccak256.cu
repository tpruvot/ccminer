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

// SM5+ cuda
extern void keccak256_cpu_init(int thr_id);
extern void keccak256_cpu_free(int thr_id);
extern void keccak256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t* resNonces, const uint2 highTarget);
extern void keccak256_setBlock_80(uint64_t *endiandata);
extern void keccak256_setOutput(int thr_id);

// compat
extern void keccak256_sm3_init(int thr_id, uint32_t threads);
extern void keccak256_sm3_free(int thr_id);
extern void keccak256_sm3_setBlock_80(void *pdata, const void *ptarget);
extern void keccak256_sm3_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t* resNonces, int order);

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
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_keccak256(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];
	uint32_t throughput;
	uint32_t intensity = 23;
	if(!use_compat_kernels[thr_id]) {
		if (strstr(device_name[dev_id], "GTX 1070")) intensity = 25;
		if (strstr(device_name[dev_id], "GTX 1080")) intensity = 26;
	}
	throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x000f;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		cuda_get_arch(thr_id);
		use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);

		if(!use_compat_kernels[thr_id]) {
			keccak256_cpu_init(thr_id);
		} else {
			// really useful ?
			keccak256_sm3_init(thr_id, throughput);
		}

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		init[thr_id] = true;
	}

	for (int k=0; k < 19; k++) {
		be32enc(&endiandata[k], pdata[k]);
	}

	const uint2 highTarget = make_uint2(ptarget[6], ptarget[7]);
	if(use_compat_kernels[thr_id])
		keccak256_sm3_setBlock_80((void*)endiandata, ptarget);
	else {
		keccak256_setBlock_80((uint64_t*)endiandata);
		keccak256_setOutput(thr_id);
	}

	do {
		int order = 0;

		*hashes_done = pdata[19] - first_nonce + throughput;

		if(use_compat_kernels[thr_id])
			keccak256_sm3_hash_80(thr_id, throughput, pdata[19], work->nonces, order++);
		else {
			keccak256_cpu_hash_80(thr_id, throughput, pdata[19], work->nonces, highTarget);
		}

		if (work->nonces[0] != UINT32_MAX && bench_algo < 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];

			be32enc(&endiandata[19], work->nonces[0]);
			keccak256_hash(vhash, endiandata);

			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (!use_compat_kernels[thr_id] && work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					keccak256_hash(vhash, endiandata);
					if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
						work->valid_nonces++;
						bn_set_target_ratio(work, vhash, 1);
					}
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1;
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				//keccak256_setOutput(thr_id);
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
extern "C" void free_keccak256(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	if(!use_compat_kernels[thr_id])
		keccak256_cpu_free(thr_id);
	else {
		keccak256_sm3_free(thr_id);
	}

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
