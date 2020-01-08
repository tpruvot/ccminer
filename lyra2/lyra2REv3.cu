extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_cubehash.h"
#include "lyra2/Lyra2.h"
}

#include <miner.h>
#include <cuda_helper.h>

static uint64_t *d_hash[MAX_GPUS];
static uint64_t* d_matrix[MAX_GPUS];

extern void blake256_cpu_init(int thr_id, uint32_t threads);
extern void blake256_cpu_setBlock_80(uint32_t *pdata);
extern void blake256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash, int order);

extern void cubehash256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, int order);

extern void lyra2v3_setTarget(const void *pTargetIn);
extern void lyra2v3_cpu_init(int thr_id, uint32_t threads, uint64_t* d_matrix);
extern void lyra2v3_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);

extern void lyra2v3_cpu_hash_32_targ(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *resultnonces);

extern void bmw256_setTarget(const void *ptarget);
extern void bmw256_cpu_init(int thr_id, uint32_t threads);
extern void bmw256_cpu_free(int thr_id);
extern void bmw256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *resultnonces);

extern "C" void lyra2v3_hash(void *state, const void *input)
{
	uint32_t hashA[8], hashB[8];

	sph_blake256_context      ctx_blake;
	sph_cubehash256_context   ctx_cube;
	sph_bmw256_context        ctx_bmw;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	LYRA2_3(hashB, 32, hashA, 32, hashA, 32, 1, 4, 4);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashB, 32);
	sph_cubehash256_close(&ctx_cube, hashA);

	LYRA2_3(hashB, 32, hashA, 32, hashA, 32, 1, 4, 4);

	sph_bmw256_init(&ctx_bmw);
	sph_bmw256(&ctx_bmw, hashB, 32);
	sph_bmw256_close(&ctx_bmw, hashA);

	memcpy(state, hashA, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_lyra2v3(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] < 500) ? 18 : is_windows() ? 19 : 20;
	if (strstr(device_name[dev_id], "GTX 1")) intensity = 20;
	if (strstr(device_name[dev_id], "RTX 20")) intensity = 20;
	uint32_t throughput = cuda_default_throughput(dev_id, 1UL << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x000f;


	if (!init[thr_id])
	{
		size_t matrix_sz = 16 * sizeof(uint64_t) * 4 * 3;
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		blake256_cpu_init(thr_id, throughput);
		bmw256_cpu_init(thr_id, throughput);

		cuda_get_arch(thr_id); // cuda_arch[] also used in cubehash256

		// SM 3 implentation requires a bit more memory
		if (device_sm[dev_id] < 500 || cuda_arch[dev_id] < 500)
			matrix_sz = 16 * sizeof(uint64_t) * 4 * 4;

		CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], matrix_sz * throughput));
		lyra2v3_cpu_init(thr_id, throughput, d_matrix[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t)32 * throughput));

		api_set_throughput(thr_id, throughput);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake256_cpu_setBlock_80(pdata);
	bmw256_setTarget(ptarget);

	do {
		int order = 0;

		blake256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		lyra2v3_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		cubehash256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		lyra2v3_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		memset(work->nonces, 0, sizeof(work->nonces));
		bmw256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], work->nonces);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			lyra2v3_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					lyra2v3_hash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
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
				gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = work->nonces[0] + 1;
				continue;
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && !abort_flag);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_lyra2v3(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_matrix[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
