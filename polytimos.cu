/*
 * Polytimos algorithm
 */
extern "C"
{
#include "sph/sph_skein.h"
#include "sph/sph_shabal.h"
#include "sph/sph_echo.h"
#include "sph/sph_luffa.h"
#include "sph/sph_fugue.h"
#include "sph/sph_streebog.h"
}

#include "miner.h"

#include "cuda_helper.h"
#include "x11/cuda_x11.h"

static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_resNonce[MAX_GPUS];

extern void skein512_cpu_setBlock_80(void *pdata);
extern void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int swap);
extern void x14_shabal512_cpu_init(int thr_id, uint32_t threads);
extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x13_fugue512_cpu_free(int thr_id);
extern void streebog_sm3_set_target(uint32_t* ptarget);
extern void streebog_sm3_hash_64_final(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t* d_resNonce);
extern void skunk_streebog_set_target(uint32_t* ptarget);
extern void skunk_cuda_streebog(int thr_id, uint32_t threads, uint32_t *d_hash, uint32_t* d_resNonce);

// CPU Hash
extern "C" void polytimos_hash(void *output, const void *input)
{
	sph_skein512_context ctx_skein;
	sph_shabal512_context ctx_shabal;
	sph_echo512_context ctx_echo;
	sph_luffa512_context ctx_luffa;
	sph_fugue512_context ctx_fugue;
	sph_gost512_context ctx_gost;

	uint32_t _ALIGN(128) hash[16];
	memset(hash, 0, sizeof hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, (void*) hash);

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, hash, 64);
	sph_shabal512_close(&ctx_shabal, hash);

	sph_echo512_init(&ctx_echo);
	sph_echo512(&ctx_echo, hash, 64);
	sph_echo512_close(&ctx_echo, hash);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, hash, 64);
	sph_luffa512_close(&ctx_luffa, hash);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, hash, 64);
	sph_fugue512_close(&ctx_fugue, hash);

	sph_gost512_init(&ctx_gost);
	sph_gost512(&ctx_gost, (const void*) hash, 64);
	sph_gost512_close(&ctx_gost, (void*) hash);

	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };

extern "C" int scanhash_polytimos(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	int dev_id = device_map[thr_id];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 20 : 19;
	uint32_t throughput =  cuda_default_throughput(thr_id, 1 << intensity); // 19=256*256*8;
	//if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x000f;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);
		use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);

		quark_skein512_cpu_init(thr_id, throughput);
		x14_shabal512_cpu_init(thr_id, throughput);
		x11_echo512_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput), 0);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonce[thr_id], 2 * sizeof(uint32_t)), -1);

		init[thr_id] = true;
	}


	uint32_t _ALIGN(64) h_resNonce[2];
	uint32_t _ALIGN(64) endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);


	cudaMemset(d_resNonce[thr_id], 0xff, 2*sizeof(uint32_t));
	skein512_cpu_setBlock_80(endiandata);
	if (use_compat_kernels[thr_id]) {
		streebog_sm3_set_target(ptarget);
	} else {
		skunk_streebog_set_target(ptarget);
	}

	do {
		int order = 0;

		skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		if (use_compat_kernels[thr_id]) {
			streebog_sm3_hash_64_final(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id]);
		} else {
			skunk_cuda_streebog(thr_id, throughput, d_hash[thr_id], d_resNonce[thr_id]);
		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		cudaMemcpy(h_resNonce, d_resNonce[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		CUDA_LOG_ERROR();

		if (h_resNonce[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			const uint32_t startNounce = pdata[19];
			uint32_t _ALIGN(64) vhash[8];

			be32enc(&endiandata[19], startNounce + h_resNonce[0]);
			polytimos_hash(vhash, endiandata);
			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work->nonces[0] = startNounce + h_resNonce[0];
				work_set_target_ratio(work, vhash);
				if (h_resNonce[1] != UINT32_MAX) {
					uint32_t secNonce = work->nonces[1] = startNounce + h_resNonce[1];
					be32enc(&endiandata[19], secNonce);
					polytimos_hash(vhash, endiandata);
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
				cudaMemset(d_resNonce[thr_id], 0xff, 2*sizeof(uint32_t));
				pdata[19] = startNounce + h_resNonce[0] + 1;
				continue;
			}
		}

		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;

	CUDA_LOG_ERROR();

	return 0;
}

// cleanup
extern "C" void free_polytimos(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	x13_fugue512_cpu_free(thr_id);
	cudaFree(d_resNonce[thr_id]);

	CUDA_LOG_ERROR();

	cudaDeviceSynchronize();
	init[thr_id] = false;
}
