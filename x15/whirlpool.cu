/*
 * whirlpool routine
 */
extern "C" {
#include <sph/sph_whirlpool.h>
#include <miner.h>
}

#include <cuda_helper.h>

#define SM3_VARIANT

#ifdef SM3_VARIANT
static uint32_t *d_hash[MAX_GPUS];
extern void whirlpool512_init_sm3(int thr_id, uint32_t threads, int mode);
extern void whirlpool512_free_sm3(int thr_id);
extern void whirlpool512_setBlock_80_sm3(void *pdata, const void *ptarget);
extern void whirlpool512_hash_64_sm3(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void whirlpool512_hash_80_sm3(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern uint32_t whirlpool512_finalhash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
//#define _DEBUG
#define _DEBUG_PREFIX "whirl"
#include <cuda_debug.cuh>
#else
extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int mode);
extern void x15_whirlpool_cpu_free(int thr_id);
extern void whirlpool512_setBlock_80(void *pdata, const void *ptarget);
extern void whirlpool512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNonces, const uint64_t target);
#endif


// CPU Hash function
extern "C" void wcoinhash(void *state, const void *input)
{
	sph_whirlpool_context ctx_whirlpool;

	unsigned char hash[128]; // uint32_t hashA[16], hashB[16];
	#define hashB hash+64

	memset(hash, 0, sizeof hash);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, input, 80);
	sph_whirlpool1_close(&ctx_whirlpool, hash);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, hash, 64);
	sph_whirlpool1_close(&ctx_whirlpool, hashB);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, hashB, 64);
	sph_whirlpool1_close(&ctx_whirlpool, hash);

	sph_whirlpool1_init(&ctx_whirlpool);
	sph_whirlpool1(&ctx_whirlpool, hash, 64);
	sph_whirlpool1_close(&ctx_whirlpool, hash);

	memcpy(state, hash, 32);
}

void whirl_midstate(void *state, const void *input)
{
	sph_whirlpool_context ctx;

	sph_whirlpool1_init(&ctx);
	sph_whirlpool1(&ctx, input, 64);

	memcpy(state, ctx.state, 64);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_whirl(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(128) endiandata[20];
	uint32_t* pdata = work->data;
	uint32_t* ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	uint32_t throughput =  cuda_default_throughput(thr_id, 1U << 19); // 19=256*256*8;
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);
	if (init[thr_id]) throughput = max(throughput, 256); // shared mem requirement

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	if (!init[thr_id]) {
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
#ifdef SM3_VARIANT
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput));
		whirlpool512_init_sm3(thr_id, throughput, 1 /* old whirlpool */);
#else
		x15_whirlpool_cpu_init(thr_id, throughput, 1 /* old whirlpool */);
#endif
		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++) {
		be32enc(&endiandata[k], pdata[k]);
	}

#ifdef SM3_VARIANT
	whirlpool512_setBlock_80_sm3((void*)endiandata, ptarget);
#else
	whirlpool512_setBlock_80((void*)endiandata, ptarget);
#endif

	do {
#ifdef SM3_VARIANT
		int order = 1;
		whirlpool512_hash_80_sm3(thr_id, throughput, pdata[19], d_hash[thr_id]);
		TRACE64(" 80 :", d_hash);
		whirlpool512_hash_64_sm3(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE64(" 64 :", d_hash);
		whirlpool512_hash_64_sm3(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE64(" 64 :", d_hash);
		work->nonces[0] = whirlpool512_finalhash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
#else
		whirlpool512_cpu_hash_80(thr_id, throughput, pdata[19], work->nonces, *(uint64_t*)&ptarget[6]);
#endif
		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != UINT32_MAX && bench_algo < 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			wcoinhash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				pdata[19] = work->nonces[0] + 1; // cursor
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
extern "C" void free_whirl(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

#ifdef SM3_VARIANT
	cudaFree(d_hash[thr_id]);
	whirlpool512_free_sm3(thr_id);
#else
	x15_whirlpool_cpu_free(thr_id);
#endif
	init[thr_id] = false;

	cudaDeviceSynchronize();
}

