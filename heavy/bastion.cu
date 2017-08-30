/**
 * bastion cuda implemention tpruvot@github 2017
 */

#include <stdio.h>
#include <string.h>
//#include <openssl/sha.h>
#include <stdint.h>
#include <miner.h>
#include <cuda_helper.h>

static uint32_t *d_hash[MAX_GPUS];
static uint32_t* d_hash_br1[MAX_GPUS];
static uint32_t* d_hash_br2[MAX_GPUS];

extern void quark_skein512_cpu_init(int thr_id, uint32_t threads);
extern void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_luffa512_cpu_init(int thr_id, uint32_t threads);
extern void x11_luffa512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x13_hamsi512_cpu_init(int thr_id, uint32_t threads);
extern void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x13_fugue512_cpu_init(int thr_id, uint32_t threads);
extern void x13_fugue512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x13_fugue512_cpu_free(int thr_id);

extern void x14_shabal512_cpu_init(int thr_id, uint32_t threads);
extern void x14_shabal512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x15_whirlpool_cpu_init(int thr_id, uint32_t threads, int mode);
extern void x15_whirlpool_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void x15_whirlpool_cpu_free(int thr_id);

extern void x11_echo512_cpu_init(int thr_id, uint32_t threads);
extern void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void bastion_init(const int thr_id, const uint32_t threads);
extern void bastion_free(const int thr_id);

extern uint32_t bastion_filter2(const int thr_id, const uint32_t threads, const uint32_t *inpHashes, uint32_t* d_hash1, uint32_t* d_hash2);
extern void bastion_merge2(const int thr_id, const uint32_t threads, uint32_t *outpHashes, uint32_t* d_hash1, uint32_t* d_hash2);

extern void hefty_cpu_hash(int thr_id, uint32_t threads, int startNounce);
extern void hefty_cpu_setBlock(int thr_id, uint32_t threads, void *data, int len);
extern void hefty_cpu_init(int thr_id, uint32_t threads);
extern void hefty_cpu_free(int thr_id);
extern void hefty_copy_hashes(int thr_id, uint32_t threads, uint32_t* d_outputhash);

#define TRACE(algo) {}

static bool init[MAX_GPUS] = { 0 };

int scanhash_bastion(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	// CUDA will process thousands of threads.
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 20);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

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


		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash_br1[thr_id], (size_t) 64 * throughput));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash_br2[thr_id], (size_t) 64 * throughput));

		bastion_init(thr_id, throughput);
		hefty_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput);

		quark_skein512_cpu_init(thr_id, throughput);
		x13_hamsi512_cpu_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		x14_shabal512_cpu_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);
		x11_echo512_cpu_init(thr_id, throughput);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	hefty_cpu_setBlock(thr_id, throughput, endiandata, 80);

	cuda_check_cpu_setTarget(ptarget);

	do {
		uint32_t branchNonces;
		int order = 0;

		// hefty
		hefty_cpu_hash(thr_id, throughput, pdata[19]);
		hefty_copy_hashes(thr_id, throughput, d_hash[thr_id]);
		TRACE("hefty  :");

		x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("luffa  :");

		// fugue or skein
		branchNonces = bastion_filter2(thr_id, throughput, d_hash[thr_id], d_hash_br1[thr_id], d_hash_br2[thr_id]);
		x13_fugue512_cpu_hash_64(thr_id, branchNonces, pdata[19], NULL, d_hash_br1[thr_id], order++);
		quark_skein512_cpu_hash_64(thr_id, throughput-branchNonces, pdata[19], NULL, d_hash_br2[thr_id], order++);
		bastion_merge2(thr_id, throughput, d_hash[thr_id], d_hash_br1[thr_id], d_hash_br2[thr_id]);
		TRACE("perm1  :");

		x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("whirl  :");
		x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		// echo or luffa
		branchNonces = bastion_filter2(thr_id, throughput, d_hash[thr_id], d_hash_br1[thr_id], d_hash_br2[thr_id]);
		x11_echo512_cpu_hash_64(thr_id, branchNonces, pdata[19], NULL, d_hash_br1[thr_id], order++);
		x11_luffa512_cpu_hash_64(thr_id, throughput-branchNonces, pdata[19], NULL, d_hash_br2[thr_id], order++);
		bastion_merge2(thr_id, throughput, d_hash[thr_id], d_hash_br1[thr_id], d_hash_br2[thr_id]);
		TRACE("perm2  :");

		x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		// shabal or whirlpool
		branchNonces = bastion_filter2(thr_id, throughput, d_hash[thr_id], d_hash_br1[thr_id], d_hash_br2[thr_id]);
		x14_shabal512_cpu_hash_64(thr_id, branchNonces, pdata[19], NULL, d_hash_br1[thr_id], order++);
		x15_whirlpool_cpu_hash_64(thr_id, throughput-branchNonces, pdata[19], NULL, d_hash_br2[thr_id], order++);
		bastion_merge2(thr_id, throughput, d_hash[thr_id], d_hash_br1[thr_id], d_hash_br2[thr_id]);
		TRACE("perm3  :");

		x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);

		// hamsi or luffa
		branchNonces = bastion_filter2(thr_id, throughput, d_hash[thr_id], d_hash_br1[thr_id], d_hash_br2[thr_id]);
		x13_hamsi512_cpu_hash_64(thr_id, branchNonces, pdata[19], NULL, d_hash_br1[thr_id], order++);
		x11_luffa512_cpu_hash_64(thr_id, throughput-branchNonces, pdata[19], NULL, d_hash_br2[thr_id], order++);
		bastion_merge2(thr_id, throughput, d_hash[thr_id], d_hash_br1[thr_id], d_hash_br2[thr_id]);
		TRACE("perm4  :");

		*hashes_done = pdata[19] - first_nonce + throughput;

		CUDA_LOG_ERROR();

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
			const uint32_t Htarg = ptarget[7];
			endiandata[19] = work->nonces[0];
			bastionhash(vhash, (uchar*) endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[0] = swab32(work->nonces[0]);
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				if (work->nonces[1] != 0) {
					endiandata[19] = work->nonces[1];
					bastionhash(vhash, (uchar*) endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					work->nonces[1] = swab32(work->nonces[1]);
					pdata[19] = max(work->nonces[0], work->nonces[1])+1;
				} else {
					pdata[19] = work->nonces[0]+1; // cursor
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet) gpulog(LOG_WARNING, thr_id,
					"result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[19] = swab32(work->nonces[0]) + 1;
				continue;
			}
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart && (uint64_t) max_nonce > (uint64_t) pdata[19] + throughput);

	*hashes_done = pdata[19] - first_nonce;

	CUDA_LOG_ERROR();

	return 0;
}

// cleanup
extern "C" void free_bastion(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_hash_br1[thr_id]);
	cudaFree(d_hash_br2[thr_id]);

	hefty_cpu_free(thr_id);
	x13_fugue512_cpu_free(thr_id);
	x15_whirlpool_cpu_free(thr_id);

	bastion_free(thr_id);
	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}

#undef SPH_C32
#undef SPH_T32
#undef SPH_C64
#undef SPH_T64
extern "C" {
#include "hefty1.h"
#include "sph/sph_luffa.h"
#include "sph/sph_fugue.h"
#include "sph/sph_skein.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_shabal.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
}

__host__
void bastionhash(void* output, const uchar* input)
{
	unsigned char _ALIGN(128) hash[64] = { 0 };

	sph_echo512_context ctx_echo;
	sph_luffa512_context ctx_luffa;
	sph_fugue512_context ctx_fugue;
	sph_whirlpool_context ctx_whirlpool;
	sph_shabal512_context ctx_shabal;
	sph_skein512_context ctx_skein;
	sph_hamsi512_context ctx_hamsi;

	HEFTY1(input, 80, hash);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512(&ctx_luffa, hash, 64);
	sph_luffa512_close(&ctx_luffa, hash);

	if (hash[0] & 0x8)
	{
		sph_fugue512_init(&ctx_fugue);
		sph_fugue512(&ctx_fugue, hash, 64);
		sph_fugue512_close(&ctx_fugue, hash);
	} else {
		sph_skein512_init(&ctx_skein);
		sph_skein512(&ctx_skein, hash, 64);
		sph_skein512_close(&ctx_skein, hash);
	}

	sph_whirlpool_init(&ctx_whirlpool);
	sph_whirlpool(&ctx_whirlpool, hash, 64);
	sph_whirlpool_close(&ctx_whirlpool, hash);

	sph_fugue512_init(&ctx_fugue);
	sph_fugue512(&ctx_fugue, hash, 64);
	sph_fugue512_close(&ctx_fugue, hash);

	if (hash[0] & 0x8)
	{
		sph_echo512_init(&ctx_echo);
		sph_echo512(&ctx_echo, hash, 64);
		sph_echo512_close(&ctx_echo, hash);
	} else {
		sph_luffa512_init(&ctx_luffa);
		sph_luffa512(&ctx_luffa, hash, 64);
		sph_luffa512_close(&ctx_luffa, hash);
	}

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, hash, 64);
	sph_shabal512_close(&ctx_shabal, hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, hash, 64);
	sph_skein512_close(&ctx_skein, hash);

	if (hash[0] & 0x8)
	{
		sph_shabal512_init(&ctx_shabal);
		sph_shabal512(&ctx_shabal, hash, 64);
		sph_shabal512_close(&ctx_shabal, hash);
	} else {
		sph_whirlpool_init(&ctx_whirlpool);
		sph_whirlpool(&ctx_whirlpool, hash, 64);
		sph_whirlpool_close(&ctx_whirlpool, hash);
	}

	sph_shabal512_init(&ctx_shabal);
	sph_shabal512(&ctx_shabal, hash, 64);
	sph_shabal512_close(&ctx_shabal, hash);

	if (hash[0] & 0x8)
	{
		sph_hamsi512_init(&ctx_hamsi);
		sph_hamsi512(&ctx_hamsi, hash, 64);
		sph_hamsi512_close(&ctx_hamsi, hash);
	} else {
		sph_luffa512_init(&ctx_luffa);
		sph_luffa512(&ctx_luffa, hash, 64);
		sph_luffa512_close(&ctx_luffa, hash);
	}

	memcpy(output, hash, 32);
}
