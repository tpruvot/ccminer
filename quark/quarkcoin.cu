extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
}

#include "miner.h"

#include "cuda_helper.h"
#include "cuda_quark.h"

#include <stdio.h>

extern uint32_t quark_filter_cpu_sm2(const int thr_id, const uint32_t threads, const uint32_t *inpHashes, uint32_t* d_branch2);
extern void quark_merge_cpu_sm2(const int thr_id, const uint32_t threads, uint32_t *outpHashes, uint32_t* d_branch2);

static uint32_t *d_hash[MAX_GPUS];
static uint32_t* d_hash_br2[MAX_GPUS];  // SM 2

// Speicher zur Generierung der Noncevektoren für die bedingten Hashes
static uint32_t *d_branch1Nonces[MAX_GPUS];
static uint32_t *d_branch2Nonces[MAX_GPUS];
static uint32_t *d_branch3Nonces[MAX_GPUS];

// Original Quarkhash Funktion aus einem miner Quelltext
extern "C" void quarkhash(void *state, const void *input)
{
	unsigned char _ALIGN(128) hash[64];

	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;

	sph_blake512_init(&ctx_blake);
	sph_blake512 (&ctx_blake, input, 80);
	sph_blake512_close(&ctx_blake, (void*) hash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
	sph_bmw512_close(&ctx_bmw, (void*) hash);

	if (hash[0] & 0x8)
	{
		sph_groestl512_init(&ctx_groestl);
		sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
		sph_groestl512_close(&ctx_groestl, (void*) hash);
	}
	else
	{
		sph_skein512_init(&ctx_skein);
		sph_skein512 (&ctx_skein, (const void*) hash, 64);
		sph_skein512_close(&ctx_skein, (void*) hash);
	}

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512 (&ctx_groestl, (const void*) hash, 64);
	sph_groestl512_close(&ctx_groestl, (void*) hash);

	sph_jh512_init(&ctx_jh);
	sph_jh512 (&ctx_jh, (const void*) hash, 64);
	sph_jh512_close(&ctx_jh, (void*) hash);

	if (hash[0] & 0x8)
	{
		sph_blake512_init(&ctx_blake);
		sph_blake512 (&ctx_blake, (const void*) hash, 64);
		sph_blake512_close(&ctx_blake, (void*) hash);
	}
	else
	{
		sph_bmw512_init(&ctx_bmw);
		sph_bmw512 (&ctx_bmw, (const void*) hash, 64);
		sph_bmw512_close(&ctx_bmw, (void*) hash);
	}

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
	sph_keccak512_close(&ctx_keccak, (void*) hash);

	sph_skein512_init(&ctx_skein);
	sph_skein512 (&ctx_skein, (const void*) hash, 64);
	sph_skein512_close(&ctx_skein, (void*) hash);

	if (hash[0] & 0x8)
	{
		sph_keccak512_init(&ctx_keccak);
		sph_keccak512 (&ctx_keccak, (const void*) hash, 64);
		sph_keccak512_close(&ctx_keccak, (void*) hash);
	}
	else
	{
		sph_jh512_init(&ctx_jh);
		sph_jh512 (&ctx_jh, (const void*) hash, 64);
		sph_jh512_close(&ctx_jh, (void*) hash);
	}

	memcpy(state, hash, 32);
}

#ifdef _DEBUG
#define TRACE(algo) { \
	if (max_nonce == 1 && pdata[19] <= 1) { \
		uint32_t* debugbuf = NULL; \
		cudaMallocHost(&debugbuf, 32); \
		cudaMemcpy(debugbuf, d_hash[thr_id], 32, cudaMemcpyDeviceToHost); \
		printf("quark %s %08x %08x %08x %08x...%08x... \n", algo, swab32(debugbuf[0]), swab32(debugbuf[1]), \
			swab32(debugbuf[2]), swab32(debugbuf[3]), swab32(debugbuf[7])); \
		cudaFreeHost(debugbuf); \
	} \
}
#else
#define TRACE(algo) {}
#endif

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_quark(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	int dev_id = device_map[thr_id];
	uint32_t def_thr = 1U << 20; // 256*4096
	uint32_t throughput = cuda_default_throughput(thr_id, def_thr);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x00F;

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

		cudaGetLastError();
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput));

		quark_blake512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		quark_compactTest_cpu_init(thr_id, throughput);

		if (cuda_arch[dev_id] >= 300) {
			cudaMalloc(&d_branch1Nonces[thr_id], sizeof(uint32_t)*throughput);
			cudaMalloc(&d_branch2Nonces[thr_id], sizeof(uint32_t)*throughput);
			cudaMalloc(&d_branch3Nonces[thr_id], sizeof(uint32_t)*throughput);
		} else {
			cudaMalloc(&d_hash_br2[thr_id], (size_t) 64 * throughput);
		}

		cuda_check_cpu_init(thr_id, throughput);
		CUDA_SAFE_CALL(cudaGetLastError());

		init[thr_id] = true;
	}

	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;
		uint32_t nrm1=0, nrm2=0, nrm3=0;

		quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
		TRACE("blake  :");
		quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
		TRACE("bmw    :");

		if (cuda_arch[dev_id] >= 300) {

			quark_compactTest_single_false_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], NULL,
				d_branch3Nonces[thr_id], &nrm3, order++);

			// nur den Skein Branch weiterverfolgen
			quark_skein512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			// das ist der unbedingte Branch für Groestl512
			quark_groestl512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			// das ist der unbedingte Branch für JH512
			quark_jh512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			// quarkNonces in branch1 und branch2 aufsplitten gemäss if (hash[0] & 0x8)
			quark_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
				d_branch1Nonces[thr_id], &nrm1,
				d_branch2Nonces[thr_id], &nrm2,
				order++);

			// das ist der bedingte Branch für Blake512
			quark_blake512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);

			// das ist der bedingte Branch für Bmw512
			quark_bmw512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);

			// das ist der unbedingte Branch für Keccak512
			quark_keccak512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			// das ist der unbedingte Branch für Skein512
			quark_skein512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

			// quarkNonces in branch1 und branch2 aufsplitten gemäss if (hash[0] & 0x8)
			quark_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
				d_branch1Nonces[thr_id], &nrm1,
				d_branch2Nonces[thr_id], &nrm2,
				order++);

			quark_keccak512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);

			work->nonces[0] = cuda_check_hash_branch(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);
			work->nonces[1] = 0;
		} else {
			/* algo permutations are made with 2 different buffers */

			quark_filter_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_br2[thr_id], order++);
			quark_merge_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			TRACE("perm1  :");

			quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("groestl:");
			quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("jh512  :");

			quark_filter_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_br2[thr_id], order++);
			quark_merge_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			TRACE("perm2  :");

			quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("keccak :");
			quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			TRACE("skein  :");

			quark_filter_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_br2[thr_id], order++);
			quark_merge_cpu_sm2(thr_id, throughput, d_hash[thr_id], d_hash_br2[thr_id]);
			TRACE("perm3  :");

			CUDA_LOG_ERROR();
			work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
			work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			quarkhash(vhash, endiandata);

			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					quarkhash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
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

		if ((uint64_t) throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
			break;
		}

		pdata[19] += throughput;

	} while (!work_restart[thr_id].restart);

	return 0;
}

// cleanup
extern "C" void free_quark(int thr_id)
{
	int dev_id = device_map[thr_id];
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);

	if (cuda_arch[dev_id] >= 300) {
		cudaFree(d_branch1Nonces[thr_id]);
		cudaFree(d_branch2Nonces[thr_id]);
		cudaFree(d_branch3Nonces[thr_id]);
	} else {
		cudaFree(d_hash_br2[thr_id]);
	}

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	quark_compactTest_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
