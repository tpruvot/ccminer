extern "C"
{
#include "sph/sph_keccak.h"
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_jh.h"
#include "sph/sph_skein.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "quark/cuda_quark.h"

static uint32_t *d_hash[MAX_GPUS] = { 0 };

// Speicher zur Generierung der Noncevektoren für die bedingten Hashes
static uint32_t *d_jackpotNonces[MAX_GPUS] = { 0 };
static uint32_t *d_branch1Nonces[MAX_GPUS] = { 0 };
static uint32_t *d_branch2Nonces[MAX_GPUS] = { 0 };
static uint32_t *d_branch3Nonces[MAX_GPUS] = { 0 };

extern void jackpot_keccak512_cpu_init(int thr_id, uint32_t threads);
extern void jackpot_keccak512_cpu_setBlock(void *pdata, size_t inlen);
extern void jackpot_keccak512_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int order);

extern void jackpot_compactTest_cpu_init(int thr_id, uint32_t threads);
extern void jackpot_compactTest_cpu_free(int thr_id);
extern void jackpot_compactTest_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable,
                                            uint32_t *d_nonces1, uint32_t *nrm1, uint32_t *d_nonces2, uint32_t *nrm2, int order);

extern uint32_t cuda_check_hash_branch(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order);

// CPU HASH JHA v8
extern "C" void jackpothash(void *state, const void *input)
{
	uint32_t hash[16];
	unsigned int rnd;

	sph_blake512_context     ctx_blake;
	sph_groestl512_context   ctx_groestl;
	sph_jh512_context        ctx_jh;
	sph_keccak512_context    ctx_keccak;
	sph_skein512_context     ctx_skein;

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512 (&ctx_keccak, input, 80);
	sph_keccak512_close(&ctx_keccak, hash);

	for (rnd = 0; rnd < 3; rnd++)
	{
		if (hash[0] & 0x01) {
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512 (&ctx_groestl, (&hash), 64);
			sph_groestl512_close(&ctx_groestl, (&hash));
		}
		else {
			sph_skein512_init(&ctx_skein);
			sph_skein512 (&ctx_skein, (&hash), 64);
			sph_skein512_close(&ctx_skein, (&hash));
		}

		if (hash[0] & 0x01) {
			sph_blake512_init(&ctx_blake);
			sph_blake512 (&ctx_blake, (&hash), 64);
			sph_blake512_close(&ctx_blake, (&hash));
		}
		else {
			sph_jh512_init(&ctx_jh);
			sph_jh512 (&ctx_jh, (&hash), 64);
			sph_jh512_close(&ctx_jh, (&hash));
		}
	}
	memcpy(state, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_jackpot(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[22];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int dev_id = device_map[thr_id];

	uint32_t throughput =  cuda_default_throughput(thr_id, 1U << 20);
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
		if (device_sm[dev_id] < 300 || cuda_arch[dev_id] < 300) {
			gpulog(LOG_ERR, thr_id, "Sorry, This algo is not supported by this GPU arch (SM 3.0 required)");
			proper_exit(EXIT_CODE_CUDA_ERROR);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput));

		jackpot_keccak512_cpu_init(thr_id, throughput);
		jackpot_compactTest_cpu_init(thr_id, throughput);
		quark_blake512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);

		cuda_check_cpu_init(thr_id, throughput);

		cudaMalloc(&d_branch1Nonces[thr_id], (size_t) sizeof(uint32_t)*throughput*2);
		cudaMalloc(&d_branch2Nonces[thr_id], (size_t) sizeof(uint32_t)*throughput*2);
		cudaMalloc(&d_branch3Nonces[thr_id], (size_t) sizeof(uint32_t)*throughput*2);

		CUDA_SAFE_CALL(cudaMalloc(&d_jackpotNonces[thr_id], (size_t) sizeof(uint32_t)*throughput*2));

		init[thr_id] = true;
	}

	for (int k=0; k < 22; k++)
		be32enc(&endiandata[k], pdata[k]);

	jackpot_keccak512_cpu_setBlock((void*)endiandata, 80);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// erstes Keccak512 Hash mit CUDA
		jackpot_keccak512_cpu_hash(thr_id, throughput, pdata[19], d_hash[thr_id], order++);

		uint32_t nrm1, nrm2, nrm3;

		// Runde 1 (ohne Gröstl)

		jackpot_compactTest_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id], NULL,
				d_branch1Nonces[thr_id], &nrm1,
				d_branch3Nonces[thr_id], &nrm3,
				order++);

		// verfolge den skein-pfad weiter
		quark_skein512_cpu_hash_64(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

		// noch schnell Blake & JH
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2,
			order++);

		if (nrm1+nrm2 == nrm3) {
			quark_blake512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);
		}

		// Runde 3 (komplett)

		// jackpotNonces in branch1/2 aufsplitten gemäss if (hash[0] & 0x01)
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2,
			order++);

		if (nrm1+nrm2 == nrm3) {
			quark_groestl512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			quark_skein512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);
		}

		// jackpotNonces in branch1/2 aufsplitten gemäss if (hash[0] & 0x01)
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2,
			order++);

		if (nrm1+nrm2 == nrm3) {
			quark_blake512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);
		}

		// Runde 3 (komplett)

		// jackpotNonces in branch1/2 aufsplitten gemäss if (hash[0] & 0x01)
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2,
			order++);

		if (nrm1+nrm2 == nrm3) {
			quark_groestl512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			quark_skein512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);
		}

		// jackpotNonces in branch1/2 aufsplitten gemäss if (hash[0] & 0x01)
		jackpot_compactTest_cpu_hash_64(thr_id, nrm3, pdata[19], d_hash[thr_id], d_branch3Nonces[thr_id],
			d_branch1Nonces[thr_id], &nrm1,
			d_branch2Nonces[thr_id], &nrm2,
			order++);

		if (nrm1+nrm2 == nrm3) {
			quark_blake512_cpu_hash_64(thr_id, nrm1, pdata[19], d_branch1Nonces[thr_id], d_hash[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, nrm2, pdata[19], d_branch2Nonces[thr_id], d_hash[thr_id], order++);
		}

		*hashes_done = pdata[19] - first_nonce + throughput;

		CUDA_LOG_ERROR();

		work->nonces[0] = cuda_check_hash_branch(thr_id, nrm3, pdata[19], d_branch3Nonces[thr_id], d_hash[thr_id], order++);

		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);

			// jackpothash function gibt die Zahl der Runden zurück
			jackpothash(vhash, endiandata);

			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
#if 0
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					jackpothash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
#else
				pdata[19] = work->nonces[0] + 1; // cursor
#endif
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

	CUDA_LOG_ERROR();

	return 0;
}

// cleanup
extern "C" void free_jackpot(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_branch1Nonces[thr_id]);
	cudaFree(d_branch2Nonces[thr_id]);
	cudaFree(d_branch3Nonces[thr_id]);
	cudaFree(d_jackpotNonces[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	jackpot_compactTest_cpu_free(thr_id);

	cudaFree(d_hash[thr_id]);

	cuda_check_cpu_free(thr_id);
	CUDA_LOG_ERROR();

	cudaDeviceSynchronize();

	init[thr_id] = false;
}
