extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_keccak.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_skein.h"
#include "sph/sph_groestl.h"
#include "lyra2/Lyra2.h"
}

#include <miner.h>
#include <cuda_helper.h>

static uint64_t* d_hash[MAX_GPUS];
static uint64_t* d_matrix[MAX_GPUS];

extern void blake256_cpu_init(int thr_id, uint32_t threads);
extern void blake256_cpu_setBlock_80(uint32_t *pdata);
//extern void blake256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash, int order);

//extern void keccak256_sm3_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
//extern void keccak256_sm3_init(int thr_id, uint32_t threads);
//extern void keccak256_sm3_free(int thr_id);

extern void blakeKeccak256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash, int order);

extern void skein256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);

extern void cubehash256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, int order);

extern void skein256_cpu_init(int thr_id, uint32_t threads);

extern void lyra2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix);
extern void lyra2_cpu_hash_32(int thr_id, uint32_t threads, uint64_t *d_outputHash, bool gtx750ti);

extern void groestl256_cpu_init(int thr_id, uint32_t threads);
extern void groestl256_cpu_free(int thr_id);
extern void groestl256_setTarget(const void *ptarget);
extern uint32_t groestl256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, int order);
extern uint32_t groestl256_getSecNonce(int thr_id, int num);


extern "C" void allium_hash(void *state, const void *input)
{
	uint32_t hashA[8], hashB[8];

	sph_blake256_context     ctx_blake;
	sph_keccak256_context    ctx_keccak;
	sph_cubehash256_context  ctx_cube;
	sph_skein256_context     ctx_skein;
	sph_groestl256_context   ctx_groestl;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	LYRA2(hashA, 32, hashB, 32, hashB, 32, 1, 8, 8);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashA, 32);
	sph_cubehash256_close(&ctx_cube, hashB);

	LYRA2(hashA, 32, hashB, 32, hashB, 32, 1, 8, 8);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashA, 32);
	sph_skein256_close(&ctx_skein, hashB);

	sph_groestl256_init(&ctx_groestl);
	sph_groestl256(&ctx_groestl, hashB, 32);
	sph_groestl256_close(&ctx_groestl, hashA);

	memcpy(state, hashA, 32);
}

static bool init[MAX_GPUS] = { 0 };
static __thread uint32_t throughput = 0;

extern "C" int scanhash_allium(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

	static __thread bool gtx750ti;
	if (!init[thr_id])
	{
		int dev_id = device_map[thr_id];
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}

		int intensity = (device_sm[dev_id] >= 500 && !is_windows()) ? 17 : 16;
		if (device_sm[device_map[thr_id]] == 500) intensity = 15;
		throughput = cuda_default_throughput(thr_id, 1U << intensity); // 18=256*256*4;
		if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, dev_id);

		if (strstr(props.name, "750 Ti")) gtx750ti = true;
		else gtx750ti = false;

		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		blake256_cpu_init(thr_id, throughput);
		//keccak256_sm3_init(thr_id, throughput);
		skein256_cpu_init(thr_id, throughput);
		groestl256_cpu_init(thr_id, throughput);

		//cuda_get_arch(thr_id);
		if (device_sm[dev_id] >= 500)
		{
			size_t matrix_sz = device_sm[dev_id] > 500 ? sizeof(uint64_t) * 4 * 4 : sizeof(uint64_t) * 8 * 8 * 3 * 4;
			CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], matrix_sz * throughput));
			lyra2_cpu_init(thr_id, throughput, d_matrix[thr_id]);
		}

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t)32 * throughput));

		init[thr_id] = true;
	}

	uint32_t _ALIGN(128) endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake256_cpu_setBlock_80(pdata);
	groestl256_setTarget(ptarget);

	do {
		int order = 0;

		//blake256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		//keccak256_sm3_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		blakeKeccak256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		lyra2_cpu_hash_32(thr_id, throughput, d_hash[thr_id], gtx750ti);
		cubehash256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		lyra2_cpu_hash_32(thr_id, throughput, d_hash[thr_id], gtx750ti);
		skein256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);

		*hashes_done = pdata[19] - first_nonce + throughput;

		work->nonces[0] = groestl256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];

			be32enc(&endiandata[19], work->nonces[0]);
			allium_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] = groestl256_getSecNonce(thr_id, 1);
				if (work->nonces[1] != UINT32_MAX) {
					be32enc(&endiandata[19], work->nonces[1]);
					allium_hash(vhash, endiandata);
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

	} while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_allium(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_matrix[thr_id]);

	//keccak256_sm3_free(thr_id);
	groestl256_cpu_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
