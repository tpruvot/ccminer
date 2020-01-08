//
//  PHI2 algo (with smart contracts header)
//  CubeHash + Lyra2 x2 + JH + Gost or Echo + Skein
//
//  Implemented by tpruvot in May 2018
//

extern "C" {
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_streebog.h"
#include "sph/sph_echo.h"
#include "lyra2/Lyra2.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "x11/cuda_x11.h"

#include <stdio.h>
#include <memory.h>

extern void cubehash512_setBlock_80(int thr_id, uint32_t* endiandata);
extern void cubehash512_cuda_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash);

extern void cubehash512_setBlock_144(int thr_id, uint32_t* endiandata);
extern void cubehash512_cuda_hash_144(const int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash);

extern void lyra2_cpu_init(int thr_id, uint32_t threads, uint64_t *d_matrix);
extern void lyra2_cuda_hash_64(int thr_id, const uint32_t threads, uint64_t* d_hash_256, uint32_t* d_hash_512, bool gtx750ti);

extern void streebog_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash);
extern void phi_streebog_hash_64_filtered(int thr_id, const uint32_t threads, uint32_t *g_hash, uint32_t *d_filter);
extern void phi_echo512_cpu_hash_64_filtered(int thr_id, const uint32_t threads, uint32_t* g_hash, uint32_t* d_filter);

extern uint32_t phi_filter_cuda(const int thr_id, const uint32_t threads, const uint32_t *inpHashes, uint32_t* d_br2, uint32_t* d_nonces);
extern void phi_merge_cuda(const int thr_id, const uint32_t threads, uint32_t *outpHashes, uint32_t* d_br2, uint32_t* d_nonces);
extern void phi_final_compress_cuda(const int thr_id, const uint32_t threads, uint32_t *d_hashes);

static uint64_t* d_matrix[MAX_GPUS];
static uint32_t* d_hash_512[MAX_GPUS];
static uint64_t* d_hash_256[MAX_GPUS];
static uint32_t* d_hash_br2[MAX_GPUS];
static uint32_t* d_nonce_br[MAX_GPUS];

static bool has_roots;

extern "C" void phi2_hash(void *output, const void *input)
{
	unsigned char _ALIGN(128) hash[64];
	unsigned char _ALIGN(128) hashA[64];
	unsigned char _ALIGN(128) hashB[64];

	sph_cubehash512_context ctx_cubehash;
	sph_jh512_context ctx_jh;
	sph_gost512_context ctx_gost;
	sph_echo512_context ctx_echo;
	sph_skein512_context ctx_skein;

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512(&ctx_cubehash, input, has_roots ? 144 : 80);
	sph_cubehash512_close(&ctx_cubehash, (void*)hashB);

	LYRA2(&hashA[ 0], 32, &hashB[ 0], 32, &hashB[ 0], 32, 1, 8, 8);
	LYRA2(&hashA[32], 32, &hashB[32], 32, &hashB[32], 32, 1, 8, 8);

	sph_jh512_init(&ctx_jh);
	sph_jh512(&ctx_jh, (const void*)hashA, 64);
	sph_jh512_close(&ctx_jh, (void*)hash);

	if (hash[0] & 1) {
		sph_gost512_init(&ctx_gost);
		sph_gost512(&ctx_gost, (const void*)hash, 64);
		sph_gost512_close(&ctx_gost, (void*)hash);
	} else {
		sph_echo512_init(&ctx_echo);
		sph_echo512(&ctx_echo, (const void*)hash, 64);
		sph_echo512_close(&ctx_echo, (void*)hash);

		sph_echo512_init(&ctx_echo);
		sph_echo512(&ctx_echo, (const void*)hash, 64);
		sph_echo512_close(&ctx_echo, (void*)hash);
	}

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, (const void*)hash, 64);
	sph_skein512_close(&ctx_skein, (void*)hash);

	for (int i=0; i<32; i++)
		hash[i] ^= hash[i+32];

	memcpy(output, hash, 32);
}

//#define _DEBUG
#define _DEBUG_PREFIX "phi-"
#include "cuda_debug.cuh"

static bool init[MAX_GPUS] = { 0 };
static bool use_compat_kernels[MAX_GPUS] = { 0 };
static __thread bool gtx750ti = false;

extern "C" int scanhash_phi2(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];
	const int dev_id = device_map[thr_id];

	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 17 : 16;
	if (device_sm[dev_id] == 500) intensity = 15;
	if (device_sm[dev_id] == 600) intensity = 17;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);
	if (init[thr_id]) throughput = max(throughput & 0xffffff80, 128); // for shared mem

	if (opt_benchmark)
		ptarget[7] = 0xff;

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		cuda_get_arch(thr_id);
		use_compat_kernels[thr_id] = (cuda_arch[dev_id] < 500);
		gtx750ti = (strstr(device_name[dev_id], "GTX 750 Ti") != NULL);

		size_t matrix_sz = device_sm[dev_id] > 500 ? sizeof(uint64_t) * 16 : sizeof(uint64_t) * 8 * 8 * 3 * 4;
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_matrix[thr_id], matrix_sz * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash_256[thr_id], (size_t)32 * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash_512[thr_id], (size_t)64 * throughput), -1);
		CUDA_CALL_OR_RET_X(cudaMalloc(&d_nonce_br[thr_id], sizeof(uint32_t) * throughput), -1);
		if (use_compat_kernels[thr_id]) {
			CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash_br2[thr_id], (size_t)64 * throughput), -1);
		}

		lyra2_cpu_init(thr_id, throughput, d_matrix[thr_id]);
		quark_jh512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		if (use_compat_kernels[thr_id]) x11_echo512_cpu_init(thr_id, throughput);

		cuda_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	has_roots = false;
	uint32_t endiandata[36];
	for (int k = 0; k < 36; k++) {
		be32enc(&endiandata[k], pdata[k]);
		if (k >= 20 && pdata[k]) has_roots = true;
	}

	cuda_check_cpu_setTarget(ptarget);
	if (has_roots)
		cubehash512_setBlock_144(thr_id, endiandata);
	else
		cubehash512_setBlock_80(thr_id, endiandata);

	do {
		int order = 0;
		if (has_roots)
			cubehash512_cuda_hash_144(thr_id, throughput, pdata[19], d_hash_512[thr_id]);
		else
			cubehash512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash_512[thr_id]);
		order++;
		TRACE("cube   ");

		lyra2_cuda_hash_64(thr_id, throughput, d_hash_256[thr_id], d_hash_512[thr_id], gtx750ti);
		order++;
		TRACE("lyra   ");

		quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_512[thr_id], order++);
		TRACE("jh     ");

		order++;
		if (!use_compat_kernels[thr_id]) {
			phi_filter_cuda(thr_id, throughput, d_hash_512[thr_id], NULL, d_nonce_br[thr_id]);
			phi_streebog_hash_64_filtered(thr_id, throughput, d_hash_512[thr_id], d_nonce_br[thr_id]);
			phi_echo512_cpu_hash_64_filtered(thr_id, throughput, d_hash_512[thr_id], d_nonce_br[thr_id]);
			phi_echo512_cpu_hash_64_filtered(thr_id, throughput, d_hash_512[thr_id], d_nonce_br[thr_id]);
		} else {
			// todo: nonces vector to reduce amount of hashes to compute
			phi_filter_cuda(thr_id, throughput, d_hash_512[thr_id], d_hash_br2[thr_id], d_nonce_br[thr_id]);
			streebog_cpu_hash_64(thr_id, throughput, d_hash_512[thr_id]);
			x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_br2[thr_id], order);
			x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_br2[thr_id], order);
			phi_merge_cuda(thr_id, throughput, d_hash_512[thr_id], d_hash_br2[thr_id], d_nonce_br[thr_id]);
		}
		TRACE("mix    ");

		quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash_512[thr_id], order++);
		TRACE("skein  ");

		phi_final_compress_cuda(thr_id, throughput, d_hash_512[thr_id]);
		TRACE("xor  ");

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash_512[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			phi2_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				*hashes_done = pdata[19] - first_nonce + throughput;
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash_512[thr_id], 1);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					phi2_hash(vhash, endiandata);
					bn_set_target_ratio(work, vhash, 1);
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor
				}
				if (pdata[19] > max_nonce) pdata[19] = max_nonce;
				return work->valid_nonces;
			}
			else if (vhash[7] > Htarg) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU! thr=%x", work->nonces[0], throughput);
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
extern "C" void free_phi2(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();
	cudaFree(d_matrix[thr_id]);
	cudaFree(d_hash_512[thr_id]);
	cudaFree(d_hash_256[thr_id]);
	cudaFree(d_nonce_br[thr_id]);
	if (use_compat_kernels[thr_id]) cudaFree(d_hash_br2[thr_id]);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
