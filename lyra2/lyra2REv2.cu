extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_skein.h"
#include "sph/sph_keccak.h"
#include "sph/sph_cubehash.h"
#include "lyra2/Lyra2.h"
}

#include <miner.h>
#include <cuda_helper.h>

static uint64_t *d_hash[MAX_GPUS];
static uint64_t* d_matrix[MAX_GPUS];

extern void blake256_cpu_init(int thr_id, uint32_t threads);
extern void blake256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash, int order);
extern void blake256_cpu_setBlock_80(uint32_t *pdata);

extern void keccak256_sm3_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void keccak256_sm3_init(int thr_id, uint32_t threads);
extern void keccak256_sm3_free(int thr_id);

extern void skein256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void skein256_cpu_init(int thr_id, uint32_t threads);
extern void cubehash256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_hash, int order);
extern void blakeKeccak256_cpu_hash_80(const int thr_id, const uint32_t threads, const uint32_t startNonce, uint64_t *Hash, int order);

extern void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNonce, uint64_t *d_outputHash, int order);
extern void lyra2v2_cpu_init(int thr_id, uint32_t threads, uint64_t* d_matrix);

extern void bmw256_setTarget(const void *ptarget);
extern void bmw256_cpu_init(int thr_id, uint32_t threads);
extern void bmw256_cpu_free(int thr_id);
extern void bmw256_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *resultnonces);

void lyra2v2_hash(void *state, const void *input)
{
	uint32_t hashA[8], hashB[8];

	sph_blake256_context      ctx_blake;
	sph_keccak256_context     ctx_keccak;
	sph_skein256_context      ctx_skein;
	sph_bmw256_context        ctx_bmw;
	sph_cubehash256_context   ctx_cube;

	sph_blake256_set_rounds(14);

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, 80);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashB, 32);
	sph_cubehash256_close(&ctx_cube, hashA);

	LYRA2(hashB, 32, hashA, 32, hashA, 32, 1, 4, 4);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashB, 32);
	sph_skein256_close(&ctx_skein, hashA);

	sph_cubehash256_init(&ctx_cube);
	sph_cubehash256(&ctx_cube, hashA, 32);
	sph_cubehash256_close(&ctx_cube, hashB);

	sph_bmw256_init(&ctx_bmw);
	sph_bmw256(&ctx_bmw, hashB, 32);
	sph_bmw256_close(&ctx_bmw, hashA);

	memcpy(state, hashA, 32);
}

static bool init[MAX_GPUS] = { 0 };
#ifndef ORG
static uint32_t throughput_buf[MAX_GPUS] = { 0 };
#endif

extern "C" int scanhash_lyra2v2(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
#ifdef ORG
	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] < 500) ? 18 : is_windows() ? 19 : 20;
	if (strstr(device_name[dev_id], "GTX 10")) intensity = 20;
	uint32_t throughput = cuda_default_throughput(dev_id, 1UL << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);
#else
	uint32_t throughput;
#endif

	if (opt_benchmark)
		ptarget[7] = 0x000f;

	if (!init[thr_id])
	{
#ifdef ORG
		size_t matrix_sz = 16 * sizeof(uint64_t) * 4 * 3;
#else
		size_t matrix_sz = sizeof(uint64_t) * 4 * 4;
		int dev_id = device_map[thr_id];
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, dev_id);

		int intensity = 0;
		// TITAN series
		if (strstr(props.name, "GTX TITAN X")) intensity = 21;
		else if (strstr(props.name, "TITAN X")) intensity = 22;
		else if (strstr(props.name, "TITAN Z")) intensity = 20;
		else if (strstr(props.name, "TITAN")) intensity = 19;
		// Pascal
		else if (strstr(props.name, "1080")) intensity = 22;
		else if (strstr(props.name, "1070")) intensity = 21;
		else if (strstr(props.name, "1060")) intensity = 21;
		else if (strstr(props.name, "1050")) intensity = 20;
		else if (strstr(props.name, "1030")) intensity = 19;
		// Maxwell
		else if (strstr(props.name, "980")) intensity = 21;
		else if (strstr(props.name, "970")) intensity = 20;
		else if (strstr(props.name, "960")) intensity = 20;
		else if (strstr(props.name, "950")) intensity = 19;
		else if (strstr(props.name, "750 Ti")) intensity = 19;
		else if (strstr(props.name, "750")) intensity = 18;
		// Kepler`Fermi
		else if (strstr(props.name, "780")) intensity = 19;
		else if (strstr(props.name, "760")) intensity = 18;
		else if (strstr(props.name, "740")) intensity = 16;
		else if (strstr(props.name, "730")) intensity = 16;
		else if (strstr(props.name, "720")) intensity = 15;
		else if (strstr(props.name, "710")) intensity = 16;
		else if (strstr(props.name, "690")) intensity = 20;
		else if (strstr(props.name, "680")) intensity = 19;
		else if (strstr(props.name, "660")) intensity = 18;
		else if (strstr(props.name, "650 Ti")) intensity = 18;
		else if (strstr(props.name, "640")) intensity = 17;
		else if (strstr(props.name, "630")) intensity = 16;
		else if (strstr(props.name, "620")) intensity = 15;
		// Tesla series
		else if (strstr(props.name, "Tesla V100")) intensity = 23;
		else if (strstr(props.name, "Tesla P100")) intensity = 22;
		else if (strstr(props.name, "Tesla P40")) intensity = 22;
		else if (strstr(props.name, "Tesla P4")) intensity = 21;
		else if (strstr(props.name, "Tesla M60")) intensity = 22;
		else if (strstr(props.name, "Tesla M6")) intensity = 20;
		else if (strstr(props.name, "Tesla M40")) intensity = 21;
		else if (strstr(props.name, "Tesla M4")) intensity = 20;
		// Quadro series
		else if (strstr(props.name, "P6000")) intensity = 22;	// àNvidia TITAN X
		else if (strstr(props.name, "P5000")) intensity = 22;	// àGTX 1080
		else if (strstr(props.name, "M6000")) intensity = 21;	// àGTX TITAN X
		else if (strstr(props.name, "M5000")) intensity = 21;	// àGTX 980
		else if (strstr(props.name, "M4000")) intensity = 20;	// àGTX 970
		else if (strstr(props.name, "M2000")) intensity = 19;	// àGTX 950
		else if (strstr(props.name, "K6000")) intensity = 19;	// àGTX 780Ti
		else if (strstr(props.name, "K5200")) intensity = 19;	// àGTX 780
		else if (strstr(props.name, "K5000")) intensity = 18;	// àGTX 770
		else if (strstr(props.name, "K600")) intensity = 15;	// àGTX 740 Half
		else if (strstr(props.name, "K420")) intensity = 15;	// àGTX 740 Half

		else if (strstr(props.name, "90")) intensity = 18;	//590
		else if (strstr(props.name, "80")) intensity = 18;	//480 580
		else if (strstr(props.name, "70")) intensity = 18;	//470 570 670 770
		else if (strstr(props.name, "65")) intensity = 17;	//465
		else if (strstr(props.name, "60")) intensity = 17;	//460 560
		else if (strstr(props.name, "55")) intensity = 17;	//555
		else if (strstr(props.name, "50")) intensity = 17;	//450 550Ti 650
		else if (strstr(props.name, "45")) intensity = 16;	//545
		else if (strstr(props.name, "40")) intensity = 15;	//440
		else if (strstr(props.name, "30")) intensity = 15;	//430 530
		else if (strstr(props.name, "20")) intensity = 14;	//420 520
		else if (strstr(props.name, "10")) intensity = 14;	//510 610

		//if (intensity != 0 && opt_eco_mode) intensity -= 3;

		if (intensity == 0)
		{
			intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 20 : 18;
			if (strstr(device_name[dev_id], "GTX 10")) intensity = 20;
		}
		throughput_buf[thr_id] = cuda_default_throughput(dev_id, 1UL << (int)intensity);

#endif
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
#ifdef ORG
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput_buf), throughput);

		blake256_cpu_init(thr_id, throughput);
		keccak256_sm3_init(thr_id,throughput);
		skein256_cpu_init(thr_id, throughput);
		bmw256_cpu_init(thr_id, throughput);
#else
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput_buf[thr_id]), throughput_buf[thr_id]);

		blake256_cpu_init(thr_id, throughput_buf[thr_id]);
		keccak256_cpu_init(thr_id, throughput_buf[thr_id]);
		skein256_cpu_init(thr_id, throughput_buf[thr_id]);
		bmw256_cpu_init(thr_id, throughput_buf[thr_id]);
#endif
		// SM 3 implentation requires a bit more memory
		if (device_sm[dev_id] < 500 || cuda_arch[dev_id] < 500)
			matrix_sz = 16 * sizeof(uint64_t) * 4 * 4;
			
#ifdef ORG
		CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], matrix_sz * throughput));
		lyra2v2_cpu_init(thr_id, throughput, d_matrix[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t)32 * throughput));

		api_set_throughput(thr_id, throughput);
#else
		CUDA_SAFE_CALL(cudaMalloc(&d_matrix[thr_id], matrix_sz * throughput_buf[thr_id]));
		lyra2v2_cpu_init(thr_id, throughput_buf[thr_id], d_matrix[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t)32 * throughput_buf[thr_id]));

		api_set_throughput(thr_id, throughput_buf[thr_id]);

		throughput = throughput_buf[thr_id];
#endif
		init[thr_id] = true;
	}
#ifndef ORG
	else throughput = min(throughput_buf[thr_id], max_nonce - first_nonce);
#endif

	uint32_t endiandata[20];
	for (int k=0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	blake256_cpu_setBlock_80(pdata);
	bmw256_setTarget(ptarget);

	do {
		int order = 0;

#ifdef ORG
		blake256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		keccak256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
#else
		blakeKeccak256_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
#endif

		cubehash256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		lyra2v2_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		skein256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
		cubehash256_cpu_hash_32(thr_id, throughput,pdata[19], d_hash[thr_id], order++);

		memset(work->nonces, 0, sizeof(work->nonces));
		bmw256_cpu_hash_32(thr_id, throughput, pdata[19], d_hash[thr_id], work->nonces);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != 0)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(64) vhash[8];
			be32enc(&endiandata[19], work->nonces[0]);
			lyra2v2_hash(vhash, endiandata);

			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (work->nonces[1] != 0) {
					be32enc(&endiandata[19], work->nonces[1]);
					lyra2v2_hash(vhash, endiandata);
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
extern "C" void free_lyra2v2(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);
	cudaFree(d_matrix[thr_id]);

	bmw256_cpu_free(thr_id);
	keccak256_sm3_free(thr_id);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
