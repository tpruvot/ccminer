/* Ziftrcoin ZR5 CUDA Implementation, (c) tpruvot 2015 */

extern "C" {
#include "sph/sph_blake.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
}

#include "miner.h"
#include "cuda_helper.h"

#include <stdio.h>
#include <memory.h>

#define ZR_BLAKE   0
#define ZR_GROESTL 1
#define ZR_JH512   2
#define ZR_SKEIN   3

#define POK_BOOL_MASK 0x00008000
#define POK_DATA_MASK 0xFFFF0000

static uint32_t* d_hash[MAX_GPUS];
static uint16_t* d_poks[MAX_GPUS];

static uint32_t**d_buffers[MAX_GPUS];
static uint8_t*  d_permut[MAX_GPUS];

static uint32_t* d_blake[MAX_GPUS];
static uint32_t* d_groes[MAX_GPUS];
static uint32_t* d_jh512[MAX_GPUS];
static uint32_t* d_skein[MAX_GPUS];

static uint8_t*  d_txs[MAX_GPUS];
__constant__ uint16_t c_txlens[POK_MAX_TXS];

__constant__ uint8_t c_permut[24][4];
static const uint8_t permut[24][4] = {
	{0, 1, 2, 3},
	{0, 1, 3, 2},
	{0, 2, 1, 3},
	{0, 2, 3, 1},
	{0, 3, 1, 2},
	{0, 3, 2, 1},
	{1, 0, 2, 3},
	{1, 0, 3, 2},
	{1, 2, 0, 3},
	{1, 2, 3, 0},
	{1, 3, 0, 2},
	{1, 3, 2, 0},
	{2, 0, 1, 3},
	{2, 0, 3, 1},
	{2, 1, 0, 3},
	{2, 1, 3, 0},
	{2, 3, 0, 1},
	{2, 3, 1, 0},
	{3, 0, 1, 2},
	{3, 0, 2, 1},
	{3, 1, 0, 2},
	{3, 1, 2, 0},
	{3, 2, 0, 1},
	{3, 2, 1, 0}
};

// CPU HASH
extern "C" void zr5hash(void *output, const void *input)
{
	sph_keccak512_context ctx_keccak;
	sph_blake512_context ctx_blake;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_skein512_context ctx_skein;

	uchar _ALIGN(64) hash[64];
	uint32_t *phash = (uint32_t *) hash;
	uint32_t norder;

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512(&ctx_keccak, (const void*) input, 80);
	sph_keccak512_close(&ctx_keccak, (void*) phash);

	norder = phash[0] % ARRAY_SIZE(permut); /* % 24 */

	for(int i = 0; i < 4; i++)
	{
		switch (permut[norder][i]) {
		case ZR_BLAKE:
			sph_blake512_init(&ctx_blake);
			sph_blake512(&ctx_blake, (const void*) phash, 64);
			sph_blake512_close(&ctx_blake, phash);
			break;
		case ZR_GROESTL:
			sph_groestl512_init(&ctx_groestl);
			sph_groestl512(&ctx_groestl, (const void*) phash, 64);
			sph_groestl512_close(&ctx_groestl, phash);
			break;
		case ZR_JH512:
			sph_jh512_init(&ctx_jh);
			sph_jh512(&ctx_jh, (const void*) phash, 64);
			sph_jh512_close(&ctx_jh, phash);
			break;
		case ZR_SKEIN:
			sph_skein512_init(&ctx_skein);
			sph_skein512(&ctx_skein, (const void*) phash, 64);
			sph_skein512_close(&ctx_skein, phash);
			break;
		default:
			break;
		}
	}
	memcpy(output, phash, 32);
}

extern "C" void zr5hash_pok(void *output, uint32_t *pdata)
{
	uint32_t _ALIGN(64) hash[8];
	const uint32_t version = (pdata[0] & (~POK_DATA_MASK)) | (use_pok ? POK_BOOL_MASK : 0);

	pdata[0] = version;
	zr5hash(hash, pdata);

	// fill PoK
	pdata[0] = version | (hash[0] & POK_DATA_MASK);
	zr5hash(hash, pdata);

	memcpy(output, hash, 32);
}

// ------------------------------------------------------------------------------------------------

__global__ __launch_bounds__(128, 8)
void zr5_init_vars_gpu(uint32_t threads, uint32_t* d_hash, uint8_t* d_permut, uint32_t** d_buffers,
        uint32_t* d_blake, uint32_t* d_groes, uint32_t* d_jh512, uint32_t* d_skein)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t offset = thread * 16U; // 64U / sizeof(uint32_t);
		uint32_t *phash = &d_hash[offset];

		// store the algos order for other procs
		const uint8_t norder = (phash[0] % ARRAY_SIZE(permut));
		const uint8_t algo = c_permut[norder][0];
		d_permut[thread] = norder;

		// init array for other procs
		d_buffers[0] = d_blake;
		d_buffers[1] = d_groes;
		d_buffers[2] = d_jh512;
		d_buffers[3] = d_skein;

		// Copy From d_hash to the first algo buffer
		// uint4 = 4x uint32_t = 16 bytes
		uint4 *psrc = (uint4*) phash;
		uint4 *pdst = (uint4*) (d_buffers[algo] + offset);
		pdst[0] = psrc[0];
		pdst[1] = psrc[1];
		pdst[2] = psrc[2];
		pdst[3] = psrc[3];
	}
}

__host__
void zr5_init_vars(int thr_id, uint32_t threads)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	zr5_init_vars_gpu <<<grid, block>>> (
		threads, d_hash[thr_id], d_permut[thr_id], d_buffers[thr_id],
		d_blake[thr_id], d_groes[thr_id], d_jh512[thr_id], d_skein[thr_id]
	);
}


__global__ __launch_bounds__(128, 8)
void zr5_move_data_to_hash_gpu(const uint32_t threads, const int rnd, uint32_t** const d_buffers, uint8_t *d_permut, uint32_t *d_hash)
{
	// copy 64 bytes hash from/to the right algo buffers
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint8_t norder = d_permut[thread];
		const uint8_t algodst = c_permut[norder][rnd];
		const uint8_t algosrc = c_permut[norder][rnd-1];

		const uint32_t offset = thread * (64 / 4);

		// uint4 = 4x uint32_t = 16 bytes
		uint4 *psrc = (uint4*) (d_buffers[algosrc] + offset);
		uint4 *pdst = (uint4*) (d_buffers[algodst] + offset);
		pdst[0] = psrc[0];
		pdst[1] = psrc[1];
		pdst[2] = psrc[2];
		pdst[3] = psrc[3];
	}
}

__host__
void zr5_move_data_to_hash(int thr_id, uint32_t threads, int rnd)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	zr5_move_data_to_hash_gpu <<<grid, block>>> (threads, rnd, d_buffers[thr_id], d_permut[thr_id], d_hash[thr_id]);
}


__global__ __launch_bounds__(128, 8)
void zr5_get_poks_gpu(uint32_t threads, uint32_t** const d_buffers, uint8_t* const d_permut, uint16_t *d_poks)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint8_t norder = d_permut[thread];
		const uint8_t algosrc = c_permut[norder][3];

		// copy only pok
		const uint32_t offset = thread * 16U; // 64 / 4;
		uint16_t* hash0 = (uint16_t*) (d_buffers[algosrc] + offset);
		d_poks[thread] = hash0[1];
	}
}

__global__ __launch_bounds__(128, 4)
void zr5_get_poks_xor_gpu(uint32_t threads, uint32_t** const d_buffers, uint8_t* d_permut, uint16_t* d_poks, uint8_t* d_txs, uint8_t txs)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint8_t norder = d_permut[thread];
		const uint8_t algo = c_permut[norder][3];
		const uint8_t ntx = norder % txs; // generally 0 on testnet...
		const uint32_t offset = thread * 16U; // 64 / 4;
		uint32_t* hash = (uint32_t*) (d_buffers[algo] + offset);
		uint32_t randNdx = hash[1] % c_txlens[ntx];
		uint8_t* ptx = &d_txs[POK_MAX_TX_SZ*ntx] + randNdx;
		uint32_t x = 0x100UL * ptx[3] + ptx[2];

		d_poks[thread] = x ^ (hash[2] >> 16);
	}
}

__host__
void zr5_get_poks(int thr_id, uint32_t threads, uint16_t* d_poks, struct work* work)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	uint8_t txs = (uint8_t) work->tx_count;

	if (txs && use_pok)
	{
		uint32_t txlens[POK_MAX_TXS];
		uint8_t* txdata = (uint8_t*) calloc(POK_MAX_TXS, POK_MAX_TX_SZ);
		if (!txdata) {
			applog(LOG_ERR, "%s: error, memory alloc failure", __func__);
			return;
		}
		// create blocs to copy on device
		for (uint8_t tx=0; tx < txs; tx++) {
			txlens[tx] = (uint32_t) (work->txs[tx].len - 3U);
			memcpy(&txdata[POK_MAX_TX_SZ*tx], work->txs[tx].data, min(POK_MAX_TX_SZ, txlens[tx]+3U));
		}
		cudaMemcpy(d_txs[thr_id], txdata, txs * POK_MAX_TX_SZ, cudaMemcpyHostToDevice);
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_txlens, txlens, txs * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
		zr5_get_poks_xor_gpu <<<grid, block>>> (threads, d_buffers[thr_id], d_permut[thr_id], d_poks, d_txs[thr_id], txs);
		free(txdata);
	} else {
		zr5_get_poks_gpu <<<grid, block>>> (threads, d_buffers[thr_id], d_permut[thr_id], d_poks);
	}
}


__global__ __launch_bounds__(128, 8)
void zr5_final_round_data_gpu(uint32_t threads, uint32_t** const d_buffers, uint8_t* const d_permut, uint32_t *d_hash, uint16_t *d_poks)
{
	// after the 4 algos rounds, copy back hash to d_hash
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint8_t norder = d_permut[thread];
		const uint8_t algosrc = c_permut[norder][3];
		const uint32_t offset = thread * 16U; // 64 / 4;

		// copy only hash[4..7]
		uint2 *psrc = (uint2*) (d_buffers[algosrc] + offset);
		uint2 *phash = (uint2*) (&d_hash[offset]);

		phash[2] = psrc[2];
		phash[3] = psrc[3];
	}
}

__host__
void zr5_final_round(int thr_id, uint32_t threads)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	zr5_final_round_data_gpu <<<grid, block>>> (threads, d_buffers[thr_id], d_permut[thr_id], d_hash[thr_id], d_poks[thr_id]);
}


extern void jackpot_keccak512_cpu_init(int thr_id, uint32_t threads);
extern void jackpot_keccak512_cpu_setBlock(void *pdata, size_t inlen);

extern void zr5_keccak512_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash);
extern void zr5_keccak512_cpu_hash_pok(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t* pdata, uint32_t *d_hash, uint16_t *d_poks);

extern void quark_blake512_cpu_init(int thr_id, uint32_t threads);
extern void quark_blake512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void quark_blake512_cpu_free(int thr_id);

extern void quark_groestl512_cpu_init(int thr_id, uint32_t threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);
extern void quark_groestl512_cpu_free(int thr_id);

extern void quark_jh512_cpu_init(int thr_id, uint32_t threads);
extern void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_skein512_cpu_init(int thr_id, uint32_t threads);
extern void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_zr5(int thr_id, struct work *work,
	uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) tmpdata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t oldp0 = pdata[0];
	const uint32_t version = (oldp0 & (~POK_DATA_MASK)) | (use_pok ? POK_BOOL_MASK : 0);
	const uint32_t first_nonce = pdata[19];
	uint32_t throughput =  cuda_default_throughput(thr_id, 1U << 18);
	throughput = min(throughput, (1U << 20)-1024);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
		ptarget[7] = 0x0000ff;

	memcpy(tmpdata, pdata, 80);

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

		// constants
		cudaMemcpyToSymbol(c_permut, permut, 24*4, 0, cudaMemcpyHostToDevice);

		// hash buffer = keccak hash 64 required
		cudaMalloc(&d_hash[thr_id], 64 * throughput);
		cudaMalloc(&d_poks[thr_id], sizeof(uint16_t) * throughput);
		cudaMalloc(&d_permut[thr_id], sizeof(uint8_t) * throughput);
		cudaMalloc(&d_buffers[thr_id], 4 * sizeof(uint32_t*));

		// data buffers for the 4 rounds
		cudaMalloc(&d_blake[thr_id], 64 * throughput);
		cudaMalloc(&d_groes[thr_id], 64 * throughput);
		cudaMalloc(&d_jh512[thr_id], 64 * throughput);
		cudaMalloc(&d_skein[thr_id], 64 * throughput);

		cudaMalloc(&d_txs[thr_id], POK_MAX_TXS * POK_MAX_TX_SZ);

		jackpot_keccak512_cpu_init(thr_id, throughput);

		quark_blake512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);

		cuda_check_cpu_init(thr_id, throughput);

		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		init[thr_id] = true;
	}

	tmpdata[0] = version;
	jackpot_keccak512_cpu_setBlock((void*)tmpdata, 80);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;

		// Keccak512 Hash with CUDA
		zr5_keccak512_cpu_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		zr5_init_vars(thr_id, throughput);

		for (int rnd=0; rnd<4; rnd++) {
			if (rnd > 0)
				zr5_move_data_to_hash(thr_id, throughput, rnd);
			quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_blake[thr_id], order++);
			quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_groes[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_jh512[thr_id], order++);
			quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_skein[thr_id], order++);
		}

		// store on device d_poks all hash[0] prefixes
		zr5_get_poks(thr_id, throughput, d_poks[thr_id], work);

		// Keccak512 with pok
		zr5_keccak512_cpu_hash_pok(thr_id, throughput, pdata[19], pdata, d_hash[thr_id], d_poks[thr_id]);
		zr5_init_vars(thr_id, throughput);

		for (int rnd=0; rnd<4; rnd++) {
			if (rnd > 0)
				zr5_move_data_to_hash(thr_id, throughput, rnd);
			quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_blake[thr_id], order++);
			quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_groes[thr_id], order++);
			quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_jh512[thr_id], order++);
			quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_skein[thr_id], order++);
		}
		zr5_final_round(thr_id, throughput);

		// do not scan results on interuption
		if (work_restart[thr_id].restart)
			return -1;

		work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];
			uint32_t oldp19 = pdata[19];
			uint32_t offset = work->nonces[0] - pdata[19];
			uint32_t pok = 0;
			uint16_t h_pok;

			*hashes_done = pdata[19] - first_nonce + throughput;

			cudaMemcpy(&h_pok, d_poks[thr_id] + offset, sizeof(uint16_t), cudaMemcpyDeviceToHost);
			pok = version | (0x10000UL * h_pok);
			pdata[0] = pok; pdata[19] = work->nonces[0];
			zr5hash(vhash, pdata);
			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, oldp19, d_hash[thr_id], 1);
				if (work->nonces[1] != 0) {
					offset = work->nonces[1] - oldp19;
					cudaMemcpy(&h_pok, d_poks[thr_id] + offset, sizeof(uint16_t), cudaMemcpyDeviceToHost);
					pok = version | (0x10000UL * h_pok);
					memcpy(tmpdata, pdata, 80);
					tmpdata[0] = pok; tmpdata[19] = work->nonces[1];
					zr5hash(vhash, tmpdata);
					if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
						bn_set_target_ratio(work, vhash, 1);
						pdata[19] = max(pdata[19], work->nonces[1]); // cursor
						pdata[20] = pok; // second nonce "pok"
						work->valid_nonces++;
					}
					pdata[19]++;
				}
				return work->valid_nonces;
			}
			else if (vhash[7] > ptarget[7]) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
					gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
				pdata[0] = oldp0;
			}
		} else
			pdata[19] += throughput;

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	pdata[0] = oldp0;

	*hashes_done = pdata[19] - first_nonce + 1;
	return 0;
}

// cleanup
extern "C" void free_zr5(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	cudaFree(d_hash[thr_id]);

	cudaFree(d_poks[thr_id]);
	cudaFree(d_permut[thr_id]);
	cudaFree(d_buffers[thr_id]);

	cudaFree(d_blake[thr_id]);
	cudaFree(d_groes[thr_id]);
	cudaFree(d_jh512[thr_id]);
	cudaFree(d_skein[thr_id]);

	cudaFree(d_txs[thr_id]);

	quark_blake512_cpu_free(thr_id);
	quark_groestl512_cpu_free(thr_id);
	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}
