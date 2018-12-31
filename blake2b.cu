/**
 * Blake2-B CUDA Implementation
 *
 * tpruvot@github July 2016
 *
 */

#include <miner.h>

#include <string.h>
#include <stdint.h>

#include <sph/blake2b.h>

#include <cuda_helper.h>
#include <cuda_vector_uint2x4.h>

#define TPB 512
#define NBN 2

static uint32_t *d_resNonces[MAX_GPUS];

__device__ uint64_t d_data[10];

static __constant__ const int8_t blake2b_sigma[12][16] = {
	{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } ,
	{ 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  } ,
	{ 11, 8,  12, 0,  5,  2,  15, 13, 10, 14, 3,  6,  7,  1,  9,  4  } ,
	{ 7,  9,  3,  1,  13, 12, 11, 14, 2,  6,  5,  10, 4,  0,  15, 8  } ,
	{ 9,  0,  5,  7,  2,  4,  10, 15, 14, 1,  11, 12, 6,  8,  3,  13 } ,
	{ 2,  12, 6,  10, 0,  11, 8,  3,  4,  13, 7,  5,  15, 14, 1,  9  } ,
	{ 12, 5,  1,  15, 14, 13, 4,  10, 0,  7,  6,  3,  9,  2,  8,  11 } ,
	{ 13, 11, 7,  14, 12, 1,  3,  9,  5,  0,  15, 4,  8,  6,  2,  10 } ,
	{ 6,  15, 14, 9,  11, 3,  0,  8,  12, 2,  13, 7,  1,  4,  10, 5  } ,
	{ 10, 2,  8,  4,  7,  6,  1,  5,  15, 11, 9,  14, 3,  12, 13, 0  } ,
	{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } ,
	{ 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  }
};

// host mem align
#define A 64

extern "C" void blake2b_hash(void *output, const void *input)
{
	uint8_t _ALIGN(A) hash[32];
	blake2b_ctx ctx;

	blake2b_init(&ctx, 32, NULL, 0);
	blake2b_update(&ctx, input, 80);
	blake2b_final(&ctx, hash);

	memcpy(output, hash, 32);
}

// ----------------------------------------------------------------

__device__ __forceinline__
static void G(const int r, const int i, uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t const m[16])
{
	a = a + b + m[ blake2b_sigma[r][2*i] ];
	((uint2*)&d)[0] = SWAPUINT2( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
	c = c + d;
	((uint2*)&b)[0] = ROR24( ((uint2*)&b)[0] ^ ((uint2*)&c)[0] );
	a = a + b + m[ blake2b_sigma[r][2*i+1] ];
	((uint2*)&d)[0] = ROR16( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
	c = c + d;
	((uint2*)&b)[0] = ROR2( ((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
}

#define ROUND(r) \
	G(r, 0, v[0], v[4], v[ 8], v[12], m); \
	G(r, 1, v[1], v[5], v[ 9], v[13], m); \
	G(r, 2, v[2], v[6], v[10], v[14], m); \
	G(r, 3, v[3], v[7], v[11], v[15], m); \
	G(r, 4, v[0], v[5], v[10], v[15], m); \
	G(r, 5, v[1], v[6], v[11], v[12], m); \
	G(r, 6, v[2], v[7], v[ 8], v[13], m); \
	G(r, 7, v[3], v[4], v[ 9], v[14], m);

__global__
//__launch_bounds__(128, 8) /* to force 64 regs */
void blake2b_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint2 target2)
{
	const uint32_t nonce = (blockDim.x * blockIdx.x + threadIdx.x) + startNonce;

	uint64_t m[16];

	m[0] = d_data[0];
	m[1] = d_data[1];
	m[2] = d_data[2];
	m[3] = d_data[3];
	m[4] = d_data[4];
	m[5] = d_data[5];
	m[6] = d_data[6];
	m[7] = d_data[7];
	m[8] = d_data[8];
	((uint32_t*)m)[18] = AS_U32(&d_data[9]);
	((uint32_t*)m)[19] = nonce;

	m[10] = m[11] = 0;
	m[12] = m[13] = 0;
	m[14] = m[15] = 0;

	uint64_t v[16] = {
		0x6a09e667f2bdc928, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
		0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
		0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
		0x510e527fade68281, 0x9b05688c2b3e6c1f, 0xe07c265404be4294, 0x5be0cd19137e2179
	};

	ROUND( 0);
	ROUND( 1);
	ROUND( 2);
	ROUND( 3);
	ROUND( 4);
	ROUND( 5);
	ROUND( 6);
	ROUND( 7);
	ROUND( 8);
	ROUND( 9);
	ROUND(10);
	ROUND(11);

	uint2 last = vectorize(v[3] ^ v[11] ^ 0xa54ff53a5f1d36f1);
	if (last.y <= target2.y && last.x <= target2.x) {
		resNonce[1] = resNonce[0];
		resNonce[0] = nonce;
	}
}

__host__
uint32_t blake2b_hash_cuda(const int thr_id, const uint32_t threads, const uint32_t startNonce, const uint2 target2, uint32_t &secNonce)
{
	uint32_t resNonces[NBN] = { UINT32_MAX, UINT32_MAX };
	uint32_t result = UINT32_MAX;

	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);

	/* Check error on Ctrl+C or kill to prevent segfaults on exit */
	if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
		return result;

	blake2b_gpu_hash <<<grid, block, 8>>> (threads, startNonce, d_resNonces[thr_id], target2);
	cudaThreadSynchronize();

	if (cudaSuccess == cudaMemcpy(resNonces, d_resNonces[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost)) {
		result = resNonces[0];
		secNonce = resNonces[1];
		if (secNonce == result) secNonce = UINT32_MAX;
	}
	return result;
}

__host__
void blake2b_setBlock(uint32_t *data)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_data, data, 80, 0, cudaMemcpyHostToDevice));
}

static bool init[MAX_GPUS] = { 0 };

int scanhash_blake2b(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(A) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];

	int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] >= 500 && !is_windows()) ? 28 : 25;
	if (device_sm[dev_id] >= 520 && is_windows()) intensity = 26;
	if (device_sm[dev_id] < 350) intensity = 22;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	if (!init[thr_id])
	{
		cudaSetDevice(dev_id);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_resNonces[thr_id], NBN * sizeof(uint32_t)), -1);
		init[thr_id] = true;
	}

	for (int i=0; i < 20; i++)
		be32enc(&endiandata[i], pdata[i]);

	const uint2 target = make_uint2(ptarget[6], ptarget[7]);
	blake2b_setBlock(endiandata);

	do {
		work->nonces[0] = blake2b_hash_cuda(thr_id, throughput, pdata[19], target, work->nonces[1]);

		*hashes_done = pdata[19] - first_nonce + throughput;

		if (work->nonces[0] != UINT32_MAX)
		{
			const uint32_t Htarg = ptarget[7];
			uint32_t _ALIGN(A) vhash[8];
			work->valid_nonces = 0;
			endiandata[19] = work->nonces[0];
			blake2b_hash(vhash, endiandata);
			if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
				work_set_target_ratio(work, vhash);
				work->valid_nonces++;
				pdata[19] = work->nonces[0] + 1;
			} else {
				gpu_increment_reject(thr_id);
			}

			if (work->nonces[1] != UINT32_MAX) {
				endiandata[19] = work->nonces[1];
				blake2b_hash(vhash, endiandata);
				if (vhash[7] <= Htarg && fulltest(vhash, ptarget)) {
					if (bn_hash_target_ratio(vhash, ptarget) > work->shareratio[0]) {
						work->sharediff[1] = work->sharediff[0];
						work->shareratio[1] = work->shareratio[0];
						xchg(work->nonces[1], work->nonces[0]);
						work_set_target_ratio(work, vhash);
					} else {
						bn_set_target_ratio(work, vhash, 1);
					}
					work->valid_nonces++;
					pdata[19] = max(work->nonces[0], work->nonces[1]) + 1; // next scan start
				} else {
					gpu_increment_reject(thr_id);
				}
			}

			if (work->valid_nonces) {
				work->nonces[0] = cuda_swab32(work->nonces[0]);
				work->nonces[1] = cuda_swab32(work->nonces[1]);
				return work->valid_nonces;
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
extern "C" void free_blake2b(int thr_id)
{
	if (!init[thr_id])
		return;

	//cudaThreadSynchronize();

	cudaFree(d_resNonces[thr_id]);

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
