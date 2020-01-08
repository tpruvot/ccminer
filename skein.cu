/**
 * SKEIN512 80 + SHA256 64
 * by tpruvot@github - 2015
 */

#include "sph/sph_skein.h"

#include "miner.h"
#include "cuda_helper.h"

#include <openssl/sha.h>

static uint32_t *d_hash[MAX_GPUS];
static __thread bool sm5 = true;

extern void quark_skein512_cpu_init(int thr_id, uint32_t threads);
extern void skein512_cpu_setBlock_80(void *pdata);
extern void skein512_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, int swap);

extern void skeincoin_init(int thr_id);
extern void skeincoin_free(int thr_id);
extern void skeincoin_setBlock_80(int thr_id, void *pdata);
extern uint32_t skeincoin_hash_sm5(int thr_id, uint32_t threads, uint32_t startNounce, int swap, uint64_t target64, uint32_t *secNonce);

static __device__ uint32_t sha256_hashTable[] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

static __device__ __constant__ uint32_t sha256_constantTable[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static __device__ __constant__ uint32_t sha256_endingTable[] = {
	0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000200,
	0x80000000, 0x01400000, 0x00205000, 0x00005088, 0x22000800, 0x22550014, 0x05089742, 0xa0000020,
	0x5a880000, 0x005c9400, 0x0016d49d, 0xfa801f00, 0xd33225d0, 0x11675959, 0xf6e6bfda, 0xb30c1549,
	0x08b2b050, 0x9d7c4c27, 0x0ce2a393, 0x88e6e1ea, 0xa52b4335, 0x67a16f49, 0xd732016f, 0x4eeb2e91,
	0x5dbf55e5, 0x8eee2335, 0xe2bc5ec2, 0xa83f4394, 0x45ad78f7, 0x36f3d0cd, 0xd99c05e8, 0xb0511dc7,
	0x69bc7ac4, 0xbd11375b, 0xe3ba71e5, 0x3b209ff2, 0x18feee17, 0xe25ad9e7, 0x13375046, 0x0515089d,
	0x4f0d0f04, 0x2627484e, 0x310128d2, 0xc668b434, 0x420841cc, 0x62d311b8, 0xe59ba771, 0x85a7a484
};

/* Elementary functions used by SHA256 */
#define SWAB32(x)     cuda_swab32(x)
//#define ROTR32(x,n)   SPH_ROTR32(x,n)

#define R(x, n)       ((x) >> (n))
#define Ch(x, y, z)   ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)  ((x & (y | z)) | (y & z))
#define S0(x)         (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define S1(x)         (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define s0(x)         (ROTR32(x, 7) ^ ROTR32(x, 18) ^ R(x, 3))
#define s1(x)         (ROTR32(x, 17) ^ ROTR32(x, 19) ^ R(x, 10))

#define ADVANCED_SHA2

#ifndef ADVANCED_SHA2

/* SHA256 round function */
#define RND(a, b, c, d, e, f, g, h, k) \
	do { \
		t0 = h + S1(e) + Ch(e, f, g) + k; \
		t1 = S0(a) + Maj(a, b, c); \
		d += t0; \
		h  = t0 + t1; \
	} while (0)

/* Adjusted round function for rotating state */
#define RNDr(S, W, i) \
	RND(S[(64 - i) & 7], S[(65 - i) & 7], \
	    S[(66 - i) & 7], S[(67 - i) & 7], \
	    S[(68 - i) & 7], S[(69 - i) & 7], \
	    S[(70 - i) & 7], S[(71 - i) & 7], \
	    W[i] + sha256_constantTable[i])

static __constant__ uint32_t sha256_ending[16] = {
	0x80000000UL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x200UL
};
__device__
void sha256_transform_gpu(uint32_t *state, uint32_t *message)
{
	uint32_t S[8];
	uint32_t W[64];
	uint32_t t0, t1;

	/* Initialize work variables. */
	for (int i = 0; i < 8; i++) {
		S[i] = state[i];
	}

	for (int i = 0; i < 16; i++) {
		W[i] = message[i];
	}

	for (int i = 16; i < 64; i += 2) {
		W[i] = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i + 1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}

	/* 3. Mix. */
	#pragma unroll
	for (int i = 0; i < 64; i++) {
		RNDr(S, W, i);
	}

	for (int i = 0; i < 8; i++)
		state[i] += S[i];
}
#endif

#ifdef ADVANCED_SHA2
__device__
void skeincoin_gpu_sha256(uint32_t *message)
{
	uint32_t W1[16];
	uint32_t W2[16];

	uint32_t regs[8];
	uint32_t hash[8];

	// Init with Hash-Table
	#pragma unroll 8
	for (int k=0; k < 8; k++) {
		hash[k] = regs[k] = sha256_hashTable[k];
	}

	#pragma unroll 16
	for (int k = 0; k<16; k++)
		W1[k] = SWAB32(message[k]);

	// Progress W1
	#pragma unroll 16
	for (int j = 0; j<16; j++)
	{
		uint32_t T1, T2;
		T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j] + W1[j];
		T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int k = 6; k >= 0; k--) regs[k + 1] = regs[k];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	// Progress W2...W3

	////// PART 1
	#pragma unroll 2
	for (int j = 0; j<2; j++)
		W2[j] = s1(W1[14 + j]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];
	#pragma unroll 5
	for (int j = 2; j<7; j++)
		W2[j] = s1(W2[j - 2]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];

	#pragma unroll 8
	for (int j = 7; j<15; j++)
		W2[j] = s1(W2[j - 2]) + W2[j - 7] + s0(W1[1 + j]) + W1[j];

	W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

	// Round function
	#pragma unroll 16
	for (int j = 0; j<16; j++)
	{
		uint32_t T1, T2;
		T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j + 16] + W2[j];
		T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	////// PART 2
	#pragma unroll 2
	for (int j = 0; j<2; j++)
		W1[j] = s1(W2[14 + j]) + W2[9 + j] + s0(W2[1 + j]) + W2[j];

	#pragma unroll 5
	for (int j = 2; j<7; j++)
		W1[j] = s1(W1[j - 2]) + W2[9 + j] + s0(W2[1 + j]) + W2[j];

	#pragma unroll 8
	for (int j = 7; j<15; j++)
		W1[j] = s1(W1[j - 2]) + W1[j - 7] + s0(W2[1 + j]) + W2[j];

	W1[15] = s1(W1[13]) + W1[8] + s0(W1[0]) + W2[15];

	// Round function
	#pragma unroll 16
	for (int j = 0; j<16; j++)
	{
		uint32_t T1, T2;
		T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j + 32] + W1[j];
		T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	////// PART 3
	#pragma unroll 2
	for (int j = 0; j<2; j++)
		W2[j] = s1(W1[14 + j]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];

	#pragma unroll 5
	for (int j = 2; j<7; j++)
		W2[j] = s1(W2[j - 2]) + W1[9 + j] + s0(W1[1 + j]) + W1[j];

	#pragma unroll 8
	for (int j = 7; j<15; j++)
		W2[j] = s1(W2[j - 2]) + W2[j - 7] + s0(W1[1 + j]) + W1[j];

	W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

	// Round function
	#pragma unroll 16
	for (int j = 0; j<16; j++)
	{
		uint32_t T1, T2;
		T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j + 48] + W2[j];
		T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int l = 6; l >= 0; l--) regs[l + 1] = regs[l];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	#pragma unroll 8
	for (int k = 0; k<8; k++)
		hash[k] += regs[k];

#if 1
	/////
	///// Second Pass (ending)
	/////
	#pragma unroll 8
	for (int k = 0; k<8; k++)
		regs[k] = hash[k];

	// Progress W1
	#pragma unroll 64
	for (int j = 0; j<64; j++)
	{
		uint32_t T1, T2;
		T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_constantTable[j] + sha256_endingTable[j];
		T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		#pragma unroll 7
		for (int k = 6; k >= 0; k--) regs[k + 1] = regs[k];
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	#pragma unroll 8
	for (int k = 0; k<8; k++)
		hash[k] += regs[k];

	// Final Hash
	#pragma unroll 8
	for (int k = 0; k<8; k++)
		message[k] = SWAB32(hash[k]);
#else
	// sha256_transform only, require an additional sha256_transform_gpu() call
	#pragma unroll 8
	for (int k = 0; k<8; k++)
		message[k] = hash[k];
#endif
}
#endif

__global__
void sha2_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *hashBuffer)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t *hash = &hashBuffer[thread << 4];
#ifdef ADVANCED_SHA2
		skeincoin_gpu_sha256(hash);
#else
		uint32_t state[16];
		uint32_t msg[16];
		#pragma unroll
		for (int i = 0; i < 8; i++)
			state[i] = sha256_hashTable[i];

		#pragma unroll
		for (int i = 0; i < 16; i++)
			msg[i] = SWAB32(hash[i]);

		sha256_transform_gpu(state, msg);
		sha256_transform_gpu(state, sha256_ending);

		#pragma unroll
		for (int i = 0; i < 8; i++)
			hash[i] = SWAB32(state[i]);
#endif
	}
}

__host__
void sha2_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_outputHashes)
{
	uint32_t threadsperblock = 128;
	dim3 block(threadsperblock);
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);

	sha2_gpu_hash_64 <<< grid, block >>>(threads, startNounce, d_outputHashes);

	// required once per scan loop to prevent cpu 100% usage (linux)
	MyStreamSynchronize(NULL, 0, thr_id);
}

extern "C" void skeincoinhash(void *output, const void *input)
{
	sph_skein512_context ctx_skein;
	SHA256_CTX sha256;

	uint32_t hash[16];

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, hash);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, (unsigned char *)hash, 64);
	SHA256_Final((unsigned char *)hash, &sha256);

	memcpy(output, hash, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_skeincoin(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;

	const uint32_t first_nonce = pdata[19];

	sm5 = (device_sm[device_map[thr_id]] >= 500);
	bool checkSecnonce = (have_stratum || have_longpoll) && !sm5;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << 20);
	if (init[thr_id]) throughput = min(throughput, (max_nonce - first_nonce));

	uint64_t target64 = 0;

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x03;

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

		cuda_get_arch(thr_id);

		if (sm5) {
			skeincoin_init(thr_id);
		} else {
			cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput);
			quark_skein512_cpu_init(thr_id, throughput);
			cuda_check_cpu_init(thr_id, throughput);
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
		}

		init[thr_id] = true;
	}

	for (int k=0; k < 19; k++)
		be32enc(&endiandata[k], pdata[k]);

	if (sm5) {
		skeincoin_setBlock_80(thr_id, (void*)endiandata);
		target64 = ((uint64_t*)ptarget)[3];
	} else {
		skein512_cpu_setBlock_80((void*)endiandata);
		cuda_check_cpu_setTarget(ptarget);
	}

	do {
		// Hash with CUDA
		*hashes_done = pdata[19] - first_nonce + throughput;

		if (sm5) {
			/* cuda_skeincoin.cu */
			work->nonces[0] = skeincoin_hash_sm5(thr_id, throughput, pdata[19], 1, target64, &work->nonces[1]);
		} else {
			/* quark/cuda_skein512.cu */
			skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
			sha2_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]);
			work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
		}

		if (work->nonces[0] != UINT32_MAX)
		{
			uint32_t _ALIGN(64) vhash[8];

			endiandata[19] = swab32(work->nonces[0]);
			skeincoinhash(vhash, endiandata);
			if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work->valid_nonces = 1;
				work_set_target_ratio(work, vhash);
				if (checkSecnonce) {
					work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], work->valid_nonces);
					if (work->nonces[1] != 0) {
						endiandata[19] = swab32(work->nonces[1]);
						skeincoinhash(vhash, endiandata);
						if (vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
							work->valid_nonces++;
							bn_set_target_ratio(work, vhash, 1);
						}
						pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
					} else {
						pdata[19] = work->nonces[0] + 1;
					}
				} else {
					pdata[19] = work->nonces[0] + 1; // cursor for next scan
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

	*hashes_done = pdata[19] - first_nonce;

	return 0;
}

// cleanup
extern "C" void free_skeincoin(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaDeviceSynchronize();

	if (sm5)
		skeincoin_free(thr_id);
	else {
		cudaFree(d_hash[thr_id]);
		cuda_check_cpu_free(thr_id);
	}

	init[thr_id] = false;

	cudaDeviceSynchronize();
}
