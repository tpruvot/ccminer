#include "miner.h"

extern "C" {
#include <stdint.h>
#include <memory.h>
}

#include "cuda_helper.h"

static const uint64_t host_keccak_round_constants[24] = {
	0x0000000000000001ull, 0x0000000000008082ull,
	0x800000000000808aull, 0x8000000080008000ull,
	0x000000000000808bull, 0x0000000080000001ull,
	0x8000000080008081ull, 0x8000000000008009ull,
	0x000000000000008aull, 0x0000000000000088ull,
	0x0000000080008009ull, 0x000000008000000aull,
	0x000000008000808bull, 0x800000000000008bull,
	0x8000000000008089ull, 0x8000000000008003ull,
	0x8000000000008002ull, 0x8000000000000080ull,
	0x000000000000800aull, 0x800000008000000aull,
	0x8000000080008081ull, 0x8000000000008080ull,
	0x0000000080000001ull, 0x8000000080008008ull
};

static uint32_t *d_KNonce[MAX_GPUS];

__constant__ uint32_t pTarget[8];
__constant__ uint64_t keccak_round_constants[24];
__constant__ uint64_t c_PaddedMessage80[10]; // padded message (80 bytes + padding?)

#if __CUDA_ARCH__ >= 350
__device__ __forceinline__
static void keccak_blockv35(uint2 *s, const uint64_t *keccak_round_constants)
{
	size_t i;
	uint2 t[5], u[5], v, w;

	#pragma unroll
	for (i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROL2(t[1], 1);
		u[1] = t[0] ^ ROL2(t[2], 1);
		u[2] = t[1] ^ ROL2(t[3], 1);
		u[3] = t[2] ^ ROL2(t[4], 1);
		u[4] = t[3] ^ ROL2(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1] = ROL2(s[6], 44);
		s[6] = ROL2(s[9], 20);
		s[9] = ROL2(s[22], 61);
		s[22] = ROL2(s[14], 39);
		s[14] = ROL2(s[20], 18);
		s[20] = ROL2(s[2], 62);
		s[2] = ROL2(s[12], 43);
		s[12] = ROL2(s[13], 25);
		s[13] = ROL2(s[19], 8);
		s[19] = ROL2(s[23], 56);
		s[23] = ROL2(s[15], 41);
		s[15] = ROL2(s[4], 27);
		s[4] = ROL2(s[24], 14);
		s[24] = ROL2(s[21], 2);
		s[21] = ROL2(s[8], 55);
		s[8] = ROL2(s[16], 45);
		s[16] = ROL2(s[5], 36);
		s[5] = ROL2(s[3], 28);
		s[3] = ROL2(s[18], 21);
		s[18] = ROL2(s[17], 15);
		s[17] = ROL2(s[11], 10);
		s[11] = ROL2(s[7], 6);
		s[7] = ROL2(s[10], 3);
		s[10] = ROL2(v, 1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
		v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= vectorize(keccak_round_constants[i]);
	}
}
#else

__device__ __forceinline__
static void keccak_blockv30(uint64_t *s, const uint64_t *keccak_round_constants)
{
	size_t i;
	uint64_t t[5], u[5], v, w;

	/* absorb input */

	for (i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROTL64(t[1], 1);
		u[1] = t[0] ^ ROTL64(t[2], 1);
		u[2] = t[1] ^ ROTL64(t[3], 1);
		u[3] = t[2] ^ ROTL64(t[4], 1);
		u[4] = t[3] ^ ROTL64(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[ 1];
		s[ 1] = ROTL64(s[ 6], 44);
		s[ 6] = ROTL64(s[ 9], 20);
		s[ 9] = ROTL64(s[22], 61);
		s[22] = ROTL64(s[14], 39);
		s[14] = ROTL64(s[20], 18);
		s[20] = ROTL64(s[ 2], 62);
		s[ 2] = ROTL64(s[12], 43);
		s[12] = ROTL64(s[13], 25);
		s[13] = ROTL64(s[19],  8);
		s[19] = ROTL64(s[23], 56);
		s[23] = ROTL64(s[15], 41);
		s[15] = ROTL64(s[ 4], 27);
		s[ 4] = ROTL64(s[24], 14);
		s[24] = ROTL64(s[21],  2);
		s[21] = ROTL64(s[ 8], 55);
		s[ 8] = ROTL64(s[16], 45);
		s[16] = ROTL64(s[ 5], 36);
		s[ 5] = ROTL64(s[ 3], 28);
		s[ 3] = ROTL64(s[18], 21);
		s[18] = ROTL64(s[17], 15);
		s[17] = ROTL64(s[11], 10);
		s[11] = ROTL64(s[ 7],  6);
		s[ 7] = ROTL64(s[10],  3);
		s[10] = ROTL64(    v,  1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[ 0]; w = s[ 1]; s[ 0] ^= (~w) & s[ 2]; s[ 1] ^= (~s[ 2]) & s[ 3]; s[ 2] ^= (~s[ 3]) & s[ 4]; s[ 3] ^= (~s[ 4]) & v; s[ 4] ^= (~v) & w;
		v = s[ 5]; w = s[ 6]; s[ 5] ^= (~w) & s[ 7]; s[ 6] ^= (~s[ 7]) & s[ 8]; s[ 7] ^= (~s[ 8]) & s[ 9]; s[ 8] ^= (~s[ 9]) & v; s[ 9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= keccak_round_constants[i];
	}
}
#endif

__global__ __launch_bounds__(128,5)
void keccak256_sm3_gpu_hash_80(uint32_t threads, uint32_t startNounce, uint32_t *resNounce)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = startNounce + thread;

#if __CUDA_ARCH__ >= 350
		uint2 keccak_gpu_state[25];
		#pragma unroll 25
		for (int i=0; i<25; i++) {
			if (i<9) keccak_gpu_state[i] = vectorize(c_PaddedMessage80[i]);
			else     keccak_gpu_state[i] = make_uint2(0, 0);
		}

		keccak_gpu_state[9]= vectorize(c_PaddedMessage80[9]);
		keccak_gpu_state[9].y = cuda_swab32(nounce);
		keccak_gpu_state[10] = make_uint2(1, 0);
		keccak_gpu_state[16] = make_uint2(0, 0x80000000);

		keccak_blockv35(keccak_gpu_state,keccak_round_constants);
		if (devectorize(keccak_gpu_state[3]) <= ((uint64_t*)pTarget)[3]) {resNounce[0] = nounce;}
#else
		uint64_t keccak_gpu_state[25];
		#pragma unroll 25
		for (int i=0; i<25; i++) {
			if (i<9) keccak_gpu_state[i] = c_PaddedMessage80[i];
			else     keccak_gpu_state[i] = 0;
		}
		keccak_gpu_state[9]  = REPLACE_HIDWORD(c_PaddedMessage80[9], cuda_swab32(nounce));
		keccak_gpu_state[10] = 0x0000000000000001;
		keccak_gpu_state[16] = 0x8000000000000000;

		keccak_blockv30(keccak_gpu_state, keccak_round_constants);
		if (keccak_gpu_state[3] <= ((uint64_t*)pTarget)[3]) { resNounce[0] = nounce; }
#endif
	}
}

__host__
void keccak256_sm3_hash_80(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *resNonces, int order)
{
	cudaMemset(d_KNonce[thr_id], 0xff, 2*sizeof(uint32_t));
	const uint32_t threadsperblock = 128;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	size_t shared_size = 0;

	keccak256_sm3_gpu_hash_80<<<grid, block, shared_size>>>(threads, startNounce, d_KNonce[thr_id]);

	cudaMemcpy(resNonces, d_KNonce[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}

#if 0
__global__ __launch_bounds__(256,3)
void keccak256_sm3_gpu_hash_32(uint32_t threads, uint32_t startNounce, uint64_t *outputHash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
#if __CUDA_ARCH__ >= 350 /* tpr: to double check if faster on SM5+ */
		uint2 keccak_gpu_state[25];
		#pragma unroll 25
		for (int i = 0; i<25; i++) {
			if (i<4) keccak_gpu_state[i] = vectorize(outputHash[i*threads+thread]);
			else     keccak_gpu_state[i] = make_uint2(0, 0);
		}
		keccak_gpu_state[4]  = make_uint2(1, 0);
		keccak_gpu_state[16] = make_uint2(0, 0x80000000);
		keccak_blockv35(keccak_gpu_state, keccak_round_constants);

		#pragma unroll 4
		for (int i=0; i<4; i++)
			outputHash[i*threads+thread] = devectorize(keccak_gpu_state[i]);
#else
		uint64_t keccak_gpu_state[25];
		#pragma unroll 25
		for (int i = 0; i<25; i++) {
			if (i<4)
				keccak_gpu_state[i] = outputHash[i*threads+thread];
			else
				keccak_gpu_state[i] = 0;
		}
		keccak_gpu_state[4]  = 0x0000000000000001;
		keccak_gpu_state[16] = 0x8000000000000000;

		keccak_blockv30(keccak_gpu_state, keccak_round_constants);
		#pragma unroll 4
		for (int i = 0; i<4; i++)
			outputHash[i*threads + thread] = keccak_gpu_state[i];
#endif
	}
}

__host__
void keccak256_sm3_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, int order)
{
	const uint32_t threadsperblock = 256;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	keccak256_sm3_gpu_hash_32 <<<grid, block>>> (threads, startNounce, d_outputHash);
	MyStreamSynchronize(NULL, order, thr_id);
}
#endif

__host__
void keccak256_sm3_setBlock_80(void *pdata,const void *pTargetIn)
{
	unsigned char PaddedMessage[80];
	memcpy(PaddedMessage, pdata, 80);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(pTarget, pTargetIn, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 10*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

__host__
void keccak256_sm3_init(int thr_id, uint32_t threads)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(keccak_round_constants, host_keccak_round_constants,
				sizeof(host_keccak_round_constants), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMalloc(&d_KNonce[thr_id], 2*sizeof(uint32_t)));
}

__host__
void keccak256_sm3_free(int thr_id)
{
	cudaFree(d_KNonce[thr_id]);
}
