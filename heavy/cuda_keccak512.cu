#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

// globaler Speicher für alle HeftyHashes aller Threads
extern uint32_t *heavy_heftyHashes[MAX_GPUS];
extern uint32_t *heavy_nonceVector[MAX_GPUS];

// globaler Speicher für unsere Ergebnisse
uint32_t *d_hash3output[MAX_GPUS];
extern uint32_t *d_hash4output[MAX_GPUS];
extern uint32_t *d_hash5output[MAX_GPUS];

// der Keccak512 State nach der ersten Runde (72 Bytes)
__constant__ uint64_t c_State[25];

// die Message (72 Bytes) für die zweite Runde auf der GPU
__constant__ uint32_t c_PaddedMessage2[18]; // 44 bytes of remaining message (Nonce at offset 4) plus padding

// ---------------------------- BEGIN CUDA keccak512 functions ------------------------------------

#define U32TO64_LE(p) \
	(((uint64_t)(*p)) | (((uint64_t)(*(p + 1))) << 32))

#define U64TO32_LE(p, v) \
	*p = (uint32_t)((v)); *(p+1) = (uint32_t)((v) >> 32);

static __device__ void mycpy72(uint32_t *d, const uint32_t *s) {
	#pragma unroll 18
	for (int k=0; k < 18; ++k) d[k] = s[k];
}

static __device__ void mycpy32(uint32_t *d, const uint32_t *s) {
	#pragma unroll 8
	for (int k=0; k < 8; ++k) d[k] = s[k];
}

typedef struct keccak_hash_state_t {
	uint64_t state[25];    // 25*2
	uint32_t buffer[72/4]; // 72
} keccak_hash_state;

__device__ void statecopy(uint64_t *d, uint64_t *s)
{
	#pragma unroll 25
	for (int i=0; i < 25; ++i)
		d[i] = s[i];
}


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

__constant__ uint64_t c_keccak_round_constants[24];

__host__ __device__ void
keccak_block(uint64_t *s, const uint32_t *in, const uint64_t *keccak_round_constants) {
	size_t i;
	uint64_t t[5], u[5], v, w;

	/* absorb input */
	for (i = 0; i < 9 /* 72/8 */; i++, in += 2)
		s[i] ^= U32TO64_LE(in);

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

// Die Hash-Funktion
template <int BLOCKSIZE> __global__ void keccak512_gpu_hash(uint32_t threads, uint32_t startNounce, void *outputHash, uint32_t *heftyHashes, uint32_t *nonceVector)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// bestimme den aktuellen Zähler
		//uint32_t nounce = startNounce + thread;
		uint32_t nounce = nonceVector[thread];

		// Index-Position des Hashes in den Hash Puffern bestimmen (Hefty1 und outputHash)
		uint32_t hashPosition = nounce - startNounce;

		// erstmal den State der ersten Runde holen
		uint64_t keccak_gpu_state[25];
#pragma unroll 25
		for (int i=0; i < 25; ++i)
			keccak_gpu_state[i] = c_State[i];

		// Message2 in den Puffer holen
		uint32_t msgBlock[18];
		mycpy72(msgBlock, c_PaddedMessage2);

		// die individuelle Nonce einsetzen
		msgBlock[1] = nounce;

		// den individuellen Hefty1 Hash einsetzen
		mycpy32(&msgBlock[(BLOCKSIZE-72)/sizeof(uint32_t)], &heftyHashes[8 * hashPosition]);

		// den Block einmal gut durchschütteln
		keccak_block(keccak_gpu_state, msgBlock, c_keccak_round_constants);

		// das Hash erzeugen
		uint32_t hash[16];

#pragma unroll 8
		for (size_t i = 0; i < 64; i += 8) {
			U64TO32_LE((&hash[i/4]), keccak_gpu_state[i / 8]);
		}

		// und ins Global Memory rausschreiben
#pragma unroll 16
		for(int k=0;k<16;k++)
			((uint32_t*)outputHash)[16*hashPosition+k] = hash[k];
	}
}

// ---------------------------- END CUDA keccak512 functions ------------------------------------

__host__
void keccak512_cpu_init(int thr_id, uint32_t threads)
{
	// Kopiere die Hash-Tabellen in den GPU-Speicher
	cudaMemcpyToSymbol( c_keccak_round_constants,
						host_keccak_round_constants,
						sizeof(host_keccak_round_constants),
						0, cudaMemcpyHostToDevice);

	// Speicher für alle Ergebnisse belegen
	cudaMalloc(&d_hash3output[thr_id], (size_t) 64 * threads);
}

__host__
void keccak512_cpu_free(int thr_id)
{
	cudaFree(d_hash3output[thr_id]);
}

// ----------------BEGIN keccak512 CPU version from scrypt-jane code --------------------

#define SCRYPT_HASH_DIGEST_SIZE 64
#define SCRYPT_KECCAK_F 1600
#define SCRYPT_KECCAK_C (SCRYPT_HASH_DIGEST_SIZE * 8 * 2) /* 1024 */
#define SCRYPT_KECCAK_R (SCRYPT_KECCAK_F - SCRYPT_KECCAK_C) /* 576 */
#define SCRYPT_HASH_BLOCK_SIZE (SCRYPT_KECCAK_R / 8) /* 72 */

// --------------- END keccak512 CPU version from scrypt-jane code --------------------

static int BLOCKSIZE = 84;

__host__
void keccak512_cpu_setBlock(void *data, int len)
	// data muss 80 oder 84-Byte haben!
	// heftyHash hat 32-Byte
{
	// CH
	// state init
	uint64_t keccak_cpu_state[25];
	memset(keccak_cpu_state, 0, sizeof(keccak_cpu_state));

	// erste Runde
	keccak_block((uint64_t*)&keccak_cpu_state, (const uint32_t*)data, host_keccak_round_constants);

	// state kopieren
	cudaMemcpyToSymbol( c_State, keccak_cpu_state, 25*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);

	// keccak hat 72-Byte blöcke, d.h. in unserem Fall zwei Blöcke
	// zu jeweils
	uint32_t msgBlock[18];
	memset(msgBlock, 0, 18 * sizeof(uint32_t));

	// kopiere die restlichen Daten rein (aber nur alles nach Byte 72)
	if (len == 84)
		memcpy(&msgBlock[0], &((uint8_t*)data)[72], 12);
	else if (len == 80)
		memcpy(&msgBlock[0], &((uint8_t*)data)[72], 8);

	// Nachricht abschließen
	if (len == 84)
		msgBlock[11] = 0x01;
	else if (len == 80)
		msgBlock[10] = 0x01;
	msgBlock[17] = 0x80000000;

	// Message 2 ins Constant Memory kopieren (die variable Nonce und
	// der Hefty1 Anteil muss aber auf der GPU erst noch ersetzt werden)
	cudaMemcpyToSymbol( c_PaddedMessage2, msgBlock, 18*sizeof(uint32_t), 0, cudaMemcpyHostToDevice );

	BLOCKSIZE = len;
}

__host__
void keccak512_cpu_copyHeftyHash(int thr_id, uint32_t threads, void *heftyHashes, int copy)
{
	// Hefty1 Hashes kopieren
	if (copy)
		CUDA_SAFE_CALL(cudaMemcpy(heavy_heftyHashes[thr_id], heftyHashes, 8 * sizeof(uint32_t) * threads, cudaMemcpyHostToDevice));
	//else cudaDeviceSynchronize();
}

__host__
void keccak512_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce)
{
	const uint32_t threadsperblock = 128;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// Größe des dynamischen Shared Memory Bereichs
	size_t shared_size = 0;

	if (BLOCKSIZE==84)
		keccak512_gpu_hash<84><<<grid, block, shared_size>>>(threads, startNounce, d_hash3output[thr_id], heavy_heftyHashes[thr_id], heavy_nonceVector[thr_id]);
	else if (BLOCKSIZE==80)
		keccak512_gpu_hash<80><<<grid, block, shared_size>>>(threads, startNounce, d_hash3output[thr_id], heavy_heftyHashes[thr_id], heavy_nonceVector[thr_id]);
}
