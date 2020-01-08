#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

// globaler Speicher für alle HeftyHashes aller Threads
extern uint32_t *heavy_heftyHashes[MAX_GPUS];
extern uint32_t *heavy_nonceVector[MAX_GPUS];

// globaler Speicher für unsere Ergebnisse
uint32_t *d_hash2output[MAX_GPUS];


/* Hash-Tabellen */
__constant__ uint32_t sha256_gpu_constantTable[64];

// muss expandiert werden
__constant__ uint32_t sha256_gpu_blockHeader[16]; // 2x512 Bit Message
__constant__ uint32_t sha256_gpu_register[8];

uint32_t sha256_cpu_hashTable[] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
uint32_t sha256_cpu_constantTable[] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

#define S(x, n)			(((x) >> (n)) | ((x) << (32 - (n))))
#define R(x, n)			((x) >> (n))
#define Ch(x, y, z)		((x & (y ^ z)) ^ z)
#define Maj(x, y, z)	((x & (y | z)) | (y & z))
#define S0(x)			(S(x, 2) ^ S(x, 13) ^ S(x, 22))
#define S1(x)			(S(x, 6) ^ S(x, 11) ^ S(x, 25))
#define s0(x)			(S(x, 7) ^ S(x, 18) ^ R(x, 3))
#define s1(x)			(S(x, 17) ^ S(x, 19) ^ R(x, 10))

#define SWAB32(x)		( ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24) )

// Die Hash-Funktion
template <int BLOCKSIZE> __global__ void sha256_gpu_hash(uint32_t threads, uint32_t startNounce, void *outputHash, uint32_t *heftyHashes, uint32_t *nonceVector)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// bestimme den aktuellen Zähler
		uint32_t nounce = startNounce + thread;
		nonceVector[thread] = nounce;

		// jeder thread in diesem  Block bekommt sein eigenes W Array im Shared memory
		uint32_t W1[16];
		uint32_t W2[16];

		// Initialisiere die register a bis h mit der Hash-Tabelle
		uint32_t regs[8];
		uint32_t hash[8];

		// pre
#pragma unroll 8
		for (int k=0; k < 8; k++)
		{
			regs[k] = sha256_gpu_register[k];
			hash[k] = regs[k];
		}

		// 2. Runde
		//memcpy(W, &sha256_gpu_blockHeader[0], sizeof(uint32_t) * 16); // TODO: aufsplitten in zwei Teilblöcke
		//memcpy(&W[5], &heftyHashes[8 * (blockDim.x * blockIdx.x + threadIdx.x)], sizeof(uint32_t) * 8); // den richtigen Hefty1 Hash holen
#pragma unroll 16
		for(int k=0;k<16;k++)
			W1[k] = sha256_gpu_blockHeader[k];

		uint32_t offset = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
#pragma unroll 8
		for(int k=0;k<8;k++)
			W1[((BLOCKSIZE-64)/4)+k] = heftyHashes[offset + k];

#pragma unroll 8
		for (int i=((BLOCKSIZE-64)/4); i < ((BLOCKSIZE-64)/4)+8; ++i) W1[i] = SWAB32(W1[i]); // die Hefty1 Hashes brauchen eine Drehung ;)
		W1[3] = SWAB32(nounce);

// Progress W1
#pragma unroll 16
		for(int j=0;j<16;j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_gpu_constantTable[j] + W1[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

			#pragma unroll 7
			for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}

// Progress W2...W3
#pragma unroll 3
		for(int k=0;k<3;k++)
		{
	#pragma unroll 2
			for(int j=0;j<2;j++)
				W2[j] = s1(W1[14+j]) + W1[9+j] + s0(W1[1+j]) + W1[j];
	#pragma unroll 5
			for(int j=2;j<7;j++)
				W2[j] = s1(W2[j-2]) + W1[9+j] + s0(W1[1+j]) + W1[j];

	#pragma unroll 8
			for(int j=7;j<15;j++)
				W2[j] = s1(W2[j-2]) + W2[j-7] + s0(W1[1+j]) + W1[j];

			W2[15] = s1(W2[13]) + W2[8] + s0(W2[0]) + W1[15];

			// Rundenfunktion
	#pragma unroll 16
			for(int j=0;j<16;j++)
			{
				uint32_t T1, T2;
				T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_gpu_constantTable[j + 16 * (k+1)] + W2[j];
				T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

				#pragma unroll 7
				for (int l=6; l >= 0; l--) regs[l+1] = regs[l];
				regs[0] = T1 + T2;
				regs[4] += T1;
			}

	#pragma unroll 16
			for(int j=0;j<16;j++)
				W1[j] = W2[j];
		}

/*
		for(int j=16;j<64;j++)
			W[j] = s1(W[j-2]) + W[j-7] + s0(W[j-15]) + W[j-16];

#pragma unroll 64
		for(int j=0;j<64;j++)
		{
			uint32_t T1, T2;
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_gpu_constantTable[j] + W[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

			#pragma unroll 7
			for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}
*/
#pragma unroll 8
		for(int k=0;k<8;k++)
			hash[k] += regs[k];

#pragma unroll 8
		for(int k=0;k<8;k++)
			((uint32_t*)outputHash)[8*thread+k] = SWAB32(hash[k]);
	}
}

// Setup Function
__host__
void sha256_cpu_init(int thr_id, uint32_t threads)
{
	// Kopiere die Hash-Tabellen in den GPU-Speicher
	cudaMemcpyToSymbol(	sha256_gpu_constantTable,
						sha256_cpu_constantTable,
						sizeof(uint32_t) * 64 );

	// Speicher für alle Ergebnisse belegen
	cudaMalloc(&d_hash2output[thr_id], (size_t) 8 * sizeof(uint32_t) * threads);
}

__host__
void sha256_cpu_free(int thr_id)
{
	cudaFree(d_hash2output[thr_id]);
}

static int BLOCKSIZE = 84;

__host__ void sha256_cpu_setBlock(void *data, int len)
	// data muss 80/84-Byte haben!
	// heftyHash hat 32-Byte
{
	// Nachricht expandieren und setzen
	uint32_t msgBlock[32];

	memset(msgBlock, 0, sizeof(uint32_t) * 32);
	memcpy(&msgBlock[0], data, len);
	if (len == 84) {
		memset(&msgBlock[21], 0, 32); // vorläufig  Nullen anstatt der Hefty1 Hashes einfüllen
		msgBlock[29] |= 0x80;
		msgBlock[31] = 928; // bitlen
	} else if (len == 80) {
		memset(&msgBlock[20], 0, 32); // vorläufig  Nullen anstatt der Hefty1 Hashes einfüllen
		msgBlock[28] |= 0x80;
		msgBlock[31] = 896; // bitlen
	}

	for(int i=0;i<31;i++) // Byteorder drehen
		msgBlock[i] = SWAB32(msgBlock[i]);

	// die erste Runde wird auf der CPU durchgeführt, da diese für
	// alle Threads gleich ist. Der Hash wird dann an die Threads
	// übergeben
	uint32_t W[64];

	// Erstelle expandierten Block W
	memcpy(W, &msgBlock[0], sizeof(uint32_t) * 16);
	for(int j=16;j<64;j++)
		W[j] = s1(W[j-2]) + W[j-7] + s0(W[j-15]) + W[j-16];

	// Initialisiere die register a bis h mit der Hash-Tabelle
	uint32_t regs[8];
	uint32_t hash[8];

	// pre
	for (int k=0; k < 8; k++)
	{
		regs[k] = sha256_cpu_hashTable[k];
		hash[k] = regs[k];
	}

	// 1. Runde
	for(int j=0;j<64;j++)
	{
		uint32_t T1, T2;
		T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_cpu_constantTable[j] + W[j];
		T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

		//#pragma unroll 7
		for (int k=6; k >= 0; k--) regs[k+1] = regs[k];
		// sollte mal noch durch memmov ersetzt werden!
//		memcpy(&regs[1], &regs[0], sizeof(uint32_t) * 7);
		regs[0] = T1 + T2;
		regs[4] += T1;
	}

	for(int k=0;k<8;k++)
		hash[k] += regs[k];

	// hash speichern
	cudaMemcpyToSymbol(	sha256_gpu_register,
						hash,
						sizeof(uint32_t) * 8 );

	// Blockheader setzen (korrekte Nonce und Hefty Hash fehlen da drin noch)
	cudaMemcpyToSymbol(	sha256_gpu_blockHeader,
						&msgBlock[16],
						64);

	BLOCKSIZE = len;
}

__host__ void sha256_cpu_copyHeftyHash(int thr_id, uint32_t threads, void *heftyHashes, int copy)
{
	// Hefty1 Hashes kopieren
	if (copy)
		CUDA_SAFE_CALL(cudaMemcpy(heavy_heftyHashes[thr_id], heftyHashes, 8 * sizeof(uint32_t) * threads, cudaMemcpyHostToDevice));
	//else cudaDeviceSynchronize();
}

__host__ void sha256_cpu_hash(int thr_id, uint32_t threads, int startNounce)
{
	const uint32_t threadsperblock = 256;

	// berechne wie viele Thread Blocks wir brauchen
	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	// Größe des dynamischen Shared Memory Bereichs
	size_t shared_size = 0;

	if (BLOCKSIZE == 84)
		sha256_gpu_hash<84><<<grid, block, shared_size>>>(threads, startNounce, d_hash2output[thr_id], heavy_heftyHashes[thr_id], heavy_nonceVector[thr_id]);
	else if (BLOCKSIZE == 80) {
		sha256_gpu_hash<80><<<grid, block, shared_size>>>(threads, startNounce, d_hash2output[thr_id], heavy_heftyHashes[thr_id], heavy_nonceVector[thr_id]);
	}
}
