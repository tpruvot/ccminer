/**
 * KECCAK-256 CUDA optimised implementation, based on ccminer-alexis code
 */

#include <miner.h>

extern "C" {
#include <stdint.h>
#include <memory.h>
}

#include <cuda_helper.h>
#include <cuda_vectors.h>

#define TPB52 1024
#define TPB50 384
#define NPT 2
#define NBN 2

static uint32_t *d_nonces[MAX_GPUS];
static uint32_t *h_nonces[MAX_GPUS];

__constant__ uint2 c_message48[6];
__constant__ uint2 c_mid[17];

__constant__ uint2 keccak_round_constants[24] = {
	{ 0x00000001, 0x00000000 }, { 0x00008082, 0x00000000 },	{ 0x0000808a, 0x80000000 }, { 0x80008000, 0x80000000 },
	{ 0x0000808b, 0x00000000 }, { 0x80000001, 0x00000000 },	{ 0x80008081, 0x80000000 }, { 0x00008009, 0x80000000 },
	{ 0x0000008a, 0x00000000 }, { 0x00000088, 0x00000000 },	{ 0x80008009, 0x00000000 }, { 0x8000000a, 0x00000000 },
	{ 0x8000808b, 0x00000000 }, { 0x0000008b, 0x80000000 },	{ 0x00008089, 0x80000000 }, { 0x00008003, 0x80000000 },
	{ 0x00008002, 0x80000000 }, { 0x00000080, 0x80000000 },	{ 0x0000800a, 0x00000000 }, { 0x8000000a, 0x80000000 },
	{ 0x80008081, 0x80000000 }, { 0x00008080, 0x80000000 },	{ 0x80000001, 0x00000000 }, { 0x80008008, 0x80000000 }
};

__device__ __forceinline__
uint2 xor3x(const uint2 a,const uint2 b,const uint2 c) {
	uint2 result;
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
	asm ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result.x) : "r"(a.x), "r"(b.x),"r"(c.x)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
	asm ("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result.y) : "r"(a.y), "r"(b.y),"r"(c.y)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
#else
	result = a^b^c;
#endif
	return result;
}

__device__ __forceinline__
uint2 chi(const uint2 a,const uint2 b,const uint2 c) { // keccak chi
	uint2 result;
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
	asm ("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(result.x) : "r"(a.x), "r"(b.x),"r"(c.x)); //0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
	asm ("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(result.y) : "r"(a.y), "r"(b.y),"r"(c.y)); //0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
#else
	result = a ^ (~b) & c;
#endif
	return result;
}

__device__ __forceinline__
uint64_t xor5(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e)
{
	uint64_t result;
	asm("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(d) ,"l"(e));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(c));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(b));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(a));
	return result;
}

#if __CUDA_ARCH__ <= 500
__global__ __launch_bounds__(TPB50, 2)
#else
__global__ __launch_bounds__(TPB52, 1)
#endif
void keccak256_gpu_hash_80(uint32_t threads, uint32_t startNonce, uint32_t *resNounce, const uint2 highTarget)
{
	uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;
	uint2 s[25], t[5], v, w, u[5];
#if __CUDA_ARCH__ > 500
	uint64_t step     = gridDim.x * blockDim.x;
	uint64_t maxNonce = startNonce + threads;
	for(uint64_t nounce = startNonce + thread; nounce<maxNonce;nounce+=step) {
#else
	uint32_t nounce = startNonce+thread;
	if(thread<threads) {
#endif
		s[ 9] = make_uint2(c_message48[0].x,cuda_swab32(nounce));
		s[10] = keccak_round_constants[0];

		t[ 4] = c_message48[1]^s[ 9];
		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[ 0] = t[4] ^ c_mid[ 0];
		u[ 1] = c_mid[ 1] ^ ROL2(t[4],1);
		u[ 2] = c_mid[ 2];
		/* thetarho pi: b[..] = rotl(a[..] ^ d[...], ..)*/
		s[ 7] = ROL2(s[10]^u[0], 3);
		s[10] = c_mid[ 3];
		    w = c_mid[ 4];
		s[20] = c_mid[ 5];
		s[ 6] = ROL2(s[ 9]^u[2],20);
		s[ 9] = c_mid[ 6];
		s[22] = c_mid[ 7];
		s[14] = ROL2(u[0],18);
		s[ 2] = c_mid[ 8];
		s[12] = ROL2(u[1],25);
		s[13] = c_mid[ 9];
		s[19] = ROR8(u[1]);
		s[23] = ROR2(u[0],23);
		s[15] = c_mid[10];
		s[ 4] = c_mid[11];
		s[24] = c_mid[12];
		s[21] = ROR2(c_message48[2]^u[1], 9);
		s[ 8] = c_mid[13];
		s[16] = ROR2(c_message48[3]^u[0],28);
		s[ 5] = ROL2(c_message48[4]^u[1],28);
		s[ 3] = ROL2(u[1],21);
		s[18] = c_mid[14];
		s[17] = c_mid[15];
		s[11] = c_mid[16];

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = c_message48[5]^u[0];
		s[ 0] = chi(v,w,s[ 2]);
		s[ 1] = chi(w,s[ 2],s[ 3]);
		s[ 2] = chi(s[ 2],s[ 3],s[ 4]);
		s[ 3] = chi(s[ 3],s[ 4],v);
		s[ 4] = chi(s[ 4],v,w);
		v = s[ 5]; w = s[ 6]; s[ 5] = chi(v,w,s[ 7]); s[ 6] = chi(w,s[ 7],s[ 8]); s[ 7] = chi(s[ 7],s[ 8],s[ 9]); s[ 8] = chi(s[ 8],s[ 9],v);s[ 9] = chi(s[ 9],v,w);
		v = s[10]; w = s[11]; s[10] = chi(v,w,s[12]); s[11] = chi(w,s[12],s[13]); s[12] = chi(s[12],s[13],s[14]); s[13] = chi(s[13],s[14],v);s[14] = chi(s[14],v,w);
		v = s[15]; w = s[16]; s[15] = chi(v,w,s[17]); s[16] = chi(w,s[17],s[18]); s[17] = chi(s[17],s[18],s[19]); s[18] = chi(s[18],s[19],v);s[19] = chi(s[19],v,w);
		v = s[20]; w = s[21]; s[20] = chi(v,w,s[22]); s[21] = chi(w,s[22],s[23]); s[22] = chi(s[22],s[23],s[24]); s[23] = chi(s[23],s[24],v);s[24] = chi(s[24],v,w);

		/* iota: a[0,0] ^= round constant */
		s[ 0] ^=keccak_round_constants[ 0];

		#if __CUDA_ARCH__ > 500
			#pragma unroll 22
		#else
			#pragma unroll 4
		#endif
		for (int i = 1; i < 23; i++) {
			#pragma unroll
			for(int j=0;j<5;j++) {
				t[ j] = vectorize(xor5(devectorize(s[ j]),devectorize(s[j+5]),devectorize(s[j+10]),devectorize(s[j+15]),devectorize(s[j+20])));
			}
			/*theta*/
			#pragma unroll
			for(int j=0;j<5;j++) {
				u[j] = ROL2(t[j], 1);
			}
			s[ 4] = xor3x(s[ 4], t[3], u[0]);s[ 9] = xor3x(s[ 9], t[3], u[0]);s[14] = xor3x(s[14], t[3], u[0]);s[19] = xor3x(s[19], t[3], u[0]);s[24] = xor3x(s[24], t[3], u[0]);
			s[ 0] = xor3x(s[ 0], t[4], u[1]);s[ 5] = xor3x(s[ 5], t[4], u[1]);s[10] = xor3x(s[10], t[4], u[1]);s[15] = xor3x(s[15], t[4], u[1]);s[20] = xor3x(s[20], t[4], u[1]);
			s[ 1] = xor3x(s[ 1], t[0], u[2]);s[ 6] = xor3x(s[ 6], t[0], u[2]);s[11] = xor3x(s[11], t[0], u[2]);s[16] = xor3x(s[16], t[0], u[2]);s[21] = xor3x(s[21], t[0], u[2]);
			s[ 2] = xor3x(s[ 2], t[1], u[3]);s[ 7] = xor3x(s[ 7], t[1], u[3]);s[12] = xor3x(s[12], t[1], u[3]);s[17] = xor3x(s[17], t[1], u[3]);s[22] = xor3x(s[22], t[1], u[3]);
			s[ 3] = xor3x(s[ 3], t[2], u[4]);s[ 8] = xor3x(s[ 8], t[2], u[4]);s[13] = xor3x(s[13], t[2], u[4]);s[18] = xor3x(s[18], t[2], u[4]);s[23] = xor3x(s[23], t[2], u[4]);
			/*rho pi: b[..] = rotl(a[..] ^ d[...], ..)*/
			v = s[ 1];
			s[ 1] = ROL2(s[ 6],44);	s[ 6] = ROL2(s[ 9],20);	s[ 9] = ROL2(s[22],61);	s[22] = ROL2(s[14],39);
			s[14] = ROL2(s[20],18);	s[20] = ROL2(s[ 2],62);	s[ 2] = ROL2(s[12],43);	s[12] = ROL2(s[13],25);
			s[13] = ROL8(s[19]);	s[19] = ROR8(s[23]);	s[23] = ROL2(s[15],41);	s[15] = ROL2(s[ 4],27);
			s[ 4] = ROL2(s[24],14);	s[24] = ROL2(s[21], 2);	s[21] = ROL2(s[ 8],55);	s[ 8] = ROL2(s[16],45);
			s[16] = ROL2(s[ 5],36);	s[ 5] = ROL2(s[ 3],28);	s[ 3] = ROL2(s[18],21);	s[18] = ROL2(s[17],15);
			s[17] = ROL2(s[11],10);	s[11] = ROL2(s[ 7], 6);	s[ 7] = ROL2(s[10], 3);	s[10] = ROL2(v, 1);
			/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
			#pragma unroll
			for(int j=0;j<25;j+=5) {
				v=s[j];w=s[j + 1];s[j] = chi(s[j],s[j+1],s[j+2]);s[j+1] = chi(s[j+1],s[j+2],s[j+3]);s[j+2]=chi(s[j+2],s[j+3],s[j+4]);s[j+3]=chi(s[j+3],s[j+4],v);s[j+4]=chi(s[j+4],v,w);
			}
			/* iota: a[0,0] ^= round constant */
			s[ 0] ^=keccak_round_constants[ i];
		}
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		#pragma unroll 5
		for(int j=0;j<5;j++) {
			t[ j] = xor3x(xor3x(s[j+0],s[j+5],s[j+10]), s[j+15], s[j+20]);
		}
		s[24] = xor3x(s[24],t[3],ROL2(t[0],1));
		s[18] = xor3x(s[18],t[2],ROL2(t[4],1));
		s[ 0] = xor3x(s[ 0],t[4],ROL2(t[1],1));
		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		s[24] = ROL2(s[24],14);
		s[18] = ROL2(s[18],21);
		if (devectorize(chi(s[18],s[24],s[ 0])) <= devectorize(highTarget)) {
//		if(chi(s[18].x,s[24].x,s[0].x)<=highTarget.x) {
//			if(chi(s[18].y,s[24].y,s[0].y)<=highTarget.y) {
				const uint32_t tmp = atomicExch(&resNounce[0], nounce);
				if (tmp != UINT32_MAX)
					resNounce[1] = tmp;
	//			return;
//			}
		}
	}
}

__host__
void keccak256_cpu_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t* resNonces, const uint2 highTarget)
{
	uint32_t tpb;
	dim3 grid;
	if (device_sm[device_map[thr_id]] <= 500) {
		tpb = TPB50;
		grid.x = (threads + tpb-1)/tpb;
	} else {
		tpb = TPB52;
		grid.x = (threads + (NPT*tpb)-1)/(NPT*tpb);
	}
	const dim3 block(tpb);

	keccak256_gpu_hash_80<<<grid, block>>>(threads, startNonce, d_nonces[thr_id], highTarget);
//	cudaDeviceSynchronize();
	cudaMemcpy(h_nonces[thr_id], d_nonces[thr_id], NBN*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	memcpy(resNonces, h_nonces[thr_id], NBN*sizeof(uint32_t));
}

#if 0
#if __CUDA_ARCH__ <= 500
__global__ __launch_bounds__(TPB50, 2)
#else
__global__ __launch_bounds__(TPB52, 1)
#endif
void keccak256_gpu_hash_32(uint32_t threads, uint2* outputHash)
{
	uint32_t thread   = blockDim.x * blockIdx.x + threadIdx.x;
	uint2 s[25], t[5], v, w, u[5];

	if(thread < threads) {
		#pragma unroll 25
		for (int i = 0; i<25; i++) {
			if (i<4) s[i] = __ldg(&outputHash[i*threads+thread]);
			else     s[i] = make_uint2(0, 0);
		}
		s[4]  = keccak_round_constants[ 0];
		s[16] = make_uint2(0, 0x80000000);
		#if __CUDA_ARCH__ > 500
			#pragma unroll
		#else
			#pragma unroll 4
		#endif
		for (uint32_t i = 0; i < 23; i++) {
			/*theta*/
			#pragma unroll 5
			for(int j=0; j<5; j++) {
				t[ j] = vectorize(xor5(devectorize(s[ j]),devectorize(s[j+5]),devectorize(s[j+10]),devectorize(s[j+15]),devectorize(s[j+20])));
			}
			/*theta*/
			#pragma unroll 5
			for(int j=0; j<5; j++) {
				u[j] = ROL2(t[j], 1);
			}
			s[ 4] = xor3x(s[ 4], t[3], u[0]);s[ 9] = xor3x(s[ 9], t[3], u[0]);s[14] = xor3x(s[14], t[3], u[0]);s[19] = xor3x(s[19], t[3], u[0]);s[24] = xor3x(s[24], t[3], u[0]);
			s[ 0] = xor3x(s[ 0], t[4], u[1]);s[ 5] = xor3x(s[ 5], t[4], u[1]);s[10] = xor3x(s[10], t[4], u[1]);s[15] = xor3x(s[15], t[4], u[1]);s[20] = xor3x(s[20], t[4], u[1]);
			s[ 1] = xor3x(s[ 1], t[0], u[2]);s[ 6] = xor3x(s[ 6], t[0], u[2]);s[11] = xor3x(s[11], t[0], u[2]);s[16] = xor3x(s[16], t[0], u[2]);s[21] = xor3x(s[21], t[0], u[2]);
			s[ 2] = xor3x(s[ 2], t[1], u[3]);s[ 7] = xor3x(s[ 7], t[1], u[3]);s[12] = xor3x(s[12], t[1], u[3]);s[17] = xor3x(s[17], t[1], u[3]);s[22] = xor3x(s[22], t[1], u[3]);
			s[ 3] = xor3x(s[ 3], t[2], u[4]);s[ 8] = xor3x(s[ 8], t[2], u[4]);s[13] = xor3x(s[13], t[2], u[4]);s[18] = xor3x(s[18], t[2], u[4]);s[23] = xor3x(s[23], t[2], u[4]);
			/*rho pi: b[..] = rotl(a[..] ^ d[...], ..)*/
			v = s[ 1];
			s[ 1] = ROL2(s[ 6],44); s[ 6] = ROL2(s[ 9],20); s[ 9] = ROL2(s[22],61); s[22] = ROL2(s[14],39);
			s[14] = ROL2(s[20],18); s[20] = ROL2(s[ 2],62); s[ 2] = ROL2(s[12],43); s[12] = ROL2(s[13],25);
			s[13] = ROL8(s[19]);    s[19] = ROR8(s[23]);    s[23] = ROL2(s[15],41); s[15] = ROL2(s[ 4],27);
			s[ 4] = ROL2(s[24],14); s[24] = ROL2(s[21], 2); s[21] = ROL2(s[ 8],55); s[ 8] = ROL2(s[16],45);
			s[16] = ROL2(s[ 5],36); s[ 5] = ROL2(s[ 3],28); s[ 3] = ROL2(s[18],21); s[18] = ROL2(s[17],15);
			s[17] = ROL2(s[11],10); s[11] = ROL2(s[ 7], 6); s[ 7] = ROL2(s[10], 3); s[10] = ROL2(v, 1);
			/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
			#pragma unroll 5
			for(int j=0; j<25; j+=5) {
				v=s[j];w=s[j + 1]; s[j] = chi(v,w,s[j+2]); s[j+1] = chi(w,s[j+2],s[j+3]); s[j+2]=chi(s[j+2],s[j+3],s[j+4]); s[j+3]=chi(s[j+3],s[j+4],v); s[j+4]=chi(s[j+4],v,w);
			}
			/* iota: a[0,0] ^= round constant */
			s[ 0] ^=keccak_round_constants[ i];
		}
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		#pragma unroll 5
		for(int j=0;j<5;j++) {
			t[ j] = xor3x(xor3x(s[j+0],s[j+5],s[j+10]), s[j+15], s[j+20]);
		}
		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		#pragma unroll 5
		for(int j=0;j<5;j++) {
			u[j] = ROL2(t[j],1);
		}
		/* thetarho pi: b[..] = rotl(a[..] ^ d[...], ..) //There's no need to perform theta and -store- the result since it's unique for each a[..]*/
		s[ 4] = xor3x(s[24], t[3], u[0]);
		s[ 0] = xor3x(s[ 0], t[4], u[1]);
		s[ 1] = xor3x(s[ 6], t[0], u[2]);
		s[ 2] = xor3x(s[12], t[1], u[3]);
		s[ 3] = xor3x(s[18], t[2], u[4]);
		s[ 1] = ROR2(s[ 1],20);
		s[ 2] = ROR2(s[ 2],21);
		s[ 3] = ROL2(s[ 3],21);
		s[ 4] = ROL2(s[ 4],14);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		outputHash[0*threads+thread] = chi(s[ 0],s[ 1],s[ 2]) ^ keccak_round_constants[23];
		outputHash[1*threads+thread] = chi(s[ 1],s[ 2],s[ 3]);
		outputHash[2*threads+thread] = chi(s[ 2],s[ 3],s[ 4]);
		outputHash[3*threads+thread] = chi(s[ 3],s[ 4],s[ 0]);
	}
}

__host__
void keccak256_cpu_hash_32(const int thr_id,const uint32_t threads, uint2* d_hash)
{
	uint32_t tpb = TPB52;
	if (device_sm[device_map[thr_id]] == 500) tpb = TPB50;
	const dim3 grid((threads + tpb-1)/tpb);
	const dim3 block(tpb);

	keccak256_gpu_hash_32 <<<grid, block>>> (threads, d_hash);
}
#endif

__host__
void keccak256_setBlock_80(uint64_t *endiandata)
{
	uint64_t midstate[17], s[25];
	uint64_t t[5], u[5];

	s[10] = 1; //(uint64_t)make_uint2(1, 0);
	s[16] = ((uint64_t)1)<<63; //(uint64_t)make_uint2(0, 0x80000000);

	t[0] = endiandata[0] ^ endiandata[5] ^ s[10];
	t[1] = endiandata[1] ^ endiandata[6] ^ s[16];
	t[2] = endiandata[2] ^ endiandata[7];
	t[3] = endiandata[3] ^ endiandata[8];

	midstate[ 0] = ROTL64(t[1], 1);         //u[0] -partial
	       u[1] = t[ 0] ^ ROTL64(t[2], 1);  //u[1]
	       u[2] = t[ 1] ^ ROTL64(t[3], 1);  //u[2]
	midstate[ 1] = t[ 2];                   //u[3] -partial
	midstate[ 2] = t[ 3] ^ ROTL64(t[0], 1); //u[4]
	midstate[ 3] = ROTL64(endiandata[1]^u[1], 1); //v
	midstate[ 4] = ROTL64(endiandata[6]^u[1], 44);
	midstate[ 5] = ROTL64(endiandata[2]^u[2], 62);
	midstate[ 6] = ROTL64(u[2], 61);
	midstate[ 7] = ROTL64(midstate[2], 39);
	midstate[ 8] = ROTL64(u[2], 43);
	midstate[ 9] = ROTL64(midstate[2], 8);
	midstate[10] = ROTL64(endiandata[4]^midstate[ 2],27);
	midstate[11] = ROTL64(midstate[2], 14);
	midstate[12] = ROTL64(u[1], 2);
	midstate[13] = ROTL64(s[16] ^ u[1], 45);
	midstate[14] = ROTL64(u[2],15);
	midstate[15] = ROTL64(u[1],10);
	midstate[16] = ROTL64(endiandata[7]^u[2], 6);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_mid, midstate,17*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));

	// pass only what's needed
	uint64_t message48[6];
	message48[0] = endiandata[9];
	message48[1] = endiandata[4];
	message48[2] = endiandata[8];
	message48[3] = endiandata[5];
	message48[4] = endiandata[3];
	message48[5] = endiandata[0];
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_message48, message48, 6*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

__host__
void keccak256_cpu_init(int thr_id)
{
	CUDA_SAFE_CALL(cudaMalloc(&d_nonces[thr_id], NBN*sizeof(uint32_t)));
	//CUDA_SAFE_CALL(cudaMallocHost(&h_nonces[thr_id], NBN*sizeof(uint32_t)));
	h_nonces[thr_id] = (uint32_t*) malloc(NBN * sizeof(uint32_t));
	if(h_nonces[thr_id] == NULL) {
		gpulog(LOG_ERR,thr_id,"Host memory allocation failed");
		exit(EXIT_FAILURE);
	}
}

__host__
void keccak256_setOutput(int thr_id)
{
	CUDA_SAFE_CALL(cudaMemset(d_nonces[thr_id], 0xff, NBN*sizeof(uint32_t)));
}

__host__
void keccak256_cpu_free(int thr_id)
{
	cudaFree(d_nonces[thr_id]);
	//cudaFreeHost(h_nonces[thr_id]);
	free(h_nonces[thr_id]);
}
