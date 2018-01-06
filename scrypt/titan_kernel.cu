/* Copyright (C) 2013 David G. Andersen. All rights reserved.
 * with modifications by Christian Buchner
 *
 * Use of this code is covered under the Apache 2.0 license, which
 * can be found in the file "LICENSE"
 */

//       attempt V.Volkov style ILP (factor 4)

#include <map>

#include <cuda_runtime.h>
#include "cuda_helper.h"
#include "miner.h"

#include "salsa_kernel.h"
#include "titan_kernel.h"

#define THREADS_PER_WU 4  // four threads per hash

typedef enum
{
	ANDERSEN,
	SIMPLE
} MemoryAccess;

#if __CUDA_ARCH__ < 320
	// Kepler (Compute 3.0)
	#define __ldg(x) (*(x))
#endif

#if !defined(__CUDA_ARCH__) ||  __CUDA_ARCH__ >= 300

// scratchbuf constants (pointers to scratch buffer for each warp, i.e. 32 hashes)
__constant__ uint32_t* c_V[TOTAL_WARP_LIMIT];

// iteration count N
__constant__ uint32_t c_N;
__constant__ uint32_t c_N_1;                   // N-1
// scratch buffer size SCRATCH
__constant__ uint32_t c_SCRATCH;
__constant__ uint32_t c_SCRATCH_WU_PER_WARP;   // (SCRATCH * WU_PER_WARP)
__constant__ uint32_t c_SCRATCH_WU_PER_WARP_1; // (SCRATCH * WU_PER_WARP)-1

template <int ALGO> __device__  __forceinline__ void block_mixer(uint4 &b, uint4 &bx, const int x1, const int x2, const int x3);

static __host__ __device__ uint4& operator ^= (uint4& left, const uint4& right) {
	left.x ^= right.x;
	left.y ^= right.y;
	left.z ^= right.z;
	left.w ^= right.w;
	return left;
}

static __host__ __device__ uint4& operator += (uint4& left, const uint4& right) {
	left.x += right.x;
	left.y += right.y;
	left.z += right.z;
	left.w += right.w;
	return left;
}

static __device__ uint4 __shfl(const uint4 bx, int target_thread) {
	return make_uint4(
#if CUDA_VERSION < 9000
            __shfl((int)bx.x, target_thread), 
            __shfl((int)bx.y, target_thread), 
            __shfl((int)bx.z, target_thread), 
            __shfl((int)bx.w, target_thread)
#else
            __shfl_sync(0xFFFFFFFF, (int)bx.x, target_thread), 
            __shfl_sync(0xFFFFFFFF, (int)bx.y, target_thread), 
            __shfl_sync(0xFFFFFFFF, (int)bx.z, target_thread), 
            __shfl_sync(0xFFFFFFFF, (int)bx.w, target_thread)
#endif
            );
}

/* write_keys writes the 8 keys being processed by a warp to the global
 * scratchpad. To effectively use memory bandwidth, it performs the writes
 * (and reads, for read_keys) 128 bytes at a time per memory location
 * by __shfl'ing the 4 entries in bx to the threads in the next-up
 * thread group. It then has eight threads together perform uint4
 * (128 bit) writes to the destination region. This seems to make
 * quite effective use of memory bandwidth. An approach that spread
 * uint32s across more threads was slower because of the increased
 * computation it required.
 *
 * "start" is the loop iteration producing the write - the offset within
 * the block's memory.
 *
 * Internally, this algorithm first __shfl's the 4 bx entries to
 * the next up thread group, and then uses a conditional move to
 * ensure that odd-numbered thread groups exchange the b/bx ordering
 * so that the right parts are written together.
 *
 * Thanks to Babu for helping design the 128-bit-per-write version.
 *
 * _direct lets the caller specify the absolute start location instead of
 * the relative start location, as an attempt to reduce some recomputation.
 */

template <MemoryAccess SCHEME> __device__ __forceinline__
void write_keys_direct(const uint4 &b, const uint4 &bx, uint32_t start)
{
	uint32_t *scratch = c_V[(blockIdx.x*blockDim.x + threadIdx.x)/32];
	if (SCHEME == ANDERSEN) {
		int target_thread = (threadIdx.x + 4)&31;
		uint4 t=b, t2=__shfl(bx, target_thread);
#if CUDA_VERSION < 9000
		int t2_start = __shfl((int)start, target_thread) + 4;
#else
        int t2_start = __shfl_sync(0xFFFFFFFF, (int)start, target_thread) + 4;
#endif
		bool c = (threadIdx.x & 0x4);
		*((uint4 *)(&scratch[c ? t2_start : start])) = (c ? t2 : t);
		*((uint4 *)(&scratch[c ? start : t2_start])) = (c ? t : t2);
	} else {
		*((uint4 *)(&scratch[start   ])) = b;
		*((uint4 *)(&scratch[start+16])) = bx;
	}
}

template <MemoryAccess SCHEME> __device__ __forceinline__
void read_keys_direct(uint4 &b, uint4 &bx, uint32_t start)
{
	uint32_t *scratch = c_V[(blockIdx.x*blockDim.x + threadIdx.x)/32];
	if (SCHEME == ANDERSEN) {
#if CUDA_VERSION < 9000
        int t2_start = __shfl((int)start, (threadIdx.x + 4)&31) + 4;
#else
        int t2_start = __shfl_sync(0xFFFFFFFF, (int)start, (threadIdx.x + 4)&31) + 4;
#endif
		bool c = (threadIdx.x & 0x4);
		b  = __ldg((uint4 *)(&scratch[c ? t2_start : start]));
		bx = __ldg((uint4 *)(&scratch[c ? start : t2_start]));
		uint4 tmp = b; b = (c ? bx : b); bx = (c ? tmp : bx);
		bx = __shfl(bx, (threadIdx.x + 28)&31);
	} else {
		b = *((uint4 *)(&scratch[start]));
		bx = *((uint4 *)(&scratch[start+16]));
	}
}

__device__  __forceinline__
void primary_order_shuffle(uint32_t b[4], uint32_t bx[4]) {
	/* Inner loop shuffle targets */
	int x1 = (threadIdx.x & 0xfc) + (((threadIdx.x & 3)+1)&3);
	int x2 = (threadIdx.x & 0xfc) + (((threadIdx.x & 3)+2)&3);
	int x3 = (threadIdx.x & 0xfc) + (((threadIdx.x & 3)+3)&3);

#if CUDA_VERSION < 9000
	b[3] = __shfl((int)b[3], x1);
	b[2] = __shfl((int)b[2], x2);
	b[1] = __shfl((int)b[1], x3);
#else
	b[3] = __shfl_sync(0xFFFFFFFF, (int)b[3], x1);
	b[2] = __shfl_sync(0xFFFFFFFF, (int)b[2], x2);
	b[1] = __shfl_sync(0xFFFFFFFF, (int)b[1], x3);
#endif
	uint32_t tmp = b[1]; b[1] = b[3]; b[3] = tmp;

#if CUDA_VERSION < 9000
	bx[3] = __shfl((int)bx[3], x1);
	bx[2] = __shfl((int)bx[2], x2);
	bx[1] = __shfl((int)bx[1], x3);
#else
	bx[3] = __shfl_sync(0xFFFFFFFF, (int)bx[3], x1);
	bx[2] = __shfl_sync(0xFFFFFFFF, (int)bx[2], x2);
	bx[1] = __shfl_sync(0xFFFFFFFF, (int)bx[1], x3);
#endif
	tmp = bx[1]; bx[1] = bx[3]; bx[3] = tmp;
}

__device__  __forceinline__
void primary_order_shuffle(uint4 &b, uint4 &bx) {
	/* Inner loop shuffle targets */
	int x1 = (threadIdx.x & 0x1c) + (((threadIdx.x & 3)+1)&3);
	int x2 = (threadIdx.x & 0x1c) + (((threadIdx.x & 3)+2)&3);
	int x3 = (threadIdx.x & 0x1c) + (((threadIdx.x & 3)+3)&3);

#if CUDA_VERSION < 9000
	b.w = __shfl((int)b.w, x1);
	b.z = __shfl((int)b.z, x2);
	b.y = __shfl((int)b.y, x3);
#else
	b.w = __shfl_sync(0xFFFFFFFF, (int)b.w, x1);
	b.z = __shfl_sync(0xFFFFFFFF, (int)b.z, x2);
	b.y = __shfl_sync(0xFFFFFFFF, (int)b.y, x3);
#endif
	uint32_t tmp = b.y; b.y = b.w; b.w = tmp;

#if CUDA_VERSION < 9000
	bx.w = __shfl((int)bx.w, x1);
	bx.z = __shfl((int)bx.z, x2);
	bx.y = __shfl((int)bx.y, x3);
#else
	bx.w = __shfl_sync(0xFFFFFFFF, (int)bx.w, x1);
	bx.z = __shfl_sync(0xFFFFFFFF, (int)bx.z, x2);
	bx.y = __shfl_sync(0xFFFFFFFF, (int)bx.y, x3);
#endif
	tmp = bx.y; bx.y = bx.w; bx.w = tmp;
}

/*
 * load_key loads a 32*32bit key from a contiguous region of memory in B.
 * The input keys are in external order (i.e., 0, 1, 2, 3, ...).
 * After loading, each thread has its four b and four bx keys stored
 * in internal processing order.
 */
__device__  __forceinline__
void load_key_salsa(const uint32_t *B, uint4 &b, uint4 &bx)
{
	uint32_t scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_WU;
	uint32_t thread_in_block = threadIdx.x & 3U;
	uint32_t key_offset = scrypt_block * 32 + (thread_in_block*4);

	// Read in permuted order. Key loads are not our bottleneck right now.
	b.x  = B[key_offset + (thread_in_block+0) & 3U];
	b.y  = B[key_offset + (thread_in_block+1) & 3U];
	b.z  = B[key_offset + (thread_in_block+2) & 3U];
	b.w  = B[key_offset + (thread_in_block+3) & 3U];

	key_offset += 16;
	bx.x = B[key_offset + (thread_in_block+0) & 3U];
	bx.y = B[key_offset + (thread_in_block+1) & 3U];
	bx.z = B[key_offset + (thread_in_block+2) & 3U];
	bx.w = B[key_offset + (thread_in_block+3) & 3U];

	primary_order_shuffle(b, bx);
}

/*
 * store_key performs the opposite transform as load_key, taking
 * internally-ordered b and bx and storing them into a contiguous
 * region of B in external order.
 */
__device__  __forceinline__
void store_key_salsa(uint32_t *B, uint4 &b, uint4 &bx)
{
	uint32_t scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_WU;
	uint32_t thread_in_block = threadIdx.x & 3U;
	uint32_t key_offset = scrypt_block * 32 + (thread_in_block*4);

	primary_order_shuffle(b, bx);

	B[key_offset + (thread_in_block+0) & 3U] = b.x;
	B[key_offset + (thread_in_block+1) & 3U] = b.y;
	B[key_offset + (thread_in_block+2) & 3U] = b.z;
	B[key_offset + (thread_in_block+3) & 3U] = b.w;

	key_offset += 16;
	B[key_offset + (thread_in_block+0) & 3U] = bx.x;
	B[key_offset + (thread_in_block+1) & 3U] = bx.y;
	B[key_offset + (thread_in_block+2) & 3U] = bx.z;
	B[key_offset + (thread_in_block+3) & 3U] = bx.w;
}


/*
 * load_key loads a 32*32bit key from a contiguous region of memory in B.
 * The input keys are in external order (i.e., 0, 1, 2, 3, ...).
 * After loading, each thread has its four b and four bx keys stored
 * in internal processing order.
 */
__device__  __forceinline__
void load_key_chacha(const uint32_t *B, uint4 &b, uint4 &bx)
{
	uint32_t scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_WU;
	uint32_t thread_in_block = threadIdx.x & 3U;
	uint32_t key_offset = scrypt_block * 32 + thread_in_block;

	// Read in permuted order. Key loads are not our bottleneck right now.
	b.x  = B[key_offset      ];
	b.y  = B[key_offset + 4*1];
	b.z  = B[key_offset + 4*2];
	b.w  = B[key_offset + 4*3];

	key_offset += 16;
	bx.x = B[key_offset      ];
	bx.y = B[key_offset + 4  ];
	bx.z = B[key_offset + 4*2];
	bx.w = B[key_offset + 4*3];
}

/*
 * store_key performs the opposite transform as load_key, taking
 * internally-ordered b and bx and storing them into a contiguous
 * region of B in external order.
 */
__device__  __forceinline__
void store_key_chacha(uint32_t *B, const uint4 &b, const uint4 &bx)
{
	uint32_t scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_WU;
	uint32_t thread_in_block = threadIdx.x & 3U;
	uint32_t key_offset = scrypt_block * 32U + thread_in_block;

	B[key_offset      ] = b.x;
	B[key_offset + 4  ] = b.y;
	B[key_offset + 4*2] = b.z;
	B[key_offset + 4*3] = b.w;

	key_offset += 16;
	B[key_offset      ] = bx.x;
	B[key_offset + 4  ] = bx.y;
	B[key_offset + 4*2] = bx.z;
	B[key_offset + 4*3] = bx.w;
}


template <int ALGO> __device__  __forceinline__
void load_key(const uint32_t *B, uint4 &b, uint4 &bx)
{
	switch(ALGO) {
	case A_SCRYPT:      load_key_salsa(B, b, bx); break;
	case A_SCRYPT_JANE: load_key_chacha(B, b, bx); break;
	}
}

template <int ALGO> __device__  __forceinline__
void store_key(uint32_t *B, uint4 &b, uint4 &bx)
{
	switch(ALGO) {
	case A_SCRYPT:      store_key_salsa(B, b, bx); break;
	case A_SCRYPT_JANE: store_key_chacha(B, b, bx); break;
	}
}


/*
 * salsa_xor_core (Salsa20/8 cypher)
 * The original scrypt called:
 * xor_salsa8(&X[0], &X[16]); <-- the "b" loop
 * xor_salsa8(&X[16], &X[0]); <-- the "bx" loop
 * This version is unrolled to handle both of these loops in a single
 * call to avoid unnecessary data movement.
 */

#if __CUDA_ARCH__ < 350
	// Kepler (Compute 3.0)
	#define XOR_ROTATE_ADD(dst, s1, s2, amt) { uint32_t tmp = s1+s2; dst ^= ((tmp<<amt)|(tmp>>(32-amt))); }
#else
	// Kepler (Compute 3.5)
	#define ROTL(a, b) __funnelshift_l( a, a, b );
	#define XOR_ROTATE_ADD(dst, s1, s2, amt) dst ^= ROTL(s1+s2, amt);
#endif


__device__  __forceinline__
void salsa_xor_core(uint4 &b, uint4 &bx, const int x1, const int x2, const int x3)
{
	uint4 x;

	b ^= bx;
	x = b;

	// Enter in "primary order" (t0 has  0,  4,  8, 12)
	//                          (t1 has  5,  9, 13,  1)
	//                          (t2 has 10, 14,  2,  6)
	//                          (t3 has 15,  3,  7, 11)

	#pragma unroll
	for (int j = 0; j < 4; j++)
	{
		// Mixing phase of salsa
		XOR_ROTATE_ADD(x.y, x.x, x.w, 7);
		XOR_ROTATE_ADD(x.z, x.y, x.x, 9);
		XOR_ROTATE_ADD(x.w, x.z, x.y, 13);
		XOR_ROTATE_ADD(x.x, x.w, x.z, 18);

		/* Transpose rows and columns. */
		/* Unclear if this optimization is needed: These are ordered based
		 * upon the dependencies needed in the later xors. Compiler should be
		 * able to figure this out, but might as well give it a hand. */
#if CUDA_VERSION < 9000
        x.y = __shfl((int)x.y, x3);
		x.w = __shfl((int)x.w, x1);
		x.z = __shfl((int)x.z, x2);
#else
        x.y = __shfl_sync(0xFFFFFFFF, (int)x.y, x3);
		x.w = __shfl_sync(0xFFFFFFFF, (int)x.w, x1);
		x.z = __shfl_sync(0xFFFFFFFF, (int)x.z, x2);
#endif
		/* The next XOR_ROTATE_ADDS could be written to be a copy-paste of the first,
		 * but the register targets are rewritten here to swap x[1] and x[3] so that
		 * they can be directly shuffled to and from our peer threads without
		 * reassignment. The reverse shuffle then puts them back in the right place.
		 */

		XOR_ROTATE_ADD(x.w, x.x, x.y, 7);
		XOR_ROTATE_ADD(x.z, x.w, x.x, 9);
		XOR_ROTATE_ADD(x.y, x.z, x.w, 13);
		XOR_ROTATE_ADD(x.x, x.y, x.z, 18);

#if CUDA_VERSION < 9000
		x.w = __shfl((int)x.w, x3);
		x.y = __shfl((int)x.y, x1);
		x.z = __shfl((int)x.z, x2);
#else
		x.w = __shfl_sync(0xFFFFFFFF, (int)x.w, x3);
		x.y = __shfl_sync(0xFFFFFFFF, (int)x.y, x1);
		x.z = __shfl_sync(0xFFFFFFFF, (int)x.z, x2);
#endif
	}

	b += x;
	// The next two lines are the beginning of the BX-centric loop iteration
	bx ^= b;
	x = bx;

	// This is a copy of the same loop above, identical but stripped of comments.
	// Duplicated so that we can complete a bx-based loop with fewer register moves.
	#pragma unroll 4
	for (int j = 0; j < 4; j++)
	{
		XOR_ROTATE_ADD(x.y, x.x, x.w, 7);
		XOR_ROTATE_ADD(x.z, x.y, x.x, 9);
		XOR_ROTATE_ADD(x.w, x.z, x.y, 13);
		XOR_ROTATE_ADD(x.x, x.w, x.z, 18);

#if CUDA_VERSION < 9000
		x.y = __shfl((int)x.y, x3);
		x.w = __shfl((int)x.w, x1);
		x.z = __shfl((int)x.z, x2);
#else
		x.y = __shfl_sync(0xFFFFFFFF, (int)x.y, x3);
		x.w = __shfl_sync(0xFFFFFFFF, (int)x.w, x1);
		x.z = __shfl_sync(0xFFFFFFFF, (int)x.z, x2);
#endif
		XOR_ROTATE_ADD(x.w, x.x, x.y, 7);
		XOR_ROTATE_ADD(x.z, x.w, x.x, 9);
		XOR_ROTATE_ADD(x.y, x.z, x.w, 13);
		XOR_ROTATE_ADD(x.x, x.y, x.z, 18);

#if CUDA_VERSION < 9000
		x.w = __shfl((int)x.w, x3);
		x.y = __shfl((int)x.y, x1);
		x.z = __shfl((int)x.z, x2);
#else
		x.w = __shfl_sync(0xFFFFFFFF, (int)x.w, x3);
		x.y = __shfl_sync(0xFFFFFFFF, (int)x.y, x1);
		x.z = __shfl_sync(0xFFFFFFFF, (int)x.z, x2);
#endif
	}

	// At the end of these iterations, the data is in primary order again.
#undef XOR_ROTATE_ADD

	bx += x;
}


/*
 * chacha_xor_core (ChaCha20/8 cypher)
 * This version is unrolled to handle both of these loops in a single
 * call to avoid unnecessary data movement.
 *
 * load_key and store_key must not use primary order when
 * using ChaCha20/8, but rather the basic transposed order
 * (referred to as "column mode" below)
 */

#if __CUDA_ARCH__ < 320
	// Kepler (Compute 3.0)
	#define CHACHA_PRIMITIVE(pt, rt, ps, amt) { uint32_t tmp = rt ^ (pt += ps); rt = ((tmp<<amt)|(tmp>>(32-amt))); }
#else
	// Kepler (Compute 3.5)
	#define ROTL(a, b) __funnelshift_l( a, a, b );
	#define CHACHA_PRIMITIVE(pt, rt, ps, amt) { pt += ps; rt = ROTL(rt ^ pt,amt); }
#endif

__device__  __forceinline__
void chacha_xor_core(uint4 &b, uint4 &bx, const int x1, const int x2, const int x3)
{
	uint4 x = b ^= bx;

	//b ^= bx;
	//x = b;

	// Enter in "column" mode (t0 has 0, 4,  8, 12)
	//                        (t1 has 1, 5,  9, 13)
	//                        (t2 has 2, 6, 10, 14)
	//                        (t3 has 3, 7, 11, 15)

	//#pragma unroll
	for (int j = 0; j < 4; j++) {

		// Column Mixing phase of chacha
		CHACHA_PRIMITIVE(x.x ,x.w, x.y, 16)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w, 12)
		CHACHA_PRIMITIVE(x.x ,x.w, x.y,  8)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w,  7)

#if CUDA_VERSION < 9000
		x.y = __shfl((int)x.y, x1);
		x.z = __shfl((int)x.z, x2);
		x.w = __shfl((int)x.w, x3);
#else
		x.y = __shfl_sync(0xFFFFFFFF, (int)x.y, x1);
		x.z = __shfl_sync(0xFFFFFFFF, (int)x.z, x2);
		x.w = __shfl_sync(0xFFFFFFFF, (int)x.w, x3);
#endif
		// Diagonal Mixing phase of chacha
		CHACHA_PRIMITIVE(x.x ,x.w, x.y, 16)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w, 12)
		CHACHA_PRIMITIVE(x.x ,x.w, x.y,  8)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w,  7)

#if CUDA_VERSION < 9000
		x.y = __shfl((int)x.y, x3);
		x.z = __shfl((int)x.z, x2);
		x.w = __shfl((int)x.w, x1);
#else
		x.y = __shfl_sync(0xFFFFFFFF, (int)x.y, x3);
		x.z = __shfl_sync(0xFFFFFFFF, (int)x.z, x2);
		x.w = __shfl_sync(0xFFFFFFFF, (int)x.w, x1);
#endif
	}

	b += x;
	// The next two lines are the beginning of the BX-centric loop iteration
	bx ^= b;
	x = bx;

	//#pragma unroll
	for (int j = 0; j < 4; j++) 
	{

		// Column Mixing phase of chacha
		CHACHA_PRIMITIVE(x.x ,x.w, x.y, 16)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w, 12)
		CHACHA_PRIMITIVE(x.x ,x.w, x.y,  8)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w,  7)

#if CUDA_VERSION < 9000
		x.y = __shfl((int)x.y, x1);
		x.z = __shfl((int)x.z, x2);
		x.w = __shfl((int)x.w, x3);
#else
		x.y = __shfl_sync(0xFFFFFFFF, (int)x.y, x1);
		x.z = __shfl_sync(0xFFFFFFFF, (int)x.z, x2);
		x.w = __shfl_sync(0xFFFFFFFF, (int)x.w, x3);
#endif

		// Diagonal Mixing phase of chacha
		CHACHA_PRIMITIVE(x.x ,x.w, x.y, 16)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w, 12)
		CHACHA_PRIMITIVE(x.x ,x.w, x.y,  8)
		CHACHA_PRIMITIVE(x.z ,x.y, x.w,  7)

#if CUDA_VERSION < 9000
		x.y = __shfl((int)x.y, x3);
		x.z = __shfl((int)x.z, x2);
		x.w = __shfl((int)x.w, x1);
#else
		x.y = __shfl_sync(0xFFFFFFFF, (int)x.y, x3);
		x.z = __shfl_sync(0xFFFFFFFF, (int)x.z, x2);
		x.w = __shfl_sync(0xFFFFFFFF, (int)x.w, x1);
#endif
	}

#undef CHACHA_PRIMITIVE

	bx += x;
}


template <int ALGO> __device__  __forceinline__
void block_mixer(uint4 &b, uint4 &bx, const int x1, const int x2, const int x3)
{
	switch(ALGO) {
	case A_SCRYPT:      salsa_xor_core(b, bx, x1, x2, x3); break;
	case A_SCRYPT_JANE: chacha_xor_core(b, bx, x1, x2, x3); break;
	}
}


/*
 * The hasher_gen_kernel operates on a group of 1024-bit input keys
 * in B, stored as:
 * B = { k1B k1Bx k2B k2Bx ... }
 * and fills up the scratchpad with the iterative hashes derived from
 * those keys:
 * scratch { k1h1B k1h1Bx K1h2B K1h2Bx ... K2h1B K2h1Bx K2h2B K2h2Bx ... }
 * scratch is 1024 times larger than the input keys B.
 * It is extremely important to stream writes effectively into scratch;
 * less important to coalesce the reads from B.
 *
 * Key ordering note: Keys are input from B in "original" order:
 * K = {k1, k2, k3, k4, k5, ..., kx15, kx16, kx17, ..., kx31 }
 * After inputting into kernel_gen, each component k and kx of the
 * key is transmuted into a permuted internal order to make processing faster:
 * K = k, kx with:
 * k = 0, 4, 8, 12, 5, 9, 13, 1, 10, 14, 2, 6, 15, 3, 7, 11
 * and similarly for kx.
 */

template <int ALGO, MemoryAccess SCHEME> __global__
void titan_scrypt_core_kernelA(const uint32_t *d_idata, int begin, int end)
{
	uint4 b, bx;

	int x1 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+1)&0x3);
	int x2 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+2)&0x3);
	int x3 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+3)&0x3);

	int scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_WU;
	int start = (scrypt_block*c_SCRATCH + (SCHEME==ANDERSEN?8:4)*(threadIdx.x%4)) % c_SCRATCH_WU_PER_WARP;

	int i=begin;

	if (i == 0) {
		load_key<ALGO>(d_idata, b, bx);
		write_keys_direct<SCHEME>(b, bx, start);
		++i;
	} else read_keys_direct<SCHEME>(b, bx, start+32*(i-1));

	while (i < end) {
		block_mixer<ALGO>(b, bx, x1, x2, x3);
		write_keys_direct<SCHEME>(b, bx, start+32*i);
		++i;
	}
}

template <int ALGO, MemoryAccess SCHEME> __global__
void titan_scrypt_core_kernelA_LG(const uint32_t *d_idata, int begin, int end, unsigned int LOOKUP_GAP)
{
	uint4 b, bx;

	int x1 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+1)&0x3);
	int x2 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+2)&0x3);
	int x3 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+3)&0x3);

	int scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_WU;
	int start = (scrypt_block*c_SCRATCH + (SCHEME==ANDERSEN?8:4)*(threadIdx.x%4)) % c_SCRATCH_WU_PER_WARP;

	int i=begin;

	if (i == 0) {
		load_key<ALGO>(d_idata, b, bx);
		write_keys_direct<SCHEME>(b, bx, start);
		++i;
	} else {
		int pos = (i-1)/LOOKUP_GAP, loop = (i-1)-pos*LOOKUP_GAP;
		read_keys_direct<SCHEME>(b, bx, start+32*pos);
		while(loop--) block_mixer<ALGO>(b, bx, x1, x2, x3);
	}

	while (i < end) {
		block_mixer<ALGO>(b, bx, x1, x2, x3);
		if (i % LOOKUP_GAP == 0)
			write_keys_direct<SCHEME>(b, bx, start+32*(i/LOOKUP_GAP));
		++i;
	}
}


/*
 * hasher_hash_kernel runs the second phase of scrypt after the scratch
 * buffer is filled with the iterative hashes: It bounces through
 * the scratch buffer in pseudorandom order, mixing the key as it goes.
 */

template <int ALGO, MemoryAccess SCHEME> __global__
void titan_scrypt_core_kernelB(uint32_t *d_odata, int begin, int end)
{
	uint4 b, bx;

	int scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_WU;
	int start = ((scrypt_block*c_SCRATCH) + (SCHEME==ANDERSEN?8:4)*(threadIdx.x%4)) % c_SCRATCH_WU_PER_WARP;

	int x1 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+1)&0x3);
	int x2 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+2)&0x3);
	int x3 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+3)&0x3);

	if (begin == 0) {
		read_keys_direct<SCHEME>(b, bx, start+32*c_N_1);
		block_mixer<ALGO>(b, bx, x1, x2, x3);
	} else load_key<ALGO>(d_odata, b, bx);

	for (int i = begin; i < end; i++) {
#if CUDA_VERSION < 9000
		int j = (__shfl((int)bx.x, (threadIdx.x & 0x1c)) & (c_N_1));
#else
		int j = (__shfl_sync(0xFFFFFFFF, (int)bx.x, (threadIdx.x & 0x1c)) & (c_N_1));
#endif
		uint4 t, tx; read_keys_direct<SCHEME>(t, tx, start+32*j);
		b ^= t; bx ^= tx;
		block_mixer<ALGO>(b, bx, x1, x2, x3);
	}

	store_key<ALGO>(d_odata, b, bx);
}

template <int ALGO, MemoryAccess SCHEME> __global__
void titan_scrypt_core_kernelB_LG(uint32_t *d_odata, int begin, int end, unsigned int LOOKUP_GAP)
{
	uint4 b, bx;

	int scrypt_block = (blockIdx.x*blockDim.x + threadIdx.x)/THREADS_PER_WU;
	int start = ((scrypt_block*c_SCRATCH) + (SCHEME==ANDERSEN?8:4)*(threadIdx.x%4)) % c_SCRATCH_WU_PER_WARP;

	int x1 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+1)&0x3);
	int x2 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+2)&0x3);
	int x3 = (threadIdx.x & 0x1c) + (((threadIdx.x & 0x03)+3)&0x3);

	if (begin == 0) {
		int pos = c_N_1/LOOKUP_GAP, loop = 1 + (c_N_1-pos*LOOKUP_GAP);
		read_keys_direct<SCHEME>(b, bx, start+32*pos);
		while(loop--)
			block_mixer<ALGO>(b, bx, x1, x2, x3);
	}
	else
			load_key<ALGO>(d_odata, b, bx);

	if (SCHEME == SIMPLE)
	{
		// better divergent thread handling submitted by nVidia engineers, but
		// supposedly this does not run with the ANDERSEN memory access scheme
#if CUDA_VERSION < 9000
        int j = (__shfl((int)bx.x, (threadIdx.x & 0x1c)) & (c_N_1));
#else
        int j = (__shfl_sync(0xFFFFFFFF, (int)bx.x, (threadIdx.x & 0x1c)) & (c_N_1));
#endif
		int pos = j/LOOKUP_GAP;
		int loop = -1;
		uint4 t, tx;

		int i = begin;
		while(i < end)
		{
			if (loop == -1) {
#if CUDA_VERSION < 9000
                j = (__shfl((int)bx.x, (threadIdx.x & 0x1c)) & (c_N_1));
#else
                j = (__shfl_sync(0xFFFFFFFF, (int)bx.x, (threadIdx.x & 0x1c)) & (c_N_1));
#endif
				pos = j/LOOKUP_GAP;
				loop = j-pos*LOOKUP_GAP;
				read_keys_direct<SCHEME>(t, tx, start+32*pos);
			}
			if (loop == 0) {
				b ^= t; bx ^= tx;
				t=b;tx=bx;
			}

			block_mixer<ALGO>(t, tx, x1, x2, x3);
			if (loop == 0) {
				b=t;bx=tx;
				i++;
			}
			loop--;
		}
	}
	else
	{
		// this is my original implementation, now used with the ANDERSEN
		// memory access scheme only.
		for (int i = begin; i < end; i++) {
#if CUDA_VERSION < 9000
            int j = (__shfl((int)bx.x, (threadIdx.x & 0x1c)) & (c_N_1));
#else
            int j = (__shfl_sync(0xFFFFFFFF, (int)bx.x, (threadIdx.x & 0x1c)) & (c_N_1));
#endif
			int pos = j/LOOKUP_GAP, loop = j-pos*LOOKUP_GAP;
			uint4 t, tx; read_keys_direct<SCHEME>(t, tx, start+32*pos);
			while (loop--)
				block_mixer<ALGO>(t, tx, x1, x2, x3);
			b ^= t; bx ^= tx;
			block_mixer<ALGO>(b, bx, x1, x2, x3);
		}
	}

	store_key<ALGO>(d_odata, b, bx);
}


TitanKernel::TitanKernel() : KernelInterface()
{
}

void TitanKernel::set_scratchbuf_constants(int MAXWARPS, uint32_t** h_V)
{
	checkCudaErrors(cudaMemcpyToSymbol(c_V, h_V, MAXWARPS*sizeof(uint32_t*), 0, cudaMemcpyHostToDevice));
}

bool TitanKernel::run_kernel(dim3 grid, dim3 threads, int WARPS_PER_BLOCK, int thr_id, cudaStream_t stream,
	uint32_t* d_idata, uint32_t* d_odata, unsigned int N, unsigned int LOOKUP_GAP, bool interactive, bool benchmark, int texture_cache)
{
	bool success = true;
	bool scrypt = IS_SCRYPT();
	bool chacha = IS_SCRYPT_JANE();

	// make some constants available to kernel, update only initially and when changing
	static uint32_t prev_N[MAX_GPUS] = { 0 };

	if (N != prev_N[thr_id]) {
		uint32_t h_N = N;
		uint32_t h_N_1 = N-1;
		uint32_t h_SCRATCH = SCRATCH;
		uint32_t h_SCRATCH_WU_PER_WARP = (SCRATCH * WU_PER_WARP);
		uint32_t h_SCRATCH_WU_PER_WARP_1 = (SCRATCH * WU_PER_WARP) - 1;

		cudaMemcpyToSymbolAsync(c_N, &h_N, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);
		cudaMemcpyToSymbolAsync(c_N_1, &h_N_1, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);
		cudaMemcpyToSymbolAsync(c_SCRATCH, &h_SCRATCH, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);
		cudaMemcpyToSymbolAsync(c_SCRATCH_WU_PER_WARP, &h_SCRATCH_WU_PER_WARP, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);
		cudaMemcpyToSymbolAsync(c_SCRATCH_WU_PER_WARP_1, &h_SCRATCH_WU_PER_WARP_1, sizeof(uint32_t), 0, cudaMemcpyHostToDevice, stream);

		prev_N[thr_id] = N;
	}

	// First phase: Sequential writes to scratchpad.

	int batch = device_batchsize[thr_id];

	unsigned int pos = 0;
	do {
		if (LOOKUP_GAP == 1) {
			if (scrypt) titan_scrypt_core_kernelA<A_SCRYPT,    ANDERSEN> <<< grid, threads, 0, stream >>>(d_idata, pos, min(pos+batch, N));
			if (chacha) titan_scrypt_core_kernelA<A_SCRYPT_JANE, SIMPLE> <<< grid, threads, 0, stream >>>(d_idata, pos, min(pos+batch, N));
		} else {
			if (scrypt) titan_scrypt_core_kernelA_LG<A_SCRYPT,    ANDERSEN> <<< grid, threads, 0, stream >>>(d_idata, pos, min(pos+batch, N), LOOKUP_GAP);
			if (chacha) titan_scrypt_core_kernelA_LG<A_SCRYPT_JANE, SIMPLE> <<< grid, threads, 0, stream >>>(d_idata, pos, min(pos+batch, N), LOOKUP_GAP);
		}
		pos += batch;

	} while (pos < N);

	// Second phase: Random read access from scratchpad.

	pos = 0;
	do {
		if (LOOKUP_GAP == 1)  {
			if (scrypt) titan_scrypt_core_kernelB<A_SCRYPT,    ANDERSEN> <<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N));
			if (chacha) titan_scrypt_core_kernelB<A_SCRYPT_JANE, SIMPLE> <<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N));
		} else {
			if (scrypt) titan_scrypt_core_kernelB_LG<A_SCRYPT,    ANDERSEN> <<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N), LOOKUP_GAP);
			if (chacha) titan_scrypt_core_kernelB_LG<A_SCRYPT_JANE, SIMPLE> <<< grid, threads, 0, stream >>>(d_odata, pos, min(pos+batch, N), LOOKUP_GAP);
		}
		pos += batch;

	} while (pos < N);

	return success;
}

#endif /* prevent SM 2 */
