/*
 * Equihash solver created by djeZo (l33tsoftw@gmail.com) for NiceHash
 * Adapted to be more compatible with older C++ compilers
 *
 * cuda_djezo solver was released by NiceHash (www.nicehash.com) under
 * GPL 3.0 license. If you don't have a copy, you can obtain one from
 * https://www.gnu.org/licenses/gpl-3.0.txt
 *
 * Based on CUDA solver by John Tromp released under MIT license.
 * Some helper functions taken out of OpenCL solver by Marc Bevand
 * released under MIT license.
 *
 * Copyright (c) 2016 John Tromp, Marc Bevand
 * Copyright (c) 2017 djeZo, Tanguy Pruvot (GPL v3)
 */

#ifdef WIN32
#include <Windows.h>
#endif

#include <stdio.h>
#include <vector>
//#include <mutex>

#include "equihash.h"
#include "eqcuda.hpp" // eq_cuda_context

#include "blake2/blake2.h"

//#define WN 200
//#define WK 9
#ifndef MAX_GPUS
#define MAX_GPUS 16
#endif

#define NDIGITS		(WK+1)
#define DIGITBITS	(WN/(NDIGITS))
#define PROOFSIZE (1<<WK)
#define BASE (1<<DIGITBITS)
#define NHASHES (2*BASE)
#define HASHESPERBLAKE (512/WN)
#define HASHOUT (HASHESPERBLAKE*WN/8)
#define NBLOCKS ((NHASHES + HASHESPERBLAKE - 1) / HASHESPERBLAKE)
#define BUCKBITS (DIGITBITS - RB)
#define NBUCKETS (1 << BUCKBITS)
#define BUCKMASK (NBUCKETS - 1)
#define SLOTBITS (RB + 2)
#define SLOTRANGE (1 << SLOTBITS)
#define NSLOTS SM
#define SLOTMASK (SLOTRANGE - 1)
#define NRESTS (1 << RB)
#define RESTMASK (NRESTS - 1)
#define CANTORBITS (2 * SLOTBITS - 2)
#define CANTORMASK ((1 << CANTORBITS) - 1)
#define CANTORMAXSQRT (2 * NSLOTS)
#define RB8_NSLOTS 640
#define RB8_NSLOTS_LD 624
#define FD_THREADS 128

#ifdef __INTELLISENSE__
// reduce vstudio editor warnings
#include <device_functions.h>
#include <device_launch_parameters.h>
#define __launch_bounds__(max_tpb, min_blocks)
#define __CUDA_ARCH__ 520
uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t z);
uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t z);
uint32_t __shfl2(uint32_t x, uint32_t y);
uint32_t __shfl_sync(uint32_t mask, uint32_t x, uint32_t y);
uint32_t atomicExch(uint32_t *x, uint32_t y);
uint32_t atomicAdd(uint32_t *x, uint32_t y);
void __syncthreads(void);
void __threadfence(void);
void __threadfence_block(void);
uint32_t __ldg(const uint32_t* address);
uint64_t __ldg(const uint64_t* address);
uint4 __ldca(const uint4 *ptr);
u32 __ldca(const u32 *ptr);
u32 umin(const u32, const u32);
u32 umax(const u32, const u32);
#endif

#define OPT_SYNC_ALL

#if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 300
#define __shfl2(var, srcLane)  __shfl_sync(0xFFFFFFFFu, var, srcLane)
#undef __any
#define __any(p) __any_sync(0xFFFFFFFFu, p)
#else
#define __shfl2 __shfl
#endif

typedef u32 proof[PROOFSIZE];

struct __align__(32) slot {
	u32 hash[8];
};

struct __align__(16) slotsmall {
	u32 hash[4];
};

struct __align__(8) slottiny {
	u32 hash[2];
};

template <u32 RB, u32 SM>
struct equi
{
	slot round0trees[4096][RB8_NSLOTS];
	slot trees[1][NBUCKETS][NSLOTS];
	struct {
		slotsmall treessmall[NSLOTS];
		slottiny treestiny[NSLOTS];
	} round2trees[NBUCKETS];
	struct {
		slotsmall treessmall[NSLOTS];
		slottiny treestiny[NSLOTS];
	} round3trees[NBUCKETS];
	slotsmall treessmall[4][NBUCKETS][NSLOTS];
	slottiny treestiny[1][4096][RB8_NSLOTS_LD];
	u32 round4bidandsids[NBUCKETS][NSLOTS];
	union {
		u64 blake_h[8];
		u32 blake_h32[16];
	};
	struct {
		u32 nslots8[4096];
		u32 nslots0[4096];
		u32 nslots[9][NBUCKETS];
		scontainerreal srealcont;
	} edata;
};

// todo: use cuda_helper.h and/or cuda_vector.h
__device__ __forceinline__ uint2 operator^ (uint2 a, uint2 b)
{
	return make_uint2(a.x ^ b.x, a.y ^ b.y);
}

__device__ __forceinline__ uint4 operator^ (uint4 a, uint4 b)
{
	return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

// for ROR 63 (or ROL 1); this func only support (32 <= offset < 64)
__device__ __forceinline__ uint2 ROR2(const uint2 a, const int offset)
{
	uint2 result;
#if __CUDA_ARCH__ > 300
	{
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
#else
	result.y = ((a.x >> (offset - 32)) | (a.y << (64 - offset)));
	result.x = ((a.y >> (offset - 32)) | (a.x << (64 - offset)));
#endif
	return result;
}


__device__ __forceinline__ uint2 SWAPUINT2(uint2 value)
{
	return make_uint2(value.y, value.x);
}

__device__ __forceinline__ uint2 ROR24(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x2107);
	result.y = __byte_perm(a.y, a.x, 0x6543);
	return result;
}

__device__ __forceinline__ uint2 ROR16(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x1076);
	result.y = __byte_perm(a.y, a.x, 0x5432);
	return result;
}

__device__ __forceinline__ void G2(u64 & a, u64 & b, u64 & c, u64 & d, u64 x, u64 y)
{
	a = a + b + x;
	((uint2*)&d)[0] = SWAPUINT2(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
	c = c + d;
	((uint2*)&b)[0] = ROR24(((uint2*)&b)[0] ^ ((uint2*)&c)[0]);
	a = a + b + y;
	((uint2*)&d)[0] = ROR16(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
	c = c + d;
	((uint2*)&b)[0] = ROR2(((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
}

// untested..
struct packer_default
{
	__device__ __forceinline__ static u32 set_bucketid_and_slots(const u32 bucketid, const u32 s0, const u32 s1, const u32 RB, const u32 SM)
	{
		return (((bucketid << SLOTBITS) | s0) << SLOTBITS) | s1;
	}

	__device__ __forceinline__ static u32 get_bucketid(const u32 bid, const u32 RB, const u32 SM)
	{
		// BUCKMASK-ed to prevent illegal memory accesses in case of memory errors
		return (bid >> (2 * SLOTBITS)) & BUCKMASK;
	}

	__device__ __forceinline__ static u32 get_slot0(const u32 bid, const u32 s1, const u32 RB, const u32 SM)
	{
		return bid & SLOTMASK;
	}

	__device__ __forceinline__ static u32 get_slot1(const u32 bid, const u32 RB, const u32 SM)
	{
		return (bid >> SLOTBITS) & SLOTMASK;
	}
};


struct packer_cantor
{
	__device__ __forceinline__ static u32 cantor(const u32 s0, const u32 s1)
	{
		u32 a = umax(s0, s1);
		u32 b = umin(s0, s1);
		return a * (a + 1) / 2 + b;
	}

	__device__ __forceinline__ static u32 set_bucketid_and_slots(const u32 bucketid, const u32 s0, const u32 s1, const u32 RB, const u32 SM)
	{
		return (bucketid << CANTORBITS) | cantor(s0, s1);
	}

	__device__ __forceinline__ static u32 get_bucketid(const u32 bid, const u32 RB, const u32 SM)
	{
		return (bid >> CANTORBITS) & BUCKMASK;
	}

	__device__ __forceinline__ static u32 get_slot0(const u32 bid, const u32 s1, const u32 RB, const u32 SM)
	{
		return ((bid & CANTORMASK) - cantor(0, s1)) & SLOTMASK;
	}

	__device__ __forceinline__ static u32 get_slot1(const u32 bid, const u32 RB, const u32 SM)
	{
		u32 k, q, sqr = 8 * (bid & CANTORMASK) + 1;
		// this k=sqrt(sqr) computing loop averages 3.4 iterations out of maximum 9
		for (k = CANTORMAXSQRT; (q = sqr / k) < k; k = (k + q) / 2);
		return ((k - 1) / 2) & SLOTMASK;
	}
};

__device__ __constant__ const u64 blake_iv[] = {
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};

#if CUDART_VERSION < 8000 || !defined(__ldca)
#define __ldca(ptr) *(ptr)
#endif

template <u32 RB, u32 SM, typename PACKER>
__global__ void digit_first(equi<RB, SM>* eq, u32 nonce)
{
	const u32 block = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ u64 hash_h[8];
	u32* hash_h32 = (u32*)hash_h;

	if (threadIdx.x < 16)
		hash_h32[threadIdx.x] = __ldca(&eq->blake_h32[threadIdx.x]);

	__syncthreads();

	u64 m = (u64)block << 32 | (u64)nonce;

	union
	{
		u64 v[16];
		u32 v32[32];
		uint4 v128[8];
	};

	v[0] = hash_h[0];
	v[1] = hash_h[1];
	v[2] = hash_h[2];
	v[3] = hash_h[3];
	v[4] = hash_h[4];
	v[5] = hash_h[5];
	v[6] = hash_h[6];
	v[7] = hash_h[7];
	v[8] = blake_iv[0];
	v[9] = blake_iv[1];
	v[10] = blake_iv[2];
	v[11] = blake_iv[3];
	v[12] = blake_iv[4] ^ (128 + 16);
	v[13] = blake_iv[5];
	v[14] = blake_iv[6] ^ 0xffffffffffffffff;
	v[15] = blake_iv[7];

	// mix 1
	G2(v[0], v[4], v[8], v[12], 0, m);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 2
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], m, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 3
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, m);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 4
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, m);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 5
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, m);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 6
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], m, 0);

	// mix 7
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], m, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 8
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, m);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 9
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], m, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 10
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], m, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 11
	G2(v[0], v[4], v[8], v[12], 0, m);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], 0, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	// mix 12
	G2(v[0], v[4], v[8], v[12], 0, 0);
	G2(v[1], v[5], v[9], v[13], 0, 0);
	G2(v[2], v[6], v[10], v[14], 0, 0);
	G2(v[3], v[7], v[11], v[15], 0, 0);
	G2(v[0], v[5], v[10], v[15], m, 0);
	G2(v[1], v[6], v[11], v[12], 0, 0);
	G2(v[2], v[7], v[8], v[13], 0, 0);
	G2(v[3], v[4], v[9], v[14], 0, 0);

	v[0] ^= hash_h[0] ^ v[8];
	v[1] ^= hash_h[1] ^ v[9];
	v[2] ^= hash_h[2] ^ v[10];
	v[3] ^= hash_h[3] ^ v[11];
	v[4] ^= hash_h[4] ^ v[12];
	v[5] ^= hash_h[5] ^ v[13];
	v32[12] ^= hash_h32[12] ^ v32[28];

	u32 bexor = __byte_perm(v32[0], 0, 0x4012); // first 20 bits
	u32 bucketid;
	asm("bfe.u32 %0, %1, 12, 12;" : "=r"(bucketid) : "r"(bexor));
	u32 slotp = atomicAdd(&eq->edata.nslots0[bucketid], 1);
	if (slotp < RB8_NSLOTS)
	{
		slot* s = &eq->round0trees[bucketid][slotp];

		uint4 tt;
		tt.x = __byte_perm(v32[0], v32[1], 0x1234);
		tt.y = __byte_perm(v32[1], v32[2], 0x1234);
		tt.z = __byte_perm(v32[2], v32[3], 0x1234);
		tt.w = __byte_perm(v32[3], v32[4], 0x1234);
		*(uint4*)(&s->hash[0]) = tt;

		tt.x = __byte_perm(v32[4], v32[5], 0x1234);
		tt.y = __byte_perm(v32[5], v32[6], 0x1234);
		tt.z = 0;
		tt.w = block << 1;
		*(uint4*)(&s->hash[4]) = tt;
	}

	bexor = __byte_perm(v32[6], 0, 0x0123);
	asm("bfe.u32 %0, %1, 12, 12;" : "=r"(bucketid) : "r"(bexor));
	slotp = atomicAdd(&eq->edata.nslots0[bucketid], 1);
	if (slotp < RB8_NSLOTS)
	{
		slot* s = &eq->round0trees[bucketid][slotp];

		uint4 tt;
		tt.x = __byte_perm(v32[6], v32[7], 0x2345);
		tt.y = __byte_perm(v32[7], v32[8], 0x2345);
		tt.z = __byte_perm(v32[8], v32[9], 0x2345);
		tt.w = __byte_perm(v32[9], v32[10], 0x2345);
		*(uint4*)(&s->hash[0]) = tt;

		tt.x = __byte_perm(v32[10], v32[11], 0x2345);
		tt.y = __byte_perm(v32[11], v32[12], 0x2345);
		tt.z = 0;
		tt.w = (block << 1) + 1;
		*(uint4*)(&s->hash[4]) = tt;
	}
}

/*
  Functions digit_1 to digit_8 works by the same principle;
  Each thread does 2-3 slot loads (loads are coalesced).
  Xorwork of slots is loaded into shared memory and is kept in registers (except for digit_1).
  At the same time, restbits (8 or 9 bits) in xorwork are used for collisions.
  Restbits determine position in ht.
  Following next is pair creation. First one (or two) pairs' xorworks are put into global memory
  as soon as possible, the rest pairs are saved in shared memory (one u32 per pair - 16 bit indices).
  In most cases, all threads have one (or two) pairs so with this trick, we offload memory writes a bit in last step.
  In last step we save xorwork of pairs in memory.
*/
template <u32 RB, u32 SM, int SSM, typename PACKER, u32 MAXPAIRS, u32 THREADS>
__global__ void digit_1(equi<RB, SM>* eq)
{
	__shared__ u16 ht[256][SSM - 1];
	__shared__ uint2 lastword1[RB8_NSLOTS];
	__shared__ uint4 lastword2[RB8_NSLOTS];
	__shared__ int ht_len[MAXPAIRS];
	__shared__ u32 pairs_len;
	__shared__ u32 next_pair;

	const u32 threadid = threadIdx.x;
	const u32 bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < 256)
		ht_len[threadid] = 0;
	else if (threadid == (THREADS - 1))
		pairs_len = 0;
	else if (threadid == (THREADS - 33))
		next_pair = 0;

	u32 bsize = umin(eq->edata.nslots0[bucketid], RB8_NSLOTS);

	u32 hr[2];
	int pos[2];
	pos[0] = pos[1] = SSM;

	uint2 ta[2];
	uint4 tb[2];

	u32 si[2];

#ifdef OPT_SYNC_ALL
	// enable this to make fully safe shared mem operations;
	// disabled gains some speed, but can rarely cause a crash
	__syncthreads();
#endif
	#pragma unroll
	for (u32 i = 0; i != 2; ++i)
	{
		si[i] = i * THREADS + threadid;
		if (si[i] >= bsize) break;

		const slot* pslot1 = eq->round0trees[bucketid] + si[i];

		// get xhash
		uint4 a1 = *(uint4*)(&pslot1->hash[0]);
		uint2 a2 = *(uint2*)(&pslot1->hash[4]);
		ta[i].x = a1.x;
		ta[i].y = a1.y;
		lastword1[si[i]] = ta[i];
		tb[i].x = a1.z;
		tb[i].y = a1.w;
		tb[i].z = a2.x;
		tb[i].w = a2.y;
		lastword2[si[i]] = tb[i];

		asm("bfe.u32 %0, %1, 20, 8;" : "=r"(hr[i]) : "r"(ta[i].x));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();
	int* pairs = ht_len;

	u32 xors[6];
	u32 xorbucketid, xorslot;

	#pragma unroll
	for (u32 i = 0; i != 2; ++i)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			u16 p = ht[hr[i]][0];

			*(uint2*)(&xors[0]) = ta[i] ^ lastword1[p];

			asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(RB), "r"(BUCKBITS));
			xorslot = atomicAdd(&eq->edata.nslots[1][xorbucketid], 1);

			if (xorslot < NSLOTS)
			{
				*(uint4*)(&xors[2]) = lastword2[si[i]] ^ lastword2[p];

				slot &xs = eq->trees[0][xorbucketid][xorslot];
				*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[1]);
				uint4 ttx;
				ttx.x = xors[5];
				ttx.y = xors[0];
				ttx.z = packer_default::set_bucketid_and_slots(bucketid, si[i], p, 8, RB8_NSLOTS);
				ttx.w = 0;
				*(uint4*)(&xs.hash[4]) = ttx;
			}

			for (int k = 1; k != pos[i]; ++k)
			{
				u32 pindex = atomicAdd(&pairs_len, 1);
				if (pindex >= MAXPAIRS) break;
				u16 prev = ht[hr[i]][k];
				pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
			}
		}
	}

	__syncthreads();

	// process pairs
	u32 plen = umin(pairs_len, MAXPAIRS);

	u32 i, k;
	for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		i = __byte_perm(pair, 0, 0x4510);
		k = __byte_perm(pair, 0, 0x4532);

		*(uint2*)(&xors[0]) = lastword1[i] ^ lastword1[k];

		asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(RB), "r"(BUCKBITS));
		xorslot = atomicAdd(&eq->edata.nslots[1][xorbucketid], 1);

		if (xorslot < NSLOTS)
		{
			*(uint4*)(&xors[2]) = lastword2[i] ^ lastword2[k];

			slot &xs = eq->trees[0][xorbucketid][xorslot];
			*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[1]);
			uint4 ttx;
			ttx.x = xors[5];
			ttx.y = xors[0];
			ttx.z = packer_default::set_bucketid_and_slots(bucketid, i, k, 8, RB8_NSLOTS);
			ttx.w = 0;
			*(uint4*)(&xs.hash[4]) = ttx;
		}
	}
}


template <u32 RB, u32 SM, int SSM, typename PACKER, u32 MAXPAIRS, u32 THREADS>
__global__ void digit_2(equi<RB, SM>* eq)
{
	__shared__ u16 ht[NRESTS][SSM - 1];
	__shared__ u32 lastword1[NSLOTS];
	__shared__ uint4 lastword2[NSLOTS];
	__shared__ int ht_len[NRESTS];
	__shared__ int pairs[MAXPAIRS];
	__shared__ u32 pairs_len;
	__shared__ u32 next_pair;

	const u32 threadid = threadIdx.x;
	const u32 bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < NRESTS)
		ht_len[threadid] = 0;
	else if (threadid == (THREADS - 1))
		pairs_len = 0;
	else if (threadid == (THREADS - 33))
		next_pair = 0;

	slot* buck = eq->trees[0][bucketid];
	u32 bsize = umin(eq->edata.nslots[1][bucketid], NSLOTS);

	u32 hr[2];
	int pos[2];
	pos[0] = pos[1] = SSM;

	u32 ta[2];
	uint4 tt[2];

	u32 si[2];
#ifdef OPT_SYNC_ALL
	__syncthreads();
#endif
	#pragma unroll 2
	for (u32 i = 0; i < 2; i++)
	{
		si[i] = i * THREADS + threadid;
		if (si[i] >= bsize) break;

		// get slot
		const slot* pslot1 = buck + si[i];

		uint4 ttx = *(uint4*)(&pslot1->hash[0]);
		lastword1[si[i]] = ta[i] = ttx.x;
		uint2 tty = *(uint2*)(&pslot1->hash[4]);
		tt[i].x = ttx.y;
		tt[i].y = ttx.z;
		tt[i].z = ttx.w;
		tt[i].w = tty.x;
		lastword2[si[i]] = tt[i];

		hr[i] = tty.y & RESTMASK;
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();

	u32 xors[5];
	u32 xorbucketid, xorslot;

	#pragma unroll 2
	for (u32 i = 0; i < 2; i++)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			u16 p = ht[hr[i]][0];

			xors[0] = ta[i] ^ lastword1[p];

			xorbucketid = xors[0] >> (12 + RB);
			xorslot = atomicAdd(&eq->edata.nslots[2][xorbucketid], 1);
			if (xorslot < NSLOTS)
			{
				*(uint4*)(&xors[1]) = tt[i] ^ lastword2[p];
				slotsmall &xs = eq->round2trees[xorbucketid].treessmall[xorslot];
				*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[0]);
				slottiny &xst = eq->round2trees[xorbucketid].treestiny[xorslot];
				uint2 ttx;
				ttx.x = xors[4];
				ttx.y = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
				*(uint2*)(&xst.hash[0]) = ttx;
			}

			for (int k = 1; k != pos[i]; ++k)
			{
				u32 pindex = atomicAdd(&pairs_len, 1);
				if (pindex >= MAXPAIRS) break;
				u16 prev = ht[hr[i]][k];
				pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
			}
		}
	}

	__syncthreads();

	// process pairs
	u32 plen = umin(pairs_len, MAXPAIRS);

	u32 i, k;
	for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		i = __byte_perm(pair, 0, 0x4510);
		k = __byte_perm(pair, 0, 0x4532);

		xors[0] = lastword1[i] ^ lastword1[k];

		xorbucketid = xors[0] >> (12 + RB);
		xorslot = atomicAdd(&eq->edata.nslots[2][xorbucketid], 1);
		if (xorslot < NSLOTS)
		{
			*(uint4*)(&xors[1]) = lastword2[i] ^ lastword2[k];
			slotsmall &xs = eq->round2trees[xorbucketid].treessmall[xorslot];
			*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[0]);
			slottiny &xst = eq->round2trees[xorbucketid].treestiny[xorslot];
			uint2 ttx;
			ttx.x = xors[4];
			ttx.y = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
			*(uint2*)(&xst.hash[0]) = ttx;
		}
	}
}


template <u32 RB, u32 SM, int SSM, typename PACKER, u32 MAXPAIRS, u32 THREADS>
__global__ void digit_3(equi<RB, SM>* eq)
{
	__shared__ u16 ht[NRESTS][(SSM - 1)];
	__shared__ uint4 lastword1[NSLOTS];
	__shared__ u32 lastword2[NSLOTS];
	__shared__ int ht_len[NRESTS];
	__shared__ int pairs[MAXPAIRS];
	__shared__ u32 pairs_len;
	__shared__ u32 next_pair;

	const u32 threadid = threadIdx.x;
	const u32 bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < NRESTS)
		ht_len[threadid] = 0;
	else if (threadid == (THREADS - 1))
		pairs_len = 0;
	else if (threadid == (THREADS - 33))
		next_pair = 0;

	u32 bsize = umin(eq->edata.nslots[2][bucketid], NSLOTS);

	u32 hr[2];
	int pos[2];
	pos[0] = pos[1] = SSM;

	u32 si[2];
	uint4 tt[2];
	u32 ta[2];

#ifdef OPT_SYNC_ALL
	__syncthreads();
#endif

	#pragma unroll 2
	for (u32 i = 0; i < 2; i++)
	{
		si[i] = i * THREADS + threadid;
		if (si[i] >= bsize) break;

		slotsmall &xs = eq->round2trees[bucketid].treessmall[si[i]];
		slottiny &xst = eq->round2trees[bucketid].treestiny[si[i]];

		tt[i] = *(uint4*)(&xs.hash[0]);
		lastword1[si[i]] = tt[i];
		ta[i] = xst.hash[0];
		lastword2[si[i]] = ta[i];
		asm("bfe.u32 %0, %1, 12, %2;" : "=r"(hr[i]) : "r"(tt[i].x), "r"(RB));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();

	u32 xors[5];
	u32 bexor, xorbucketid, xorslot;

	#pragma unroll 2
	for (u32 i = 0; i < 2; i++)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			u16 p = ht[hr[i]][0];

			xors[4] = ta[i] ^ lastword2[p];

			if (xors[4] != 0)
			{
				*(uint4*)(&xors[0]) = tt[i] ^ lastword1[p];

				bexor = __byte_perm(xors[0], xors[1], 0x2107);
				asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(bexor), "r"(RB), "r"(BUCKBITS));
				xorslot = atomicAdd(&eq->edata.nslots[3][xorbucketid], 1);

				if (xorslot < NSLOTS)
				{
					slotsmall &xs = eq->round3trees[xorbucketid].treessmall[xorslot];
					*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[1]);
					slottiny &xst = eq->round3trees[xorbucketid].treestiny[xorslot];
					uint2 ttx;
					ttx.x = bexor;
					ttx.y = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
					*(uint2*)(&xst.hash[0]) = ttx;
				}
			}

			for (int k = 1; k != pos[i]; ++k)
			{
				u32 pindex = atomicAdd(&pairs_len, 1);
				if (pindex >= MAXPAIRS) break;
				u16 prev = ht[hr[i]][k];
				pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
			}
		}
	}

	__syncthreads();

	// process pairs
	u32 plen = umin(pairs_len, MAXPAIRS);

	u32 i, k;
	for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		i = __byte_perm(pair, 0, 0x4510);
		k = __byte_perm(pair, 0, 0x4532);

		xors[4] = lastword2[i] ^ lastword2[k];

		if (xors[4] != 0)
		{
			*(uint4*)(&xors[0]) = lastword1[i] ^ lastword1[k];

			bexor = __byte_perm(xors[0], xors[1], 0x2107);
			asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(bexor), "r"(RB), "r"(BUCKBITS));
			xorslot = atomicAdd(&eq->edata.nslots[3][xorbucketid], 1);

			if (xorslot < NSLOTS)
			{
				slotsmall &xs = eq->round3trees[xorbucketid].treessmall[xorslot];
				*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[1]);
				slottiny &xst = eq->round3trees[xorbucketid].treestiny[xorslot];
				uint2 ttx;
				ttx.x = bexor;
				ttx.y = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
				*(uint2*)(&xst.hash[0]) = ttx;
			}
		}
	}
}


template <u32 RB, u32 SM, int SSM, typename PACKER, u32 MAXPAIRS, u32 THREADS>
__global__ void digit_4(equi<RB, SM>* eq)
{
	__shared__ u16 ht[NRESTS][(SSM - 1)];
	__shared__ uint4 lastword[NSLOTS];
	__shared__ int ht_len[NRESTS];
	__shared__ int pairs[MAXPAIRS];
	__shared__ u32 pairs_len;
	__shared__ u32 next_pair;

	const u32 threadid = threadIdx.x;
	const u32 bucketid = blockIdx.x;

	// reset hashtable len
	if (threadid < NRESTS)
		ht_len[threadid] = 0;
	else if (threadid == (THREADS - 1))
		pairs_len = 0;
	else if (threadid == (THREADS - 33))
		next_pair = 0;

	u32 bsize = umin(eq->edata.nslots[3][bucketid], NSLOTS);

	u32 hr[2];
	int pos[2];
	pos[0] = pos[1] = SSM;

	u32 si[2];
	uint4 tt[2];
#ifdef OPT_SYNC_ALL
	__syncthreads();
#endif
	#pragma unroll 2
	for (u32 i = 0; i < 2; i++)
	{
		si[i] = i * THREADS + threadid;
		if (si[i] >= bsize) break;

		slotsmall &xs = eq->round3trees[bucketid].treessmall[si[i]];
		slottiny &xst = eq->round3trees[bucketid].treestiny[si[i]];

		// get xhash
		tt[i] = *(uint4*)(&xs.hash[0]);
		lastword[si[i]] = tt[i];
		hr[i] = xst.hash[0] & RESTMASK;
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();
	u32 xors[4];
	u32 xorbucketid, xorslot;

	#pragma unroll 2
	for (u32 i = 0; i < 2; i++)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			u16 p = ht[hr[i]][0];

			*(uint4*)(&xors[0]) = tt[i] ^ lastword[p];

			if (xors[3] != 0)
			{
				asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(4 + RB), "r"(BUCKBITS));
				xorslot = atomicAdd(&eq->edata.nslots[4][xorbucketid], 1);
				if (xorslot < NSLOTS)
				{
					slotsmall &xs = eq->treessmall[3][xorbucketid][xorslot];
					*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[0]);

					eq->round4bidandsids[xorbucketid][xorslot] = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
				}
			}

			for (int k = 1; k != pos[i]; ++k)
			{
				u32 pindex = atomicAdd(&pairs_len, 1);
				if (pindex >= MAXPAIRS) break;
				u16 prev = ht[hr[i]][k];
				pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
			}
		}
	}

	__syncthreads();

	// process pairs
	u32 plen = umin(pairs_len, MAXPAIRS);
	u32 i, k;
	for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		i = __byte_perm(pair, 0, 0x4510);
		k = __byte_perm(pair, 0, 0x4532);

		*(uint4*)(&xors[0]) = lastword[i] ^ lastword[k];
		if (xors[3] != 0)
		{
			asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(4 + RB), "r"(BUCKBITS));
			xorslot = atomicAdd(&eq->edata.nslots[4][xorbucketid], 1);
			if (xorslot < NSLOTS)
			{
				slotsmall &xs = eq->treessmall[3][xorbucketid][xorslot];
				*(uint4*)(&xs.hash[0]) = *(uint4*)(&xors[0]);
				eq->round4bidandsids[xorbucketid][xorslot] = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
			}
		}
	}
}


template <u32 RB, u32 SM, int SSM, typename PACKER, u32 MAXPAIRS, u32 THREADS>
__global__ void digit_5(equi<RB, SM>* eq)
{
	__shared__ u16 ht[NRESTS][(SSM - 1)];
	__shared__ uint4 lastword[NSLOTS];
	__shared__ int ht_len[NRESTS];
	__shared__ int pairs[MAXPAIRS];
	__shared__ u32 pairs_len;
	__shared__ u32 next_pair;

	const u32 threadid = threadIdx.x;
	const u32 bucketid = blockIdx.x;

	if (threadid < NRESTS)
		ht_len[threadid] = 0;
	else if (threadid == (THREADS - 1))
		pairs_len = 0;
	else if (threadid == (THREADS - 33))
		next_pair = 0;

	slotsmall* buck = eq->treessmall[3][bucketid];
	u32 bsize = umin(eq->edata.nslots[4][bucketid], NSLOTS);

	u32 hr[2];
	int pos[2];
	pos[0] = pos[1] = SSM;

	u32 si[2];
	uint4 tt[2];
#ifdef OPT_SYNC_ALL
	__syncthreads();
#endif
	#pragma unroll 2
	for (u32 i = 0; i < 2; i++)
	{
		si[i] = i * THREADS + threadid;
		if (si[i] >= bsize) break;

		const slotsmall* pslot1 = buck + si[i];

		tt[i] = *(uint4*)(&pslot1->hash[0]);
		lastword[si[i]] = tt[i];
		asm("bfe.u32 %0, %1, 4, %2;" : "=r"(hr[i]) : "r"(tt[i].x), "r"(RB));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();
	u32 xors[4];
	u32 bexor, xorbucketid, xorslot;

	#pragma unroll 2
	for (u32 i = 0; i < 2; i++)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			u16 p = ht[hr[i]][0];

			*(uint4*)(&xors[0]) = tt[i] ^ lastword[p];

			if (xors[3] != 0)
			{
				bexor = __byte_perm(xors[0], xors[1], 0x1076);
				asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(bexor), "r"(RB), "r"(BUCKBITS));
				xorslot = atomicAdd(&eq->edata.nslots[5][xorbucketid], 1);
				if (xorslot < NSLOTS)
				{
					slotsmall &xs = eq->treessmall[2][xorbucketid][xorslot];
					uint4 ttx;
					ttx.x = xors[1];
					ttx.y = xors[2];
					ttx.z = xors[3];
					ttx.w = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
					*(uint4*)(&xs.hash[0]) = ttx;
				}
			}

			for (int k = 1; k != pos[i]; ++k)
			{
				u32 pindex = atomicAdd(&pairs_len, 1);
				if (pindex >= MAXPAIRS) break;
				u16 prev = ht[hr[i]][k];
				pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
			}
		}
	}

	__syncthreads();

	// process pairs
	u32 plen = umin(pairs_len, MAXPAIRS);
	u32 i, k;
	for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		i = __byte_perm(pair, 0, 0x4510);
		k = __byte_perm(pair, 0, 0x4532);

		*(uint4*)(&xors[0]) = lastword[i] ^ lastword[k];

		if (xors[3] != 0)
		{
			bexor = __byte_perm(xors[0], xors[1], 0x1076);
			asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(bexor), "r"(RB), "r"(BUCKBITS));
			xorslot = atomicAdd(&eq->edata.nslots[5][xorbucketid], 1);
			if (xorslot < NSLOTS)
			{
				slotsmall &xs = eq->treessmall[2][xorbucketid][xorslot];
				uint4 tt;
				tt.x = xors[1];
				tt.y = xors[2];
				tt.z = xors[3];
				tt.w = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
				*(uint4*)(&xs.hash[0]) = tt;
			}
		}
	}
}


template <u32 RB, u32 SM, int SSM, typename PACKER, u32 MAXPAIRS>
__global__ void digit_6(equi<RB, SM>* eq)
{
	__shared__ u16 ht[NRESTS][(SSM - 1)];
	__shared__ uint2 lastword1[NSLOTS];
	__shared__ u32 lastword2[NSLOTS];
	__shared__ int ht_len[MAXPAIRS];
	__shared__ u32 pairs_len;
	__shared__ u32 bsize_sh;
	__shared__ u32 next_pair;

	const u32 threadid = threadIdx.x;
	const u32 bucketid = blockIdx.x;

	// reset hashtable len
	ht_len[threadid] = 0;
	if (threadid == (NRESTS - 1))
	{
		pairs_len = 0;
		next_pair = 0;
	}
	else if (threadid == (NRESTS - 33))
		bsize_sh = umin(eq->edata.nslots[5][bucketid], NSLOTS);

	slotsmall* buck = eq->treessmall[2][bucketid];

	u32 hr[3];
	int pos[3];
	pos[0] = pos[1] = pos[2] = SSM;

	u32 si[3];
	uint4 tt[3];

	__syncthreads();

	u32 bsize = bsize_sh;

	#pragma unroll 3
	for (u32 i = 0; i < 3; i++)
	{
		si[i] = i * NRESTS + threadid;
		if (si[i] >= bsize) break;

		const slotsmall* pslot1 = buck + si[i];

		tt[i] = *(uint4*)(&pslot1->hash[0]);
		lastword1[si[i]] = *(uint2*)(&tt[i].x);
		lastword2[si[i]] = tt[i].z;
		asm("bfe.u32 %0, %1, 16, %2;" : "=r"(hr[i]) : "r"(tt[i].x), "r"(RB));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	// doing this to save shared memory
	int* pairs = ht_len;
	__syncthreads();

	u32 xors[3];
	u32 bexor, xorbucketid, xorslot;

	#pragma unroll 3
	for (u32 i = 0; i < 3; i++)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			u16 p = ht[hr[i]][0];

			xors[2] = tt[i].z ^ lastword2[p];

			if (xors[2] != 0)
			{
				*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ lastword1[p];

				bexor = __byte_perm(xors[0], xors[1], 0x1076);
				xorbucketid = bexor >> (12 + RB);
				xorslot = atomicAdd(&eq->edata.nslots[6][xorbucketid], 1);
				if (xorslot < NSLOTS)
				{
					slotsmall &xs = eq->treessmall[0][xorbucketid][xorslot];
					uint4 ttx;
					ttx.x = xors[1];
					ttx.y = xors[2];
					ttx.z = bexor;
					ttx.w = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
					*(uint4*)(&xs.hash[0]) = ttx;
				}
			}

			if (pos[i] > 1)
			{
				p = ht[hr[i]][1];

				xors[2] = tt[i].z ^ lastword2[p];

				if (xors[2] != 0)
				{
					*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ lastword1[p];

					bexor = __byte_perm(xors[0], xors[1], 0x1076);
					xorbucketid = bexor >> (12 + RB);
					xorslot = atomicAdd(&eq->edata.nslots[6][xorbucketid], 1);
					if (xorslot < NSLOTS)
					{
						slotsmall &xs = eq->treessmall[0][xorbucketid][xorslot];
						uint4 ttx;
						ttx.x = xors[1];
						ttx.y = xors[2];
						ttx.z = bexor;
						ttx.w = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
						*(uint4*)(&xs.hash[0]) = ttx;
					}
				}

				for (int k = 2; k != pos[i]; ++k)
				{
					u32 pindex = atomicAdd(&pairs_len, 1);
					if (pindex >= MAXPAIRS) break;
					u16 prev = ht[hr[i]][k];
					pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
				}
			}
		}
	}

	__syncthreads();

	// process pairs
	u32 plen = umin(pairs_len, MAXPAIRS);
	for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		u32 pair = pairs[s];
		u32 i = __byte_perm(pair, 0, 0x4510);
		u32 k = __byte_perm(pair, 0, 0x4532);

		xors[2] = lastword2[i] ^ lastword2[k];
		if (xors[2] == 0)
			continue;

		*(uint2*)(&xors[0]) = lastword1[i] ^ lastword1[k];

		bexor = __byte_perm(xors[0], xors[1], 0x1076);
		xorbucketid = bexor >> (12 + RB);
		xorslot = atomicAdd(&eq->edata.nslots[6][xorbucketid], 1);
		if (xorslot >= NSLOTS) continue;
		slotsmall &xs = eq->treessmall[0][xorbucketid][xorslot];
		uint4 ttx;
		ttx.x = xors[1];
		ttx.y = xors[2];
		ttx.z = bexor;
		ttx.w = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
		*(uint4*)(&xs.hash[0]) = ttx;
	}
}


template <u32 RB, u32 SM, int SSM, typename PACKER, u32 MAXPAIRS>
__global__ void digit_7(equi<RB, SM>* eq)
{
	__shared__ u16 ht[NRESTS][(SSM - 1)];
	__shared__ u32 lastword[NSLOTS][2];
	__shared__ int ht_len[NRESTS];
	__shared__ int pairs[MAXPAIRS];
	__shared__ u32 pairs_len;
	__shared__ u32 bsize_sh;
	__shared__ u32 next_pair;

	const u32 threadid = threadIdx.x;
	const u32 bucketid = blockIdx.x;

	// reset hashtable len
	ht_len[threadid] = 0;
	if (threadid == (NRESTS - 1))
	{
		pairs_len = 0;
		next_pair = 0;
	}
	else if (threadid == (NRESTS - 33))
		bsize_sh = umin(eq->edata.nslots[6][bucketid], NSLOTS);

	slotsmall* buck = eq->treessmall[0][bucketid];

	u32 hr[3];
	int pos[3];
	pos[0] = pos[1] = pos[2] = SSM;

	u32 si[3];
	uint4 tt[3];

	__syncthreads();

	u32 bsize = bsize_sh;

	#pragma unroll 3
	for (u32 i = 0; i < 3; i++)
	{
		si[i] = i * NRESTS + threadid;
		if (si[i] >= bsize) break;

		const slotsmall* pslot1 = buck + si[i];

		// get xhash
		tt[i] = *(uint4*)(&pslot1->hash[0]);
		*(uint2*)(&lastword[si[i]][0]) = *(uint2*)(&tt[i].x);
		asm("bfe.u32 %0, %1, 12, %2;" : "=r"(hr[i]) : "r"(tt[i].z), "r"(RB));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();

	u32 xors[2];
	u32 xorbucketid, xorslot;

	#pragma unroll 3
	for (u32 i = 0; i < 3; i++)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			u16 p = ht[hr[i]][0];

			*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ *(uint2*)(&lastword[p][0]);

			if (xors[1] != 0)
			{
				asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(8 + RB), "r"(BUCKBITS));
				xorslot = atomicAdd(&eq->edata.nslots[7][xorbucketid], 1);
				if (xorslot < NSLOTS)
				{
					slotsmall &xs = eq->treessmall[1][xorbucketid][xorslot];
					uint4 ttx;
					ttx.x = xors[0];
					ttx.y = xors[1];
					ttx.z = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
					ttx.w = 0;
					*(uint4*)(&xs.hash[0]) = ttx;
				}
			}

			if (pos[i] > 1)
			{
				p = ht[hr[i]][1];

				*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ *(uint2*)(&lastword[p][0]);

				if (xors[1] != 0)
				{
					asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(8 + RB), "r"(BUCKBITS));
					xorslot = atomicAdd(&eq->edata.nslots[7][xorbucketid], 1);
					if (xorslot < NSLOTS)
					{
						slotsmall &xs = eq->treessmall[1][xorbucketid][xorslot];
						uint4 ttx;
						ttx.x = xors[0];
						ttx.y = xors[1];
						ttx.z = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
						ttx.w = 0;
						*(uint4*)(&xs.hash[0]) = ttx;
					}
				}

				for (int k = 2; k != pos[i]; ++k)
				{
					u32 pindex = atomicAdd(&pairs_len, 1);
					if (pindex >= MAXPAIRS) break;
					u16 prev = ht[hr[i]][k];
					pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
				}
			}
		}
	}

	__syncthreads();

	// process pairs
	u32 plen = umin(pairs_len, MAXPAIRS);
	for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		u32 i = __byte_perm(pair, 0, 0x4510);
		u32 k = __byte_perm(pair, 0, 0x4532);

		*(uint2*)(&xors[0]) = *(uint2*)(&lastword[i][0]) ^ *(uint2*)(&lastword[k][0]);

		if (xors[1] == 0)
			continue;

		asm("bfe.u32 %0, %1, %2, %3;" : "=r"(xorbucketid) : "r"(xors[0]), "r"(8 + RB), "r"(BUCKBITS));
		xorslot = atomicAdd(&eq->edata.nslots[7][xorbucketid], 1);
		if (xorslot >= NSLOTS) continue;
		slotsmall &xs = eq->treessmall[1][xorbucketid][xorslot];
		uint4 tt;
		tt.x = xors[0];
		tt.y = xors[1];
		tt.z = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
		tt.w = 0;
		*(uint4*)(&xs.hash[0]) = tt;
	}
}


template <u32 RB, u32 SM, int SSM, typename PACKER, u32 MAXPAIRS>
__global__ void digit_8(equi<RB, SM>* eq)
{
	__shared__ u16 ht[NRESTS][(SSM - 1)];
	__shared__ u32 lastword[NSLOTS][2];
	__shared__ int ht_len[NRESTS];
	__shared__ int pairs[MAXPAIRS];
	__shared__ u32 pairs_len;
	__shared__ u32 bsize_sh;
	__shared__ u32 next_pair;

	const u32 threadid = threadIdx.x;
	const u32 bucketid = blockIdx.x;

	// reset hashtable len
	ht_len[threadid] = 0;
	if (threadid == (NRESTS - 1))
	{
		next_pair = 0;
		pairs_len = 0;
	}
	else if (threadid == (NRESTS - 33))
		bsize_sh = umin(eq->edata.nslots[7][bucketid], NSLOTS);

	slotsmall* buck = eq->treessmall[1][bucketid];

	u32 hr[3];
	int pos[3];
	pos[0] = pos[1] = pos[2] = SSM;

	u32 si[3];
	uint2 tt[3];

	__syncthreads();

	u32 bsize = bsize_sh;

	#pragma unroll 3
	for (u32 i = 0; i < 3; i++)
	{
		si[i] = i * NRESTS + threadid;
		if (si[i] >= bsize) break;

		const slotsmall* pslot1 = buck + si[i];

		// get xhash
		tt[i] = *(uint2*)(&pslot1->hash[0]);
		*(uint2*)(&lastword[si[i]][0]) = *(uint2*)(&tt[i].x);
		asm("bfe.u32 %0, %1, 8, %2;" : "=r"(hr[i]) : "r"(tt[i].x), "r"(RB));
		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1)) ht[hr[i]][pos[i]] = si[i];
	}

	__syncthreads();

	u32 xors[2];
	u32 bexor, xorbucketid, xorslot;

	#pragma unroll 3
	for (u32 i = 0; i < 3; i++)
	{
		if (pos[i] >= SSM) continue;

		if (pos[i] > 0)
		{
			u16 p = ht[hr[i]][0];

			*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ *(uint2*)(&lastword[p][0]);

			if (xors[1] != 0)
			{
				bexor = __byte_perm(xors[0], xors[1], 0x0765);
				xorbucketid = bexor >> (12 + 8);
				xorslot = atomicAdd(&eq->edata.nslots8[xorbucketid], 1);
				if (xorslot < RB8_NSLOTS_LD)
				{
					slottiny &xs = eq->treestiny[0][xorbucketid][xorslot];
					uint2 tt;
					tt.x = xors[1];
					tt.y = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
					*(uint2*)(&xs.hash[0]) = tt;
				}
			}

			if (pos[i] > 1)
			{
				p = ht[hr[i]][1];

				*(uint2*)(&xors[0]) = *(uint2*)(&tt[i].x) ^ *(uint2*)(&lastword[p][0]);

				if (xors[1] != 0)
				{
					bexor = __byte_perm(xors[0], xors[1], 0x0765);
					xorbucketid = bexor >> (12 + 8);
					xorslot = atomicAdd(&eq->edata.nslots8[xorbucketid], 1);
					if (xorslot < RB8_NSLOTS_LD)
					{
						slottiny &xs = eq->treestiny[0][xorbucketid][xorslot];
						uint2 tt;
						tt.x = xors[1];
						tt.y = PACKER::set_bucketid_and_slots(bucketid, si[i], p, RB, SM);
						*(uint2*)(&xs.hash[0]) = tt;
					}
				}

				for (int k = 2; k != pos[i]; ++k)
				{
					u32 pindex = atomicAdd(&pairs_len, 1);
					if (pindex >= MAXPAIRS) break;
					u16 prev = ht[hr[i]][k];
					pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
				}
			}
		}
	}

	__syncthreads();

	// process pairs
	u32 plen = umin(pairs_len, MAXPAIRS);
	for (u32 s = atomicAdd(&next_pair, 1); s < plen; s = atomicAdd(&next_pair, 1))
	{
		int pair = pairs[s];
		u32 i = __byte_perm(pair, 0, 0x4510);
		u32 k = __byte_perm(pair, 0, 0x4532);

		*(uint2*)(&xors[0]) = *(uint2*)(&lastword[i][0]) ^ *(uint2*)(&lastword[k][0]);

		if (xors[1] == 0)
			continue;

		bexor = __byte_perm(xors[0], xors[1], 0x0765);
		xorbucketid = bexor >> (12 + 8);
		xorslot = atomicAdd(&eq->edata.nslots8[xorbucketid], 1);
		if (xorslot >= RB8_NSLOTS_LD) continue;
		slottiny &xs = eq->treestiny[0][xorbucketid][xorslot];
		uint2 tt;
		tt.x = xors[1];
		tt.y = PACKER::set_bucketid_and_slots(bucketid, i, k, RB, SM);
		*(uint2*)(&xs.hash[0]) = tt;
	}
}

/*
  Last round function is similar to previous ones but has different ending.
  We use warps to process final candidates. Each warp process one candidate.
  First two bidandsids (u32 of stored bucketid and two slotids) are retreived by
  lane 0 and lane 16, next four bidandsids by lane 0, 8, 16 and 24, ... until
  all lanes in warp have bidandsids from round 4. Next, each thread retreives
  16 indices. While doing so, indices are put into comparison using atomicExch
  to determine if there are duplicates (tromp's method). At the end, if no
  duplicates are found, candidate solution is saved (all indices). Note that this
  dup check method is not exact so CPU dup checking is needed after.
*/
template <u32 RB, u32 SM, int SSM, u32 FCT, typename PACKER, u32 MAXPAIRS, u32 DUPBITS, u32 W>
__global__ void digit_last_wdc(equi<RB, SM>* eq)
{
	__shared__ u8 shared_data[8192];
	int* ht_len = (int*)(&shared_data[0]);
	int* pairs = ht_len;
	u32* lastword = (u32*)(&shared_data[256 * 4]);
	u16* ht = (u16*)(&shared_data[256 * 4 + RB8_NSLOTS_LD * 4]);
	u32* pairs_len = (u32*)(&shared_data[8188]);

	const u32 threadid = threadIdx.x;
	const u32 bucketid = blockIdx.x;

	// reset hashtable len
	#pragma unroll
	for (u32 i = 0; i < FCT; i++)
		ht_len[(i * (256 / FCT)) + threadid] = 0;

	if (threadid == ((256 / FCT) - 1))
		*pairs_len = 0;

	slottiny* buck = eq->treestiny[0][bucketid];
	u32 bsize = umin(eq->edata.nslots8[bucketid], RB8_NSLOTS_LD);

	u32 si[3 * FCT];
	u32 hr[3 * FCT];
	int pos[3 * FCT];
	u32 lw[3 * FCT];

	#pragma unroll
	for (u32 i = 0; i < (3 * FCT); i++)
		pos[i] = SSM;

	__syncthreads();

	#pragma unroll
	for (u32 i = 0; i < (3 * FCT); i++)
	{
		si[i] = i * (256 / FCT) + threadid;
		if (si[i] >= bsize) break;

		const slottiny* pslot1 = buck + si[i];

		// get xhash
		uint2 tt = *(uint2*)(&pslot1->hash[0]);
		lw[i] = tt.x;
		lastword[si[i]] = lw[i];

		u32 a;
		asm("bfe.u32 %0, %1, 20, 8;" : "=r"(a) : "r"(lw[i]));
		hr[i] = a;

		pos[i] = atomicAdd(&ht_len[hr[i]], 1);
		if (pos[i] < (SSM - 1))
			ht[hr[i] * (SSM - 1) + pos[i]] = si[i];
	}

	__syncthreads();

	#pragma unroll
	for (u32 i = 0; i < (3 * FCT); i++)
	{
		if (pos[i] >= SSM) continue;

		for (int k = 0; k != pos[i]; ++k)
		{
			u16 prev = ht[hr[i] * (SSM - 1) + k];
			if (lw[i] != lastword[prev]) continue;
			u32 pindex = atomicAdd(pairs_len, 1);
			if (pindex >= MAXPAIRS) break;
			pairs[pindex] = __byte_perm(si[i], prev, 0x1054);
		}
	}

	__syncthreads();
	u32 plen = umin(*pairs_len, 64);

#define CALC_LEVEL(a, b, c, d) { \
	u32 plvl = levels[b]; \
	u32* bucks = eq->round4bidandsids[PACKER::get_bucketid(plvl, RB, SM)]; \
	u32 slot1 = PACKER::get_slot1(plvl, RB, SM); \
	u32 slot0 = PACKER::get_slot0(plvl, slot1, RB, SM); \
	levels[b] = bucks[slot1]; \
	levels[c] = bucks[slot0]; \
				}

#define CALC_LEVEL_SMALL(a, b, c, d) { \
	u32 plvl = levels[b]; \
	slotsmall* bucks = eq->treessmall[a][PACKER::get_bucketid(plvl, RB, SM)]; \
	u32 slot1 = PACKER::get_slot1(plvl, RB, SM); \
	u32 slot0 = PACKER::get_slot0(plvl, slot1, RB, SM); \
	levels[b] = bucks[slot1].hash[d]; \
	levels[c] = bucks[slot0].hash[d]; \
				}

	u32 lane = threadIdx.x & 0x1f;
	u32 par = threadIdx.x >> 5;

	u32* levels = (u32*)&pairs[MAXPAIRS + (par << DUPBITS)];
	u32* susp = levels;

	while (par < plen)
	{
		int pair = pairs[par];
		par += W;

		if (lane % 16 == 0)
		{
			u32 plvl;
			if (lane == 0) plvl = buck[__byte_perm(pair, 0, 0x4510)].hash[1];
			else plvl = buck[__byte_perm(pair, 0, 0x4532)].hash[1];
			slotsmall* bucks = eq->treessmall[1][PACKER::get_bucketid(plvl, RB, SM)];
			u32 slot1 = PACKER::get_slot1(plvl, RB, SM);
			u32 slot0 = PACKER::get_slot0(plvl, slot1, RB, SM);
			levels[lane] = bucks[slot1].hash[2];
			levels[lane + 8] = bucks[slot0].hash[2];
		}

		if (lane % 8 == 0)
			CALC_LEVEL_SMALL(0, lane, lane + 4, 3);

		if (lane % 4 == 0)
			CALC_LEVEL_SMALL(2, lane, lane + 2, 3);

		if (lane % 2 == 0)
			CALC_LEVEL(0, lane, lane + 1, 4);

		u32 ind[16];

		u32 f1 = levels[lane];
		const slottiny* buck_v4 = &eq->round3trees[PACKER::get_bucketid(f1, RB, SM)].treestiny[0];
		const u32 slot1_v4 = PACKER::get_slot1(f1, RB, SM);
		const u32 slot0_v4 = PACKER::get_slot0(f1, slot1_v4, RB, SM);

		susp[lane] = 0xffffffff;
		susp[32 + lane] = 0xffffffff;

#define CHECK_DUP(a) \
	__any(atomicExch(&susp[(ind[a] & ((1 << DUPBITS) - 1))], (ind[a] >> DUPBITS)) == (ind[a] >> DUPBITS))

		u32 f2 = buck_v4[slot1_v4].hash[1];
		const slottiny* buck_v3_1 = &eq->round2trees[PACKER::get_bucketid(f2, RB, SM)].treestiny[0];
		const u32 slot1_v3_1 = PACKER::get_slot1(f2, RB, SM);
		const u32 slot0_v3_1 = PACKER::get_slot0(f2, slot1_v3_1, RB, SM);

		susp[64 + lane] = 0xffffffff;
		susp[96 + lane] = 0xffffffff;

		u32 f0 = buck_v3_1[slot1_v3_1].hash[1];
		const slot* buck_v2_1 = eq->trees[0][PACKER::get_bucketid(f0, RB, SM)];
		const u32 slot1_v2_1 = PACKER::get_slot1(f0, RB, SM);
		const u32 slot0_v2_1 = PACKER::get_slot0(f0, slot1_v2_1, RB, SM);

		susp[128 + lane] = 0xffffffff;
		susp[160 + lane] = 0xffffffff;

		u32 f3 = buck_v2_1[slot1_v2_1].hash[6];
		const slot* buck_fin_1 = eq->round0trees[packer_default::get_bucketid(f3, 8, RB8_NSLOTS)];
		const u32 slot1_fin_1 = packer_default::get_slot1(f3, 8, RB8_NSLOTS);
		const u32 slot0_fin_1 = packer_default::get_slot0(f3, slot1_fin_1, 8, RB8_NSLOTS);

		susp[192 + lane] = 0xffffffff;
		susp[224 + lane] = 0xffffffff;

		ind[0] = buck_fin_1[slot1_fin_1].hash[7];
		if (CHECK_DUP(0)) continue;
		ind[1] = buck_fin_1[slot0_fin_1].hash[7];
		if (CHECK_DUP(1)) continue;

		u32 f4 = buck_v2_1[slot0_v2_1].hash[6];
		const slot* buck_fin_2 = eq->round0trees[packer_default::get_bucketid(f4, 8, RB8_NSLOTS)];
		const u32 slot1_fin_2 = packer_default::get_slot1(f4, 8, RB8_NSLOTS);
		const u32 slot0_fin_2 = packer_default::get_slot0(f4, slot1_fin_2, 8, RB8_NSLOTS);

		ind[2] = buck_fin_2[slot1_fin_2].hash[7];
		if (CHECK_DUP(2)) continue;
		ind[3] = buck_fin_2[slot0_fin_2].hash[7];
		if (CHECK_DUP(3)) continue;

		u32 f5 = buck_v3_1[slot0_v3_1].hash[1];
		const slot* buck_v2_2 = eq->trees[0][PACKER::get_bucketid(f5, RB, SM)];
		const u32 slot1_v2_2 = PACKER::get_slot1(f5, RB, SM);
		const u32 slot0_v2_2 = PACKER::get_slot0(f5, slot1_v2_2, RB, SM);

		u32 f6 = buck_v2_2[slot1_v2_2].hash[6];
		const slot* buck_fin_3 = eq->round0trees[packer_default::get_bucketid(f6, 8, RB8_NSLOTS)];
		const u32 slot1_fin_3 = packer_default::get_slot1(f6, 8, RB8_NSLOTS);
		const u32 slot0_fin_3 = packer_default::get_slot0(f6, slot1_fin_3, 8, RB8_NSLOTS);

		ind[4] = buck_fin_3[slot1_fin_3].hash[7];
		if (CHECK_DUP(4)) continue;
		ind[5] = buck_fin_3[slot0_fin_3].hash[7];
		if (CHECK_DUP(5)) continue;

		u32 f7 = buck_v2_2[slot0_v2_2].hash[6];
		const slot* buck_fin_4 = eq->round0trees[packer_default::get_bucketid(f7, 8, RB8_NSLOTS)];
		const u32 slot1_fin_4 = packer_default::get_slot1(f7, 8, RB8_NSLOTS);
		const u32 slot0_fin_4 = packer_default::get_slot0(f7, slot1_fin_4, 8, RB8_NSLOTS);

		ind[6] = buck_fin_4[slot1_fin_4].hash[7];
		if (CHECK_DUP(6)) continue;
		ind[7] = buck_fin_4[slot0_fin_4].hash[7];
		if (CHECK_DUP(7)) continue;

		u32 f8 = buck_v4[slot0_v4].hash[1];
		const slottiny* buck_v3_2 = &eq->round2trees[PACKER::get_bucketid(f8, RB, SM)].treestiny[0];
		const u32 slot1_v3_2 = PACKER::get_slot1(f8, RB, SM);
		const u32 slot0_v3_2 = PACKER::get_slot0(f8, slot1_v3_2, RB, SM);

		u32 f9 = buck_v3_2[slot1_v3_2].hash[1];
		const slot* buck_v2_3 = eq->trees[0][PACKER::get_bucketid(f9, RB, SM)];
		const u32 slot1_v2_3 = PACKER::get_slot1(f9, RB, SM);
		const u32 slot0_v2_3 = PACKER::get_slot0(f9, slot1_v2_3, RB, SM);

		u32 f10 = buck_v2_3[slot1_v2_3].hash[6];
		const slot* buck_fin_5 = eq->round0trees[packer_default::get_bucketid(f10, 8, RB8_NSLOTS)];
		const u32 slot1_fin_5 = packer_default::get_slot1(f10, 8, RB8_NSLOTS);
		const u32 slot0_fin_5 = packer_default::get_slot0(f10, slot1_fin_5, 8, RB8_NSLOTS);

		ind[8] = buck_fin_5[slot1_fin_5].hash[7];
		if (CHECK_DUP(8)) continue;
		ind[9] = buck_fin_5[slot0_fin_5].hash[7];
		if (CHECK_DUP(9)) continue;

		u32 f11 = buck_v2_3[slot0_v2_3].hash[6];
		const slot* buck_fin_6 = eq->round0trees[packer_default::get_bucketid(f11, 8, RB8_NSLOTS)];
		const u32 slot1_fin_6 = packer_default::get_slot1(f11, 8, RB8_NSLOTS);
		const u32 slot0_fin_6 = packer_default::get_slot0(f11, slot1_fin_6, 8, RB8_NSLOTS);

		ind[10] = buck_fin_6[slot1_fin_6].hash[7];
		if (CHECK_DUP(10)) continue;
		ind[11] = buck_fin_6[slot0_fin_6].hash[7];
		if (CHECK_DUP(11)) continue;

		u32 f12 = buck_v3_2[slot0_v3_2].hash[1];
		const slot* buck_v2_4 = eq->trees[0][PACKER::get_bucketid(f12, RB, SM)];
		const u32 slot1_v2_4 = PACKER::get_slot1(f12, RB, SM);
		const u32 slot0_v2_4 = PACKER::get_slot0(f12, slot1_v2_4, RB, SM);

		u32 f13 = buck_v2_4[slot1_v2_4].hash[6];
		const slot* buck_fin_7 = eq->round0trees[packer_default::get_bucketid(f13, 8, RB8_NSLOTS)];
		const u32 slot1_fin_7 = packer_default::get_slot1(f13, 8, RB8_NSLOTS);
		const u32 slot0_fin_7 = packer_default::get_slot0(f13, slot1_fin_7, 8, RB8_NSLOTS);

		ind[12] = buck_fin_7[slot1_fin_7].hash[7];
		if (CHECK_DUP(12)) continue;
		ind[13] = buck_fin_7[slot0_fin_7].hash[7];
		if (CHECK_DUP(13)) continue;

		u32 f14 = buck_v2_4[slot0_v2_4].hash[6];
		const slot* buck_fin_8 = eq->round0trees[packer_default::get_bucketid(f14, 8, RB8_NSLOTS)];
		const u32 slot1_fin_8 = packer_default::get_slot1(f14, 8, RB8_NSLOTS);
		const u32 slot0_fin_8 = packer_default::get_slot0(f14, slot1_fin_8, 8, RB8_NSLOTS);

		ind[14] = buck_fin_8[slot1_fin_8].hash[7];
		if (CHECK_DUP(14)) continue;
		ind[15] = buck_fin_8[slot0_fin_8].hash[7];
		if (CHECK_DUP(15)) continue;

		u32 soli;
		if (lane == 0) {
			soli = atomicAdd(&eq->edata.srealcont.nsols, 1);
		}
#if __CUDA_ARCH__ >= 300
		// all threads get the value from lane 0
		soli = __shfl2(soli, 0);
#else
		__syncthreads();
		soli = eq->edata.srealcont.nsols;
#endif
		if (soli < MAXREALSOLS)
		{
			u32 pos = lane << 4;
			*(uint4*)(&eq->edata.srealcont.sols[soli][pos     ]) = *(uint4*)(&ind[ 0]);
			*(uint4*)(&eq->edata.srealcont.sols[soli][pos +  4]) = *(uint4*)(&ind[ 4]);
			*(uint4*)(&eq->edata.srealcont.sols[soli][pos +  8]) = *(uint4*)(&ind[ 8]);
			*(uint4*)(&eq->edata.srealcont.sols[soli][pos + 12]) = *(uint4*)(&ind[12]);
		}
	}
}

//std::mutex dev_init;
int dev_init_done[MAX_GPUS] = { 0 };

__host__
static int compu32(const void *pa, const void *pb)
{
	uint32_t a = *(uint32_t *)pa, b = *(uint32_t *)pb;
	return a<b ? -1 : a == b ? 0 : +1;
}

__host__
static bool duped(uint32_t* prf)
{
	uint32_t sortprf[512];
	memcpy(sortprf, prf, sizeof(uint32_t) * 512);
	qsort(sortprf, 512, sizeof(uint32_t), &compu32);
	for (uint32_t i = 1; i<512; i++) {
		if (sortprf[i] <= sortprf[i - 1])
			return true;
	}
	return false;
}

__host__
static void sort_pair(uint32_t *a, uint32_t len)
{
	uint32_t *b = a + len;
	uint32_t  tmp, need_sorting = 0;
	for (uint32_t i = 0; i < len; i++) {
		if (need_sorting || a[i] > b[i])
		{
			need_sorting = 1;
			tmp = a[i];
			a[i] = b[i];
			b[i] = tmp;
		}
		else if (a[i] < b[i])
			return;
	}
}

__host__
static void setheader(blake2b_state *ctx, const char *header, const u32 headerLen, const char* nce, const u32 nonceLen)
{
	uint32_t le_N = WN;
	uint32_t le_K = WK;
	uchar personal[] = "ZcashPoW01230123";
	memcpy(personal + 8, &le_N, 4);
	memcpy(personal + 12, &le_K, 4);
	blake2b_param P[1];
	P->digest_length = HASHOUT;
	P->key_length = 0;
	P->fanout = 1;
	P->depth = 1;
	P->leaf_length = 0;
	P->node_offset = 0;
	P->node_depth = 0;
	P->inner_length = 0;
	memset(P->reserved, 0, sizeof(P->reserved));
	memset(P->salt, 0, sizeof(P->salt));
	memcpy(P->personal, (const uint8_t *)personal, 16);
	eq_blake2b_init_param(ctx, P);
	eq_blake2b_update(ctx, (const uchar *)header, headerLen);
	if (nonceLen) eq_blake2b_update(ctx, (const uchar *)nce, nonceLen);
}

#ifdef WIN32
typedef CUresult(CUDAAPI *dec_cuDeviceGet)(CUdevice*, int);
typedef CUresult(CUDAAPI *dec_cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
typedef CUresult(CUDAAPI *dec_cuCtxPushCurrent)(CUcontext);
typedef CUresult(CUDAAPI *dec_cuCtxDestroy)(CUcontext);

dec_cuDeviceGet _cuDeviceGet = nullptr;
dec_cuCtxCreate _cuCtxCreate = nullptr;
dec_cuCtxPushCurrent _cuCtxPushCurrent = nullptr;
dec_cuCtxDestroy _cuCtxDestroy = nullptr;
#endif

template <u32 RB, u32 SM, u32 SSM, u32 THREADS, typename PACKER>
__host__ eq_cuda_context<RB, SM, SSM, THREADS, PACKER>::eq_cuda_context(int thr_id, int dev_id)
{
	thread_id = thr_id;
	device_id = dev_id;
	solutions = nullptr;
	equi_mem_sz = sizeof(equi<RB, SM>);
	throughput = NBLOCKS;
	totalblocks = NBLOCKS/FD_THREADS;
	threadsperblock = FD_THREADS;
	threadsperblock_digits = THREADS;

	//dev_init.lock();
	if (!dev_init_done[device_id])
	{
		// only first thread shall init device
		checkCudaErrors(cudaSetDevice(device_id));
		checkCudaErrors(cudaDeviceReset());
		checkCudaErrors(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

		pctx = nullptr;
	}
	else
	{
		// create new context
		CUdevice dev;

#ifdef WIN32
		if (_cuDeviceGet == nullptr)
		{
			HMODULE hmod = LoadLibraryA("nvcuda.dll");
			if (hmod == NULL)
				throw std::runtime_error("Failed to load nvcuda.dll");
			_cuDeviceGet = (dec_cuDeviceGet)GetProcAddress(hmod, "cuDeviceGet");
			if (_cuDeviceGet == nullptr)
				throw std::runtime_error("Failed to get cuDeviceGet address");
			_cuCtxCreate = (dec_cuCtxCreate)GetProcAddress(hmod, "cuCtxCreate_v2");
			if (_cuCtxCreate == nullptr)
				throw std::runtime_error("Failed to get cuCtxCreate address");
			_cuCtxPushCurrent = (dec_cuCtxPushCurrent)GetProcAddress(hmod, "cuCtxPushCurrent_v2");
			if (_cuCtxPushCurrent == nullptr)
				throw std::runtime_error("Failed to get cuCtxPushCurrent address");
			_cuCtxDestroy = (dec_cuCtxDestroy)GetProcAddress(hmod, "cuCtxDestroy_v2");
			if (_cuCtxDestroy == nullptr)
				throw std::runtime_error("Failed to get cuCtxDestroy address");
		}

		checkCudaDriverErrors(_cuDeviceGet(&dev, device_id));
		checkCudaDriverErrors(_cuCtxCreate(&pctx, CU_CTX_SCHED_BLOCKING_SYNC, dev));
		checkCudaDriverErrors(_cuCtxPushCurrent(pctx));
#else
		checkCudaDriverErrors(cuDeviceGet(&dev, device_id));
		checkCudaDriverErrors(cuCtxCreate(&pctx, CU_CTX_SCHED_BLOCKING_SYNC, dev));
		checkCudaDriverErrors(cuCtxPushCurrent(pctx));
#endif
	}
	++dev_init_done[device_id];
	//dev_init.unlock();

	if (cudaMalloc((void**)&device_eq, equi_mem_sz) != cudaSuccess)
		throw std::runtime_error("CUDA: failed to alloc memory");

	solutions = (scontainerreal*) malloc(sizeof(scontainerreal));
	if (!solutions)
		throw std::runtime_error("EOM: failed to alloc solutions memory");
}

template <u32 RB, u32 SM, u32 SSM, u32 THREADS, typename PACKER>
__host__ void eq_cuda_context<RB, SM, SSM, THREADS, PACKER>::solve(const char *tequihash_header,
	unsigned int tequihash_header_len,
	const char* nonce,
	unsigned int nonce_len,
	fn_cancel cancelf,
	fn_solution solutionf,
	fn_hashdone hashdonef)
{
	blake2b_state blake_ctx;

	int blocks = NBUCKETS;

	setheader(&blake_ctx, tequihash_header, tequihash_header_len, nonce, nonce_len);

	// todo: improve
	// djezo solver allows last 4 bytes of nonce to be iterrated
	// this can be used to create internal loop - calc initial blake hash only once, then load 8*8 bytes on device (blake state h)
	// then just iterate nn++
	// less CPU load, 1 cudaMemcpy less -> faster
	//u32 nn = *(u32*)&nonce[28];
	u32 nn = 0;

	checkCudaErrors(cudaMemcpy(&device_eq->blake_h, &blake_ctx.h, sizeof(u64) * 8, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(&device_eq->edata, 0, sizeof(device_eq->edata)));

	digit_first<RB, SM, PACKER> <<<NBLOCKS / FD_THREADS, FD_THREADS >>>(device_eq, nn);

	digit_1<RB, SM, SSM, PACKER, 4 * NRESTS, 512> <<<4096, 512 >>>(device_eq);
	digit_2<RB, SM, SSM, PACKER, 4 * NRESTS, THREADS> <<<blocks, THREADS >>>(device_eq);
	digit_3<RB, SM, SSM, PACKER, 4 * NRESTS, THREADS> <<<blocks, THREADS >>>(device_eq);

	if (cancelf(thread_id)) return;

	digit_4<RB, SM, SSM, PACKER, 4 * NRESTS, THREADS> <<<blocks, THREADS >>>(device_eq);
	digit_5<RB, SM, SSM, PACKER, 4 * NRESTS, THREADS> <<<blocks, THREADS >>>(device_eq);

	digit_6<RB, SM, SSM - 1, PACKER, 3 * NRESTS> <<<blocks, NRESTS >>>(device_eq);
	digit_7<RB, SM, SSM - 1, PACKER, 3 * NRESTS> <<<blocks, NRESTS >>>(device_eq);
	digit_8<RB, SM, SSM - 1, PACKER, 3 * NRESTS> <<<blocks, NRESTS >>>(device_eq);

	digit_last_wdc<RB, SM, SSM - 3, 2, PACKER, 64, 8, 4> <<<4096, 256 / 2 >>>(device_eq);

	checkCudaErrors(cudaMemcpy(solutions, &device_eq->edata.srealcont, (MAXREALSOLS * (512 * 4)) + 4, cudaMemcpyDeviceToHost));

	//printf("T%d nsols: %u\n", thread_id, solutions->nsols);
	//if (solutions->nsols > 9)
	//	printf("missing sol, total: %u\n", solutions->nsols);

	for (u32 s = 0; (s < solutions->nsols) && (s < MAXREALSOLS); s++)
	{
		// remove dups on CPU (dup removal on GPU is not fully exact and can pass on some invalid solutions)
		if (duped(solutions->sols[s])) continue;

		// perform sort of pairs
		for (uint32_t level = 0; level < 9; level++)
			for (uint32_t i = 0; i < (1 << 9); i += (2 << level))
				sort_pair(&solutions->sols[s][i], 1 << level);

		std::vector<uint32_t> index_vector(PROOFSIZE);
		for (u32 i = 0; i < PROOFSIZE; i++) {
			index_vector[i] = solutions->sols[s][i];
		}

		solutionf(thread_id, index_vector, DIGITBITS, nullptr);
	}

	// ccminer: only use hashdonef if no solutions...
	if (!solutions->nsols)
		hashdonef(thread_id);
}

// destructor
template <u32 RB, u32 SM, u32 SSM, u32 THREADS, typename PACKER>
__host__
void eq_cuda_context<RB, SM, SSM, THREADS, PACKER>::freemem()
{
	if (solutions)
		free(solutions);

	if (device_eq) {
		cudaFree(device_eq);
		device_eq = NULL;
	}

	if (pctx) {
		// non primary thread, destroy context
#ifdef WIN32
		checkCudaDriverErrors(_cuCtxDestroy(pctx));
#else
		checkCudaDriverErrors(cuCtxDestroy(pctx));
#endif
	} else {
		checkCudaErrors(cudaDeviceReset());
		dev_init_done[device_id] = 0;
	}
}

template <u32 RB, u32 SM, u32 SSM, u32 THREADS, typename PACKER>
__host__
eq_cuda_context<RB, SM, SSM, THREADS, PACKER>::~eq_cuda_context()
{
	freemem();
}

#ifdef CONFIG_MODE_1
template class eq_cuda_context<CONFIG_MODE_1>;
#endif

#ifdef CONFIG_MODE_2
template class eq_cuda_context<CONFIG_MODE_2>;
#endif

#ifdef CONFIG_MODE_3
template class eq_cuda_context<CONFIG_MODE_3>;
#endif
