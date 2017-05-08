/*
 * Copyright 2009 Colin Percival, 2011 ArtForz, 2011-2013 pooler
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file was originally written by Colin Percival as part of the Tarsnap
 * online backup system.
 */

#ifdef WIN32
#include <ppl.h>
using namespace Concurrency;
#else
#include <omp.h>
#endif

#include "miner.h"
#include "scrypt/salsa_kernel.h"
#include "scrypt/sha256.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <emmintrin.h>
#if !((defined(__FreeBSD__) && __FreeBSD__ >= 10) || defined(__APPLE__))
#include <malloc.h>
#else
#include <stdlib.h>
#endif
#include <new>

// A thin wrapper around the builtin __m128i type
class uint32x4_t
{
public:
#if WIN32
	void * operator new(size_t size) _THROW1(_STD bad_alloc) { void *p; if ((p = _aligned_malloc(size, 16)) == 0) { static const std::bad_alloc nomem; _RAISE(nomem); } return (p); }
	void operator delete(void *p) { _aligned_free(p); }
	void * operator new[](size_t size) _THROW1(_STD bad_alloc) { void *p; if ((p = _aligned_malloc(size, 16)) == 0) { static const std::bad_alloc nomem; _RAISE(nomem); } return (p); }
	void operator delete[](void *p) { _aligned_free(p); }
#else
	void * operator new(size_t size) throw(std::bad_alloc) { void *p; if (posix_memalign(&p, 16, size) < 0) { static const std::bad_alloc nomem; throw nomem; } return (p); }
	void operator delete(void *p) { free(p); }
	void * operator new[](size_t size) throw(std::bad_alloc) { void *p; if (posix_memalign(&p, 16, size) < 0) { static const std::bad_alloc nomem; throw nomem; } return (p); }
	void operator delete[](void *p) { free(p); }
#endif
	uint32x4_t() { };
	uint32x4_t(const __m128i init) { val = init; }
	uint32x4_t(const uint32_t init) { val = _mm_set1_epi32((int)init); }
	uint32x4_t(const uint32_t a, const uint32_t b, const uint32_t c, const uint32_t d) { val = _mm_setr_epi32((int)a,(int)b,(int)c,(int)d); }
	inline operator const __m128i() const { return val; }
	inline const uint32x4_t operator+(const uint32x4_t &other) const { return _mm_add_epi32(val, other); }
	inline const uint32x4_t operator+(const uint32_t other) const { return _mm_add_epi32(val, _mm_set1_epi32((int)other)); }
	inline uint32x4_t& operator+=(const uint32x4_t other) { val = _mm_add_epi32(val, other); return *this; }
	inline uint32x4_t& operator+=(const uint32_t other) { val = _mm_add_epi32(val, _mm_set1_epi32((int)other)); return *this; }
	inline const uint32x4_t operator&(const uint32_t other) const { return _mm_and_si128(val, _mm_set1_epi32((int)other)); }
	inline const uint32x4_t operator&(const uint32x4_t &other) const { return _mm_and_si128(val, other); }
	inline const uint32x4_t operator|(const uint32x4_t &other) const { return _mm_or_si128(val, other); }
	inline const uint32x4_t operator^(const uint32x4_t &other) const { return _mm_xor_si128(val, other); }
	inline const uint32x4_t operator<<(const int num) const { return _mm_slli_epi32(val, num); }
	inline const uint32x4_t operator>>(const int num) const { return _mm_srli_epi32(val, num); }
	inline const uint32_t operator[](const int num) const { return ((uint32_t*)&val)[num]; }
 protected:
	__m128i val;
};

// non-member overload
inline const uint32x4_t operator+(const uint32_t left, const uint32x4_t &right) { return _mm_add_epi32(_mm_set1_epi32((int)left), right); }


//
// Code taken from sha2.cpp and vectorized, with minimal changes where required
// Not all subroutines are actually used.
//

#define bswap_32x4(x) ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) \
					 | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))

static __inline uint32x4_t swab32x4(const uint32x4_t &v)
{
	return bswap_32x4(v);
}

static const uint32_t sha256_h[8] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

static const uint32_t sha256_k[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
	0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
	0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
	0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
	0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
	0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
	0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
	0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
	0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

void sha256_initx4(uint32x4_t *statex4)
{
	for (int i=0; i<8; ++i)
		statex4[i] = sha256_h[i];
}

/* Elementary functions used by SHA256 */
#define Ch(x, y, z)     ((x & (y ^ z)) ^ z)
#define Maj(x, y, z)    ((x & (y | z)) | (y & z))
#define ROTR(x, n)      ((x >> n) | (x << (32 - n)))
#define S0(x)           (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define S1(x)           (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define s0(x)           (ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3))
#define s1(x)           (ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10))

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
	RND(S[(64 - i) % 8], S[(65 - i) % 8], \
		S[(66 - i) % 8], S[(67 - i) % 8], \
		S[(68 - i) % 8], S[(69 - i) % 8], \
		S[(70 - i) % 8], S[(71 - i) % 8], \
		W[i] + sha256_k[i])

/*
 * SHA256 block compression function.  The 256-bit state is transformed via
 * the 512-bit input block to produce a new state.
 */
void sha256_transformx4(uint32x4_t *state, const uint32x4_t *block, int swap)
{
	uint32x4_t W[64];
	uint32x4_t S[8];
	uint32x4_t t0, t1;
	int i;

	/* 1. Prepare message schedule W. */
	if (swap) {
		for (i = 0; i < 16; i++)
			W[i] = swab32x4(block[i]);
	} else
		memcpy(W, block, 4*64);
	for (i = 16; i < 64; i += 2) {
		W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}

	/* 2. Initialize working variables. */
	memcpy(S, state, 4*32);

	/* 3. Mix. */
	RNDr(S, W,  0);
	RNDr(S, W,  1);
	RNDr(S, W,  2);
	RNDr(S, W,  3);
	RNDr(S, W,  4);
	RNDr(S, W,  5);
	RNDr(S, W,  6);
	RNDr(S, W,  7);
	RNDr(S, W,  8);
	RNDr(S, W,  9);
	RNDr(S, W, 10);
	RNDr(S, W, 11);
	RNDr(S, W, 12);
	RNDr(S, W, 13);
	RNDr(S, W, 14);
	RNDr(S, W, 15);
	RNDr(S, W, 16);
	RNDr(S, W, 17);
	RNDr(S, W, 18);
	RNDr(S, W, 19);
	RNDr(S, W, 20);
	RNDr(S, W, 21);
	RNDr(S, W, 22);
	RNDr(S, W, 23);
	RNDr(S, W, 24);
	RNDr(S, W, 25);
	RNDr(S, W, 26);
	RNDr(S, W, 27);
	RNDr(S, W, 28);
	RNDr(S, W, 29);
	RNDr(S, W, 30);
	RNDr(S, W, 31);
	RNDr(S, W, 32);
	RNDr(S, W, 33);
	RNDr(S, W, 34);
	RNDr(S, W, 35);
	RNDr(S, W, 36);
	RNDr(S, W, 37);
	RNDr(S, W, 38);
	RNDr(S, W, 39);
	RNDr(S, W, 40);
	RNDr(S, W, 41);
	RNDr(S, W, 42);
	RNDr(S, W, 43);
	RNDr(S, W, 44);
	RNDr(S, W, 45);
	RNDr(S, W, 46);
	RNDr(S, W, 47);
	RNDr(S, W, 48);
	RNDr(S, W, 49);
	RNDr(S, W, 50);
	RNDr(S, W, 51);
	RNDr(S, W, 52);
	RNDr(S, W, 53);
	RNDr(S, W, 54);
	RNDr(S, W, 55);
	RNDr(S, W, 56);
	RNDr(S, W, 57);
	RNDr(S, W, 58);
	RNDr(S, W, 59);
	RNDr(S, W, 60);
	RNDr(S, W, 61);
	RNDr(S, W, 62);
	RNDr(S, W, 63);

	/* 4. Mix local working variables into global state */
	for (i = 0; i < 8; i++)
		state[i] += S[i];
}

static const uint32_t sha256d_hash1[16] = {
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x80000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000100
};

static void sha256dx4(uint32x4_t *hash, uint32x4_t *data)
{
	uint32x4_t S[16];

	sha256_initx4(S);
	sha256_transformx4(S, data, 0);
	sha256_transformx4(S, data + 16, 0);
	for (int i=8; i<16; ++i)
		S[i] = sha256d_hash1[i];
	sha256_initx4(hash);
	sha256_transformx4(hash, S, 0);
}

static inline void sha256d_preextendx4(uint32x4_t *W)
{
	W[16] = s1(W[14]) + W[ 9] + s0(W[ 1]) + W[ 0];
	W[17] = s1(W[15]) + W[10] + s0(W[ 2]) + W[ 1];
	W[18] = s1(W[16]) + W[11]             + W[ 2];
	W[19] = s1(W[17]) + W[12] + s0(W[ 4]);
	W[20] =             W[13] + s0(W[ 5]) + W[ 4];
	W[21] =             W[14] + s0(W[ 6]) + W[ 5];
	W[22] =             W[15] + s0(W[ 7]) + W[ 6];
	W[23] =             W[16] + s0(W[ 8]) + W[ 7];
	W[24] =             W[17] + s0(W[ 9]) + W[ 8];
	W[25] =                     s0(W[10]) + W[ 9];
	W[26] =                     s0(W[11]) + W[10];
	W[27] =                     s0(W[12]) + W[11];
	W[28] =                     s0(W[13]) + W[12];
	W[29] =                     s0(W[14]) + W[13];
	W[30] =                     s0(W[15]) + W[14];
	W[31] =                     s0(W[16]) + W[15];
}

static inline void sha256d_prehashx4(uint32x4_t *S, const uint32x4_t *W)
{
	uint32x4_t t0, t1;
	RNDr(S, W, 0);
	RNDr(S, W, 1);
	RNDr(S, W, 2);
}

static inline void sha256d_msx4(uint32x4_t *hash, uint32x4_t *W,
	const uint32_t *midstate, const uint32_t *prehash)
{
	uint32x4_t S[64];
	uint32x4_t t0, t1;
	int i;

	S[18] = W[18];
	S[19] = W[19];
	S[20] = W[20];
	S[22] = W[22];
	S[23] = W[23];
	S[24] = W[24];
	S[30] = W[30];
	S[31] = W[31];

	W[18] += s0(W[3]);
	W[19] += W[3];
	W[20] += s1(W[18]);
	W[21]  = s1(W[19]);
	W[22] += s1(W[20]);
	W[23] += s1(W[21]);
	W[24] += s1(W[22]);
	W[25]  = s1(W[23]) + W[18];
	W[26]  = s1(W[24]) + W[19];
	W[27]  = s1(W[25]) + W[20];
	W[28]  = s1(W[26]) + W[21];
	W[29]  = s1(W[27]) + W[22];
	W[30] += s1(W[28]) + W[23];
	W[31] += s1(W[29]) + W[24];
	for (i = 32; i < 64; i += 2) {
		W[i]   = s1(W[i - 2]) + W[i - 7] + s0(W[i - 15]) + W[i - 16];
		W[i+1] = s1(W[i - 1]) + W[i - 6] + s0(W[i - 14]) + W[i - 15];
	}

	for (i=0; i<8; ++i)
		S[i] = prehash[i];

	RNDr(S, W,  3);
	RNDr(S, W,  4);
	RNDr(S, W,  5);
	RNDr(S, W,  6);
	RNDr(S, W,  7);
	RNDr(S, W,  8);
	RNDr(S, W,  9);
	RNDr(S, W, 10);
	RNDr(S, W, 11);
	RNDr(S, W, 12);
	RNDr(S, W, 13);
	RNDr(S, W, 14);
	RNDr(S, W, 15);
	RNDr(S, W, 16);
	RNDr(S, W, 17);
	RNDr(S, W, 18);
	RNDr(S, W, 19);
	RNDr(S, W, 20);
	RNDr(S, W, 21);
	RNDr(S, W, 22);
	RNDr(S, W, 23);
	RNDr(S, W, 24);
	RNDr(S, W, 25);
	RNDr(S, W, 26);
	RNDr(S, W, 27);
	RNDr(S, W, 28);
	RNDr(S, W, 29);
	RNDr(S, W, 30);
	RNDr(S, W, 31);
	RNDr(S, W, 32);
	RNDr(S, W, 33);
	RNDr(S, W, 34);
	RNDr(S, W, 35);
	RNDr(S, W, 36);
	RNDr(S, W, 37);
	RNDr(S, W, 38);
	RNDr(S, W, 39);
	RNDr(S, W, 40);
	RNDr(S, W, 41);
	RNDr(S, W, 42);
	RNDr(S, W, 43);
	RNDr(S, W, 44);
	RNDr(S, W, 45);
	RNDr(S, W, 46);
	RNDr(S, W, 47);
	RNDr(S, W, 48);
	RNDr(S, W, 49);
	RNDr(S, W, 50);
	RNDr(S, W, 51);
	RNDr(S, W, 52);
	RNDr(S, W, 53);
	RNDr(S, W, 54);
	RNDr(S, W, 55);
	RNDr(S, W, 56);
	RNDr(S, W, 57);
	RNDr(S, W, 58);
	RNDr(S, W, 59);
	RNDr(S, W, 60);
	RNDr(S, W, 61);
	RNDr(S, W, 62);
	RNDr(S, W, 63);

	for (i = 0; i < 8; i++)
		S[i] += midstate[i];

	W[18] = S[18];
	W[19] = S[19];
	W[20] = S[20];
	W[22] = S[22];
	W[23] = S[23];
	W[24] = S[24];
	W[30] = S[30];
	W[31] = S[31];

	for (i=8; i<16; ++i)
		S[i] = sha256d_hash1[i];
	S[16] = s1(sha256d_hash1[14]) + sha256d_hash1[ 9] + s0(S[ 1]) + S[ 0];
	S[17] = s1(sha256d_hash1[15]) + sha256d_hash1[10] + s0(S[ 2]) + S[ 1];
	S[18] = s1(S[16]) + sha256d_hash1[11] + s0(S[ 3]) + S[ 2];
	S[19] = s1(S[17]) + sha256d_hash1[12] + s0(S[ 4]) + S[ 3];
	S[20] = s1(S[18]) + sha256d_hash1[13] + s0(S[ 5]) + S[ 4];
	S[21] = s1(S[19]) + sha256d_hash1[14] + s0(S[ 6]) + S[ 5];
	S[22] = s1(S[20]) + sha256d_hash1[15] + s0(S[ 7]) + S[ 6];
	S[23] = s1(S[21]) + S[16] + s0(sha256d_hash1[ 8]) + S[ 7];
	S[24] = s1(S[22]) + S[17] + s0(sha256d_hash1[ 9]) + sha256d_hash1[ 8];
	S[25] = s1(S[23]) + S[18] + s0(sha256d_hash1[10]) + sha256d_hash1[ 9];
	S[26] = s1(S[24]) + S[19] + s0(sha256d_hash1[11]) + sha256d_hash1[10];
	S[27] = s1(S[25]) + S[20] + s0(sha256d_hash1[12]) + sha256d_hash1[11];
	S[28] = s1(S[26]) + S[21] + s0(sha256d_hash1[13]) + sha256d_hash1[12];
	S[29] = s1(S[27]) + S[22] + s0(sha256d_hash1[14]) + sha256d_hash1[13];
	S[30] = s1(S[28]) + S[23] + s0(sha256d_hash1[15]) + sha256d_hash1[14];
	S[31] = s1(S[29]) + S[24] + s0(S[16])             + sha256d_hash1[15];
	for (i = 32; i < 60; i += 2) {
		S[i]   = s1(S[i - 2]) + S[i - 7] + s0(S[i - 15]) + S[i - 16];
		S[i+1] = s1(S[i - 1]) + S[i - 6] + s0(S[i - 14]) + S[i - 15];
	}
	S[60] = s1(S[58]) + S[53] + s0(S[45]) + S[44];

	sha256_initx4(hash);

	RNDr(hash, S,  0);
	RNDr(hash, S,  1);
	RNDr(hash, S,  2);
	RNDr(hash, S,  3);
	RNDr(hash, S,  4);
	RNDr(hash, S,  5);
	RNDr(hash, S,  6);
	RNDr(hash, S,  7);
	RNDr(hash, S,  8);
	RNDr(hash, S,  9);
	RNDr(hash, S, 10);
	RNDr(hash, S, 11);
	RNDr(hash, S, 12);
	RNDr(hash, S, 13);
	RNDr(hash, S, 14);
	RNDr(hash, S, 15);
	RNDr(hash, S, 16);
	RNDr(hash, S, 17);
	RNDr(hash, S, 18);
	RNDr(hash, S, 19);
	RNDr(hash, S, 20);
	RNDr(hash, S, 21);
	RNDr(hash, S, 22);
	RNDr(hash, S, 23);
	RNDr(hash, S, 24);
	RNDr(hash, S, 25);
	RNDr(hash, S, 26);
	RNDr(hash, S, 27);
	RNDr(hash, S, 28);
	RNDr(hash, S, 29);
	RNDr(hash, S, 30);
	RNDr(hash, S, 31);
	RNDr(hash, S, 32);
	RNDr(hash, S, 33);
	RNDr(hash, S, 34);
	RNDr(hash, S, 35);
	RNDr(hash, S, 36);
	RNDr(hash, S, 37);
	RNDr(hash, S, 38);
	RNDr(hash, S, 39);
	RNDr(hash, S, 40);
	RNDr(hash, S, 41);
	RNDr(hash, S, 42);
	RNDr(hash, S, 43);
	RNDr(hash, S, 44);
	RNDr(hash, S, 45);
	RNDr(hash, S, 46);
	RNDr(hash, S, 47);
	RNDr(hash, S, 48);
	RNDr(hash, S, 49);
	RNDr(hash, S, 50);
	RNDr(hash, S, 51);
	RNDr(hash, S, 52);
	RNDr(hash, S, 53);
	RNDr(hash, S, 54);
	RNDr(hash, S, 55);
	RNDr(hash, S, 56);

	hash[2] += hash[6] + S1(hash[3]) + Ch(hash[3], hash[4], hash[5])
			 + S[57] + sha256_k[57];
	hash[1] += hash[5] + S1(hash[2]) + Ch(hash[2], hash[3], hash[4])
			 + S[58] + sha256_k[58];
	hash[0] += hash[4] + S1(hash[1]) + Ch(hash[1], hash[2], hash[3])
			 + S[59] + sha256_k[59];
	hash[7] += hash[3] + S1(hash[0]) + Ch(hash[0], hash[1], hash[2])
			 + S[60] + sha256_k[60]
			 + sha256_h[7];
}

//
// Code taken from original scrypt.cpp and vectorized with minimal changes.
//

static const uint32x4_t keypadx4[12] = {
	0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00000280
};
static const uint32x4_t innerpadx4[11] = {
	0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x000004a0
};
static const uint32x4_t outerpadx4[8] = {
	0x80000000, 0, 0, 0, 0, 0, 0, 0x00000300
};
static const uint32x4_t finalblkx4[16] = {
	0x00000001, 0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00000620
};

static inline void HMAC_SHA256_80_initx4(const uint32x4_t *key,
	uint32x4_t *tstate, uint32x4_t *ostate)
{
	uint32x4_t ihash[8];
	uint32x4_t pad[16];
	int i;

	/* tstate is assumed to contain the midstate of key */
	memcpy(pad, key + 16, 4*16);
	memcpy(pad + 4, keypadx4, 4*48);
	sha256_transformx4(tstate, pad, 0);
	memcpy(ihash, tstate, 4*32);

	sha256_initx4(ostate);
	for (i = 0; i < 8; i++)
		pad[i] = ihash[i] ^ 0x5c5c5c5c;
	for (; i < 16; i++)
		pad[i] = 0x5c5c5c5c;
	sha256_transformx4(ostate, pad, 0);

	sha256_initx4(tstate);
	for (i = 0; i < 8; i++)
		pad[i] = ihash[i] ^ 0x36363636;
	for (; i < 16; i++)
		pad[i] = 0x36363636;
	sha256_transformx4(tstate, pad, 0);
}

static inline void PBKDF2_SHA256_80_128x4(const uint32x4_t *tstate,
	const uint32x4_t *ostate, const uint32x4_t *salt, uint32x4_t *output)
{
	uint32x4_t istate[8], ostate2[8];
	uint32x4_t ibuf[16], obuf[16];
	int i, j;

	memcpy(istate, tstate, 4*32);
	sha256_transformx4(istate, salt, 0);

	memcpy(ibuf, salt + 16, 4*16);
	memcpy(ibuf + 5, innerpadx4, 4*44);
	memcpy(obuf + 8, outerpadx4, 4*32);

	for (i = 0; i < 4; i++) {
		memcpy(obuf, istate, 4*32);
		ibuf[4] = i + 1;
		sha256_transformx4(obuf, ibuf, 0);

		memcpy(ostate2, ostate, 4*32);
		sha256_transformx4(ostate2, obuf, 0);
		for (j = 0; j < 8; j++)
			output[8 * i + j] = swab32x4(ostate2[j]);
	}
}

static inline void PBKDF2_SHA256_128_32x4(uint32x4_t *tstate, uint32x4_t *ostate,
	const uint32x4_t *salt, uint32x4_t *output)
{
	uint32x4_t buf[16];
	int i;

	sha256_transformx4(tstate, salt, 1);
	sha256_transformx4(tstate, salt + 16, 1);
	sha256_transformx4(tstate, finalblkx4, 0);
	memcpy(buf, tstate, 4*32);
	memcpy(buf + 8, outerpadx4, 4*32);

	sha256_transformx4(ostate, buf, 0);
	for (i = 0; i < 8; i++)
		output[i] = swab32x4(ostate[i]);
}


//
// Original scrypt.cpp HMAC SHA256 functions
//

static const uint32_t keypad[12] = {
	0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00000280
};
static const uint32_t innerpad[11] = {
	0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x000004a0
};
static const uint32_t outerpad[8] = {
	0x80000000, 0, 0, 0, 0, 0, 0, 0x00000300
};
static const uint32_t finalblk[16] = {
	0x00000001, 0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00000620
};

static inline void HMAC_SHA256_80_init(const uint32_t *key,
	uint32_t *tstate, uint32_t *ostate)
{
	uint32_t ihash[8];
	uint32_t pad[16];
	int i;

	/* tstate is assumed to contain the midstate of key */
	memcpy(pad, key + 16, 16);
	memcpy(pad + 4, keypad, 48);
	sha256_transform(tstate, pad, 0);
	memcpy(ihash, tstate, 32);

	sha256_init(ostate);
	for (i = 0; i < 8; i++)
		pad[i] = ihash[i] ^ 0x5c5c5c5c;
	for (; i < 16; i++)
		pad[i] = 0x5c5c5c5c;
	sha256_transform(ostate, pad, 0);

	sha256_init(tstate);
	for (i = 0; i < 8; i++)
		pad[i] = ihash[i] ^ 0x36363636;
	for (; i < 16; i++)
		pad[i] = 0x36363636;
	sha256_transform(tstate, pad, 0);
}

static inline void PBKDF2_SHA256_80_128(const uint32_t *tstate,
	const uint32_t *ostate, const uint32_t *salt, uint32_t *output)
{
	uint32_t istate[8], ostate2[8];
	uint32_t ibuf[16], obuf[16];
	int i, j;

	memcpy(istate, tstate, 32);
	sha256_transform(istate, salt, 0);

	memcpy(ibuf, salt + 16, 16);
	memcpy(ibuf + 5, innerpad, 44);
	memcpy(obuf + 8, outerpad, 32);

	for (i = 0; i < 4; i++) {
		memcpy(obuf, istate, 32);
		ibuf[4] = i + 1;
		sha256_transform(obuf, ibuf, 0);

		memcpy(ostate2, ostate, 32);
		sha256_transform(ostate2, obuf, 0);
		for (j = 0; j < 8; j++)
			output[8 * i + j] = swab32(ostate2[j]);
	}
}

static inline void PBKDF2_SHA256_128_32(uint32_t *tstate, uint32_t *ostate,
	const uint32_t *salt, uint32_t *output)
{
	uint32_t buf[16];

	sha256_transform(tstate, salt, 1);
	sha256_transform(tstate, salt + 16, 1);
	sha256_transform(tstate, finalblk, 0);
	memcpy(buf, tstate, 32);
	memcpy(buf + 8, outerpad, 32);

	sha256_transform(ostate, buf, 0);
	for (int i = 0; i < 8; i++)
		output[i] = swab32(ostate[i]);
}

static int lastFactor = 0;

static void computeGold(uint32_t* const input, uint32_t *reference, uchar *scratchpad);

static bool init[MAX_GPUS] = { 0 };

// cleanup
void free_scrypt(int thr_id)
{
	int dev_id = device_map[thr_id];

	if (!init[thr_id])
		return;

	// trivial way to free all...
	cudaSetDevice(dev_id);
	cudaDeviceSynchronize();
	cudaDeviceReset();

	init[thr_id] = false;
}

// Scrypt proof of work algorithm
// using SSE2 vectorized HMAC SHA256 on CPU and
// a salsa core implementation on GPU with CUDA
//
int scanhash_scrypt(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done,
	unsigned char *scratchbuf, struct timeval *tv_start, struct timeval *tv_end)
{
	int result = 0;
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	static __thread int throughput = 0;

	if (!init[thr_id]) {
		int dev_id = device_map[thr_id];
		cudaSetDevice(dev_id);
		cudaDeviceSynchronize();
		cudaDeviceReset();
		cudaSetDevice(dev_id);

		throughput = cuda_throughput(thr_id);
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		init[thr_id] = true;
	}

	if (throughput == 0)
		return -1;

	gettimeofday(tv_start, NULL);

	uint32_t n = pdata[19];
	const uint32_t Htarg = ptarget[7];

	// no default set with --cputest
	if (opt_nfactor == 0) opt_nfactor = 9;
	uint32_t N = (1UL<<(opt_nfactor+1));
	uint32_t *scratch = new uint32_t[N*32]; // scratchbuffer for CPU based validation

	uint32_t nonce[2];
	uint32_t* hash[2]   = { cuda_hashbuffer(thr_id,0), cuda_hashbuffer(thr_id,1) };
	uint32_t* X[2]      = { cuda_transferbuffer(thr_id,0), cuda_transferbuffer(thr_id,1) };

	bool sha_on_cpu = (parallel < 2);
	bool sha_multithreaded = (parallel == 1);
	uint32x4_t* datax4[2]   = { sha_on_cpu ? new uint32x4_t[throughput/4 * 20] : NULL, sha_on_cpu ? new uint32x4_t[throughput/4 * 20] : NULL };
	uint32x4_t* hashx4[2]   = { sha_on_cpu ? new uint32x4_t[throughput/4 * 8]  : NULL, sha_on_cpu ? new uint32x4_t[throughput/4 * 8]  : NULL };
	uint32x4_t* tstatex4[2] = { sha_on_cpu ? new uint32x4_t[throughput/4 * 8]  : NULL, sha_on_cpu ? new uint32x4_t[throughput/4 * 8]  : NULL };
	uint32x4_t* ostatex4[2] = { sha_on_cpu ? new uint32x4_t[throughput/4 * 8]  : NULL, sha_on_cpu ? new uint32x4_t[throughput/4 * 8]  : NULL };
	uint32x4_t* Xx4[2]      = { sha_on_cpu ? new uint32x4_t[throughput/4 * 32] : NULL, sha_on_cpu ? new uint32x4_t[throughput/4 * 32] : NULL };

	// log n-factor
	if (!opt_quiet && lastFactor != opt_nfactor) {
		applog(LOG_WARNING, "scrypt factor set to %d (%u)", opt_nfactor, N);
		lastFactor = opt_nfactor;
	}

	uint32_t _ALIGN(64) midstate[8];
	sha256_init(midstate);
	sha256_transform(midstate, pdata, 0);

	if (sha_on_cpu) {
		for (int i = 0; i < throughput/4; ++i) {
			for (int j = 0; j < 20; j++) {
				datax4[0][20*i+j] = uint32x4_t(pdata[j]);
				datax4[1][20*i+j] = uint32x4_t(pdata[j]);
			}
		}
	}
	else prepare_sha256(thr_id, pdata, midstate);

	int cur = 1, nxt = 0;
	int iteration = 0;
	int num_shares = (4*opt_n_threads) || 1; // opt_n_threads can be 0 with --cputest
	int share_workload = ((((throughput + num_shares-1) / num_shares) + 3) / 4) * 4;

	do {
		nonce[nxt] = n;

		if (sha_on_cpu)
		{
			for (int i = 0; i < throughput/4; i++) {
				datax4[nxt][i * 20 + 19] = uint32x4_t(n+0, n+1, n+2, n+3);
				n += 4;
			}
			if (sha_multithreaded)
			{
#ifdef WIN32
				parallel_for (0, num_shares, [&](int share) {
					for (int k = (share_workload*share)/4; k < (share_workload*(share+1))/4 && k < throughput/4; k++) {
						for (int l = 0; l < 8; l++)
							tstatex4[nxt][k * 8 + l] = uint32x4_t(midstate[l]);
							HMAC_SHA256_80_initx4(&datax4[nxt][k * 20], &tstatex4[nxt][k * 8], &ostatex4[nxt][k * 8]);
							PBKDF2_SHA256_80_128x4(&tstatex4[nxt][k * 8], &ostatex4[nxt][k * 8], &datax4[nxt][k * 20], &Xx4[nxt][k * 32]);
					}
				} );
#else
			#pragma omp parallel for
				for (int share = 0; share < num_shares; share++) {
					for (int k = (share_workload*share)/4; k < (share_workload*(share+1))/4 && k < throughput/4; k++) {
						for (int l = 0; l < 8; l++)
							tstatex4[nxt][k * 8 + l] = uint32x4_t(midstate[l]);
							HMAC_SHA256_80_initx4(&datax4[nxt][k * 20], &tstatex4[nxt][k * 8], &ostatex4[nxt][k * 8]);
							PBKDF2_SHA256_80_128x4(&tstatex4[nxt][k * 8], &ostatex4[nxt][k * 8], &datax4[nxt][k * 20], &Xx4[nxt][k * 32]);
					}
				}
#endif
			}
			else /* sha_multithreaded */
			{
				for (int k = 0; k < throughput/4; k++) {
					for (int l = 0; l < 8; l++)
						tstatex4[nxt][k * 8 + l] = uint32x4_t(midstate[l]);
						HMAC_SHA256_80_initx4(&datax4[nxt][k * 20], &tstatex4[nxt][k * 8], &ostatex4[nxt][k * 8]);
						PBKDF2_SHA256_80_128x4(&tstatex4[nxt][k * 8], &ostatex4[nxt][k * 8], &datax4[nxt][k * 20], &Xx4[nxt][k * 32]);
				}
			}

			for (int i = 0; i < throughput/4; i++) {
				for (int j = 0; j < 32; j++) {
					uint32x4_t &t = Xx4[nxt][i * 32 + j];
					X[nxt][(4*i+0)*32+j] = t[0]; X[nxt][(4*i+1)*32+j] = t[1];
					X[nxt][(4*i+2)*32+j] = t[2]; X[nxt][(4*i+3)*32+j] = t[3];
				}
			}

			cuda_scrypt_serialize(thr_id, nxt);
			cuda_scrypt_HtoD(thr_id, X[nxt], nxt);

			cuda_scrypt_core(thr_id, nxt, N);
			cuda_scrypt_done(thr_id, nxt);

			cuda_scrypt_DtoH(thr_id, X[nxt], nxt, false);
			//cuda_scrypt_flush(thr_id, nxt);
			if(!cuda_scrypt_sync(thr_id, nxt))
			{
				result = -1;
				break;
			}

			for (int i = 0; i < throughput/4; i++) {
				for (int j = 0; j < 32; j++) {
					Xx4[cur][i * 32 + j] = uint32x4_t(
						X[cur][(4*i+0)*32+j], X[cur][(4*i+1)*32+j],
						X[cur][(4*i+2)*32+j], X[cur][(4*i+3)*32+j]
					);
				}
			}

			if (sha_multithreaded)
			{
#ifdef WIN32
				parallel_for (0, num_shares, [&](int share) {
					for (int k = (share_workload*share)/4; k < (share_workload*(share+1))/4 && k < throughput/4; k++) {
						PBKDF2_SHA256_128_32x4(&tstatex4[cur][k * 8], &ostatex4[cur][k * 8], &Xx4[cur][k * 32], &hashx4[cur][k * 8]);
					}
				} );
#else
				#pragma omp parallel for
				for (int share = 0; share < num_shares; share++) {
					for (int k = (share_workload*share)/4; k < (share_workload*(share+1))/4 && k < throughput/4; k++) {
						PBKDF2_SHA256_128_32x4(&tstatex4[cur][k * 8], &ostatex4[cur][k * 8], &Xx4[cur][k * 32], &hashx4[cur][k * 8]);
					}
				}
#endif
			} else {

				for (int k = 0; k < throughput/4; k++) {
					PBKDF2_SHA256_128_32x4(&tstatex4[cur][k * 8], &ostatex4[cur][k * 8], &Xx4[cur][k * 32], &hashx4[cur][k * 8]);
				}
			}

			for (int i = 0; i < throughput/4; i++) {
				for (int j = 0; j < 8; j++) {
					uint32x4_t &t = hashx4[cur][i * 8 + j];
					hash[cur][(4*i+0)*8+j] = t[0]; hash[cur][(4*i+1)*8+j] = t[1];
					hash[cur][(4*i+2)*8+j] = t[2]; hash[cur][(4*i+3)*8+j] = t[3];
				}
			}
		}
		else /* sha_on_cpu */
		{
			n += throughput;

			cuda_scrypt_serialize(thr_id, nxt);
			pre_sha256(thr_id, nxt, nonce[nxt], throughput);

			cuda_scrypt_core(thr_id, nxt, N);
			// cuda_scrypt_flush(thr_id, nxt);
			if (!cuda_scrypt_sync(thr_id, nxt)) {
				printf("error\n");
				result = -1;
				break;
			}

			post_sha256(thr_id, nxt, throughput);
			cuda_scrypt_done(thr_id, nxt);

			cuda_scrypt_DtoH(thr_id, hash[nxt], nxt, true);
			// cuda_scrypt_flush(thr_id, nxt);
			if (!cuda_scrypt_sync(thr_id, nxt)) {
				printf("error\n");
				result = -1;
				break;
			}
		}

		if (iteration > 0 || opt_n_threads == 0)
		{
			for (int i = 0; i < throughput; i++)
			{
				if (hash[cur][i * 8 + 7] <= Htarg && fulltest(hash[cur] + i * 8, ptarget))
				{
					// CPU based validation to rule out GPU errors (scalar CPU code)
					uint32_t _ALIGN(64) inp[32], ref[32], tstate[8], ostate[8], refhash[8], ldata[20];

					memcpy(ldata, pdata, 80); ldata[19] = nonce[cur] + i;
					memcpy(tstate, midstate, 32);
					HMAC_SHA256_80_init(ldata, tstate, ostate);
					PBKDF2_SHA256_80_128(tstate, ostate, ldata, inp);
					computeGold(inp, ref, (uchar*)scratch);
					bool good = true;

					if (sha_on_cpu) {
						if (memcmp(&X[cur][i * 32], ref, 32*sizeof(uint32_t)) != 0) good = false;
					} else {
						PBKDF2_SHA256_128_32(tstate, ostate, ref, refhash);
						if (memcmp(&hash[cur][i * 8], refhash, 32) != 0) good = false;
					}

					if (!good) {
						gpulog(LOG_WARNING, thr_id, "result does not validate on CPU! (i=%d, s=%d)", i, cur);
					} else {
						*hashes_done = n - pdata[19];
						work_set_target_ratio(work, refhash);
						pdata[19] = nonce[cur] + i;
						result = 1;
						goto byebye;
					}
				}
			}
		}

		cur = (cur+1)&1;
		nxt = (nxt+1)&1;
		++iteration;

		//printf("n=%d, thr=%d, max=%d, rest=%d\n", n, throughput, max_nonce, work_restart[thr_id].restart);
	} while (n <= max_nonce && !work_restart[thr_id].restart);

	*hashes_done = n - pdata[19];
	pdata[19] = n;
byebye:
	delete[] datax4[0]; delete[] datax4[1]; delete[] hashx4[0]; delete[] hashx4[1];
	delete[] tstatex4[0]; delete[] tstatex4[1]; delete[] ostatex4[0]; delete[] ostatex4[1];
	delete[] Xx4[0]; delete[] Xx4[1];
	delete [] scratch;
	gettimeofday(tv_end, NULL);
	return result;
}

#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

static void xor_salsa8(uint32_t * const B, const uint32_t * const C)
{
	uint32_t x0 = (B[ 0] ^= C[ 0]), x1 = (B[ 1] ^= C[ 1]), x2 = (B[ 2] ^= C[ 2]), x3 = (B[ 3] ^= C[ 3]);
	uint32_t x4 = (B[ 4] ^= C[ 4]), x5 = (B[ 5] ^= C[ 5]), x6 = (B[ 6] ^= C[ 6]), x7 = (B[ 7] ^= C[ 7]);
	uint32_t x8 = (B[ 8] ^= C[ 8]), x9 = (B[ 9] ^= C[ 9]), xa = (B[10] ^= C[10]), xb = (B[11] ^= C[11]);
	uint32_t xc = (B[12] ^= C[12]), xd = (B[13] ^= C[13]), xe = (B[14] ^= C[14]), xf = (B[15] ^= C[15]);

	/* Operate on columns. */
	x4 ^= ROTL(x0 + xc,  7);  x9 ^= ROTL(x5 + x1,  7); xe ^= ROTL(xa + x6,  7);  x3 ^= ROTL(xf + xb,  7);
	x8 ^= ROTL(x4 + x0,  9);  xd ^= ROTL(x9 + x5,  9); x2 ^= ROTL(xe + xa,  9);  x7 ^= ROTL(x3 + xf,  9);
	xc ^= ROTL(x8 + x4, 13);  x1 ^= ROTL(xd + x9, 13); x6 ^= ROTL(x2 + xe, 13);  xb ^= ROTL(x7 + x3, 13);
	x0 ^= ROTL(xc + x8, 18);  x5 ^= ROTL(x1 + xd, 18); xa ^= ROTL(x6 + x2, 18);  xf ^= ROTL(xb + x7, 18);

	/* Operate on rows. */
	x1 ^= ROTL(x0 + x3,  7);  x6 ^= ROTL(x5 + x4,  7); xb ^= ROTL(xa + x9,  7);  xc ^= ROTL(xf + xe,  7);
	x2 ^= ROTL(x1 + x0,  9);  x7 ^= ROTL(x6 + x5,  9); x8 ^= ROTL(xb + xa,  9);  xd ^= ROTL(xc + xf,  9);
	x3 ^= ROTL(x2 + x1, 13);  x4 ^= ROTL(x7 + x6, 13); x9 ^= ROTL(x8 + xb, 13);  xe ^= ROTL(xd + xc, 13);
	x0 ^= ROTL(x3 + x2, 18);  x5 ^= ROTL(x4 + x7, 18); xa ^= ROTL(x9 + x8, 18);  xf ^= ROTL(xe + xd, 18);

	/* Operate on columns. */
	x4 ^= ROTL(x0 + xc,  7);  x9 ^= ROTL(x5 + x1,  7); xe ^= ROTL(xa + x6,  7);  x3 ^= ROTL(xf + xb,  7);
	x8 ^= ROTL(x4 + x0,  9);  xd ^= ROTL(x9 + x5,  9); x2 ^= ROTL(xe + xa,  9);  x7 ^= ROTL(x3 + xf,  9);
	xc ^= ROTL(x8 + x4, 13);  x1 ^= ROTL(xd + x9, 13); x6 ^= ROTL(x2 + xe, 13);  xb ^= ROTL(x7 + x3, 13);
	x0 ^= ROTL(xc + x8, 18);  x5 ^= ROTL(x1 + xd, 18); xa ^= ROTL(x6 + x2, 18);  xf ^= ROTL(xb + x7, 18);

	/* Operate on rows. */
	x1 ^= ROTL(x0 + x3,  7);  x6 ^= ROTL(x5 + x4,  7); xb ^= ROTL(xa + x9,  7);  xc ^= ROTL(xf + xe,  7);
	x2 ^= ROTL(x1 + x0,  9);  x7 ^= ROTL(x6 + x5,  9); x8 ^= ROTL(xb + xa,  9);  xd ^= ROTL(xc + xf,  9);
	x3 ^= ROTL(x2 + x1, 13);  x4 ^= ROTL(x7 + x6, 13); x9 ^= ROTL(x8 + xb, 13);  xe ^= ROTL(xd + xc, 13);
	x0 ^= ROTL(x3 + x2, 18);  x5 ^= ROTL(x4 + x7, 18); xa ^= ROTL(x9 + x8, 18);  xf ^= ROTL(xe + xd, 18);

	/* Operate on columns. */
	x4 ^= ROTL(x0 + xc,  7);  x9 ^= ROTL(x5 + x1,  7); xe ^= ROTL(xa + x6,  7);  x3 ^= ROTL(xf + xb,  7);
	x8 ^= ROTL(x4 + x0,  9);  xd ^= ROTL(x9 + x5,  9); x2 ^= ROTL(xe + xa,  9);  x7 ^= ROTL(x3 + xf,  9);
	xc ^= ROTL(x8 + x4, 13);  x1 ^= ROTL(xd + x9, 13); x6 ^= ROTL(x2 + xe, 13);  xb ^= ROTL(x7 + x3, 13);
	x0 ^= ROTL(xc + x8, 18);  x5 ^= ROTL(x1 + xd, 18); xa ^= ROTL(x6 + x2, 18);  xf ^= ROTL(xb + x7, 18);

	/* Operate on rows. */
	x1 ^= ROTL(x0 + x3,  7);  x6 ^= ROTL(x5 + x4,  7); xb ^= ROTL(xa + x9,  7);  xc ^= ROTL(xf + xe,  7);
	x2 ^= ROTL(x1 + x0,  9);  x7 ^= ROTL(x6 + x5,  9); x8 ^= ROTL(xb + xa,  9);  xd ^= ROTL(xc + xf,  9);
	x3 ^= ROTL(x2 + x1, 13);  x4 ^= ROTL(x7 + x6, 13); x9 ^= ROTL(x8 + xb, 13);  xe ^= ROTL(xd + xc, 13);
	x0 ^= ROTL(x3 + x2, 18);  x5 ^= ROTL(x4 + x7, 18); xa ^= ROTL(x9 + x8, 18);  xf ^= ROTL(xe + xd, 18);

	/* Operate on columns. */
	x4 ^= ROTL(x0 + xc,  7);  x9 ^= ROTL(x5 + x1,  7); xe ^= ROTL(xa + x6,  7);  x3 ^= ROTL(xf + xb,  7);
	x8 ^= ROTL(x4 + x0,  9);  xd ^= ROTL(x9 + x5,  9); x2 ^= ROTL(xe + xa,  9);  x7 ^= ROTL(x3 + xf,  9);
	xc ^= ROTL(x8 + x4, 13);  x1 ^= ROTL(xd + x9, 13); x6 ^= ROTL(x2 + xe, 13);  xb ^= ROTL(x7 + x3, 13);
	x0 ^= ROTL(xc + x8, 18);  x5 ^= ROTL(x1 + xd, 18); xa ^= ROTL(x6 + x2, 18);  xf ^= ROTL(xb + x7, 18);

	/* Operate on rows. */
	x1 ^= ROTL(x0 + x3,  7);  x6 ^= ROTL(x5 + x4,  7); xb ^= ROTL(xa + x9,  7);  xc ^= ROTL(xf + xe,  7);
	x2 ^= ROTL(x1 + x0,  9);  x7 ^= ROTL(x6 + x5,  9); x8 ^= ROTL(xb + xa,  9);  xd ^= ROTL(xc + xf,  9);
	x3 ^= ROTL(x2 + x1, 13);  x4 ^= ROTL(x7 + x6, 13); x9 ^= ROTL(x8 + xb, 13);  xe ^= ROTL(xd + xc, 13);
	x0 ^= ROTL(x3 + x2, 18);  x5 ^= ROTL(x4 + x7, 18); xa ^= ROTL(x9 + x8, 18);  xf ^= ROTL(xe + xd, 18);

	B[ 0] += x0; B[ 1] += x1; B[ 2] += x2; B[ 3] += x3; B[ 4] += x4; B[ 5] += x5; B[ 6] += x6; B[ 7] += x7;
	B[ 8] += x8; B[ 9] += x9; B[10] += xa; B[11] += xb; B[12] += xc; B[13] += xd; B[14] += xe; B[15] += xf;
}

/**
 * @param X input/ouput
 * @param V scratch buffer
 * @param N factor (def. 1024)
 */
static void scrypt_core(uint32_t *X, uint32_t *V, uint32_t N)
{
	for (uint32_t i = 0; i < N; i++) {
		memcpy(&V[i * 32], X, 128);
		xor_salsa8(&X[0], &X[16]);
		xor_salsa8(&X[16], &X[0]);
	}
	for (uint32_t i = 0; i < N; i++) {
		uint32_t j = 32 * (X[16] & (N - 1));
		for (uint8_t k = 0; k < 32; k++)
			X[k] ^= V[j + k];
		xor_salsa8(&X[0], &X[16]);
		xor_salsa8(&X[16], &X[0]);
	}
}

/**
 * Compute reference data set on the CPU
 * @param input      input data as provided to device
 * @param reference  reference data, computed but preallocated
 * @param scratchpad scrypt scratchpad
 **/
static void computeGold(uint32_t* const input, uint32_t *reference, uchar *scratchpad)
{
	uint32_t X[32] = { 0 };
	uint32_t *V = (uint32_t*) scratchpad;
	uint32_t N = (1<<(opt_nfactor+1)); // default 9 = 1024

	for (int k = 0; k < 32; k++)
		X[k] = input[k];

	scrypt_core(X, V, N);

	for (int k = 0; k < 32; k++)
		reference[k] = X[k];
}

/* cputest */
void scrypthash(void* output, const void* input)
{
	uint32_t _ALIGN(64) X[32], ref[32] = { 0 }, tstate[8], ostate[8], midstate[8];
	uint32_t _ALIGN(64) data[20];
	uchar *scratchbuf;

	// no default set with --cputest
	if (opt_nfactor == 0) opt_nfactor = 9;

	scratchbuf = (uchar*) calloc(4 * 128 + 63, 1UL << (opt_nfactor+1));

	memcpy(data, input, 80);

	sha256_init(midstate);
	sha256_transform(midstate, data, 0); /* ok */

	memcpy(tstate, midstate, 32);
	HMAC_SHA256_80_init(data, tstate, ostate);
	PBKDF2_SHA256_80_128(tstate, ostate, data, X); /* ok */

	if (scratchbuf) {
		computeGold(X, ref, scratchbuf);
		PBKDF2_SHA256_128_32(tstate, ostate, ref, (uint32_t*) output);
	} else {
		memset(output, 0, 32);
	}

	free(scratchbuf);
}
