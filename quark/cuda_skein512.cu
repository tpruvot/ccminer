#include <stdint.h>
#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h" 
#include "cuda_vector.h" 

static __constant__ __align__(16) uint2 c_PaddedMessage80[2];
__constant__ uint2 __align__(16)  precalcvalues[8];
__constant__ __align__(16) uint32_t sha256_endingTable[64];

static uint32_t *d_found[MAX_GPUS];

// Take a look at: https://www.schneier.com/skein1.3.pdf

#define SHL(x, n)			((x) << (n))
#define SHR(x, n)			((x) >> (n))

uint32_t *d_nonce[MAX_GPUS];

/*
 * M9_ ## s ## _ ## i  evaluates to s+i mod 9 (0 <= s <= 18, 0 <= i <= 7).
 */

#define M9_0_0    0
#define M9_0_1    1
#define M9_0_2    2
#define M9_0_3    3
#define M9_0_4    4
#define M9_0_5    5
#define M9_0_6    6
#define M9_0_7    7

#define M9_1_0    1
#define M9_1_1    2
#define M9_1_2    3
#define M9_1_3    4
#define M9_1_4    5
#define M9_1_5    6
#define M9_1_6    7
#define M9_1_7    8

#define M9_2_0    2
#define M9_2_1    3
#define M9_2_2    4
#define M9_2_3    5
#define M9_2_4    6
#define M9_2_5    7
#define M9_2_6    8
#define M9_2_7    0

#define M9_3_0    3
#define M9_3_1    4
#define M9_3_2    5
#define M9_3_3    6
#define M9_3_4    7
#define M9_3_5    8
#define M9_3_6    0
#define M9_3_7    1

#define M9_4_0    4
#define M9_4_1    5
#define M9_4_2    6
#define M9_4_3    7
#define M9_4_4    8
#define M9_4_5    0
#define M9_4_6    1
#define M9_4_7    2

#define M9_5_0    5
#define M9_5_1    6
#define M9_5_2    7
#define M9_5_3    8
#define M9_5_4    0
#define M9_5_5    1
#define M9_5_6    2
#define M9_5_7    3

#define M9_6_0    6
#define M9_6_1    7
#define M9_6_2    8
#define M9_6_3    0
#define M9_6_4    1
#define M9_6_5    2
#define M9_6_6    3
#define M9_6_7    4

#define M9_7_0    7
#define M9_7_1    8
#define M9_7_2    0
#define M9_7_3    1
#define M9_7_4    2
#define M9_7_5    3
#define M9_7_6    4
#define M9_7_7    5

#define M9_8_0    8
#define M9_8_1    0
#define M9_8_2    1
#define M9_8_3    2
#define M9_8_4    3
#define M9_8_5    4
#define M9_8_6    5
#define M9_8_7    6

#define M9_9_0    0
#define M9_9_1    1
#define M9_9_2    2
#define M9_9_3    3
#define M9_9_4    4
#define M9_9_5    5
#define M9_9_6    6
#define M9_9_7    7

#define M9_10_0   1
#define M9_10_1   2
#define M9_10_2   3
#define M9_10_3   4
#define M9_10_4   5
#define M9_10_5   6
#define M9_10_6   7
#define M9_10_7   8

#define M9_11_0   2
#define M9_11_1   3
#define M9_11_2   4
#define M9_11_3   5
#define M9_11_4   6
#define M9_11_5   7
#define M9_11_6   8
#define M9_11_7   0

#define M9_12_0   3
#define M9_12_1   4
#define M9_12_2   5
#define M9_12_3   6
#define M9_12_4   7
#define M9_12_5   8
#define M9_12_6   0
#define M9_12_7   1

#define M9_13_0   4
#define M9_13_1   5
#define M9_13_2   6
#define M9_13_3   7
#define M9_13_4   8
#define M9_13_5   0
#define M9_13_6   1
#define M9_13_7   2

#define M9_14_0   5
#define M9_14_1   6
#define M9_14_2   7
#define M9_14_3   8
#define M9_14_4   0
#define M9_14_5   1
#define M9_14_6   2
#define M9_14_7   3

#define M9_15_0   6
#define M9_15_1   7
#define M9_15_2   8
#define M9_15_3   0
#define M9_15_4   1
#define M9_15_5   2
#define M9_15_6   3
#define M9_15_7   4

#define M9_16_0   7
#define M9_16_1   8
#define M9_16_2   0
#define M9_16_3   1
#define M9_16_4   2
#define M9_16_5   3
#define M9_16_6   4
#define M9_16_7   5

#define M9_17_0   8
#define M9_17_1   0
#define M9_17_2   1
#define M9_17_3   2
#define M9_17_4   3
#define M9_17_5   4
#define M9_17_6   5
#define M9_17_7   6

#define M9_18_0   0
#define M9_18_1   1
#define M9_18_2   2
#define M9_18_3   3
#define M9_18_4   4
#define M9_18_5   5
#define M9_18_6   6
#define M9_18_7   7

/*
 * M3_ ## s ## _ ## i  evaluates to s+i mod 3 (0 <= s <= 18, 0 <= i <= 1).
 */

#define M3_0_0    0
#define M3_0_1    1
#define M3_1_0    1
#define M3_1_1    2
#define M3_2_0    2
#define M3_2_1    0
#define M3_3_0    0
#define M3_3_1    1
#define M3_4_0    1
#define M3_4_1    2
#define M3_5_0    2
#define M3_5_1    0
#define M3_6_0    0
#define M3_6_1    1
#define M3_7_0    1
#define M3_7_1    2
#define M3_8_0    2
#define M3_8_1    0
#define M3_9_0    0
#define M3_9_1    1
#define M3_10_0   1
#define M3_10_1   2
#define M3_11_0   2
#define M3_11_1   0
#define M3_12_0   0
#define M3_12_1   1
#define M3_13_0   1
#define M3_13_1   2
#define M3_14_0   2
#define M3_14_1   0
#define M3_15_0   0
#define M3_15_1   1
#define M3_16_0   1
#define M3_16_1   2
#define M3_17_0   2
#define M3_17_1   0
#define M3_18_0   0
#define M3_18_1   1

#define XCAT(x, y)     XCAT_(x, y)
#define XCAT_(x, y)    x ## y

#define SKBI(k, s, i)   XCAT(k, XCAT(XCAT(XCAT(M9_, s), _), i))
#define SKBT(t, s, v)   XCAT(t, XCAT(XCAT(XCAT(M3_, s), _), v))

#define TFBIG_KINIT(k0, k1, k2, k3, k4, k5, k6, k7, k8, t0, t1, t2) { \
		k8 = ((k0 ^ k1) ^ (k2 ^ k3)) ^ ((k4 ^ k5) ^ (k6 ^ k7))); \
		k8.x=k8.x ^0xA9FC1A22; \
		K8.y=k8.y ^0x1BD11BDA; \
		t2 = t0 ^ t1; \
	}
//vectorize(0x1BD11BDAA9FC1A22ULL);
#define TFBIG_ADDKEY(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
		w0 = (w0 + SKBI(k, s, 0)); \
		w1 = (w1 + SKBI(k, s, 1)); \
		w2 = (w2 + SKBI(k, s, 2)); \
		w3 = (w3 + SKBI(k, s, 3)); \
		w4 = (w4 + SKBI(k, s, 4)); \
		w5 = (w5 + SKBI(k, s, 5) + SKBT(t, s, 0)); \
		w6 = (w6 + SKBI(k, s, 6) + SKBT(t, s, 1)); \
		w7 = (w7 + SKBI(k, s, 7) + vectorizelow(s)); \
	}

#define TFBIG_MIX(x0, x1, rc) { \
		x0 = x0 + x1; \
		x1 = ROL2(x1, rc) ^ x0; \
	}
#define TFBIG_MIX_24(x0, x1, rc) { \
		x0 = x0 + x1; \
		x1 = ROL24(x1) ^ x0; \
		}

#define TFBIG_MIX_8(x0, x1) { \
		x0 = x0 + x1; \
		x1 = ROL8(x1) ^ x0; \
		}

#define TFBIG_MIX_56(x0, x1) { \
		x0 = x0 + x1; \
		x1 = ROR8(x1) ^ x0; \
		}


#define TFBIG_MIX8(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX(w0, w1, rc0); \
		TFBIG_MIX(w2, w3, rc1); \
		TFBIG_MIX(w4, w5, rc2); \
		TFBIG_MIX(w6, w7, rc3); \
	}

#define TFBIG_MIX8_8(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX_8(w0, w1); \
		TFBIG_MIX(w2, w3, rc1); \
		TFBIG_MIX_56(w4, w5); \
		TFBIG_MIX(w6, w7, rc3); \
		}

#define TFBIG_MIX8_24(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX_24(w0, w1, rc0); \
		TFBIG_MIX(w2, w3, rc1); \
		TFBIG_MIX(w4, w5, rc2); \
		TFBIG_MIX(w6, w7, rc3); \
		}


#define TFBIG_MIX8_56(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX(w0, w1, rc0); \
		TFBIG_MIX(w2, w3, rc1); \
		TFBIG_MIX(w4, w5, rc2); \
		TFBIG_MIX_56(w6, w7, rc3); \
		}


#define TFBIG_4e(s)  { \
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
		TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
		TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
		TFBIG_MIX8_56(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
	}

#define TFBIG_4o(s)  { \
		TFBIG_ADDKEY(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_24(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
		TFBIG_MIX8(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
		TFBIG_MIX8(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
		TFBIG_MIX8_8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
	}

/* uint2 variant for SM3.2+ */

#define TFBIG_KINIT_UI2(k0, k1, k2, k3, k4, k5, k6, k7, k8, t0, t1, t2) { \
		k8 = (k0 ^ k1 ^ k2 ^ k3 ^ k4 ^ k5 ^ k6 ^ k7) \
			^ vectorize((0x1BD11BDAA9FC1A22)); \
		t2 = t0 ^ t1; \
		}

#define TFBIG_ADDKEY_UI2(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
		w0+= (SKBI(k, s, 0)); \
		w1+= (SKBI(k, s, 1)); \
		w2+= (SKBI(k, s, 2)); \
		w3+= (SKBI(k, s, 3)); \
		w4+= (SKBI(k, s, 4)); \
		w5+= (SKBI(k, s, 5) + SKBT(t, s, 0)); \
		w6+= (SKBI(k, s, 6) + SKBT(t, s, 1)); \
		w7+= (SKBI(k, s, 7) + vectorize(s)); \
		}

#define TFBIG_ADDKEY_PRE(w0, w1, w2, w3, w4, w5, w6, w7, k, t, s) { \
		w0+= (SKBI(k, s, 0)); \
		w1+= (SKBI(k, s, 1)); \
		w2+= (SKBI(k, s, 2)); \
		w3+= (SKBI(k, s, 3)); \
		w4+= (SKBI(k, s, 4)); \
		w5+= (SKBI(k, s, 5) + SKBT(t, s, 0)); \
		w6+= (SKBI(k, s, 6) + SKBT(t, s, 1)); \
		w7+= (SKBI(k, s, 7) + (s)); \
				}

#define TFBIG_MIX_UI2(x0, x1, rc) { \
		x0+= x1; \
		x1 = ROL2(x1, rc) ^ x0; \
		}

#define TFBIG_MIX_UI2_8(x0, x1) { \
		x0+=x1; \
		x1 = ROL8(x1) ^ x0; \
				}

#define TFBIG_MIX_UI2_56(x0, x1) { \
		x0+=x1; \
		x1 = ROR8(x1) ^ x0; \
								}

#define TFBIG_MIX_PRE(x0, x1, rc) { \
		x0+= x1; \
		x1 = ROTL64(x1, rc) ^ x0; \
				}

#define TFBIG_MIX8_UI2(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX_UI2(w0, w1, rc0); \
		TFBIG_MIX_UI2(w2, w3, rc1); \
		TFBIG_MIX_UI2(w4, w5, rc2); \
		TFBIG_MIX_UI2(w6, w7, rc3); \
		}

#define TFBIG_MIX8_UI2_8(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX_UI2_8(w0, w1); \
		TFBIG_MIX_UI2(w2, w3, rc1); \
		TFBIG_MIX_UI2_56(w4, w5); \
		TFBIG_MIX_UI2(w6, w7, rc3); \
				}


#define TFBIG_MIX8_PRE(w0, w1, w2, w3, w4, w5, w6, w7, rc0, rc1, rc2, rc3) { \
		TFBIG_MIX_PRE(w0, w1, rc0); \
		TFBIG_MIX_PRE(w2, w3, rc1); \
		TFBIG_MIX_PRE(w4, w5, rc2); \
		TFBIG_MIX_PRE(w6, w7, rc3); \
			}

#define TFBIG_4e_UI2(s)  { \
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
		TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
		TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
		TFBIG_MIX8_UI2(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
		}

#define TFBIG_4e_PRE(s)  { \
		TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 46, 36, 19, 37); \
		TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 33, 27, 14, 42); \
		TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 17, 49, 36, 39); \
		TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3], 44,  9, 54, 56); \
				}

#define TFBIG_4o_UI2(s)  { \
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
		TFBIG_MIX8_UI2(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
		TFBIG_MIX8_UI2(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
		TFBIG_MIX8_UI2_8(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
		}

#define TFBIG_4o_PRE(s)  { \
		TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, s); \
		TFBIG_MIX8_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 39, 30, 34, 24); \
		TFBIG_MIX8_PRE(p[2], p[1], p[4], p[7], p[6], p[5], p[0], p[3], 13, 50, 10, 17); \
		TFBIG_MIX8_PRE(p[4], p[1], p[6], p[3], p[0], p[5], p[2], p[7], 25, 29, 39, 43); \
		TFBIG_MIX8_PRE(p[6], p[1], p[0], p[7], p[2], p[5], p[4], p[3],  8, 35, 56, 22); \
		}
__global__
#if __CUDA_ARCH__ > 500
__launch_bounds__(480, 3)
#else
__launch_bounds__(240, 6)
#endif
void quark_skein512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t * const __restrict__ g_hash, const uint32_t *const __restrict__ g_nonceVector)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// Skein
		uint2 skein_p[8], h[9];

		const uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		const int hashPosition = nounce - startNounce;
		uint64_t *Hash = &g_hash[8 * hashPosition];


		uint2 msg[8];

		uint28 *phash = (uint28*)Hash;
		uint28 *outpt = (uint28*)msg;
		outpt[0] = phash[0];
		outpt[1] = phash[1];

		h[0] = skein_p[0] = (msg[0]);
		h[1] = skein_p[1] = (msg[1]);
		h[2] = skein_p[2] = (msg[2]);
		h[3] = skein_p[3] = (msg[3]);
		h[4] = skein_p[4] = (msg[4]);
		h[5] = skein_p[5] = (msg[5]);
		h[6] = skein_p[6] = (msg[6]);
		h[7] = skein_p[7] = (msg[7]);

		skein_p[0] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[1] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[2] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[3] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[4] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[5] += vectorize(0xEABE394CA9D5C434ULL);
		skein_p[6] += vectorize(0x891112C71A75B523ULL);
		skein_p[7] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROR8(skein_p[3]) ^ skein_p[4];
		skein_p[0] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[1] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[2] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[3] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[4] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[5] += vectorize(0x891112C71A75B523ULL);
		skein_p[6] += vectorize(0x9E18A40B660FCC73ULL);
		skein_p[7] += vectorize(0xcab2076d98173ec4ULL+1);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL24(skein_p[7]) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL8(skein_p[1]) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROR8(skein_p[5]) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[1] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[2] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[3] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[4] += vectorize(0x991112C71A75B523ULL);
		skein_p[5] += vectorize(0x9E18A40B660FCC73ULL);
		skein_p[6] += vectorize(0xCAB2076D98173F04ULL);
		skein_p[7] += vectorize(0x4903ADFF749C51D0ULL);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROR8(skein_p[3]) ^ skein_p[4];
		skein_p[0] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[1] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[2] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[3] += vectorize(0x991112C71A75B523ULL);
		skein_p[4] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[5] += vectorize(0xcab2076d98173f04ULL);
		skein_p[6] += vectorize(0x3903ADFF749C51CEULL);
		skein_p[7] += vectorize(0x0D95DE399746DF03ULL+3);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL24(skein_p[7]) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL8(skein_p[1]) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROR8(skein_p[5]) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[1] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[2] += vectorize(0x991112C71A75B523ULL);
		skein_p[3] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[4] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[5] += vectorize(0x3903ADFF749C51CEULL);
		skein_p[6] += vectorize(0xFD95DE399746DF43ULL);
		skein_p[7] += vectorize(0x8FD1934127C79BD2ULL);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROR8(skein_p[3]) ^ skein_p[4];
		skein_p[0] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[1] += vectorize(0x991112C71A75B523ULL);
		skein_p[2] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[3] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[4] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[5] += vectorize(0x0D95DE399746DF03ULL + 0xf000000000000040ULL);
		skein_p[6] += vectorize(0x8FD1934127C79BCEULL + 0x0000000000000040ULL);
		skein_p[7] += vectorize(0x9A255629FF352CB1ULL + 5);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL24(skein_p[7]) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL8(skein_p[1]) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROR8(skein_p[5]) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0x991112C71A75B523ULL);
		skein_p[1] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[2] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[3] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[4] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[5] += vectorize(0x8FD1934127C79BCEULL + 0x0000000000000040ULL);
		skein_p[6] += vectorize(0x8A255629FF352CB1ULL);
		skein_p[7] += vectorize(0x5DB62599DF6CA7B0ULL + 6);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROR8(skein_p[3]) ^ skein_p[4];
		skein_p[0] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[1] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[2] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[3] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[4] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[5] += vectorize(0x8A255629FF352CB1ULL);
		skein_p[6] += vectorize(0x4DB62599DF6CA7F0ULL);
		skein_p[7] += vectorize(0xEABE394CA9D5C3F4ULL + 7);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL24(skein_p[7]) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL8(skein_p[1]) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROR8(skein_p[5]) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[1] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[2] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[3] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[4] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[5] += vectorize(0x4DB62599DF6CA7F0ULL);
		skein_p[6] += vectorize(0xEABE394CA9D5C434ULL);
		skein_p[7] += vectorize(0x991112C71A75B52BULL);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROR8(skein_p[3]) ^ skein_p[4];
		skein_p[0] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[1] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[2] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[3] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[4] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[5] += vectorize(0xEABE394CA9D5C434ULL);
		skein_p[6] += vectorize(0x891112C71A75B523ULL);
		skein_p[7] += vectorize(0xAE18A40B660FCC33ULL + 9);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL24(skein_p[7]) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL8(skein_p[1]) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROR8(skein_p[5]) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[1] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[2] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[3] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[4] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[5] += vectorize(0x891112C71A75B523ULL);
		skein_p[6] += vectorize(0x9E18A40B660FCC73ULL);
		skein_p[7] += vectorize(0xcab2076d98173eceULL);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROR8(skein_p[3]) ^ skein_p[4];
		skein_p[0] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[1] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[2] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[3] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[4] += vectorize(0x991112C71A75B523ULL);
		skein_p[5] += vectorize(0x9E18A40B660FCC73ULL);
		skein_p[6] += vectorize(0xcab2076d98173ec4ULL + 0x0000000000000040ULL);
		skein_p[7] += vectorize(0x4903ADFF749C51CEULL + 11);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL24(skein_p[7]) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL8(skein_p[1]) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROR8(skein_p[5]) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[1] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[2] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[3] += vectorize(0x991112C71A75B523ULL);
		skein_p[4] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[5] += vectorize(0xcab2076d98173ec4ULL + 0x0000000000000040ULL);
		skein_p[6] += vectorize(0x3903ADFF749C51CEULL);
		skein_p[7] += vectorize(0x0D95DE399746DF03ULL + 12);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROR8(skein_p[3]) ^ skein_p[4];
		skein_p[0] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[1] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[2] += vectorize(0x991112C71A75B523ULL);
		skein_p[3] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[4] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[5] += vectorize(0x3903ADFF749C51CEULL);
		skein_p[6] += vectorize(0x0D95DE399746DF03ULL + 0xf000000000000040ULL);
		skein_p[7] += vectorize(0x8FD1934127C79BCEULL + 13);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL24(skein_p[7]) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL8(skein_p[1]) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROR8(skein_p[5]) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0xEABE394CA9D5C3F4ULL);
		skein_p[1] += vectorize(0x991112C71A75B523ULL);
		skein_p[2] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[3] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[4] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[5] += vectorize(0x0D95DE399746DF03ULL + 0xf000000000000040ULL);
		skein_p[6] += vectorize(0x8FD1934127C79BCEULL + 0x0000000000000040ULL);
		skein_p[7] += vectorize(0x9A255629FF352CB1ULL + 14);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROR8(skein_p[3]) ^ skein_p[4];
		skein_p[0] += vectorize(0x991112C71A75B523ULL);
		skein_p[1] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[2] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[3] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[4] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[5] += vectorize(0x8FD1934127C79BCEULL + 0x0000000000000040ULL);
		skein_p[6] += vectorize(0x8A255629FF352CB1ULL);
		skein_p[7] += vectorize(0x5DB62599DF6CA7B0ULL + 15);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL24(skein_p[7]) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL8(skein_p[1]) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROR8(skein_p[5]) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0xAE18A40B660FCC33ULL);
		skein_p[1] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[2] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[3] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[4] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[5] += vectorize(0x8A255629FF352CB1ULL);
		skein_p[6] += vectorize(0x4DB62599DF6CA7F0ULL);
		skein_p[7] += vectorize(0xEABE394CA9D5C3F4ULL +16ULL);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 46) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 36) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 19) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 37) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 33) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 27) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 14) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 42) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 17) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 49) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 36) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 39) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 44) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 9) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 54) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROR8(skein_p[3]) ^ skein_p[4];
		skein_p[0] += vectorize(0xcab2076d98173ec4ULL);
		skein_p[1] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[2] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[3] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[4] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[5] += vectorize(0x4DB62599DF6CA7F0ULL);
		skein_p[6] += vectorize(0xEABE394CA9D5C3F4ULL + 0x0000000000000040ULL);
		skein_p[7] += vectorize(0x991112C71A75B523ULL + 17);
		skein_p[0] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 39) ^ skein_p[0];
		skein_p[2] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 30) ^ skein_p[2];
		skein_p[4] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 34) ^ skein_p[4];
		skein_p[6] += skein_p[7];
		skein_p[7] = ROL24(skein_p[7]) ^ skein_p[6];
		skein_p[2] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 13) ^ skein_p[2];
		skein_p[4] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 50) ^ skein_p[4];
		skein_p[6] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 10) ^ skein_p[6];
		skein_p[0] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 17) ^ skein_p[0];
		skein_p[4] += skein_p[1];
		skein_p[1] = ROL2(skein_p[1], 25) ^ skein_p[4];
		skein_p[6] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 29) ^ skein_p[6];
		skein_p[0] += skein_p[5];
		skein_p[5] = ROL2(skein_p[5], 39) ^ skein_p[0];
		skein_p[2] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 43) ^ skein_p[2];
		skein_p[6] += skein_p[1];
		skein_p[1] = ROL8(skein_p[1]) ^ skein_p[6];
		skein_p[0] += skein_p[7];
		skein_p[7] = ROL2(skein_p[7], 35) ^ skein_p[0];
		skein_p[2] += skein_p[5];
		skein_p[5] = ROR8(skein_p[5]) ^ skein_p[2];
		skein_p[4] += skein_p[3];
		skein_p[3] = ROL2(skein_p[3], 22) ^ skein_p[4];
		skein_p[0] += vectorize(0x4903ADFF749C51CEULL);
		skein_p[1] += vectorize(0x0D95DE399746DF03ULL);
		skein_p[2] += vectorize(0x8FD1934127C79BCEULL);
		skein_p[3] += vectorize(0x9A255629FF352CB1ULL);
		skein_p[4] += vectorize(0x5DB62599DF6CA7B0ULL);
		skein_p[5] += vectorize(0xEABE394CA9D5C3F4ULL + 0x0000000000000040ULL);
		skein_p[6] += vectorize(0x891112C71A75B523ULL);
		skein_p[7] += vectorize(0xAE18A40B660FCC33ULL + 18);

#define h0 skein_p[0]
#define h1 skein_p[1]
#define h2 skein_p[2]
#define h3 skein_p[3]
#define h4 skein_p[4]
#define h5 skein_p[5]
#define h6 skein_p[6]
#define h7 skein_p[7]
		h0 ^= h[0];
		h1 ^= h[1];
		h2 ^= h[2];
		h3 ^= h[3];
		h4 ^= h[4];
		h5 ^= h[5];
		h6 ^= h[6];
		h7 ^= h[7];

		uint2 skein_h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ vectorize(0x1BD11BDAA9FC1A22ULL);

		uint2 hash64[8];

		hash64[0] = (h0);
//		hash64[1] = (h1);
		hash64[2] = (h2);
//		hash64[3] = (h3);
		hash64[4] = (h4);
		hash64[5] = (h5 + vectorizelow(8ULL));
		hash64[6] = (h6 + vectorizehigh(0xff000000UL));
//		hash64[7] = (h7);

		hash64[0] += h1;
		hash64[1] = ROL2(h1, 46) ^ hash64[0];
		hash64[2] += h3;
		hash64[3] = ROL2(h3, 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += h7;
		hash64[7] = ROL2(h7, 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];
		hash64[0] = (hash64[0] + h1);
		hash64[1] = (hash64[1] + h2);
		hash64[2] = (hash64[2] + h3);
		hash64[3] = (hash64[3] + h4);
		hash64[4] = (hash64[4] + h5);
		hash64[5] = (hash64[5] + h6 + vectorizehigh(0xff000000UL));
		hash64[6] = (hash64[6] + h7 + vectorize(0xff00000000000008ULL));
		hash64[7] = (hash64[7] + skein_h8 + vectorizelow(1));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL24(hash64[7]) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL8(hash64[1]) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROR8(hash64[5]) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + h2);
		hash64[1] = (hash64[1] + h3);
		hash64[2] = (hash64[2] + h4);
		hash64[3] = (hash64[3] + h5);
		hash64[4] = (hash64[4] + h6);
		hash64[5] = (hash64[5] + h7 + vectorize(0xff00000000000008ULL));
		hash64[6] = (hash64[6] + skein_h8 + vectorizelow(8ULL));
		hash64[7] = (hash64[7] + h0 + vectorize(2));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];
		hash64[0] = (hash64[0] + h3);
		hash64[1] = (hash64[1] + h4);
		hash64[2] = (hash64[2] + h5);
		hash64[3] = (hash64[3] + h6);
		hash64[4] = (hash64[4] + h7);
		hash64[5] = (hash64[5] + skein_h8 + vectorizelow(8));
		hash64[6] = (hash64[6] + h0 + vectorizehigh(0xff000000UL));
		hash64[7] = (hash64[7] + h1 + vectorizelow(3));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL24(hash64[7]) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL8(hash64[1]) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROR8(hash64[5]) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + h4);
		hash64[1] = (hash64[1] + h5);
		hash64[2] = (hash64[2] + h6);
		hash64[3] = (hash64[3] + h7);
		hash64[4] = (hash64[4] + skein_h8);
		hash64[5] = (hash64[5] + h0 + vectorizehigh(0xff000000UL));
		hash64[6] = (hash64[6] + h1 + vectorize(0xff00000000000008ULL));
		hash64[7] = (hash64[7] + h2 + vectorizelow(4));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];
		hash64[0] = (hash64[0] + h5);
		hash64[1] = (hash64[1] + h6);
		hash64[2] = (hash64[2] + h7);
		hash64[3] = (hash64[3] + skein_h8);
		hash64[4] = (hash64[4] + h0);
		hash64[5] = (hash64[5] + h1 + vectorize(0xff00000000000008ULL));
		hash64[6] = (hash64[6] + h2 + vectorizelow(8ULL));
		hash64[7] = (hash64[7] + h3 + vectorizelow(5));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL24(hash64[7]) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL8(hash64[1]) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROR8(hash64[5]) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + h6);
		hash64[1] = (hash64[1] + h7);
		hash64[2] = (hash64[2] + skein_h8);
		hash64[3] = (hash64[3] + h0);
		hash64[4] = (hash64[4] + h1);
		hash64[5] = (hash64[5] + h2 + vectorizelow(8ULL));
		hash64[6] = (hash64[6] + h3 + vectorizehigh(0xff000000UL));
		hash64[7] = (hash64[7] + h4 + vectorizelow(6));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];
		hash64[0] = (hash64[0] + h7);
		hash64[1] = (hash64[1] + skein_h8);
		hash64[2] = (hash64[2] + h0);
		hash64[3] = (hash64[3] + h1);
		hash64[4] = (hash64[4] + h2);
		hash64[5] = (hash64[5] + h3 + vectorizehigh(0xff000000UL));
		hash64[6] = (hash64[6] + h4 + vectorize(0xff00000000000008ULL));
		hash64[7] = (hash64[7] + h5 + vectorizelow(7));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL24(hash64[7]) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL8(hash64[1]) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROR8(hash64[5]) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + skein_h8);
		hash64[1] = (hash64[1] + h0);
		hash64[2] = (hash64[2] + h1);
		hash64[3] = (hash64[3] + h2);
		hash64[4] = (hash64[4] + h3);
		hash64[5] = (hash64[5] + h4 + vectorize(0xff00000000000008ULL));
		hash64[6] = (hash64[6] + h5 + vectorizelow(8));
		hash64[7] = (hash64[7] + h6 + vectorizelow(8));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];
		hash64[0] = (hash64[0] + h0);
		hash64[1] = (hash64[1] + h1);
		hash64[2] = (hash64[2] + h2);
		hash64[3] = (hash64[3] + h3);
		hash64[4] = (hash64[4] + h4);
		hash64[5] = (hash64[5] + h5 + vectorizelow(8));
		hash64[6] = (hash64[6] + h6 + vectorizehigh(0xff000000UL));
		hash64[7] = (hash64[7] + h7 + vectorizelow(9));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL24(hash64[7]) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL8(hash64[1]) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROR8(hash64[5]) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];

		hash64[0] = (hash64[0] + h1);
		hash64[1] = (hash64[1] + h2);
		hash64[2] = (hash64[2] + h3);
		hash64[3] = (hash64[3] + h4);
		hash64[4] = (hash64[4] + h5);
		hash64[5] = (hash64[5] + h6 + vectorizehigh(0xff000000UL));
		hash64[6] = (hash64[6] + h7 + vectorize(0xff00000000000008ULL));
		hash64[7] = (hash64[7] + skein_h8 + (vectorizelow(10)));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];
		hash64[0] = (hash64[0] + h2);
		hash64[1] = (hash64[1] + h3);
		hash64[2] = (hash64[2] + h4);
		hash64[3] = (hash64[3] + h5);
		hash64[4] = (hash64[4] + h6);
		hash64[5] = (hash64[5] + h7 + vectorize(0xff00000000000008ULL));
		hash64[6] = (hash64[6] + skein_h8 + vectorizelow(8ULL));
		hash64[7] = (hash64[7] + h0 + vectorizelow(11));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL24(hash64[7]) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL8(hash64[1]) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROR8(hash64[5]) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + h3);
		hash64[1] = (hash64[1] + h4);
		hash64[2] = (hash64[2] + h5);
		hash64[3] = (hash64[3] + h6);
		hash64[4] = (hash64[4] + h7);
		hash64[5] = (hash64[5] + skein_h8 + vectorizelow(8));
		hash64[6] = (hash64[6] + h0 + vectorizehigh(0xff000000UL));
		hash64[7] = (hash64[7] + h1 + vectorizelow(12));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];
		hash64[0] = (hash64[0] + h4);
		hash64[1] = (hash64[1] + h5);
		hash64[2] = (hash64[2] + h6);
		hash64[3] = (hash64[3] + h7);
		hash64[4] = (hash64[4] + skein_h8);
		hash64[5] = (hash64[5] + h0 + vectorizehigh(0xff000000UL));
		hash64[6] = (hash64[6] + h1 + vectorize(0xff00000000000008ULL));
		hash64[7] = (hash64[7] + h2 + vectorizelow(13));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL24(hash64[7]) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL8(hash64[1]) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROR8(hash64[5]) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + h5);
		hash64[1] = (hash64[1] + h6);
		hash64[2] = (hash64[2] + h7);
		hash64[3] = (hash64[3] + skein_h8);
		hash64[4] = (hash64[4] + h0);
		hash64[5] = (hash64[5] + h1 + vectorize(0xff00000000000008ULL));
		hash64[6] = (hash64[6] + h2 + vectorizelow(8ULL));
		hash64[7] = (hash64[7] + h3 + vectorizelow(14));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];
		hash64[0] = (hash64[0] + h6);
		hash64[1] = (hash64[1] + h7);
		hash64[2] = (hash64[2] + skein_h8);
		hash64[3] = (hash64[3] + h0);
		hash64[4] = (hash64[4] + h1);
		hash64[5] = (hash64[5] + h2 + vectorizelow(8ULL));
		hash64[6] = (hash64[6] + h3 + vectorizehigh(0xff000000UL));
		hash64[7] = (hash64[7] + h4 + vectorizelow(15));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL24(hash64[7]) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL8(hash64[1]) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROR8(hash64[5]) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];
		hash64[0] = (hash64[0] + h7);
		hash64[1] = (hash64[1] + skein_h8);
		hash64[2] = (hash64[2] + h0);
		hash64[3] = (hash64[3] + h1);
		hash64[4] = (hash64[4] + h2);
		hash64[5] = (hash64[5] + h3 + vectorizehigh(0xff000000UL));
		hash64[6] = (hash64[6] + h4 + vectorize(0xff00000000000008ULL));
		hash64[7] = (hash64[7] + h5 + vectorizelow(16));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 46) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 36) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 19) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL2(hash64[7], 37) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 33) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 27) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 14) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 42) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 17) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 49) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 36) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 39) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL2(hash64[1], 44) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 9) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROL2(hash64[5], 54) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROR8(hash64[3]) ^ hash64[4];
		hash64[0] = (hash64[0] + skein_h8);
		hash64[1] = (hash64[1] + h0);
		hash64[2] = (hash64[2] + h1);
		hash64[3] = (hash64[3] + h2);
		hash64[4] = (hash64[4] + h3);
		hash64[5] = (hash64[5] + h4 + vectorize(0xff00000000000008ULL));
		hash64[6] = (hash64[6] + h5 + vectorizelow(8ULL));
		hash64[7] = (hash64[7] + h6 + vectorizelow(17));
		hash64[0] += hash64[1];
		hash64[1] = ROL2(hash64[1], 39) ^ hash64[0];
		hash64[2] += hash64[3];
		hash64[3] = ROL2(hash64[3], 30) ^ hash64[2];
		hash64[4] += hash64[5];
		hash64[5] = ROL2(hash64[5], 34) ^ hash64[4];
		hash64[6] += hash64[7];
		hash64[7] = ROL24(hash64[7]) ^ hash64[6];
		hash64[2] += hash64[1];
		hash64[1] = ROL2(hash64[1], 13) ^ hash64[2];
		hash64[4] += hash64[7];
		hash64[7] = ROL2(hash64[7], 50) ^ hash64[4];
		hash64[6] += hash64[5];
		hash64[5] = ROL2(hash64[5], 10) ^ hash64[6];
		hash64[0] += hash64[3];
		hash64[3] = ROL2(hash64[3], 17) ^ hash64[0];
		hash64[4] += hash64[1];
		hash64[1] = ROL2(hash64[1], 25) ^ hash64[4];
		hash64[6] += hash64[3];
		hash64[3] = ROL2(hash64[3], 29) ^ hash64[6];
		hash64[0] += hash64[5];
		hash64[5] = ROL2(hash64[5], 39) ^ hash64[0];
		hash64[2] += hash64[7];
		hash64[7] = ROL2(hash64[7], 43) ^ hash64[2];
		hash64[6] += hash64[1];
		hash64[1] = ROL8(hash64[1]) ^ hash64[6];
		hash64[0] += hash64[7];
		hash64[7] = ROL2(hash64[7], 35) ^ hash64[0];
		hash64[2] += hash64[5];
		hash64[5] = ROR8(hash64[5]) ^ hash64[2];
		hash64[4] += hash64[3];
		hash64[3] = ROL2(hash64[3], 22) ^ hash64[4];

		Hash[0] = devectorize(hash64[0] + h0);
		Hash[1] = devectorize(hash64[1] + h1);
		Hash[2] = devectorize(hash64[2] + h2);
		Hash[3] = devectorize(hash64[3] + h3);
		Hash[4] = devectorize(hash64[4] + h4);
		Hash[5] = devectorize(hash64[5] + h5)+ 8;
		Hash[6] = devectorize(hash64[6] + h6)+ 0xff00000000000000ULL;
		Hash[7] = devectorize(hash64[7] + h7)+ 18;

#undef h0
#undef h1
#undef h2
#undef h3
#undef h4
#undef h5
#undef h6
#undef h7
	}
}


__host__ void quark_skein512_cpu_init(int thr_id)
{
	cudaMalloc(&d_nonce[thr_id], 2*sizeof(uint32_t));
}

__host__ void quark_skein512_setTarget(const void *ptarget)
{
}
__host__ void quark_skein512_cpu_free(int32_t thr_id)
{
	cudaFreeHost(&d_nonce[thr_id]);
}



/* Elementary functions used by SHA256 */
#define SWAB32(x)     cuda_swab32(x)
#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

#define R(x, n)       ((x) >> (n))
//#define Ch(x, y, z)   ((x & (y ^ z)) ^ z)

//#define F2(x, y, z)   ((x & (y ^ z)) ^ z)
__device__ __forceinline__
uint32_t Ch(const uint32_t a, const uint32_t b, const uint32_t c)
{
	uint32_t result;
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
	asm volatile ("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(result) : "r"(a), "r"(b), "r"(c)); //0xCA=((F0∧(CC⊻AA))⊻AA)
#else
	result = ((a & (b ^ c)) ^ c);
#endif
	return result;
}

//#define Maj(x, y, z)  ((x & (y | z)) | (y & z))

__device__ __forceinline__
uint32_t Maj(const uint32_t a, const uint32_t b, const uint32_t c)
{ //Sha256 - Maj - andor
	uint32_t result;
#if __CUDA_ARCH__ >= 500 && CUDA_VERSION >= 7050
	asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(result) : "r"(a), "r"(b), "r"(c)); // 0xE8 = ((0xF0 & (0xCC | 0xAA)) | (0xCC & 0xAA))
#else
	result = ((a & (b | c)) | (b & c));
#endif
	return result;
}


#define S0(x)         (ROTR32(x, 2) ^ ROTR32(x, 13) ^ ROTR32(x, 22))
#define S1(x)         (ROTR32(x, 6) ^ ROTR32(x, 11) ^ ROTR32(x, 25))
#define s0(x)         (ROTR32(x, 7) ^ ROTR32(x, 18) ^ R(x, 3))
#define s1(x)         (ROTR32(x, 17) ^ ROTR32(x, 19) ^ R(x, 10))



__global__ //__launch_bounds__(1024,1)
void skein512_gpu_hash_80_52(uint32_t threads, uint32_t startNounce, uint32_t *const __restrict__ d_found, uint2 target)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint2 t0, t1, t2;
		
		const uint32_t sha256_hashTable[] = {
			0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
		};


		const uint32_t sha256_constantTable[64] = {
			0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
			0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
			0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
			0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
			0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
			0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
			0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
			0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
		};


		uint32_t nonce = (startNounce + thread);


		h0 = precalcvalues[0];
		h1 = precalcvalues[1];
		h2 = precalcvalues[2];
		h3 = precalcvalues[3];
		h4 = precalcvalues[4];
		h5 = precalcvalues[5];
		h6 = precalcvalues[6];
		h7 = precalcvalues[7];
		t2.x = 0x70000000;
		t2.y = 0x00000040;	//precalcvalues[8];

	//	uint2 nounce2 = make_uint2(_LOWORD(c_PaddedMessage80[1]), cuda_swab32(startNounce + thread));

		// skein_big_close -> etype = 0x160, ptr = 16, bcount = 1, extra = 16
//		p[0] = vectorize(c_PaddedMessage80[0]);
//		p[1] = nounce2;

//		#pragma unroll
//		for (int i = 2; i < 8; i++)
//			p[i] = make_uint2(0,0);

		uint2 p[8] = { 0 };
		p[0] = c_PaddedMessage80[0];
		p[1].x = c_PaddedMessage80[1].x;
		p[1].y = cuda_swab32(startNounce + thread);
		uint2 nounce2 = p[1];
		//		c_PaddedMessage80[0], nounce2, 0, 0, 0, 0, 0, 0



		t0 = vectorizelow(0x50ull); // SPH_T64(bcount << 6) + (sph_u64)(extra);
		t1 = vectorizehigh(0xB0000000ul); // (bcount >> 58) + ((sph_u64)(etype) << 55);
		TFBIG_KINIT_UI2(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);


		t0 = vectorizelow(8); // extra
		t1 = vectorizehigh(0xFF000000ul); // etype

		h0 = (c_PaddedMessage80[0]) ^ p[0];
		h1 = nounce2 ^ p[1];
		h2 = p[2];
		h3 = p[3];
		h4 = p[4];
		h5 = p[5];
		h6 = p[6];
		h7 = p[7];

		h8 = h0 ^ h1 ^ p[2] ^ p[3] ^ p[4] ^ p[5] ^ p[6] ^ p[7] ^ vectorize(0x1BD11BDAA9FC1A22);
		t2 = vectorize(0xFF00000000000008ull);

		// p[8] = { 0 };
		#pragma unroll 8
		for (int i = 0; i < 8; i++)
		{
			p[i].x = 0;
			p[i].y = 0;
		}
		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		uint32_t *message = (uint32_t *)p;	

		uint32_t W1[16];
		uint32_t W2[16];

		uint32_t regs[8];
		uint32_t hash[8];

		// Init with Hash-Table
#pragma unroll 8
		for (int k = 0; k < 8; k++)
		{
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

		/////
		///// Second Pass (ending)
		/////
#pragma unroll 8
		for (int k = 0; k<8; k++)
			regs[k] = hash[k];

		// Progress W1
		uint32_t T1, T2;
#pragma unroll 
		for (int j = 0; j<56; j++)
		{
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_endingTable[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

#pragma unroll 7
			for (int k = 6; k >= 0; k--)
				regs[k + 1] = regs[k];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}
		T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_endingTable[56];
		T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
		regs[7] = T1 + T2;
		regs[3] += T1;

		T1 = regs[6] + S1(regs[3]) + Ch(regs[3], regs[4], regs[5]) + sha256_endingTable[57];
		T2 = S0(regs[7]) + Maj(regs[7], regs[0], regs[1]);
		regs[6] = T1 + T2;
		regs[2] += T1;
		//************
		regs[1] += regs[5] + S1(regs[2]) + Ch(regs[2], regs[3], regs[4]) + sha256_endingTable[58];
		regs[0] += regs[4] + S1(regs[1]) + Ch(regs[1], regs[2], regs[3]) + sha256_endingTable[59];
		regs[7] += regs[3] + S1(regs[0]) + Ch(regs[0], regs[1], regs[2]) + sha256_endingTable[60];
		regs[6] += regs[2] + S1(regs[7]) + Ch(regs[7], regs[0], regs[1]) + sha256_endingTable[61];

		uint2 test;
		test.y= SWAB32(hash[7] + regs[7]);
		test.x= SWAB32(hash[6] + regs[6]);

		uint64_t test_64 = devectorize(test);
		uint64_t target_64 = devectorize(target);
		if (test_64 <= target_64)
		{
			uint32_t tmp = atomicExch(&(d_found[0]), nonce);
			if (tmp != 0xffffffff)
				d_found[1] = nonce;
		}
	}
}
__global__
void skein512_gpu_hash_80_50(uint32_t threads, uint32_t startNounce, uint32_t *const __restrict__ d_found, uint2 target)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint2 h0, h1, h2, h3, h4, h5, h6, h7, h8;
		uint2 t0, t1, t2;

		uint32_t nounce = (startNounce + thread);
//		uint32_t target32 = target.x;

		const uint32_t sha256_hashTable[] = {
			0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
		};

		const uint32_t sha256_constantTable[64] = {
			0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
			0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
			0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
			0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
			0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
			0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
			0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
			0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
		};


		h0 = precalcvalues[0];
		h1 = precalcvalues[1];
		h2 = precalcvalues[2];
		h3 = precalcvalues[3];
		h4 = precalcvalues[4];
		h5 = precalcvalues[5];
		h6 = precalcvalues[6];
		h7 = precalcvalues[7];
		//		t2 = precalcvalues[8];
		t2.x = 0x70000000;
		t2.y = 0x00000040;	//precalcvalues[8];

		uint2 p[8] = { 0 };
		p[0] = c_PaddedMessage80[0];
		p[1].x = c_PaddedMessage80[1].x;
		p[1].y = cuda_swab32(startNounce + thread);
		uint2 nounce2 = p[1];
#pragma unroll
		for (int i = 2; i < 8; i++)
		{
			p[i].x = 0;
			p[i].y = 0;
		}
		t0 = vectorizelow(0x50ull); // SPH_T64(bcount << 6) + (sph_u64)(extra);
		t1 = vectorizehigh(0xB0000000ul); // (bcount >> 58) + ((sph_u64)(etype) << 55);
		TFBIG_KINIT_UI2(h0, h1, h2, h3, h4, h5, h6, h7, h8, t0, t1, t2);
		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);


		t0 = vectorizelow(8); // extra
		t1 = vectorizehigh(0xFF000000ul); // etype

		h0 = (c_PaddedMessage80[0]) ^ p[0];
		h1 = nounce2 ^ p[1];
		h2 = p[2];
		h3 = p[3];
		h4 = p[4];
		h5 = p[5];
		h6 = p[6];
		h7 = p[7];

		h8 = h0 ^ h1 ^ p[2] ^ p[3] ^ p[4] ^ p[5] ^ p[6] ^ p[7] ^ vectorize(0x1BD11BDAA9FC1A22);
		t2 = vectorize(0xFF00000000000008ull);

		// p[8] = { 0 };
#pragma unroll 8
		for (int i = 0; i < 8; i++)
		{
			p[i].x = 0;
			p[i].y = 0;
		}

		TFBIG_4e_UI2(0);
		TFBIG_4o_UI2(1);
		TFBIG_4e_UI2(2);
		TFBIG_4o_UI2(3);
		TFBIG_4e_UI2(4);
		TFBIG_4o_UI2(5);
		TFBIG_4e_UI2(6);
		TFBIG_4o_UI2(7);
		TFBIG_4e_UI2(8);
		TFBIG_4o_UI2(9);
		TFBIG_4e_UI2(10);
		TFBIG_4o_UI2(11);
		TFBIG_4e_UI2(12);
		TFBIG_4o_UI2(13);
		TFBIG_4e_UI2(14);
		TFBIG_4o_UI2(15);
		TFBIG_4e_UI2(16);
		TFBIG_4o_UI2(17);
		TFBIG_ADDKEY_UI2(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

		uint32_t *message = (uint32_t *)p;

		uint32_t W1[16];
		uint32_t W2[16];

		uint32_t regs[8];
		uint32_t hash[8];

		// Init with Hash-Table
#pragma unroll 8
		for (int k = 0; k < 8; k++)
		{
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

		/////
		///// Second Pass (ending)
		/////
#pragma unroll 8
		for (int k = 0; k<8; k++)
			regs[k] = hash[k];

		// Progress W1
		uint32_t T1, T2;
#pragma unroll 1
		for (int j = 0; j<56; j++)//62
		{
			T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6]) + sha256_endingTable[j];
			T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);

#pragma unroll 7
			for (int k = 6; k >= 0; k--)
				regs[k + 1] = regs[k];
			regs[0] = T1 + T2;
			regs[4] += T1;
		}
		T1 = regs[7] + S1(regs[4]) + Ch(regs[4], regs[5], regs[6])+sha256_endingTable[56];
		T2 = S0(regs[0]) + Maj(regs[0], regs[1], regs[2]);
		regs[7] = T1 + T2;
		regs[3] += T1;

		T1 = regs[6] + S1(regs[3]) + Ch(regs[3], regs[4], regs[5]) + sha256_endingTable[57];
		T2 = S0(regs[7]) + Maj(regs[7], regs[0], regs[1]);
		regs[6] = T1 + T2;
		regs[2] += T1;
		//************
		regs[1] += regs[5] + S1(regs[2]) + Ch(regs[2], regs[3], regs[4]) + sha256_endingTable[58];
		regs[0] += regs[4] + S1(regs[1]) + Ch(regs[1], regs[2], regs[3]) + sha256_endingTable[59];
		regs[7] += regs[3] + S1(regs[0]) + Ch(regs[0], regs[1], regs[2]) + sha256_endingTable[60];
		regs[6] += regs[2] + S1(regs[7]) + Ch(regs[7], regs[0], regs[1]) + sha256_endingTable[61];


		uint2 test;
		test.y = SWAB32(hash[7] + regs[7]);
		test.x = SWAB32(hash[6] + regs[6]);

		uint64_t test_64 = devectorize(test);
		uint64_t target_64 = devectorize(target);
		if (test_64 <= target_64)
		{
			uint32_t tmp = atomicCAS(d_found, 0xffffffff, nounce);
			if (tmp != 0xffffffff)
				d_found[1] = nounce;
		}
	}
}
#if __CUDA_ARCH__ > 500
#define tp 448
#else
#define tp 128
#endif

__host__
void quark_skein512_cpu_hash_64(uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash)
{
	dim3 grid((threads + tp - 1) / tp);
	dim3 block(tp);
	quark_skein512_gpu_hash_64 << <grid, block >> >(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

}

static uint64_t PaddedMessage[16];

static void precalc()
{
	uint64_t h0, h1, h2, h3, h4, h5, h6, h7, h8;
	uint64_t t0, t1, t2;

	h0 = 0x4903ADFF749C51CEull;
	h1 = 0x0D95DE399746DF03ull;
	h2 = 0x8FD1934127C79BCEull;
	h3 = 0x9A255629FF352CB1ull;
	h4 = 0x5DB62599DF6CA7B0ull;
	h5 = 0xEABE394CA9D5C3F4ull;
	h6 = 0x991112C71A75B523ull;
	h7 = 0xAE18A40B660FCC33ull;
	h8 = h0 ^ h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6 ^ h7 ^ SPH_C64(0x1BD11BDAA9FC1A22);

	t0 = 64; // ptr
	t1 = 0x7000000000000000ull;
	t2 = 0x7000000000000040ull;

	uint64_t p[8];
	for (int i = 0; i<8; i++)
		p[i] = PaddedMessage[i];

	TFBIG_4e_PRE(0);
	TFBIG_4o_PRE(1);
	TFBIG_4e_PRE(2);
	TFBIG_4o_PRE(3);
	TFBIG_4e_PRE(4);
	TFBIG_4o_PRE(5);
	TFBIG_4e_PRE(6);
	TFBIG_4o_PRE(7);
	TFBIG_4e_PRE(8);
	TFBIG_4o_PRE(9);
	TFBIG_4e_PRE(10);
	TFBIG_4o_PRE(11);
	TFBIG_4e_PRE(12);
	TFBIG_4o_PRE(13);
	TFBIG_4e_PRE(14);
	TFBIG_4o_PRE(15);
	TFBIG_4e_PRE(16);
	TFBIG_4o_PRE(17);
	TFBIG_ADDKEY_PRE(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], h, t, 18);

	uint64_t buffer[8];

	buffer[0] = PaddedMessage[0] ^ p[0];
	buffer[1] = PaddedMessage[1] ^ p[1];
	buffer[2] = PaddedMessage[2] ^ p[2];
	buffer[3] = PaddedMessage[3] ^ p[3];
	buffer[4] = PaddedMessage[4] ^ p[4];
	buffer[5] = PaddedMessage[5] ^ p[5];
	buffer[6] = PaddedMessage[6] ^ p[6];
	buffer[7] = PaddedMessage[7] ^ p[7];
//	buffer[8] = 0x7000000000000040ull; //t2;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(precalcvalues, buffer, sizeof(buffer), 0, cudaMemcpyHostToDevice));

	int endingTable[] = {
		0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
		0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000200,
		0x80000000, 0x01400000, 0x00205000, 0x00005088, 0x22000800, 0x22550014, 0x05089742, 0xa0000020,
		0x5a880000, 0x005c9400, 0x0016d49d, 0xfa801f00, 0xd33225d0, 0x11675959, 0xf6e6bfda, 0xb30c1549,
		0x08b2b050, 0x9d7c4c27, 0x0ce2a393, 0x88e6e1ea, 0xa52b4335, 0x67a16f49, 0xd732016f, 0x4eeb2e91,
		0x5dbf55e5, 0x8eee2335, 0xe2bc5ec2, 0xa83f4394, 0x45ad78f7, 0x36f3d0cd, 0xd99c05e8, 0xb0511dc7,
		0x69bc7ac4, 0xbd11375b, 0xe3ba71e5, 0x3b209ff2, 0x18feee17, 0xe25ad9e7, 0x13375046, 0x0515089d,
		0x4f0d0f04, 0x2627484e, 0x310128d2, 0xc668b434, 0x420841cc, 0x62d311b8, 0xe59ba771, 0x85a7a484
	};

	int constantTable[64] = {
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
	};
	for (int i = 0; i < 64; i++)
	{
		endingTable[i] = constantTable[i] + endingTable[i];
	}
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(sha256_endingTable, endingTable, sizeof(uint32_t) * 64, 0, cudaMemcpyHostToDevice));

}



__host__
void skein512_cpu_setBlock_80(uint32_t thr_id, void *pdata)
{
	memcpy(&PaddedMessage[0], pdata, 80);

	CUDA_SAFE_CALL(
		cudaMemcpyToSymbol(c_PaddedMessage80, &PaddedMessage[8], 8*2, 0, cudaMemcpyHostToDevice)
	);
	CUDA_SAFE_CALL(cudaMalloc(&(d_found[thr_id]), 2 * sizeof(uint32_t)));

	precalc();
}

__host__
void skein512_cpu_hash_80_52(int thr_id, uint32_t threads, uint32_t startNounce, int swapu,uint2 target, uint32_t *h_found)
{
	dim3 grid((threads + 1024 - 1) / 1024);
	dim3 block(1024);
	cudaMemset(d_found[thr_id], 0xffffffff, 2 * sizeof(uint32_t));
	skein512_gpu_hash_80_52 << < grid, block >> > (threads, startNounce, d_found[thr_id], (uint2)target);
	cudaMemcpy(h_found, d_found[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
__host__
void skein512_cpu_hash_80_50(int thr_id, uint32_t threads, uint32_t startNounce, int swapu, uint2 target, uint32_t *h_found)
{
	dim3 grid((threads + 32 - 1) / 32);
	dim3 block(32);
	cudaMemset(d_found[thr_id], 0xffffffff, 2 * sizeof(uint32_t));
	skein512_gpu_hash_80_50 << < grid, block >> > (threads, startNounce, d_found[thr_id], target);
	cudaMemcpy(h_found, d_found[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
