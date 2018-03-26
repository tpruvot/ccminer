
#include <stdio.h>
#include <cuda_helper.h>

#define TPB 256

/*
 * fugue512-80 x16r kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2018 tpruvot
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 */

#ifdef __INTELLISENSE__
#define __byte_perm(x, y, m) (x|y)
#define tex1Dfetch(t, n) (n)
#define __CUDACC__
#include <cuda_texture_types.h>
#endif

// store allocated textures device addresses
static unsigned int* d_textures[MAX_GPUS][1];

#define mixtab0(x) mixtabs[(x)]
#define mixtab1(x) mixtabs[(x)+256]
#define mixtab2(x) mixtabs[(x)+512]
#define mixtab3(x) mixtabs[(x)+768]

static texture<unsigned int, 1, cudaReadModeElementType> mixTab0Tex;

static const uint32_t mixtab0[] = {
	0x63633297, 0x7c7c6feb, 0x77775ec7, 0x7b7b7af7, 0xf2f2e8e5, 0x6b6b0ab7, 0x6f6f16a7, 0xc5c56d39,
	0x303090c0, 0x01010704, 0x67672e87, 0x2b2bd1ac, 0xfefeccd5, 0xd7d71371, 0xabab7c9a, 0x767659c3,
	0xcaca4005, 0x8282a33e, 0xc9c94909, 0x7d7d68ef, 0xfafad0c5, 0x5959947f, 0x4747ce07, 0xf0f0e6ed,
	0xadad6e82, 0xd4d41a7d, 0xa2a243be, 0xafaf608a, 0x9c9cf946, 0xa4a451a6, 0x727245d3, 0xc0c0762d,
	0xb7b728ea, 0xfdfdc5d9, 0x9393d47a, 0x2626f298, 0x363682d8, 0x3f3fbdfc, 0xf7f7f3f1, 0xcccc521d,
	0x34348cd0, 0xa5a556a2, 0xe5e58db9, 0xf1f1e1e9, 0x71714cdf, 0xd8d83e4d, 0x313197c4, 0x15156b54,
	0x04041c10, 0xc7c76331, 0x2323e98c, 0xc3c37f21, 0x18184860, 0x9696cf6e, 0x05051b14, 0x9a9aeb5e,
	0x0707151c, 0x12127e48, 0x8080ad36, 0xe2e298a5, 0xebeba781, 0x2727f59c, 0xb2b233fe, 0x757550cf,
	0x09093f24, 0x8383a43a, 0x2c2cc4b0, 0x1a1a4668, 0x1b1b416c, 0x6e6e11a3, 0x5a5a9d73, 0xa0a04db6,
	0x5252a553, 0x3b3ba1ec, 0xd6d61475, 0xb3b334fa, 0x2929dfa4, 0xe3e39fa1, 0x2f2fcdbc, 0x8484b126,
	0x5353a257, 0xd1d10169, 0x00000000, 0xededb599, 0x2020e080, 0xfcfcc2dd, 0xb1b13af2, 0x5b5b9a77,
	0x6a6a0db3, 0xcbcb4701, 0xbebe17ce, 0x3939afe4, 0x4a4aed33, 0x4c4cff2b, 0x5858937b, 0xcfcf5b11,
	0xd0d0066d, 0xefefbb91, 0xaaaa7b9e, 0xfbfbd7c1, 0x4343d217, 0x4d4df82f, 0x333399cc, 0x8585b622,
	0x4545c00f, 0xf9f9d9c9, 0x02020e08, 0x7f7f66e7, 0x5050ab5b, 0x3c3cb4f0, 0x9f9ff04a, 0xa8a87596,
	0x5151ac5f, 0xa3a344ba, 0x4040db1b, 0x8f8f800a, 0x9292d37e, 0x9d9dfe42, 0x3838a8e0, 0xf5f5fdf9,
	0xbcbc19c6, 0xb6b62fee, 0xdada3045, 0x2121e784, 0x10107040, 0xffffcbd1, 0xf3f3efe1, 0xd2d20865,
	0xcdcd5519, 0x0c0c2430, 0x1313794c, 0xececb29d, 0x5f5f8667, 0x9797c86a, 0x4444c70b, 0x1717655c,
	0xc4c46a3d, 0xa7a758aa, 0x7e7e61e3, 0x3d3db3f4, 0x6464278b, 0x5d5d886f, 0x19194f64, 0x737342d7,
	0x60603b9b, 0x8181aa32, 0x4f4ff627, 0xdcdc225d, 0x2222ee88, 0x2a2ad6a8, 0x9090dd76, 0x88889516,
	0x4646c903, 0xeeeebc95, 0xb8b805d6, 0x14146c50, 0xdede2c55, 0x5e5e8163, 0x0b0b312c, 0xdbdb3741,
	0xe0e096ad, 0x32329ec8, 0x3a3aa6e8, 0x0a0a3628, 0x4949e43f, 0x06061218, 0x2424fc90, 0x5c5c8f6b,
	0xc2c27825, 0xd3d30f61, 0xacac6986, 0x62623593, 0x9191da72, 0x9595c662, 0xe4e48abd, 0x797974ff,
	0xe7e783b1, 0xc8c84e0d, 0x373785dc, 0x6d6d18af, 0x8d8d8e02, 0xd5d51d79, 0x4e4ef123, 0xa9a97292,
	0x6c6c1fab, 0x5656b943, 0xf4f4fafd, 0xeaeaa085, 0x6565208f, 0x7a7a7df3, 0xaeae678e, 0x08083820,
	0xbaba0bde, 0x787873fb, 0x2525fb94, 0x2e2ecab8, 0x1c1c5470, 0xa6a65fae, 0xb4b421e6, 0xc6c66435,
	0xe8e8ae8d, 0xdddd2559, 0x747457cb, 0x1f1f5d7c, 0x4b4bea37, 0xbdbd1ec2, 0x8b8b9c1a, 0x8a8a9b1e,
	0x70704bdb, 0x3e3ebaf8, 0xb5b526e2, 0x66662983, 0x4848e33b, 0x0303090c, 0xf6f6f4f5, 0x0e0e2a38,
	0x61613c9f, 0x35358bd4, 0x5757be47, 0xb9b902d2, 0x8686bf2e, 0xc1c17129, 0x1d1d5374, 0x9e9ef74e,
	0xe1e191a9, 0xf8f8decd, 0x9898e556, 0x11117744, 0x696904bf, 0xd9d93949, 0x8e8e870e, 0x9494c166,
	0x9b9bec5a, 0x1e1e5a78, 0x8787b82a, 0xe9e9a989, 0xcece5c15, 0x5555b04f, 0x2828d8a0, 0xdfdf2b51,
	0x8c8c8906, 0xa1a14ab2, 0x89899212, 0x0d0d2334, 0xbfbf10ca, 0xe6e684b5, 0x4242d513, 0x686803bb,
	0x4141dc1f, 0x9999e252, 0x2d2dc3b4, 0x0f0f2d3c, 0xb0b03df6, 0x5454b74b, 0xbbbb0cda, 0x16166258
};

#define TIX4(q, x00, x01, x04, x07, x08, x22, x24, x27, x30) { \
	x22 ^= x00; \
	x00 = (q); \
	x08 ^= x00; \
	x01 ^= x24; \
	x04 ^= x27; \
	x07 ^= x30; \
}

#define CMIX36(x00, x01, x02, x04, x05, x06, x18, x19, x20) { \
	x00 ^= x04; \
	x01 ^= x05; \
	x02 ^= x06; \
	x18 ^= x04; \
	x19 ^= x05; \
	x20 ^= x06; \
}

#define SMIX(x0, x1, x2, x3) { \
	uint32_t tmp; \
	uint32_t r0 = 0; \
	uint32_t r1 = 0; \
	uint32_t r2 = 0; \
	uint32_t r3 = 0; \
	uint32_t c0 = mixtab0(x0 >> 24); \
	tmp = mixtab1((x0 >> 16) & 0xFF); \
	c0 ^= tmp; \
	r1 ^= tmp; \
	tmp = mixtab2((x0 >>  8) & 0xFF); \
	c0 ^= tmp; \
	r2 ^= tmp; \
	tmp = mixtab3(x0 & 0xFF); \
	c0 ^= tmp; \
	r3 ^= tmp; \
	tmp = mixtab0(x1 >> 24); \
	uint32_t c1 = tmp; \
	r0 ^= tmp; \
	tmp = mixtab1((x1 >> 16) & 0xFF); \
	c1 ^= tmp; \
	tmp = mixtab2((x1 >>  8) & 0xFF); \
	c1 ^= tmp; \
	r2 ^= tmp; \
	tmp = mixtab3(x1 & 0xFF); \
	c1 ^= tmp; \
	r3 ^= tmp; \
	tmp = mixtab0(x2 >> 24); \
	uint32_t c2 = tmp; \
	r0 ^= tmp; \
	tmp = mixtab1((x2 >> 16) & 0xFF); \
	c2 ^= tmp; \
	r1 ^= tmp; \
	tmp = mixtab2((x2 >>  8) & 0xFF); \
	c2 ^= tmp; \
	tmp = mixtab3(x2 & 0xFF); \
	c2 ^= tmp; \
	r3 ^= tmp; \
	tmp = mixtab0(x3 >> 24); \
	uint32_t c3 = tmp; \
	r0 ^= tmp; \
	tmp = mixtab1((x3 >> 16) & 0xFF); \
	c3 ^= tmp; \
	r1 ^= tmp; \
	tmp = mixtab2((x3 >>  8) & 0xFF); \
	c3 ^= tmp; \
	r2 ^= tmp; \
	tmp = mixtab3(x3 & 0xFF); \
	c3 ^= tmp; \
	x0 = ((c0 ^ r0) & 0xFF000000) | ((c1 ^ r1) & 0x00FF0000) \
		| ((c2 ^ r2) & 0x0000FF00) | ((c3 ^ r3) & 0x000000FF); \
	x1 = ((c1 ^ (r0 <<  8)) & 0xFF000000) | ((c2 ^ (r1 <<  8)) & 0x00FF0000) \
		| ((c3 ^ (r2 <<  8)) & 0x0000FF00) | ((c0 ^ (r3 >> 24)) & 0x000000FF); \
	x2 = ((c2 ^ (r0 << 16)) & 0xFF000000) | ((c3 ^ (r1 << 16)) & 0x00FF0000) \
		| ((c0 ^ (r2 >> 16)) & 0x0000FF00) | ((c1 ^ (r3 >> 16)) & 0x000000FF); \
	x3 = ((c3 ^ (r0 << 24)) & 0xFF000000) | ((c0 ^ (r1 >>  8)) & 0x00FF0000) \
		| ((c1 ^ (r2 >>  8)) & 0x0000FF00) | ((c2 ^ (r3 >>  8)) & 0x000000FF); \
}

#define SUB_ROR3 { \
	B33 = S33, B34 = S34, B35 = S35; \
	S35 = S32; S34 = S31; S33 = S30; S32 = S29; S31 = S28; S30 = S27; S29 = S26; S28 = S25; S27 = S24; \
	S26 = S23; S25 = S22; S24 = S21; S23 = S20; S22 = S19; S21 = S18; S20 = S17; S19 = S16; S18 = S15; \
	S17 = S14; S16 = S13; S15 = S12; S14 = S11; S13 = S10; S12 = S09; S11 = S08; S10 = S07; S09 = S06; \
	S08 = S05; S07 = S04; S06 = S03; S05 = S02; S04 = S01; S03 = S00; S02 = B35; S01 = B34; S00 = B33; \
}

#define SUB_ROR8 { \
	B28 = S28, B29 = S29, B30 = S30, B31 = S31, B32 = S32, B33 = S33, B34 = S34, B35 = S35; \
	S35 = S27; S34 = S26; S33 = S25; S32 = S24; S31 = S23; S30 = S22; S29 = S21; S28 = S20; S27 = S19; \
	S26 = S18; S25 = S17; S24 = S16; S23 = S15; S22 = S14; S21 = S13; S20 = S12; S19 = S11; S18 = S10; \
	S17 = S09; S16 = S08; S15 = S07; S14 = S06; S13 = S05; S12 = S04; S11 = S03; S10 = S02; S09 = S01; \
	S08 = S00; S07 = B35; S06 = B34; S05 = B33; S04 = B32; S03 = B31; S02 = B30; S01 = B29; S00 = B28; \
}

#define SUB_ROR9 { \
	B27 = S27, B28 = S28, B29 = S29, B30 = S30, B31 = S31, B32 = S32, B33 = S33, B34 = S34, B35 = S35; \
	S35 = S26; S34 = S25; S33 = S24; S32 = S23; S31 = S22; S30 = S21; S29 = S20; S28 = S19; S27 = S18; \
	S26 = S17; S25 = S16; S24 = S15; S23 = S14; S22 = S13; S21 = S12; S20 = S11; S19 = S10; S18 = S09; \
	S17 = S08; S16 = S07; S15 = S06; S14 = S05; S13 = S04; S12 = S03; S11 = S02; S10 = S01; S09 = S00; \
	S08 = B35; S07 = B34; S06 = B33; S05 = B32; S04 = B31; S03 = B30; S02 = B29; S01 = B28; S00 = B27; \
}

#define SUB_ROR9_3 { \
	SUB_ROR3; SUB_ROR3; SUB_ROR3; \
}

#define SUB_ROR12 { /* to fix */ \
	B24 = S00; B25 = S01; B26 = S02; B27 = S03; B28 = S04; B29 = S05; B30 = S06; B31 = S07; B32 = S08; B33 = S09; B34 = S10; B35 = S11; \
	S00 = S12; S01 = S13; S02 = S14; S03 = S15; S04 = S16; S05 = S17; S06 = S18; S07 = S19; S08 = S20; S09 = S21; S10 = S22; S11 = S23; \
	S12 = S24; S13 = S25; S14 = S26; S15 = S27; S16 = S28; S17 = S29; S18 = S30; S19 = S31; S20 = S32; S21 = S33; S22 = S34; S23 = S35; \
	S24 = B24; S25 = B25; S26 = B26; S27 = B27; S28 = B28; S29 = B29; S30 = B30; S31 = B31; S32 = B32; S33 = B33; S34 = B34; S35 = B35; \
}

#define FUGUE512_3(x, y, z) { \
	TIX4(x, S00, S01, S04, S07, S08, S22, S24, S27, S30); \
	CMIX36(S33, S34, S35, S01, S02, S03, S15, S16, S17); \
	SMIX(S33, S34, S35, S00); \
	CMIX36(S30, S31, S32, S34, S35, S00, S12, S13, S14); \
	SMIX(S30, S31, S32, S33); \
	CMIX36(S27, S28, S29, S31, S32, S33, S09, S10, S11); \
	SMIX(S27, S28, S29, S30); \
	CMIX36(S24, S25, S26, S28, S29, S30, S06, S07, S08); \
	SMIX(S24, S25, S26, S27); \
	\
	TIX4(y, S24, S25, S28, S31, S32, S10, S12, S15, S18); \
	CMIX36(S21, S22, S23, S25, S26, S27, S03, S04, S05); \
	SMIX(S21, S22, S23, S24); \
	CMIX36(S18, S19, S20, S22, S23, S24, S00, S01, S02); \
	SMIX(S18, S19, S20, S21); \
	CMIX36(S15, S16, S17, S19, S20, S21, S33, S34, S35); \
	SMIX(S15, S16, S17, S18); \
	CMIX36(S12, S13, S14, S16, S17, S18, S30, S31, S32); \
	SMIX(S12, S13, S14, S15); \
	\
	TIX4(z, S12, S13, S16, S19, S20, S34, S00, S03, S06); \
	CMIX36(S09, S10, S11, S13, S14, S15, S27, S28, S29); \
	SMIX(S09, S10, S11, S12); \
	CMIX36(S06, S07, S08, S10, S11, S12, S24, S25, S26); \
	SMIX(S06, S07, S08, S09); \
	CMIX36(S03, S04, S05, S07, S08, S09, S21, S22, S23); \
	SMIX(S03, S04, S05, S06); \
	CMIX36(S00, S01, S02, S04, S05, S06, S18, S19, S20); \
	SMIX(S00, S01, S02, S03); \
}

#define FUGUE512_F(w, x, y, z) { \
	TIX4(w, S00, S01, S04, S07, S08, S22, S24, S27, S30); \
	CMIX36(S33, S34, S35, S01, S02, S03, S15, S16, S17); \
	SMIX(S33, S34, S35, S00); \
	CMIX36(S30, S31, S32, S34, S35, S00, S12, S13, S14); \
	SMIX(S30, S31, S32, S33); \
	CMIX36(S27, S28, S29, S31, S32, S33, S09, S10, S11); \
	SMIX(S27, S28, S29, S30); \
	CMIX36(S24, S25, S26, S28, S29, S30, S06, S07, S08); \
	SMIX(S24, S25, S26, S27); \
	\
	TIX4(x, S24, S25, S28, S31, S32, S10, S12, S15, S18); \
	CMIX36(S21, S22, S23, S25, S26, S27, S03, S04, S05); \
	SMIX(S21, S22, S23, S24); \
	CMIX36(S18, S19, S20, S22, S23, S24, S00, S01, S02); \
	SMIX(S18, S19, S20, S21); \
	CMIX36(S15, S16, S17, S19, S20, S21, S33, S34, S35); \
	SMIX(S15, S16, S17, S18); \
	CMIX36(S12, S13, S14, S16, S17, S18, S30, S31, S32); \
	SMIX(S12, S13, S14, S15); \
	\
	TIX4(y, S12, S13, S16, S19, S20, S34, S00, S03, S06); \
	CMIX36(S09, S10, S11, S13, S14, S15, S27, S28, S29); \
	SMIX(S09, S10, S11, S12); \
	CMIX36(S06, S07, S08, S10, S11, S12, S24, S25, S26); \
	SMIX(S06, S07, S08, S09); \
	CMIX36(S03, S04, S05, S07, S08, S09, S21, S22, S23); \
	SMIX(S03, S04, S05, S06); \
	CMIX36(S00, S01, S02, S04, S05, S06, S18, S19, S20); \
	SMIX(S00, S01, S02, S03); \
	\
	TIX4(z, S00, S01, S04, S07, S08, S22, S24, S27, S30); \
	CMIX36(S33, S34, S35, S01, S02, S03, S15, S16, S17); \
	SMIX(S33, S34, S35, S00); \
	CMIX36(S30, S31, S32, S34, S35, S00, S12, S13, S14); \
	SMIX(S30, S31, S32, S33); \
	CMIX36(S27, S28, S29, S31, S32, S33, S09, S10, S11); \
	SMIX(S27, S28, S29, S30); \
	CMIX36(S24, S25, S26, S28, S29, S30, S06, S07, S08); \
	SMIX(S24, S25, S26, S27); \
}

#undef ROL8
#ifdef __CUDA_ARCH__
__device__ __forceinline__
uint32_t ROL8(const uint32_t a) {
	return __byte_perm(a, 0, 0x2103);
}
__device__ __forceinline__
uint32_t ROR8(const uint32_t a) {
	return __byte_perm(a, 0, 0x0321);
}
__device__ __forceinline__
uint32_t ROL16(const uint32_t a) {
	return __byte_perm(a, 0, 0x1032);
}
#else
#define ROL8(u)  ROTL32(u, 8)
#define ROR8(u)  ROTR32(u, 8)
#define ROL16(u) ROTL32(u,16)
#endif

//#define AS_UINT4(addr) *((uint4*)(addr))

__constant__ static uint64_t c_PaddedMessage80[10];

__host__
void x16_fugue512_setBlock_80(void *pdata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, pdata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}

/***************************************************/

__global__
__launch_bounds__(TPB)
void x16_fugue512_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint64_t *g_hash)
{
	__shared__ uint32_t mixtabs[1024];

	// load shared mem (with 256 threads)
	const uint32_t thr = threadIdx.x & 0xFF;
	const uint32_t tmp = tex1Dfetch(mixTab0Tex, thr);
	mixtabs[thr] = tmp;
	mixtabs[thr+256] = ROR8(tmp);
	mixtabs[thr+512] = ROL16(tmp);
	mixtabs[thr+768] = ROL8(tmp);
#if TPB <= 256
	if (blockDim.x < 256) {
		const uint32_t thr = (threadIdx.x + 0x80) & 0xFF;
		const uint32_t tmp = tex1Dfetch(mixTab0Tex, thr);
		mixtabs[thr] = tmp;
		mixtabs[thr + 256] = ROR8(tmp);
		mixtabs[thr + 512] = ROL16(tmp);
		mixtabs[thr + 768] = ROL8(tmp);
	}
#endif

	__syncthreads();

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t Data[20];

		#pragma unroll
		for(int i = 0; i < 10; i++)
			AS_UINT2(&Data[i * 2]) = AS_UINT2(&c_PaddedMessage80[i]);
		Data[19] = (startNonce + thread);

		uint32_t S00, S01, S02, S03, S04, S05, S06, S07, S08, S09, S10, S11;
		uint32_t S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22, S23;
		uint32_t S24, S25, S26, S27, S28, S29, S30, S31, S32, S33, S34, S35;
		//uint32_t B24, B25, B26,
		uint32_t B27, B28, B29, B30, B31, B32, B33, B34, B35;
		//const uint64_t bc = 640 bits to hash
		//const uint32_t bclo = (uint32_t)(bc);
		//const uint32_t bchi = (uint32_t)(bc >> 32);

		S00 = S01 = S02 = S03 = S04 = S05 = S06 = S07 = S08 = S09 = 0;
		S10 = S11 = S12 = S13 = S14 = S15 = S16 = S17 = S18 = S19 = 0;
		S20 = 0x8807a57e; S21 = 0xe616af75; S22 = 0xc5d3e4db; S23 = 0xac9ab027;
		S24 = 0xd915f117; S25 = 0xb6eecc54; S26 = 0x06e8020b; S27 = 0x4a92efd1;
		S28 = 0xaac6e2c9; S29 = 0xddb21398; S30 = 0xcae65838; S31 = 0x437f203f;
		S32 = 0x25ea78e7; S33 = 0x951fddd6; S34 = 0xda6ed11d; S35 = 0xe13e3567;

		FUGUE512_3((Data[ 0]), (Data[ 1]), (Data[ 2]));
		FUGUE512_3((Data[ 3]), (Data[ 4]), (Data[ 5]));
		FUGUE512_3((Data[ 6]), (Data[ 7]), (Data[ 8]));
		FUGUE512_3((Data[ 9]), (Data[10]), (Data[11]));
		FUGUE512_3((Data[12]), (Data[13]), (Data[14]));
		FUGUE512_3((Data[15]), (Data[16]), (Data[17]));
		FUGUE512_F((Data[18]), (Data[19]), 0/*bchi*/, (80*8)/*bclo*/);

		// rotate right state by 3 dwords (S00 = S33, S03 = S00)
		SUB_ROR3;
		SUB_ROR9;

		#pragma unroll 32
		for (int i = 0; i < 32; i++) {
			SUB_ROR3;
			CMIX36(S00, S01, S02, S04, S05, S06, S18, S19, S20);
			SMIX(S00, S01, S02, S03);
		}
		#pragma unroll 13
		for (int i = 0; i < 13; i++) {
			S04 ^= S00;
			S09 ^= S00;
			S18 ^= S00;
			S27 ^= S00;
			SUB_ROR9;
			SMIX(S00, S01, S02, S03);
			S04 ^= S00;
			S10 ^= S00;
			S18 ^= S00;
			S27 ^= S00;
			SUB_ROR9;
			SMIX(S00, S01, S02, S03);
			S04 ^= S00;
			S10 ^= S00;
			S19 ^= S00;
			S27 ^= S00;
			SUB_ROR9;
			SMIX(S00, S01, S02, S03);
			S04 ^= S00;
			S10 ^= S00;
			S19 ^= S00;
			S28 ^= S00;
			SUB_ROR8;
			SMIX(S00, S01, S02, S03);
		}
		S04 ^= S00;
		S09 ^= S00;
		S18 ^= S00;
		S27 ^= S00;

		Data[ 0] = cuda_swab32(S01);
		Data[ 1] = cuda_swab32(S02);
		Data[ 2] = cuda_swab32(S03);
		Data[ 3] = cuda_swab32(S04);
		Data[ 4] = cuda_swab32(S09);
		Data[ 5] = cuda_swab32(S10);
		Data[ 6] = cuda_swab32(S11);
		Data[ 7] = cuda_swab32(S12);
		Data[ 8] = cuda_swab32(S18);
		Data[ 9] = cuda_swab32(S19);
		Data[10] = cuda_swab32(S20);
		Data[11] = cuda_swab32(S21);
		Data[12] = cuda_swab32(S27);
		Data[13] = cuda_swab32(S28);
		Data[14] = cuda_swab32(S29);
		Data[15] = cuda_swab32(S30);

		const size_t hashPosition = thread;
		uint64_t* pHash = &g_hash[hashPosition << 3];
		#pragma unroll 4
		for(int i = 0; i < 4; i++)
			AS_UINT4(&pHash[i * 2]) = AS_UINT4(&Data[i * 4]);
	}
}

#define texDef(id, texname, texmem, texsource, texsize) { \
	unsigned int *texmem; \
	cudaMalloc(&texmem, texsize); \
	d_textures[thr_id][id] = texmem; \
	cudaMemcpy(texmem, texsource, texsize, cudaMemcpyHostToDevice); \
	texname.normalized = 0; \
	texname.filterMode = cudaFilterModePoint; \
	texname.addressMode[0] = cudaAddressModeClamp; \
	{ cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned int>(); \
	  cudaBindTexture(NULL, &texname, texmem, &channelDesc, texsize ); \
	} \
}

__host__
void x16_fugue512_cpu_init(int thr_id, uint32_t threads)
{
	texDef(0, mixTab0Tex, mixTab0m, mixtab0, sizeof(uint32_t)*256);
}

__host__
void x16_fugue512_cpu_free(int thr_id)
{
	cudaFree(d_textures[thr_id][0]);
}

__host__
void x16_fugue512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNonce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = TPB;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x16_fugue512_gpu_hash_80 <<<grid, block>>> (threads, startNonce, (uint64_t*)d_hash);
}
