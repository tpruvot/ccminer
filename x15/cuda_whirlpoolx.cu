/*
 * Built on cbuchner1's implementation, actual hashing code
 * based on sphlib 3.0
 */
#include <stdio.h>
#include <memory.h>

#define threadsPerBlock 1024

#include "cuda_helper.h"

__constant__ uint64_t c_PaddedMessage80[16]; // padded message (80 bytes + padding)
__constant__ uint64_t c_xtra[8];
__constant__ uint64_t c_tmp[72];
__constant__ uint64_t pTarget[4];

static uint32_t *h_wxnounce[MAX_GPUS] = { 0 };
static uint32_t *d_WXNonce[MAX_GPUS] = { 0 };

/**
 * Whirlpool CUDA kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014 djm34 & tpruvot & SP & Provos Alexis
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
 * @author djm34
 * @author tpruvot
 * @author SP
 * @author Provos Alexis
 */

__constant__ __align__(64) uint64_t mixTob0Tox[256];

const uint64_t plain_T0[256]= {
	0xD83078C018601818,0x2646AF05238C2323,0xB891F97EC63FC6C6,0xFBCD6F13E887E8E8,0xCB13A14C87268787,0x116D62A9B8DAB8B8,0x0902050801040101,0x0D9E6E424F214F4F,0x9B6CEEAD36D83636,
	0xFF510459A6A2A6A6,0x0CB9BDDED26FD2D2,0x0EF706FBF5F3F5F5,0x96F280EF79F97979,0x30DECE5F6FA16F6F,0x6D3FEFFC917E9191,0xF8A407AA52555252,0x47C0FD27609D6060,0x35657689BCCABCBC,
	0x372BCDAC9B569B9B,0x8A018C048E028E8E,0xD25B1571A3B6A3A3,0x6C183C600C300C0C,0x84F68AFF7BF17B7B,0x806AE1B535D43535,0xF53A69E81D741D1D,0xB3DD4753E0A7E0E0,0x21B3ACF6D77BD7D7,
	0x9C99ED5EC22FC2C2,0x435C966D2EB82E2E,0x29967A624B314B4B,0x5DE121A3FEDFFEFE,0xD5AE168257415757,0xBD2A41A815541515,0xE8EEB69F77C17777,0x926EEBA537DC3737,0x9ED7567BE5B3E5E5,
	0x1323D98C9F469F9F,0x23FD17D3F0E7F0F0,0x20947F6A4A354A4A,0x44A9959EDA4FDADA,0xA2B025FA587D5858,0xCF8FCA06C903C9C9,0x7C528D5529A42929,0x5A1422500A280A0A,0x507F4FE1B1FEB1B1,
	0xC95D1A69A0BAA0A0,0x14D6DA7F6BB16B6B,0xD917AB5C852E8585,0x3C677381BDCEBDBD,0x8FBA34D25D695D5D,0x9020508010401010,0x07F503F3F4F7F4F4,0xDD8BC016CB0BCBCB,0xD37CC6ED3EF83E3E,
	0x2D0A112805140505,0x78CEE61F67816767,0x97D55373E4B7E4E4,0x024EBB25279C2727,0x7382583241194141,0xA70B9D2C8B168B8B,0xF6530151A7A6A7A7,0xB2FA94CF7DE97D7D,0x4937FBDC956E9595,
	0x56AD9F8ED847D8D8,0x70EB308BFBCBFBFB,0xCDC17123EE9FEEEE,0xBBF891C77CED7C7C,0x71CCE31766856666,0x7BA78EA6DD53DDDD,0xAF2E4BB8175C1717,0x458E460247014747,0x1A21DC849E429E9E,
	0xD489C51ECA0FCACA,0x585A99752DB42D2D,0x2E637991BFC6BFBF,0x3F0E1B38071C0707,0xAC472301AD8EADAD,0xB0B42FEA5A755A5A,0xEF1BB56C83368383,0xB666FF8533CC3333,0x5CC6F23F63916363,
	0x12040A1002080202,0x93493839AA92AAAA,0xDEE2A8AF71D97171,0xC68DCF0EC807C8C8,0xD1327DC819641919,0x3B92707249394949,0x5FAF9A86D943D9D9,0x31F91DC3F2EFF2F2,0xA8DB484BE3ABE3E3,
	0xB9B62AE25B715B5B,0xBC0D9234881A8888,0x3E29C8A49A529A9A,0x0B4CBE2D26982626,0xBF64FA8D32C83232,0x597D4AE9B0FAB0B0,0xF2CF6A1BE983E9E9,0x771E33780F3C0F0F,0x33B7A6E6D573D5D5,
	0xF41DBA74803A8080,0x27617C99BEC2BEBE,0xEB87DE26CD13CDCD,0x8968E4BD34D03434,0x3290757A483D4848,0x54E324ABFFDBFFFF,0x8DF48FF77AF57A7A,0x643DEAF4907A9090,0x9DBE3EC25F615F5F,
	0x3D40A01D20802020,0x0FD0D56768BD6868,0xCA3472D01A681A1A,0xB7412C19AE82AEAE,0x7D755EC9B4EAB4B4,0xCEA8199A544D5454,0x7F3BE5EC93769393,0x2F44AA0D22882222,0x63C8E907648D6464,
	0x2AFF12DBF1E3F1F1,0xCCE6A2BF73D17373,0x82245A9012481212,0x7A805D3A401D4040,0x4810284008200808,0x959BE856C32BC3C3,0xDFC57B33EC97ECEC,0x4DAB9096DB4BDBDB,0xC05F1F61A1BEA1A1,
	0x9107831C8D0E8D8D,0xC87AC9F53DF43D3D,0x5B33F1CC97669797,0x0000000000000000,0xF983D436CF1BCFCF,0x6E5687452BAC2B2B,0xE1ECB39776C57676,0xE619B06482328282,0x28B1A9FED67FD6D6,
	0xC33677D81B6C1B1B,0x74775BC1B5EEB5B5,0xBE432911AF86AFAF,0x1DD4DF776AB56A6A,0xEAA00DBA505D5050,0x578A4C1245094545,0x38FB18CBF3EBF3F3,0xAD60F09D30C03030,0xC4C3742BEF9BEFEF,
	0xDA7EC3E53FFC3F3F,0xC7AA1C9255495555,0xDB591079A2B2A2A2,0xE9C96503EA8FEAEA,0x6ACAEC0F65896565,0x036968B9BAD2BABA,0x4A5E93652FBC2F2F,0x8E9DE74EC027C0C0,0x60A181BEDE5FDEDE,
	0xFC386CE01C701C1C,0x46E72EBBFDD3FDFD,0x1F9A64524D294D4D,0x7639E0E492729292,0xFAEABC8F75C97575,0x360C1E3006180606,0xAE0998248A128A8A,0x4B7940F9B2F2B2B2,0x85D15963E6BFE6E6,
	0x7E1C36700E380E0E,0xE73E63F81F7C1F1F,0x55C4F73762956262,0x3AB5A3EED477D4D4,0x814D3229A89AA8A8,0x5231F4C496629696,0x62EF3A9BF9C3F9F9,0xA397F666C533C5C5,0x104AB13525942525,
	0xABB220F259795959,0xD015AE54842A8484,0xC5E4A7B772D57272,0xEC72DDD539E43939,0x1698615A4C2D4C4C,0x94BC3BCA5E655E5E,0x9FF085E778FD7878,0xE570D8DD38E03838,0x980586148C0A8C8C,
	0x17BFB2C6D163D1D1,0xE4570B41A5AEA5A5,0xA1D94D43E2AFE2E2,0x4EC2F82F61996161,0x427B45F1B3F6B3B3,0x3442A51521842121,0x0825D6949C4A9C9C,0xEE3C66F01E781E1E,0x6186522243114343,
	0xB193FC76C73BC7C7,0x4FE52BB3FCD7FCFC,0x2408142004100404,0xE3A208B251595151,0x252FC7BC995E9999,0x22DAC44F6DA96D6D,0x651A39680D340D0D,0x79E93583FACFFAFA,0x69A384B6DF5BDFDF,
	0xA9FC9BD77EE57E7E,0x1948B43D24902424,0xFE76D7C53BEC3B3B,0x9A4B3D31AB96ABAB,0xF081D13ECE1FCECE,0x9922558811441111,0x8303890C8F068F8F,0x049C6B4A4E254E4E,0x667351D1B7E6B7B7,
	0xE0CB600BEB8BEBEB,0xC178CCFD3CF03C3C,0xFD1FBF7C813E8181,0x4035FED4946A9494,0x1CF30CEBF7FBF7F7,0x186F67A1B9DEB9B9,0x8B265F98134C1313,0x51589C7D2CB02C2C,0x05BBB8D6D36BD3D3,
	0x8CD35C6BE7BBE7E7,0x39DCCB576EA56E6E,0xAA95F36EC437C4C4,0x1B060F18030C0303,0xDCAC138A56455656,0x5E88491A440D4444,0xA0FE9EDF7FE17F7F,0x884F3721A99EA9A9,0x6754824D2AA82A2A,
	0x0A6B6DB1BBD6BBBB,0x879FE246C123C1C1,0xF1A602A253515353,0x72A58BAEDC57DCDC,0x531627580B2C0B0B,0x0127D39C9D4E9D9D,0x2BD8C1476CAD6C6C,0xA462F59531C43131,0xF3E8B98774CD7474,
	0x15F109E3F6FFF6F6,0x4C8C430A46054646,0xA5452609AC8AACAC,0xB50F973C891E8989,0xB42844A014501414,0xBADF425BE1A3E1E1,0xA62C4EB016581616,0xF774D2CD3AE83A3A,0x06D2D06F69B96969,
	0x41122D4809240909,0xD7E0ADA770DD7070,0x6F7154D9B6E2B6B6,0x1EBDB7CED067D0D0,0xD6C77E3BED93EDED,0xE285DB2ECC17CCCC,0x6884572A42154242,0x2C2DC2B4985A9898,0xED550E49A4AAA4A4,
	0x7550885D28A02828,0x86B831DA5C6D5C5C,0x6BED3F93F8C7F8F8,0xC211A44486228686
};

/**
 * Round constants.
 */
__constant__ uint64_t InitVector_RC[10];

const uint64_t plain_RC[10] = {
	0x4F01B887E8C62318,0x52916F79F5D2A636,0x357B0CA38E9BBC60,0x57FE4B2EC2D7E01D,0xDA4AF09FE5377715,
	0x856BA0B10A29C958,0x67053ECBF4105DBD,0xD8957DA78B4127E4,0x9E4717DD667CEEFB,0x33835AAD07BF2DCA
};

/* ====================================================================== */

__device__ __forceinline__
static uint64_t ROUND_ELT(const uint64_t* sharedMemory, const uint64_t* __restrict__ in, const int i0, const int i1, const int i2, const int i3, const int i4, const int i5, const int i6, const int i7)
{
	uint32_t* in32 = (uint32_t*)in;
	return xor8(	sharedMemory[__byte_perm(in32[(i0 << 1)], 0, 0x4440)],
			sharedMemory[__byte_perm(in32[(i1 << 1)], 0, 0x4441) + 256],
			sharedMemory[__byte_perm(in32[(i2 << 1)], 0, 0x4442) + 512],
			sharedMemory[__byte_perm(in32[(i3 << 1)], 0, 0x4443) + 768],
			sharedMemory[__byte_perm(in32[(i4 << 1) + 1], 0, 0x4440) + 1024],
			sharedMemory[__byte_perm(in32[(i5 << 1) + 1], 0, 0x4441) + 1280],
			sharedMemory[__byte_perm(in32[(i6 << 1) + 1], 0, 0x4442) + 1536],
			sharedMemory[__byte_perm(in32[(i7 << 1) + 1], 0, 0x4443) + 1792]);
}

#define TRANSFER(dst, src) { \
	dst[0] = src ## 0; \
	dst[1] = src ## 1; \
	dst[2] = src ## 2; \
	dst[3] = src ## 3; \
	dst[4] = src ## 4; \
	dst[5] = src ## 5; \
	dst[6] = src ## 6; \
	dst[7] = src ## 7; \
}

#define ROUND(table, in, out, c0, c1, c2, c3, c4, c5, c6, c7) { \
	out ## 0 = xor1(ROUND_ELT(table, in, 0, 7, 6, 5, 4, 3, 2, 1), c0); \
	out ## 1 = xor1(ROUND_ELT(table, in, 1, 0, 7, 6, 5, 4, 3, 2), c1); \
	out ## 2 = xor1(ROUND_ELT(table, in, 2, 1, 0, 7, 6, 5, 4, 3), c2); \
	out ## 3 = xor1(ROUND_ELT(table, in, 3, 2, 1, 0, 7, 6, 5, 4), c3); \
	out ## 4 = xor1(ROUND_ELT(table, in, 4, 3, 2, 1, 0, 7, 6, 5), c4); \
	out ## 5 = xor1(ROUND_ELT(table, in, 5, 4, 3, 2, 1, 0, 7, 6), c5); \
	out ## 6 = xor1(ROUND_ELT(table, in, 6, 5, 4, 3, 2, 1, 0, 7), c6); \
	out ## 7 = xor1(ROUND_ELT(table, in, 7, 6, 5, 4, 3, 2, 1, 0), c7); \
}

#define ROUND1(table, in, out, c) { \
	out ## 0 = xor1(ROUND_ELT(table, in, 0, 7, 6, 5, 4, 3, 2, 1), c); \
	out ## 1 = ROUND_ELT(table, in, 1, 0, 7, 6, 5, 4, 3, 2); \
	out ## 2 = ROUND_ELT(table, in, 2, 1, 0, 7, 6, 5, 4, 3); \
	out ## 3 = ROUND_ELT(table, in, 3, 2, 1, 0, 7, 6, 5, 4); \
	out ## 4 = ROUND_ELT(table, in, 4, 3, 2, 1, 0, 7, 6, 5); \
	out ## 5 = ROUND_ELT(table, in, 5, 4, 3, 2, 1, 0, 7, 6); \
	out ## 6 = ROUND_ELT(table, in, 6, 5, 4, 3, 2, 1, 0, 7); \
	out ## 7 = ROUND_ELT(table, in, 7, 6, 5, 4, 3, 2, 1, 0); \
}

#define ROUND_KSCHED(table, in, out, c) \
	ROUND1(table, in, out, c) \
	TRANSFER(in, out)

#define ROUND_WENC(table, in, key, out) \
	ROUND(table, in, out, key[0], key[1], key[2],key[3], key[4], key[5], key[6], key[7]) \
	TRANSFER(in, out)

static uint64_t* d_xtra[MAX_GPUS] = { 0 };
static uint64_t* d_tmp[MAX_GPUS] = { 0 };

__device__ __forceinline__
static void whirlpoolx_getShared(uint64_t* sharedMemory)
{
	if (threadIdx.x < 256) {
		sharedMemory[threadIdx.x] = mixTob0Tox[threadIdx.x];
		sharedMemory[threadIdx.x+256]  = ROTL64(sharedMemory[threadIdx.x], 8);
		sharedMemory[threadIdx.x+512]  = ROTL64(sharedMemory[threadIdx.x],16);
		sharedMemory[threadIdx.x+768]  = ROTL64(sharedMemory[threadIdx.x],24);
		sharedMemory[threadIdx.x+1024] = ROTL64(sharedMemory[threadIdx.x],32);
		sharedMemory[threadIdx.x+1280] = ROTR64(sharedMemory[threadIdx.x],24);
		sharedMemory[threadIdx.x+1536] = ROTR64(sharedMemory[threadIdx.x],16);
		sharedMemory[threadIdx.x+1792] = ROTR64(sharedMemory[threadIdx.x], 8);
	}
	__syncthreads();
}


__global__
void whirlpoolx_gpu_precompute(uint32_t threads, uint64_t* d_xtra, uint64_t* d_tmp)
{
	__shared__ uint64_t sharedMemory[2048];

	whirlpoolx_getShared(sharedMemory);
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t n[8];
		uint64_t h[8] = { 0 };

		#pragma unroll 8
		for (int i=0; i<8; i++) {
			n[i] = c_PaddedMessage80[i];  // read data
		}
		//#pragma unroll 10
		for (unsigned int r=0; r < 10; r++) {
			uint64_t tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
			ROUND_KSCHED(sharedMemory, h, tmp, InitVector_RC[r]);
			ROUND_WENC(sharedMemory, n, h, tmp);
		}
		#pragma unroll 8
		for (int i=0; i < 8; i++) {
			h[i] = xor1(n[i],c_PaddedMessage80[i]);
		}

		if(threadIdx.x==0)d_xtra[threadIdx.x]=h[1];
		uint64_t atLastCalc=xor1(h[3],h[5]);

		//////////////////////////////////
		n[0] = c_PaddedMessage80[8];    //read data
		n[1] = c_PaddedMessage80[9]; //whirlpool
		n[2] = 0x0000000000000080; //whirlpool
		n[3] = 0;
		n[4] = 0;
		n[5] = 0;
		n[6] = 0;
		n[7] = 0x8002000000000000;

		n[0] = xor1(n[0],h[0]);
		n[2] = xor1(n[2],h[2]);	n[3] = h[3];
		n[4] = h[4];	n[5] = h[5];
		n[6] = h[6];	n[7] = xor1(n[7],h[7]);
		uint64_t tmp[8];
		tmp[0] = xor1(ROUND_ELT(sharedMemory, h, 0, 7, 6, 5, 4, 3, 2, 1),InitVector_RC[0]);
		tmp[1] = ROUND_ELT(sharedMemory, h, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp[2] = ROUND_ELT(sharedMemory, h, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp[3] = ROUND_ELT(sharedMemory, h, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp[4] = ROUND_ELT(sharedMemory, h, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp[5] = ROUND_ELT(sharedMemory, h, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp[6] = ROUND_ELT(sharedMemory, h, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp[7] = ROUND_ELT(sharedMemory, h, 7, 6, 5, 4, 3, 2, 1, 0);

		uint64_t tmp2[8];
		uint32_t* n32 = (uint32_t*)n;
		tmp2[0]=xor8(	sharedMemory[__byte_perm(n32[ 0], 0, 0x4440)]  		,sharedMemory[__byte_perm(n32[14], 0, 0x4441) + 256],
				sharedMemory[__byte_perm(n32[12], 0, 0x4442) + 512]	,sharedMemory[__byte_perm(n32[10], 0, 0x4443) + 768],
				sharedMemory[__byte_perm(n32[ 9], 0, 0x4440) + 1024]	,sharedMemory[__byte_perm(n32[ 7], 0, 0x4441) + 1280],
				sharedMemory[__byte_perm(n32[ 5], 0, 0x4442) + 1536]	,tmp[0]);

		tmp2[1]=xor8(	tmp[1]							,sharedMemory[__byte_perm(n32[ 0], 0, 0x4441) + 256],
				sharedMemory[__byte_perm(n32[14], 0, 0x4442) +  512]	,sharedMemory[__byte_perm(n32[12], 0, 0x4443) + 768],
				sharedMemory[__byte_perm(n32[11], 0, 0x4440) + 1024]	,sharedMemory[__byte_perm(n32[ 9], 0, 0x4441) + 1280],
				sharedMemory[__byte_perm(n32[ 7], 0, 0x4442) + 1536]	,sharedMemory[__byte_perm(n32[ 5], 0, 0x4443) + 1792]);

		tmp2[2]=xor8(	sharedMemory[__byte_perm(n32[ 4], 0, 0x4440)]  		,tmp[2]						    ,
				sharedMemory[__byte_perm(n32[ 0], 0, 0x4442) +  512]	,sharedMemory[__byte_perm(n32[14], 0, 0x4443) + 768],
				sharedMemory[__byte_perm(n32[13], 0, 0x4440) + 1024]	,sharedMemory[__byte_perm(n32[11], 0, 0x4441) + 1280],
				sharedMemory[__byte_perm(n32[ 9], 0, 0x4442) + 1536]	,sharedMemory[__byte_perm(n32[ 7], 0, 0x4443) + 1792]);

		tmp2[3]=xor8(	sharedMemory[__byte_perm(n32[ 6], 0, 0x4440)]  		,sharedMemory[__byte_perm(n32[ 4], 0, 0x4441) + 256],
				tmp[3]							,sharedMemory[__byte_perm(n32[ 0], 0, 0x4443) + 768],
				sharedMemory[__byte_perm(n32[15], 0, 0x4440) + 1024]	,sharedMemory[__byte_perm(n32[13], 0, 0x4441) + 1280],
				sharedMemory[__byte_perm(n32[11], 0, 0x4442) + 1536]	,sharedMemory[__byte_perm(n32[ 9], 0, 0x4443) + 1792]);

		tmp2[4]=xor8(	sharedMemory[__byte_perm(n32[ 8], 0, 0x4440)]  		,sharedMemory[__byte_perm(n32[ 6], 0, 0x4441) + 256]  ,
				sharedMemory[__byte_perm(n32[ 4], 0, 0x4442) +  512]	,tmp[4]						      ,
				sharedMemory[__byte_perm(n32[ 1], 0, 0x4440) + 1024]	,sharedMemory[__byte_perm(n32[15], 0, 0x4441) + 1280] ,
				sharedMemory[__byte_perm(n32[13], 0, 0x4442) + 1536]	,sharedMemory[__byte_perm(n32[11], 0, 0x4443) + 1792]);

		tmp2[5]=xor8(	sharedMemory[__byte_perm(n32[10], 0, 0x4440)]  		,sharedMemory[__byte_perm(n32[ 8], 0, 0x4441) + 256],
				sharedMemory[__byte_perm(n32[ 6], 0, 0x4442) +  512]	,sharedMemory[__byte_perm(n32[ 4], 0, 0x4443) + 768],
				tmp[5]							,sharedMemory[__byte_perm(n32[ 1], 0, 0x4441) + 1280],
				sharedMemory[__byte_perm(n32[15], 0, 0x4442) + 1536]	,sharedMemory[__byte_perm(n32[13], 0, 0x4443) + 1792]);

		tmp2[6]=xor8(	sharedMemory[__byte_perm(n32[12], 0, 0x4440)]  		,sharedMemory[__byte_perm(n32[10], 0, 0x4441) + 256],
				sharedMemory[__byte_perm(n32[ 8], 0, 0x4442) +  512]	,sharedMemory[__byte_perm(n32[ 6], 0, 0x4443) + 768],
				sharedMemory[__byte_perm(n32[ 5], 0, 0x4440) + 1024]	,tmp[6],
				sharedMemory[__byte_perm(n32[ 1], 0, 0x4442) + 1536]	,sharedMemory[__byte_perm(n32[15], 0, 0x4443) + 1792]);

		tmp2[7]=xor8(	sharedMemory[__byte_perm(n32[14], 0, 0x4440)]  		,sharedMemory[__byte_perm(n32[12], 0, 0x4441) + 256],
				sharedMemory[__byte_perm(n32[10], 0, 0x4442) +  512]	,sharedMemory[__byte_perm(n32[ 8], 0, 0x4443) + 768],
				sharedMemory[__byte_perm(n32[ 7], 0, 0x4440) + 1024]	,sharedMemory[__byte_perm(n32[ 5], 0, 0x4441) + 1280],
				tmp[7]							,sharedMemory[__byte_perm(n32[ 1], 0, 0x4443) + 1792]);

		n[1] ^= h[1];
		tmp2[1]^=sharedMemory[__byte_perm(n32[2], 0, 0x4440)];
		tmp2[2]^=sharedMemory[__byte_perm(n32[2], 0, 0x4441) + 256];
		tmp2[3]^=sharedMemory[__byte_perm(n32[2], 0, 0x4442) + 512];
		tmp2[4]^=sharedMemory[__byte_perm(n32[2], 0, 0x4443) + 768];

		d_tmp[threadIdx.x]=tmp2[threadIdx.x];

		uint64_t tmp3[8];
		tmp3[0] = xor1(ROUND_ELT(sharedMemory, tmp, 0, 7, 6, 5, 4, 3, 2, 1), InitVector_RC[1]);
		tmp3[1] = ROUND_ELT(sharedMemory, tmp, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp3[2] = ROUND_ELT(sharedMemory, tmp, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp3[3] = ROUND_ELT(sharedMemory, tmp, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp3[4] = ROUND_ELT(sharedMemory, tmp, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp3[5] = ROUND_ELT(sharedMemory, tmp, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp3[6] = ROUND_ELT(sharedMemory, tmp, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp3[7] = ROUND_ELT(sharedMemory, tmp, 7, 6, 5, 4, 3, 2, 1, 0);

		n32 = (uint32_t*)tmp2;
		uint64_t tmp4[8];
		tmp4[0]=(	sharedMemory[__byte_perm(n32[ 9], 0, 0x4440) + 1024]	^sharedMemory[__byte_perm(n32[ 7], 0, 0x4441) + 1280]^
			sharedMemory[__byte_perm(n32[ 5], 0, 0x4442) + 1536]	^sharedMemory[__byte_perm(n32[ 3], 0, 0x4443) + 1792]) ^tmp3[0];

		tmp4[1]=(sharedMemory[__byte_perm(n32[ 2], 0, 0x4440)]		^sharedMemory[__byte_perm(n32[ 9], 0, 0x4441) + 1280]^
			sharedMemory[__byte_perm(n32[ 7], 0, 0x4442) + 1536]	^sharedMemory[__byte_perm(n32[ 5], 0, 0x4443) + 1792]) ^tmp3[1];

		tmp4[2]=(sharedMemory[__byte_perm(n32[ 4], 0, 0x4440)]  	^sharedMemory[__byte_perm(n32[ 2], 0, 0x4441) + 256]^
			sharedMemory[__byte_perm(n32[ 9], 0, 0x4442) + 1536]	^sharedMemory[__byte_perm(n32[ 7], 0, 0x4443) + 1792]) ^tmp3[2];

		tmp4[3]=(sharedMemory[__byte_perm(n32[ 6], 0, 0x4440)]  	^sharedMemory[__byte_perm(n32[ 4], 0, 0x4441) + 256]^
			sharedMemory[__byte_perm(n32[ 2], 0, 0x4442) +  512]	^sharedMemory[__byte_perm(n32[ 9], 0, 0x4443) + 1792]) ^tmp3[3];

		tmp4[4]=(sharedMemory[__byte_perm(n32[ 8], 0, 0x4440)]  	^sharedMemory[__byte_perm(n32[ 6], 0, 0x4441) + 256]^
			sharedMemory[__byte_perm(n32[ 4], 0, 0x4442) +  512]	^sharedMemory[__byte_perm(n32[ 2], 0, 0x4443) + 768]) ^tmp3[4];

		tmp4[5]=(sharedMemory[__byte_perm(n32[ 8], 0, 0x4441) + 256]	^sharedMemory[__byte_perm(n32[ 6], 0, 0x4442) +  512]^
			sharedMemory[__byte_perm(n32[ 4], 0, 0x4443) + 768]	^sharedMemory[__byte_perm(n32[ 3], 0, 0x4440) + 1024]) ^tmp3[5];

		tmp4[6]=(sharedMemory[__byte_perm(n32[ 8], 0, 0x4442) + 512]	^sharedMemory[__byte_perm(n32[ 6], 0, 0x4443) + 768]^
			sharedMemory[__byte_perm(n32[ 5], 0, 0x4440) + 1024]	^sharedMemory[__byte_perm(n32[ 3], 0, 0x4441) + 1280]) ^tmp3[6];

		tmp4[7]=(sharedMemory[__byte_perm(n32[ 8], 0, 0x4443) + 768]	^sharedMemory[__byte_perm(n32[ 7], 0, 0x4440) + 1024]^
			sharedMemory[__byte_perm(n32[ 5], 0, 0x4441) + 1280]	^sharedMemory[__byte_perm(n32[ 3], 0, 0x4442) + 1536]) ^tmp3[7];

		d_tmp[threadIdx.x+16]=tmp4[threadIdx.x];

		uint64_t tmp5[8];
		tmp5[0] = xor1(ROUND_ELT(sharedMemory, tmp3, 0, 7, 6, 5, 4, 3, 2, 1), InitVector_RC[2]);
		tmp5[1] = ROUND_ELT(sharedMemory, tmp3, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp5[2] = ROUND_ELT(sharedMemory, tmp3, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp5[3] = ROUND_ELT(sharedMemory, tmp3, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp5[4] = ROUND_ELT(sharedMemory, tmp3, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp5[5] = ROUND_ELT(sharedMemory, tmp3, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp5[6] = ROUND_ELT(sharedMemory, tmp3, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp5[7] = ROUND_ELT(sharedMemory, tmp3, 7, 6, 5, 4, 3, 2, 1, 0);

		d_tmp[threadIdx.x+8]=tmp5[threadIdx.x];

		uint64_t tmp6[8];
		tmp6[0] = xor1(ROUND_ELT(sharedMemory, tmp5, 0, 7, 6, 5, 4, 3, 2, 1), InitVector_RC[3]);
		tmp6[1] = ROUND_ELT(sharedMemory, tmp5, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp6[2] = ROUND_ELT(sharedMemory, tmp5, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp6[3] = ROUND_ELT(sharedMemory, tmp5, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp6[4] = ROUND_ELT(sharedMemory, tmp5, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp6[5] = ROUND_ELT(sharedMemory, tmp5, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp6[6] = ROUND_ELT(sharedMemory, tmp5, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp6[7] = ROUND_ELT(sharedMemory, tmp5, 7, 6, 5, 4, 3, 2, 1, 0);

		d_tmp[threadIdx.x+24]=tmp6[threadIdx.x];

		uint64_t tmp7[8];
		tmp7[0] = xor1(ROUND_ELT(sharedMemory, tmp6, 0, 7, 6, 5, 4, 3, 2, 1), InitVector_RC[4]);
		tmp7[1] = ROUND_ELT(sharedMemory, tmp6, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp7[2] = ROUND_ELT(sharedMemory, tmp6, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp7[3] = ROUND_ELT(sharedMemory, tmp6, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp7[4] = ROUND_ELT(sharedMemory, tmp6, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp7[5] = ROUND_ELT(sharedMemory, tmp6, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp7[6] = ROUND_ELT(sharedMemory, tmp6, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp7[7] = ROUND_ELT(sharedMemory, tmp6, 7, 6, 5, 4, 3, 2, 1, 0);

		d_tmp[threadIdx.x+32]=tmp7[threadIdx.x];

		uint64_t tmp8[8];
		tmp8[0] = xor1(ROUND_ELT(sharedMemory, tmp7, 0, 7, 6, 5, 4, 3, 2, 1), InitVector_RC[5]);
		tmp8[1] = ROUND_ELT(sharedMemory, tmp7, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp8[2] = ROUND_ELT(sharedMemory, tmp7, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp8[3] = ROUND_ELT(sharedMemory, tmp7, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp8[4] = ROUND_ELT(sharedMemory, tmp7, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp8[5] = ROUND_ELT(sharedMemory, tmp7, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp8[6] = ROUND_ELT(sharedMemory, tmp7, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp8[7] = ROUND_ELT(sharedMemory, tmp7, 7, 6, 5, 4, 3, 2, 1, 0);

		d_tmp[threadIdx.x+40]=tmp8[threadIdx.x];

		uint64_t tmp9[8];
		tmp9[0] = xor1(ROUND_ELT(sharedMemory, tmp8, 0, 7, 6, 5, 4, 3, 2, 1), InitVector_RC[6]);
		tmp9[1] = ROUND_ELT(sharedMemory, tmp8, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp9[2] = ROUND_ELT(sharedMemory, tmp8, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp9[3] = ROUND_ELT(sharedMemory, tmp8, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp9[4] = ROUND_ELT(sharedMemory, tmp8, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp9[5] = ROUND_ELT(sharedMemory, tmp8, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp9[6] = ROUND_ELT(sharedMemory, tmp8, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp9[7] = ROUND_ELT(sharedMemory, tmp8, 7, 6, 5, 4, 3, 2, 1, 0);

		d_tmp[threadIdx.x+48]=tmp9[threadIdx.x];

		uint64_t tmp10[8];
		tmp10[0] = xor1(ROUND_ELT(sharedMemory, tmp9, 0, 7, 6, 5, 4, 3, 2, 1), InitVector_RC[7]);
		tmp10[1] = ROUND_ELT(sharedMemory, tmp9, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp10[2] = ROUND_ELT(sharedMemory, tmp9, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp10[3] = ROUND_ELT(sharedMemory, tmp9, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp10[4] = ROUND_ELT(sharedMemory, tmp9, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp10[5] = ROUND_ELT(sharedMemory, tmp9, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp10[6] = ROUND_ELT(sharedMemory, tmp9, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp10[7] = ROUND_ELT(sharedMemory, tmp9, 7, 6, 5, 4, 3, 2, 1, 0);


		d_tmp[threadIdx.x+56]=tmp10[threadIdx.x];

		uint64_t tmp11[8];
		tmp11[0] = xor1(ROUND_ELT(sharedMemory, tmp10, 0, 7, 6, 5, 4, 3, 2, 1), InitVector_RC[8]);
		tmp11[1] = ROUND_ELT(sharedMemory, tmp10, 1, 0, 7, 6, 5, 4, 3, 2);
		tmp11[2] = ROUND_ELT(sharedMemory, tmp10, 2, 1, 0, 7, 6, 5, 4, 3);
		tmp11[3] = ROUND_ELT(sharedMemory, tmp10, 3, 2, 1, 0, 7, 6, 5, 4);
		tmp11[4] = ROUND_ELT(sharedMemory, tmp10, 4, 3, 2, 1, 0, 7, 6, 5);
		tmp11[5] = ROUND_ELT(sharedMemory, tmp10, 5, 4, 3, 2, 1, 0, 7, 6);
		tmp11[6] = ROUND_ELT(sharedMemory, tmp10, 6, 5, 4, 3, 2, 1, 0, 7);
		tmp11[7] = ROUND_ELT(sharedMemory, tmp10, 7, 6, 5, 4, 3, 2, 1, 0);

		d_tmp[threadIdx.x+64]=tmp11[threadIdx.x];

		if(threadIdx.x==1){
			tmp[0]=ROUND_ELT(sharedMemory,tmp11, 3, 2, 1, 0, 7, 6, 5, 4);
			tmp[1]=ROUND_ELT(sharedMemory,tmp11, 5, 4, 3, 2, 1, 0, 7, 6);
			tmp[4] = xor3(tmp[0],tmp[1],atLastCalc);
			d_xtra[threadIdx.x]=tmp[4];
		}
	}
}

__global__ __launch_bounds__(threadsPerBlock,2)
void whirlpoolx_gpu_hash(uint32_t threads, uint32_t startNounce, uint32_t *resNounce)
{
	__shared__ uint64_t sharedMemory[2048];

	whirlpoolx_getShared(sharedMemory);

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint64_t n[8];
		uint64_t tmp[8];
		uint32_t nounce = startNounce + thread;

		n[1] = xor1(REPLACE_HIDWORD(c_PaddedMessage80[9], cuda_swab32(nounce)),c_xtra[0]);

		uint32_t* n32 = (uint32_t*)&n[0];
		n[0]=sharedMemory[__byte_perm(n32[3], 0, 0x4443) + 1792];
		n[5]=sharedMemory[__byte_perm(n32[3], 0, 0x4440) + 1024];
		n[6]=sharedMemory[__byte_perm(n32[3], 0, 0x4441) + 1280];
		n[7]=sharedMemory[__byte_perm(n32[3], 0, 0x4442) + 1536];
		n[0]=xor1(c_tmp[0],n[0]);
		n[1]=c_tmp[1];
		n[2]=c_tmp[2];
		n[3]=c_tmp[3];
		n[4]=c_tmp[4];
		n[5]=xor1(c_tmp[5],n[5]);
		n[6]=xor1(c_tmp[6],n[6]);
		n[7]=xor1(c_tmp[7],n[7]);

		tmp[0]=xor3(sharedMemory[__byte_perm(n32[10],0,0x4443)+768],sharedMemory[__byte_perm(n32[12],0,0x4442)+512],sharedMemory[__byte_perm(n32[14],0,0x4441)+256]);
		tmp[1]=xor3(sharedMemory[__byte_perm(n32[11],0,0x4440)+1024],sharedMemory[__byte_perm(n32[12],0,0x4443)+768],sharedMemory[__byte_perm(n32[14],0,0x4442)+512]);
		tmp[2]=xor3(sharedMemory[__byte_perm(n32[11],0,0x4441)+1280],sharedMemory[__byte_perm(n32[13],0,0x4440)+1024],sharedMemory[__byte_perm(n32[14],0,0x4443)+768]);
		tmp[3]=xor3(sharedMemory[__byte_perm(n32[11],0,0x4442)+1536],sharedMemory[__byte_perm(n32[13],0,0x4441)+1280],sharedMemory[__byte_perm(n32[15],0,0x4440)+1024]);
		tmp[4]=xor3(sharedMemory[__byte_perm(n32[11],0,0x4443)+1792],sharedMemory[__byte_perm(n32[13],0,0x4442)+1536],sharedMemory[__byte_perm(n32[15],0,0x4441)+1280]);
		tmp[5]=xor3(sharedMemory[__byte_perm(n32[10],0,0x4440)],sharedMemory[__byte_perm(n32[13],0,0x4443)+1792],sharedMemory[__byte_perm(n32[15],0,0x4442)+1536]);
		tmp[6]=xor3(sharedMemory[__byte_perm(n32[12],0,0x4440)],sharedMemory[__byte_perm(n32[10],0,0x4441)+256],sharedMemory[__byte_perm(n32[15],0,0x4443)+1792]);
		tmp[7]=xor3(sharedMemory[__byte_perm(n32[14],0,0x4440)],sharedMemory[__byte_perm(n32[12],0,0x4441)+256],sharedMemory[__byte_perm(n32[10],0,0x4442)+ 512]);

		tmp[0]=xor3(sharedMemory[__byte_perm(n32[ 0], 0, 0x4440)],tmp[0],c_tmp[0+16]);
		tmp[1]=xor3(sharedMemory[__byte_perm(n32[ 0], 0, 0x4441) + 256],tmp[1],c_tmp[1+16]);
		tmp[2]=xor3(sharedMemory[__byte_perm(n32[ 0], 0, 0x4442) +  512],tmp[2],c_tmp[2+16]);
		tmp[3]=xor3(sharedMemory[__byte_perm(n32[ 0], 0, 0x4443) + 768],tmp[3],c_tmp[3+16]);
		tmp[4]=xor3(sharedMemory[__byte_perm(n32[ 1], 0, 0x4440) + 1024],tmp[4],c_tmp[4+16]);
		tmp[5]=xor3(sharedMemory[__byte_perm(n32[ 1], 0, 0x4441) + 1280],tmp[5],c_tmp[5+16]);
		tmp[6]=xor3(sharedMemory[__byte_perm(n32[ 1], 0, 0x4442) + 1536],tmp[6],c_tmp[6+16]);
		tmp[7]=xor3(sharedMemory[__byte_perm(n32[ 1], 0, 0x4443) + 1792],tmp[7],c_tmp[7+16]);

		n[0]=tmp[0];
		n[1]=tmp[1];
		n[2]=tmp[2];
		n[3]=tmp[3];
		n[4]=tmp[4];
		n[5]=tmp[5];
		n[6]=tmp[6];
		n[7]=tmp[7];

		tmp[0] = xor1(ROUND_ELT(sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1), c_tmp[0+8]);
		tmp[1] = xor1(ROUND_ELT(sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2), c_tmp[1+8]);
		tmp[2] = xor1(ROUND_ELT(sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3), c_tmp[2+8]);
		tmp[3] = xor1(ROUND_ELT(sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4), c_tmp[3+8]);
		tmp[4] = xor1(ROUND_ELT(sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5), c_tmp[4+8]);
		tmp[5] = xor1(ROUND_ELT(sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6), c_tmp[5+8]);
		tmp[6] = xor1(ROUND_ELT(sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7), c_tmp[6+8]);
		tmp[7] = xor1(ROUND_ELT(sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0), c_tmp[7+8]);

		n[0] = xor1(ROUND_ELT(sharedMemory, tmp, 0, 7, 6, 5, 4, 3, 2, 1), c_tmp[0+24]);
		n[1] = xor1(ROUND_ELT(sharedMemory, tmp, 1, 0, 7, 6, 5, 4, 3, 2), c_tmp[1+24]);
		n[2] = xor1(ROUND_ELT(sharedMemory, tmp, 2, 1, 0, 7, 6, 5, 4, 3), c_tmp[2+24]);
		n[3] = xor1(ROUND_ELT(sharedMemory, tmp, 3, 2, 1, 0, 7, 6, 5, 4), c_tmp[3+24]);
		n[4] = xor1(ROUND_ELT(sharedMemory, tmp, 4, 3, 2, 1, 0, 7, 6, 5), c_tmp[4+24]);
		n[5] = xor1(ROUND_ELT(sharedMemory, tmp, 5, 4, 3, 2, 1, 0, 7, 6), c_tmp[5+24]);
		n[6] = xor1(ROUND_ELT(sharedMemory, tmp, 6, 5, 4, 3, 2, 1, 0, 7), c_tmp[6+24]);
		n[7] = xor1(ROUND_ELT(sharedMemory, tmp, 7, 6, 5, 4, 3, 2, 1, 0), c_tmp[7+24]);

		tmp[0] = xor1(ROUND_ELT(sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1), c_tmp[0+32]);
		tmp[1] = xor1(ROUND_ELT(sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2), c_tmp[1+32]);
		tmp[2] = xor1(ROUND_ELT(sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3), c_tmp[2+32]);
		tmp[3] = xor1(ROUND_ELT(sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4), c_tmp[3+32]);
		tmp[4] = xor1(ROUND_ELT(sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5), c_tmp[4+32]);
		tmp[5] = xor1(ROUND_ELT(sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6), c_tmp[5+32]);
		tmp[6] = xor1(ROUND_ELT(sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7), c_tmp[6+32]);
		tmp[7] = xor1(ROUND_ELT(sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0), c_tmp[7+32]);

		n[0] = xor1(ROUND_ELT(sharedMemory, tmp, 0, 7, 6, 5, 4, 3, 2, 1), c_tmp[0+40]);
		n[1] = xor1(ROUND_ELT(sharedMemory, tmp, 1, 0, 7, 6, 5, 4, 3, 2), c_tmp[1+40]);
		n[2] = xor1(ROUND_ELT(sharedMemory, tmp, 2, 1, 0, 7, 6, 5, 4, 3), c_tmp[2+40]);
		n[3] = xor1(ROUND_ELT(sharedMemory, tmp, 3, 2, 1, 0, 7, 6, 5, 4), c_tmp[3+40]);
		n[4] = xor1(ROUND_ELT(sharedMemory, tmp, 4, 3, 2, 1, 0, 7, 6, 5), c_tmp[4+40]);
		n[5] = xor1(ROUND_ELT(sharedMemory, tmp, 5, 4, 3, 2, 1, 0, 7, 6), c_tmp[5+40]);
		n[6] = xor1(ROUND_ELT(sharedMemory, tmp, 6, 5, 4, 3, 2, 1, 0, 7), c_tmp[6+40]);
		n[7] = xor1(ROUND_ELT(sharedMemory, tmp, 7, 6, 5, 4, 3, 2, 1, 0), c_tmp[7+40]);

		tmp[0] = xor1(ROUND_ELT(sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1), c_tmp[0+48]);
		tmp[1] = xor1(ROUND_ELT(sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2), c_tmp[1+48]);
		tmp[2] = xor1(ROUND_ELT(sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3), c_tmp[2+48]);
		tmp[3] = xor1(ROUND_ELT(sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4), c_tmp[3+48]);
		tmp[4] = xor1(ROUND_ELT(sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5), c_tmp[4+48]);
		tmp[5] = xor1(ROUND_ELT(sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6), c_tmp[5+48]);
		tmp[6] = xor1(ROUND_ELT(sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7), c_tmp[6+48]);
		tmp[7] = xor1(ROUND_ELT(sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0), c_tmp[7+48]);

		n[0] = xor1(ROUND_ELT(sharedMemory, tmp, 0, 7, 6, 5, 4, 3, 2, 1), c_tmp[0+56]);
		n[1] = xor1(ROUND_ELT(sharedMemory, tmp, 1, 0, 7, 6, 5, 4, 3, 2), c_tmp[1+56]);
		n[2] = xor1(ROUND_ELT(sharedMemory, tmp, 2, 1, 0, 7, 6, 5, 4, 3), c_tmp[2+56]);
		n[3] = xor1(ROUND_ELT(sharedMemory, tmp, 3, 2, 1, 0, 7, 6, 5, 4), c_tmp[3+56]);
		n[4] = xor1(ROUND_ELT(sharedMemory, tmp, 4, 3, 2, 1, 0, 7, 6, 5), c_tmp[4+56]);
		n[5] = xor1(ROUND_ELT(sharedMemory, tmp, 5, 4, 3, 2, 1, 0, 7, 6), c_tmp[5+56]);
		n[6] = xor1(ROUND_ELT(sharedMemory, tmp, 6, 5, 4, 3, 2, 1, 0, 7), c_tmp[6+56]);
		n[7] = xor1(ROUND_ELT(sharedMemory, tmp, 7, 6, 5, 4, 3, 2, 1, 0), c_tmp[7+56]);

		tmp[0] = xor1(ROUND_ELT(sharedMemory, n, 0, 7, 6, 5, 4, 3, 2, 1), c_tmp[0+64]);
		tmp[1] = xor1(ROUND_ELT(sharedMemory, n, 1, 0, 7, 6, 5, 4, 3, 2), c_tmp[1+64]);
		tmp[2] = xor1(ROUND_ELT(sharedMemory, n, 2, 1, 0, 7, 6, 5, 4, 3), c_tmp[2+64]);
		tmp[3] = xor1(ROUND_ELT(sharedMemory, n, 3, 2, 1, 0, 7, 6, 5, 4), c_tmp[3+64]);
		tmp[4] = xor1(ROUND_ELT(sharedMemory, n, 4, 3, 2, 1, 0, 7, 6, 5), c_tmp[4+64]);
		tmp[5] = xor1(ROUND_ELT(sharedMemory, n, 5, 4, 3, 2, 1, 0, 7, 6), c_tmp[5+64]);
		tmp[6] = xor1(ROUND_ELT(sharedMemory, n, 6, 5, 4, 3, 2, 1, 0, 7), c_tmp[6+64]);
		tmp[7] = xor1(ROUND_ELT(sharedMemory, n, 7, 6, 5, 4, 3, 2, 1, 0), c_tmp[7+64]);

		if (xor3(c_xtra[1], ROUND_ELT(sharedMemory, tmp, 3, 2, 1, 0, 7, 6, 5, 4), ROUND_ELT(sharedMemory, tmp, 5, 4, 3, 2, 1, 0, 7, 6)) <= pTarget[3]) {
			atomicMin(&resNounce[0], nounce);
		}
	}
}

__host__
extern void whirlpoolx_cpu_init(int thr_id, uint32_t threads)
{
	cudaMemcpyToSymbol(InitVector_RC, plain_RC, sizeof(plain_RC), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(mixTob0Tox, plain_T0, sizeof(plain_T0), 0, cudaMemcpyHostToDevice);
	cudaMalloc(&d_WXNonce[thr_id], sizeof(uint32_t));
	cudaMallocHost(&h_wxnounce[thr_id], sizeof(uint32_t));
	cudaMalloc(&d_xtra[thr_id], 8 * sizeof(uint64_t));
	CUDA_SAFE_CALL(cudaMalloc(&d_tmp[thr_id], 8 * 9 * sizeof(uint64_t))); // d_tmp[threadIdx.x+64] (7+64)
}

__host__
extern void whirlpoolx_cpu_free(int thr_id)
{
	cudaFree(d_WXNonce[thr_id]);
	cudaFreeHost(h_wxnounce[thr_id]);
	cudaFree(d_xtra[thr_id]);
	cudaFree(d_tmp[thr_id]);
}

__host__
void whirlpoolx_setBlock_80(void *pdata, const void *ptarget)
{
	uint64_t PaddedMessage[16];
	memcpy(PaddedMessage, pdata, 80);
	memset((uint8_t*)&PaddedMessage+80, 0, 48);
	((uint8_t*)PaddedMessage)[80] = 0x80; /* ending */
	cudaMemcpyToSymbol(pTarget, ptarget, 4*sizeof(uint64_t), 0, cudaMemcpyHostToDevice);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_PaddedMessage80, PaddedMessage, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

__host__
void whirlpoolx_precompute(int thr_id)
{
	dim3 grid(1);
	dim3 block(256);

	whirlpoolx_gpu_precompute <<<grid, block>>>(8, d_xtra[thr_id], d_tmp[thr_id]);
	cudaDeviceSynchronize();

	cudaMemcpyToSymbol(c_xtra, d_xtra[thr_id], 8 * sizeof(uint64_t), 0, cudaMemcpyDeviceToDevice);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_tmp, d_tmp[thr_id], 8 * 9 * sizeof(uint64_t), 0, cudaMemcpyDeviceToDevice));
}

__host__
uint32_t whirlpoolx_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce)
{
	dim3 grid((threads + threadsPerBlock-1) / threadsPerBlock);
	dim3 block(threadsPerBlock);

	cudaMemset(d_WXNonce[thr_id], 0xff, sizeof(uint32_t));

	whirlpoolx_gpu_hash<<<grid, block>>>(threads, startNounce, d_WXNonce[thr_id]);
	cudaDeviceSynchronize();

	cudaMemcpy(h_wxnounce[thr_id], d_WXNonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	return *(h_wxnounce[thr_id]);
}
