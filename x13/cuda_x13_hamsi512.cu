/*
 * Quick Hamsi-512 for X13 by tsiv - 2014
 * + Hamsi-512 80 by tpruvot - 2018
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include "cuda_helper.h"

typedef unsigned char BitSequence;

static __constant__ uint32_t d_alpha_n[32];
static __constant__ uint32_t d_alpha_f[32];
static __constant__ uint32_t d_T512[64][16];

static const uint32_t alpha_n[] = {
	0xff00f0f0, 0xccccaaaa, 0xf0f0cccc, 0xff00aaaa, 0xccccaaaa, 0xf0f0ff00, 0xaaaacccc, 0xf0f0ff00,
	0xf0f0cccc, 0xaaaaff00, 0xccccff00, 0xaaaaf0f0, 0xaaaaf0f0, 0xff00cccc, 0xccccf0f0, 0xff00aaaa,
	0xccccaaaa, 0xff00f0f0, 0xff00aaaa, 0xf0f0cccc, 0xf0f0ff00, 0xccccaaaa, 0xf0f0ff00, 0xaaaacccc,
	0xaaaaff00, 0xf0f0cccc, 0xaaaaf0f0, 0xccccff00, 0xff00cccc, 0xaaaaf0f0, 0xff00aaaa, 0xccccf0f0
};

static const uint32_t alpha_f[] = {
	0xcaf9639c, 0x0ff0f9c0, 0x639c0ff0, 0xcaf9f9c0, 0x0ff0f9c0, 0x639ccaf9, 0xf9c00ff0, 0x639ccaf9,
	0x639c0ff0, 0xf9c0caf9, 0x0ff0caf9, 0xf9c0639c, 0xf9c0639c, 0xcaf90ff0, 0x0ff0639c, 0xcaf9f9c0,
	0x0ff0f9c0, 0xcaf9639c, 0xcaf9f9c0, 0x639c0ff0, 0x639ccaf9, 0x0ff0f9c0, 0x639ccaf9, 0xf9c00ff0,
	0xf9c0caf9, 0x639c0ff0, 0xf9c0639c, 0x0ff0caf9, 0xcaf90ff0, 0xf9c0639c, 0xcaf9f9c0, 0x0ff0639c
};

#define hamsi_s00   m0
#define hamsi_s01   m1
#define hamsi_s02   c0
#define hamsi_s03   c1
#define hamsi_s04   m2
#define hamsi_s05   m3
#define hamsi_s06   c2
#define hamsi_s07   c3
#define hamsi_s08   c4
#define hamsi_s09   c5
#define hamsi_s0A   m4
#define hamsi_s0B   m5
#define hamsi_s0C   c6
#define hamsi_s0D   c7
#define hamsi_s0E   m6
#define hamsi_s0F   m7
#define hamsi_s10   m8
#define hamsi_s11   m9
#define hamsi_s12   c8
#define hamsi_s13   c9
#define hamsi_s14   mA
#define hamsi_s15   mB
#define hamsi_s16   cA
#define hamsi_s17   cB
#define hamsi_s18   cC
#define hamsi_s19   cD
#define hamsi_s1A   mC
#define hamsi_s1B   mD
#define hamsi_s1C   cE
#define hamsi_s1D   cF
#define hamsi_s1E   mE
#define hamsi_s1F   mF

#define SBOX(a, b, c, d) { \
		uint32_t t; \
		t = (a); \
		(a) &= (c); \
		(a) ^= (d); \
		(c) ^= (b); \
		(c) ^= (a); \
		(d) |= t; \
		(d) ^= (b); \
		t ^= (c); \
		(b) = (d); \
		(d) |= t; \
		(d) ^= (a); \
		(a) &= (b); \
		t ^= (a); \
		(b) ^= (d); \
		(b) ^= t; \
		(a) = (c); \
		(c) = (b); \
		(b) = (d); \
		(d) = SPH_T32(~t); \
	}

#define HAMSI_L(a, b, c, d) { \
		(a) = ROTL32(a, 13); \
		(c) = ROTL32(c, 3); \
		(b) ^= (a) ^ (c); \
		(d) ^= (c) ^ ((a) << 3); \
		(b) = ROTL32(b, 1); \
		(d) = ROTL32(d, 7); \
		(a) ^= (b) ^ (d); \
		(c) ^= (d) ^ ((b) << 7); \
		(a) = ROTL32(a, 5); \
		(c) = ROTL32(c, 22); \
	}

#define ROUND_BIG(rc, alpha) { \
		hamsi_s00 ^= alpha[0x00]; \
		hamsi_s08 ^= alpha[0x08]; \
		hamsi_s10 ^= alpha[0x10]; \
		hamsi_s18 ^= alpha[0x18]; \
		hamsi_s01 ^= alpha[0x01] ^ (uint32_t)(rc); \
		hamsi_s09 ^= alpha[0x09]; \
		hamsi_s11 ^= alpha[0x11]; \
		hamsi_s19 ^= alpha[0x19]; \
		hamsi_s02 ^= alpha[0x02]; \
		hamsi_s0A ^= alpha[0x0A]; \
		hamsi_s12 ^= alpha[0x12]; \
		hamsi_s1A ^= alpha[0x1A]; \
		hamsi_s03 ^= alpha[0x03]; \
		hamsi_s0B ^= alpha[0x0B]; \
		hamsi_s13 ^= alpha[0x13]; \
		hamsi_s1B ^= alpha[0x1B]; \
		hamsi_s04 ^= alpha[0x04]; \
		hamsi_s0C ^= alpha[0x0C]; \
		hamsi_s14 ^= alpha[0x14]; \
		hamsi_s1C ^= alpha[0x1C]; \
		hamsi_s05 ^= alpha[0x05]; \
		hamsi_s0D ^= alpha[0x0D]; \
		hamsi_s15 ^= alpha[0x15]; \
		hamsi_s1D ^= alpha[0x1D]; \
		hamsi_s06 ^= alpha[0x06]; \
		hamsi_s0E ^= alpha[0x0E]; \
		hamsi_s16 ^= alpha[0x16]; \
		hamsi_s1E ^= alpha[0x1E]; \
		hamsi_s07 ^= alpha[0x07]; \
		hamsi_s0F ^= alpha[0x0F]; \
		hamsi_s17 ^= alpha[0x17]; \
		hamsi_s1F ^= alpha[0x1F]; \
		SBOX(hamsi_s00, hamsi_s08, hamsi_s10, hamsi_s18); \
		SBOX(hamsi_s01, hamsi_s09, hamsi_s11, hamsi_s19); \
		SBOX(hamsi_s02, hamsi_s0A, hamsi_s12, hamsi_s1A); \
		SBOX(hamsi_s03, hamsi_s0B, hamsi_s13, hamsi_s1B); \
		SBOX(hamsi_s04, hamsi_s0C, hamsi_s14, hamsi_s1C); \
		SBOX(hamsi_s05, hamsi_s0D, hamsi_s15, hamsi_s1D); \
		SBOX(hamsi_s06, hamsi_s0E, hamsi_s16, hamsi_s1E); \
		SBOX(hamsi_s07, hamsi_s0F, hamsi_s17, hamsi_s1F); \
		HAMSI_L(hamsi_s00, hamsi_s09, hamsi_s12, hamsi_s1B); \
		HAMSI_L(hamsi_s01, hamsi_s0A, hamsi_s13, hamsi_s1C); \
		HAMSI_L(hamsi_s02, hamsi_s0B, hamsi_s14, hamsi_s1D); \
		HAMSI_L(hamsi_s03, hamsi_s0C, hamsi_s15, hamsi_s1E); \
		HAMSI_L(hamsi_s04, hamsi_s0D, hamsi_s16, hamsi_s1F); \
		HAMSI_L(hamsi_s05, hamsi_s0E, hamsi_s17, hamsi_s18); \
		HAMSI_L(hamsi_s06, hamsi_s0F, hamsi_s10, hamsi_s19); \
		HAMSI_L(hamsi_s07, hamsi_s08, hamsi_s11, hamsi_s1A); \
		HAMSI_L(hamsi_s00, hamsi_s02, hamsi_s05, hamsi_s07); \
		HAMSI_L(hamsi_s10, hamsi_s13, hamsi_s15, hamsi_s16); \
		HAMSI_L(hamsi_s09, hamsi_s0B, hamsi_s0C, hamsi_s0E); \
		HAMSI_L(hamsi_s19, hamsi_s1A, hamsi_s1C, hamsi_s1F); \
	}


#define P_BIG  { \
		for( int r = 0; r < 6; r++ ) \
			ROUND_BIG(r, d_alpha_n); \
	}

#define PF_BIG { \
		for( int r = 0; r < 12; r++ ) \
			ROUND_BIG(r, d_alpha_f); \
	}

#define T_BIG  { \
		/* order is important */ \
		cF = (h[0xF] ^= hamsi_s17); \
		cE = (h[0xE] ^= hamsi_s16); \
		cD = (h[0xD] ^= hamsi_s15); \
		cC = (h[0xC] ^= hamsi_s14); \
		cB = (h[0xB] ^= hamsi_s13); \
		cA = (h[0xA] ^= hamsi_s12); \
		c9 = (h[0x9] ^= hamsi_s11); \
		c8 = (h[0x8] ^= hamsi_s10); \
		c7 = (h[0x7] ^= hamsi_s07); \
		c6 = (h[0x6] ^= hamsi_s06); \
		c5 = (h[0x5] ^= hamsi_s05); \
		c4 = (h[0x4] ^= hamsi_s04); \
		c3 = (h[0x3] ^= hamsi_s03); \
		c2 = (h[0x2] ^= hamsi_s02); \
		c1 = (h[0x1] ^= hamsi_s01); \
		c0 = (h[0x0] ^= hamsi_s00); \
	}


static const uint32_t T512[64][16] = {
{ 0xef0b0270, 0x3afd0000, 0x5dae0000, 0x69490000, 0x9b0f3c06, 0x4405b5f9, 0x66140a51, 0x924f5d0a, // 0
  0xc96b0030, 0xe7250000, 0x2f840000, 0x264f0000, 0x08695bf9, 0x6dfcf137, 0x509f6984, 0x9e69af68 },
{ 0xc96b0030, 0xe7250000, 0x2f840000, 0x264f0000, 0x08695bf9, 0x6dfcf137, 0x509f6984, 0x9e69af68,
  0x26600240, 0xddd80000, 0x722a0000, 0x4f060000, 0x936667ff, 0x29f944ce, 0x368b63d5, 0x0c26f262 },
{ 0x145a3c00, 0xb9e90000, 0x61270000, 0xf1610000, 0xce613d6c, 0xb0493d78, 0x47a96720, 0xe18e24c5,
  0x23671400, 0xc8b90000, 0xf4c70000, 0xfb750000, 0x73cd2465, 0xf8a6a549, 0x02c40a3f, 0xdc24e61f },
{ 0x23671400, 0xc8b90000, 0xf4c70000, 0xfb750000, 0x73cd2465, 0xf8a6a549, 0x02c40a3f, 0xdc24e61f,
  0x373d2800, 0x71500000, 0x95e00000, 0x0a140000, 0xbdac1909, 0x48ef9831, 0x456d6d1f, 0x3daac2da },
{ 0x54285c00, 0xeaed0000, 0xc5d60000, 0xa1c50000, 0xb3a26770, 0x94a5c4e1, 0x6bb0419d, 0x551b3782,
  0x9cbb1800, 0xb0d30000, 0x92510000, 0xed930000, 0x593a4345, 0xe114d5f4, 0x430633da, 0x78cace29 },
{ 0x9cbb1800, 0xb0d30000, 0x92510000, 0xed930000, 0x593a4345, 0xe114d5f4, 0x430633da, 0x78cace29,
  0xc8934400, 0x5a3e0000, 0x57870000, 0x4c560000, 0xea982435, 0x75b11115, 0x28b67247, 0x2dd1f9ab },
{ 0x29449c00, 0x64e70000, 0xf24b0000, 0xc2f30000, 0x0ede4e8f, 0x56c23745, 0xf3e04259, 0x8d0d9ec4,
  0x466d0c00, 0x08620000, 0xdd5d0000, 0xbadd0000, 0x6a927942, 0x441f2b93, 0x218ace6f, 0xbf2c0be2 },
{ 0x466d0c00, 0x08620000, 0xdd5d0000, 0xbadd0000, 0x6a927942, 0x441f2b93, 0x218ace6f, 0xbf2c0be2, // 7
  0x6f299000, 0x6c850000, 0x2f160000, 0x782e0000, 0x644c37cd, 0x12dd1cd6, 0xd26a8c36, 0x32219526 },
{ 0xf6800005, 0x3443c000, 0x24070000, 0x8f3d0000, 0x21373bfb, 0x0ab8d5ae, 0xcdc58b19, 0xd795ba31,
  0xa67f0001, 0x71378000, 0x19fc0000, 0x96db0000, 0x3a8b6dfd, 0xebcaaef3, 0x2c6d478f, 0xac8e6c88 },
{ 0xa67f0001, 0x71378000, 0x19fc0000, 0x96db0000, 0x3a8b6dfd, 0xebcaaef3, 0x2c6d478f, 0xac8e6c88,
  0x50ff0004, 0x45744000, 0x3dfb0000, 0x19e60000, 0x1bbc5606, 0xe1727b5d, 0xe1a8cc96, 0x7b1bd6b9 },
{ 0xf7750009, 0xcf3cc000, 0xc3d60000, 0x04920000, 0x029519a9, 0xf8e836ba, 0x7a87f14e, 0x9e16981a,
  0xd46a0000, 0x8dc8c000, 0xa5af0000, 0x4a290000, 0xfc4e427a, 0xc9b4866c, 0x98369604, 0xf746c320 },
{ 0xd46a0000, 0x8dc8c000, 0xa5af0000, 0x4a290000, 0xfc4e427a, 0xc9b4866c, 0x98369604, 0xf746c320,
  0x231f0009, 0x42f40000, 0x66790000, 0x4ebb0000, 0xfedb5bd3, 0x315cb0d6, 0xe2b1674a, 0x69505b3a },
{ 0x774400f0, 0xf15a0000, 0xf5b20000, 0x34140000, 0x89377e8c, 0x5a8bec25, 0x0bc3cd1e, 0xcf3775cb,
  0xf46c0050, 0x96180000, 0x14a50000, 0x031f0000, 0x42947eb8, 0x66bf7e19, 0x9ca470d2, 0x8a341574 },
{ 0xf46c0050, 0x96180000, 0x14a50000, 0x031f0000, 0x42947eb8, 0x66bf7e19, 0x9ca470d2, 0x8a341574,
  0x832800a0, 0x67420000, 0xe1170000, 0x370b0000, 0xcba30034, 0x3c34923c, 0x9767bdcc, 0x450360bf },
{ 0xe8870170, 0x9d720000, 0x12db0000, 0xd4220000, 0xf2886b27, 0xa921e543, 0x4ef8b518, 0x618813b1, // 14
  0xb4370060, 0x0c4c0000, 0x56c20000, 0x5cae0000, 0x94541f3f, 0x3b3ef825, 0x1b365f3d, 0xf3d45758 },
{ 0xb4370060, 0x0c4c0000, 0x56c20000, 0x5cae0000, 0x94541f3f, 0x3b3ef825, 0x1b365f3d, 0xf3d45758,
  0x5cb00110, 0x913e0000, 0x44190000, 0x888c0000, 0x66dc7418, 0x921f1d66, 0x55ceea25, 0x925c44e9 },
{ 0x0c720000, 0x49e50f00, 0x42790000, 0x5cea0000, 0x33aa301a, 0x15822514, 0x95a34b7b, 0xb44b0090,
  0xfe220000, 0xa7580500, 0x25d10000, 0xf7600000, 0x893178da, 0x1fd4f860, 0x4ed0a315, 0xa123ff9f },
{ 0xfe220000, 0xa7580500, 0x25d10000, 0xf7600000, 0x893178da, 0x1fd4f860, 0x4ed0a315, 0xa123ff9f,
  0xf2500000, 0xeebd0a00, 0x67a80000, 0xab8a0000, 0xba9b48c0, 0x0a56dd74, 0xdb73e86e, 0x1568ff0f },
{ 0x45180000, 0xa5b51700, 0xf96a0000, 0x3b480000, 0x1ecc142c, 0x231395d6, 0x16bca6b0, 0xdf33f4df,
  0xb83d0000, 0x16710600, 0x379a0000, 0xf5b10000, 0x228161ac, 0xae48f145, 0x66241616, 0xc5c1eb3e },
{ 0xb83d0000, 0x16710600, 0x379a0000, 0xf5b10000, 0x228161ac, 0xae48f145, 0x66241616, 0xc5c1eb3e,
  0xfd250000, 0xb3c41100, 0xcef00000, 0xcef90000, 0x3c4d7580, 0x8d5b6493, 0x7098b0a6, 0x1af21fe1 },
{ 0x75a40000, 0xc28b2700, 0x94a40000, 0x90f50000, 0xfb7857e0, 0x49ce0bae, 0x1767c483, 0xaedf667e,
  0xd1660000, 0x1bbc0300, 0x9eec0000, 0xf6940000, 0x03024527, 0xcf70fcf2, 0xb4431b17, 0x857f3c2b },
{ 0xd1660000, 0x1bbc0300, 0x9eec0000, 0xf6940000, 0x03024527, 0xcf70fcf2, 0xb4431b17, 0x857f3c2b, // 21
  0xa4c20000, 0xd9372400, 0x0a480000, 0x66610000, 0xf87a12c7, 0x86bef75c, 0xa324df94, 0x2ba05a55 },
{ 0x75c90003, 0x0e10c000, 0xd1200000, 0xbaea0000, 0x8bc42f3e, 0x8758b757, 0xbb28761d, 0x00b72e2b,
  0xeecf0001, 0x6f564000, 0xf33e0000, 0xa79e0000, 0xbdb57219, 0xb711ebc5, 0x4a3b40ba, 0xfeabf254 },
{ 0xeecf0001, 0x6f564000, 0xf33e0000, 0xa79e0000, 0xbdb57219, 0xb711ebc5, 0x4a3b40ba, 0xfeabf254,
  0x9b060002, 0x61468000, 0x221e0000, 0x1d740000, 0x36715d27, 0x30495c92, 0xf11336a7, 0xfe1cdc7f },
{ 0x86790000, 0x3f390002, 0xe19ae000, 0x98560000, 0x9565670e, 0x4e88c8ea, 0xd3dd4944, 0x161ddab9,
  0x30b70000, 0xe5d00000, 0xf4f46000, 0x42c40000, 0x63b83d6a, 0x78ba9460, 0x21afa1ea, 0xb0a51834 },
{ 0x30b70000, 0xe5d00000, 0xf4f46000, 0x42c40000, 0x63b83d6a, 0x78ba9460, 0x21afa1ea, 0xb0a51834,
  0xb6ce0000, 0xdae90002, 0x156e8000, 0xda920000, 0xf6dd5a64, 0x36325c8a, 0xf272e8ae, 0xa6b8c28d },
{ 0x14190000, 0x23ca003c, 0x50df0000, 0x44b60000, 0x1b6c67b0, 0x3cf3ac75, 0x61e610b0, 0xdbcadb80,
  0xe3430000, 0x3a4e0014, 0xf2c60000, 0xaa4e0000, 0xdb1e42a6, 0x256bbe15, 0x123db156, 0x3a4e99d7 },
{ 0xe3430000, 0x3a4e0014, 0xf2c60000, 0xaa4e0000, 0xdb1e42a6, 0x256bbe15, 0x123db156, 0x3a4e99d7,
  0xf75a0000, 0x19840028, 0xa2190000, 0xeef80000, 0xc0722516, 0x19981260, 0x73dba1e6, 0xe1844257 },
{ 0x54500000, 0x0671005c, 0x25ae0000, 0x6a1e0000, 0x2ea54edf, 0x664e8512, 0xbfba18c3, 0x7e715d17, // 28
  0xbc8d0000, 0xfc3b0018, 0x19830000, 0xd10b0000, 0xae1878c4, 0x42a69856, 0x0012da37, 0x2c3b504e },
{ 0xbc8d0000, 0xfc3b0018, 0x19830000, 0xd10b0000, 0xae1878c4, 0x42a69856, 0x0012da37, 0x2c3b504e,
  0xe8dd0000, 0xfa4a0044, 0x3c2d0000, 0xbb150000, 0x80bd361b, 0x24e81d44, 0xbfa8c2f4, 0x524a0d59 },
{ 0x69510000, 0xd4e1009c, 0xc3230000, 0xac2f0000, 0xe4950bae, 0xcea415dc, 0x87ec287c, 0xbce1a3ce,
  0xc6730000, 0xaf8d000c, 0xa4c10000, 0x218d0000, 0x23111587, 0x7913512f, 0x1d28ac88, 0x378dd173 },
{ 0xc6730000, 0xaf8d000c, 0xa4c10000, 0x218d0000, 0x23111587, 0x7913512f, 0x1d28ac88, 0x378dd173,
  0xaf220000, 0x7b6c0090, 0x67e20000, 0x8da20000, 0xc7841e29, 0xb7b744f3, 0x9ac484f4, 0x8b6c72bd },
{ 0xcc140000, 0xa5630000, 0x5ab90780, 0x3b500000, 0x4bd013ff, 0x879b3418, 0x694348c1, 0xca5a87fe,
  0x819e0000, 0xec570000, 0x66320280, 0x95f30000, 0x5da92802, 0x48f43cbc, 0xe65aa22d, 0x8e67b7fa },
{ 0x819e0000, 0xec570000, 0x66320280, 0x95f30000, 0x5da92802, 0x48f43cbc, 0xe65aa22d, 0x8e67b7fa,
  0x4d8a0000, 0x49340000, 0x3c8b0500, 0xaea30000, 0x16793bfd, 0xcf6f08a4, 0x8f19eaec, 0x443d3004 },
{ 0x78230000, 0x12fc0000, 0xa93a0b80, 0x90a50000, 0x713e2879, 0x7ee98924, 0xf08ca062, 0x636f8bab,
  0x02af0000, 0xb7280000, 0xba1c0300, 0x56980000, 0xba8d45d3, 0x8048c667, 0xa95c149a, 0xf4f6ea7b },
{ 0x02af0000, 0xb7280000, 0xba1c0300, 0x56980000, 0xba8d45d3, 0x8048c667, 0xa95c149a, 0xf4f6ea7b, // 35
  0x7a8c0000, 0xa5d40000, 0x13260880, 0xc63d0000, 0xcbb36daa, 0xfea14f43, 0x59d0b4f8, 0x979961d0 },
{ 0xac480000, 0x1ba60000, 0x45fb1380, 0x03430000, 0x5a85316a, 0x1fb250b6, 0xfe72c7fe, 0x91e478f6,
  0x1e4e0000, 0xdecf0000, 0x6df80180, 0x77240000, 0xec47079e, 0xf4a0694e, 0xcda31812, 0x98aa496e },
{ 0x1e4e0000, 0xdecf0000, 0x6df80180, 0x77240000, 0xec47079e, 0xf4a0694e, 0xcda31812, 0x98aa496e,
  0xb2060000, 0xc5690000, 0x28031200, 0x74670000, 0xb6c236f4, 0xeb1239f8, 0x33d1dfec, 0x094e3198 },
{ 0xaec30000, 0x9c4f0001, 0x79d1e000, 0x2c150000, 0x45cc75b3, 0x6650b736, 0xab92f78f, 0xa312567b,
  0xdb250000, 0x09290000, 0x49aac000, 0x81e10000, 0xcafe6b59, 0x42793431, 0x43566b76, 0xe86cba2e },
{ 0xdb250000, 0x09290000, 0x49aac000, 0x81e10000, 0xcafe6b59, 0x42793431, 0x43566b76, 0xe86cba2e,
  0x75e60000, 0x95660001, 0x307b2000, 0xadf40000, 0x8f321eea, 0x24298307, 0xe8c49cf9, 0x4b7eec55 },
{ 0x58430000, 0x807e0000, 0x78330001, 0xc66b3800, 0xe7375cdc, 0x79ad3fdd, 0xac73fe6f, 0x3a4479b1,
  0x1d5a0000, 0x2b720000, 0x488d0000, 0xaf611800, 0x25cb2ec5, 0xc879bfd0, 0x81a20429, 0x1e7536a6 },
{ 0x1d5a0000, 0x2b720000, 0x488d0000, 0xaf611800, 0x25cb2ec5, 0xc879bfd0, 0x81a20429, 0x1e7536a6,
  0x45190000, 0xab0c0000, 0x30be0001, 0x690a2000, 0xc2fc7219, 0xb1d4800d, 0x2dd1fa46, 0x24314f17 },
{ 0xa53b0000, 0x14260000, 0x4e30001e, 0x7cae0000, 0x8f9e0dd5, 0x78dfaa3d, 0xf73168d8, 0x0b1b4946, // 42
  0x07ed0000, 0xb2500000, 0x8774000a, 0x970d0000, 0x437223ae, 0x48c76ea4, 0xf4786222, 0x9075b1ce },
{ 0x07ed0000, 0xb2500000, 0x8774000a, 0x970d0000, 0x437223ae, 0x48c76ea4, 0xf4786222, 0x9075b1ce,
  0xa2d60000, 0xa6760000, 0xc9440014, 0xeba30000, 0xccec2e7b, 0x3018c499, 0x03490afa, 0x9b6ef888 },
{ 0x88980000, 0x1f940000, 0x7fcf002e, 0xfb4e0000, 0xf158079a, 0x61ae9167, 0xa895706c, 0xe6107494,
  0x0bc20000, 0xdb630000, 0x7e88000c, 0x15860000, 0x91fd48f3, 0x7581bb43, 0xf460449e, 0xd8b61463 },
{ 0x0bc20000, 0xdb630000, 0x7e88000c, 0x15860000, 0x91fd48f3, 0x7581bb43, 0xf460449e, 0xd8b61463,
  0x835a0000, 0xc4f70000, 0x01470022, 0xeec80000, 0x60a54f69, 0x142f2a24, 0x5cf534f2, 0x3ea660f7 },
{ 0x52500000, 0x29540000, 0x6a61004e, 0xf0ff0000, 0x9a317eec, 0x452341ce, 0xcf568fe5, 0x5303130f,
  0x538d0000, 0xa9fc0000, 0x9ef70006, 0x56ff0000, 0x0ae4004e, 0x92c5cdf9, 0xa9444018, 0x7f975691 },
{ 0x538d0000, 0xa9fc0000, 0x9ef70006, 0x56ff0000, 0x0ae4004e, 0x92c5cdf9, 0xa9444018, 0x7f975691,
  0x01dd0000, 0x80a80000, 0xf4960048, 0xa6000000, 0x90d57ea2, 0xd7e68c37, 0x6612cffd, 0x2c94459e },
{ 0xe6280000, 0x4c4b0000, 0xa8550000, 0xd3d002e0, 0xd86130b8, 0x98a7b0da, 0x289506b4, 0xd75a4897,
  0xf0c50000, 0x59230000, 0x45820000, 0xe18d00c0, 0x3b6d0631, 0xc2ed5699, 0xcbe0fe1c, 0x56a7b19f },
{ 0xf0c50000, 0x59230000, 0x45820000, 0xe18d00c0, 0x3b6d0631, 0xc2ed5699, 0xcbe0fe1c, 0x56a7b19f, // 49
  0x16ed0000, 0x15680000, 0xedd70000, 0x325d0220, 0xe30c3689, 0x5a4ae643, 0xe375f8a8, 0x81fdf908 },
{ 0xb4310000, 0x77330000, 0xb15d0000, 0x7fd004e0, 0x78a26138, 0xd116c35d, 0xd256d489, 0x4e6f74de,
  0xe3060000, 0xbdc10000, 0x87130000, 0xbff20060, 0x2eba0a1a, 0x8db53751, 0x73c5ab06, 0x5bd61539 },
{ 0xe3060000, 0xbdc10000, 0x87130000, 0xbff20060, 0x2eba0a1a, 0x8db53751, 0x73c5ab06, 0x5bd61539,
  0x57370000, 0xcaf20000, 0x364e0000, 0xc0220480, 0x56186b22, 0x5ca3f40c, 0xa1937f8f, 0x15b961e7 },
{ 0x02f20000, 0xa2810000, 0x873f0000, 0xe36c7800, 0x1e1d74ef, 0x073d2bd6, 0xc4c23237, 0x7f32259e,
  0xbadd0000, 0x13ad0000, 0xb7e70000, 0xf7282800, 0xdf45144d, 0x361ac33a, 0xea5a8d14, 0x2a2c18f0 },
{ 0xbadd0000, 0x13ad0000, 0xb7e70000, 0xf7282800, 0xdf45144d, 0x361ac33a, 0xea5a8d14, 0x2a2c18f0,
  0xb82f0000, 0xb12c0000, 0x30d80000, 0x14445000, 0xc15860a2, 0x3127e8ec, 0x2e98bf23, 0x551e3d6e },
{ 0x1e6c0000, 0xc4420000, 0x8a2e0000, 0xbcb6b800, 0x2c4413b6, 0x8bfdd3da, 0x6a0c1bc8, 0xb99dc2eb,
  0x92560000, 0x1eda0000, 0xea510000, 0xe8b13000, 0xa93556a5, 0xebfb6199, 0xb15c2254, 0x33c5244f },
{ 0x92560000, 0x1eda0000, 0xea510000, 0xe8b13000, 0xa93556a5, 0xebfb6199, 0xb15c2254, 0x33c5244f,
  0x8c3a0000, 0xda980000, 0x607f0000, 0x54078800, 0x85714513, 0x6006b243, 0xdb50399c, 0x8a58e6a4 },
{ 0x033d0000, 0x08b30000, 0xf33a0000, 0x3ac20007, 0x51298a50, 0x6b6e661f, 0x0ea5cfe3, 0xe6da7ffe, // 56
  0xa8da0000, 0x96be0000, 0x5c1d0000, 0x07da0002, 0x7d669583, 0x1f98708a, 0xbb668808, 0xda878000 },
{ 0xa8da0000, 0x96be0000, 0x5c1d0000, 0x07da0002, 0x7d669583, 0x1f98708a, 0xbb668808, 0xda878000,
  0xabe70000, 0x9e0d0000, 0xaf270000, 0x3d180005, 0x2c4f1fd3, 0x74f61695, 0xb5c347eb, 0x3c5dfffe },
{ 0x01930000, 0xe7820000, 0xedfb0000, 0xcf0c000b, 0x8dd08d58, 0xbca3b42e, 0x063661e1, 0x536f9e7b,
  0x92280000, 0xdc850000, 0x57fa0000, 0x56dc0003, 0xbae92316, 0x5aefa30c, 0x90cef752, 0x7b1675d7 },
{ 0x92280000, 0xdc850000, 0x57fa0000, 0x56dc0003, 0xbae92316, 0x5aefa30c, 0x90cef752, 0x7b1675d7,
  0x93bb0000, 0x3b070000, 0xba010000, 0x99d00008, 0x3739ae4e, 0xe64c1722, 0x96f896b3, 0x2879ebac },
{ 0x5fa80000, 0x56030000, 0x43ae0000, 0x64f30013, 0x257e86bf, 0x1311944e, 0x541e95bf, 0x8ea4db69,
  0x00440000, 0x7f480000, 0xda7c0000, 0x2a230001, 0x3badc9cc, 0xa9b69c87, 0x030a9e60, 0xbe0a679e },
{ 0x00440000, 0x7f480000, 0xda7c0000, 0x2a230001, 0x3badc9cc, 0xa9b69c87, 0x030a9e60, 0xbe0a679e,
  0x5fec0000, 0x294b0000, 0x99d20000, 0x4ed00012, 0x1ed34f73, 0xbaa708c9, 0x57140bdf, 0x30aebcf7 },
{ 0xee930000, 0xd6070000, 0x92c10000, 0x2b9801e0, 0x9451287c, 0x3b6cfb57, 0x45312374, 0x201f6a64,
  0x7b280000, 0x57420000, 0xa9e50000, 0x634300a0, 0x9edb442f, 0x6d9995bb, 0x27f83b03, 0xc7ff60f0 },
{ 0x7b280000, 0x57420000, 0xa9e50000, 0x634300a0, 0x9edb442f, 0x6d9995bb, 0x27f83b03, 0xc7ff60f0,
  0x95bb0000, 0x81450000, 0x3b240000, 0x48db0140, 0x0a8a6c53, 0x56f56eec, 0x62c91877, 0xe7e00a94 }
};

__global__
void x13_hamsi512_gpu_hash_64(uint32_t threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

		int hashPosition = nounce - startNounce;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition<<3];
		unsigned char *h1 = (unsigned char *)Hash;

		uint32_t c0 = 0x73746565, c1 = 0x6c706172, c2 = 0x6b204172, c3 = 0x656e6265;
		uint32_t c4 = 0x72672031, c5 = 0x302c2062, c6 = 0x75732032, c7 = 0x3434362c;
		uint32_t c8 = 0x20422d33, c9 = 0x30303120, cA = 0x4c657576, cB = 0x656e2d48;
		uint32_t cC = 0x65766572, cD = 0x6c65652c, cE = 0x2042656c, cF = 0x6769756d;
		uint32_t h[16] = { c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, cA, cB, cC, cD, cE, cF };
		uint32_t m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, mA, mB, mC, mD, mE, mF;
		uint32_t *tp, db, dm;

		for(int i = 0; i < 64; i += 8) {

			m0 = 0; m1 = 0; m2 = 0; m3 = 0; m4 = 0; m5 = 0; m6 = 0; m7 = 0;
			m8 = 0; m9 = 0; mA = 0; mB = 0; mC = 0; mD = 0; mE = 0; mF = 0;
			tp = &d_T512[0][0];

			#pragma unroll 2
			for (int u = 0; u < 8; u ++) {
				db = h1[i+u];
				#pragma unroll 2
				for (int v = 0; v < 8; v ++, db >>= 1) {
					dm = -(uint32_t)(db & 1);
					m0 ^= dm & *(tp+ 0); m1 ^= dm & *(tp+ 1);
					m2 ^= dm & *(tp+ 2); m3 ^= dm & *(tp+ 3);
					m4 ^= dm & *(tp+ 4); m5 ^= dm & *(tp+ 5);
					m6 ^= dm & *(tp+ 6); m7 ^= dm & *(tp+ 7);
					m8 ^= dm & *(tp+ 8); m9 ^= dm & *(tp+ 9);
					mA ^= dm & *(tp+10); mB ^= dm & *(tp+11);
					mC ^= dm & *(tp+12); mD ^= dm & *(tp+13);
					mE ^= dm & *(tp+14); mF ^= dm & *(tp+15);
					tp += 16;
				}
			}

			for( int r = 0; r < 6; r += 2 ) {
				ROUND_BIG(r, d_alpha_n);
				ROUND_BIG(r+1, d_alpha_n);
			}
			T_BIG;
		}

		// precomputed for 64 bytes blocks ?
		tp = &d_T512[0][0] + 112;
		m0 = tp[ 0]; m1 = tp[ 1];
		m2 = tp[ 2]; m3 = tp[ 3];
		m4 = tp[ 4]; m5 = tp[ 5];
		m6 = tp[ 6]; m7 = tp[ 7];
		m8 = tp[ 8]; m9 = tp[ 9];
		mA = tp[10]; mB = tp[11];
		mC = tp[12]; mD = tp[13];
		mE = tp[14]; mF = tp[15];

		for( int r = 0; r < 6; r += 2 ) {
			ROUND_BIG(r, d_alpha_n);
			ROUND_BIG(r+1, d_alpha_n);
		}
		T_BIG;

		tp = &d_T512[0][0] + 784;
		m0 = tp[ 0]; m1 = tp[ 1];
		m2 = tp[ 2]; m3 = tp[ 3];
		m4 = tp[ 4]; m5 = tp[ 5];
		m6 = tp[ 6]; m7 = tp[ 7];
		m8 = tp[ 8]; m9 = tp[ 9];
		mA = tp[10]; mB = tp[11];
		mC = tp[12]; mD = tp[13];
		mE = tp[14]; mF = tp[15];

		for( int r = 0; r < 12; r += 2 ) {
			ROUND_BIG(r, d_alpha_f);
			ROUND_BIG(r+1, d_alpha_f);
		}
		T_BIG;

		#pragma unroll 16
		for (int i = 0; i < 16; i++)
			Hash[i] = cuda_swab32(h[i]);
	}
}

__host__
void x13_hamsi512_cpu_init(int thr_id, uint32_t threads)
{
	cudaMemcpyToSymbol(d_alpha_n, alpha_n, sizeof(uint32_t)*32, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_alpha_f, alpha_f, sizeof(uint32_t)*32, 0, cudaMemcpyHostToDevice);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_T512, T512, sizeof(uint32_t)*64*16, 0, cudaMemcpyHostToDevice));
}

__host__
void x13_hamsi512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
	const uint32_t threadsperblock = 128;

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	x13_hamsi512_gpu_hash_64<<<grid, block>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
	//MyStreamSynchronize(NULL, order, thr_id);
}

__constant__ static uint64_t c_PaddedMessage80[10];

__host__
void x16_hamsi512_setBlock_80(void *pdata)
{
	cudaMemcpyToSymbol(c_PaddedMessage80, pdata, sizeof(c_PaddedMessage80), 0, cudaMemcpyHostToDevice);
}

__global__
void x16_hamsi512_gpu_hash_80(const uint32_t threads, const uint32_t startNonce, uint64_t *g_hash)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		unsigned char h1[80];
		#pragma unroll
		for (int i = 0; i < 10; i++)
			((uint2*)h1)[i] = ((uint2*)c_PaddedMessage80)[i];
		//((uint64_t*)h1)[9] = REPLACE_HIDWORD(c_PaddedMessage80[9], cuda_swab32(startNonce + thread));
		((uint32_t*)h1)[19] = cuda_swab32(startNonce + thread);

		uint32_t c0 = 0x73746565, c1 = 0x6c706172, c2 = 0x6b204172, c3 = 0x656e6265;
		uint32_t c4 = 0x72672031, c5 = 0x302c2062, c6 = 0x75732032, c7 = 0x3434362c;
		uint32_t c8 = 0x20422d33, c9 = 0x30303120, cA = 0x4c657576, cB = 0x656e2d48;
		uint32_t cC = 0x65766572, cD = 0x6c65652c, cE = 0x2042656c, cF = 0x6769756d;
		uint32_t h[16] = { c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, cA, cB, cC, cD, cE, cF };
		uint32_t m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, mA, mB, mC, mD, mE, mF;
		uint32_t *tp, db, dm;

		for(int i = 0; i < 80; i += 8)
		{
			m0 = 0; m1 = 0; m2 = 0; m3 = 0; m4 = 0; m5 = 0; m6 = 0; m7 = 0;
			m8 = 0; m9 = 0; mA = 0; mB = 0; mC = 0; mD = 0; mE = 0; mF = 0;
			tp = &d_T512[0][0];

			#pragma unroll
			for (int u = 0; u < 8; u++) {
				db = h1[i + u];
				#pragma unroll 2
				for (int v = 0; v < 8; v++, db >>= 1) {
					dm = -(uint32_t)(db & 1);
					m0 ^= dm & tp[ 0]; m1 ^= dm & tp[ 1];
					m2 ^= dm & tp[ 2]; m3 ^= dm & tp[ 3];
					m4 ^= dm & tp[ 4]; m5 ^= dm & tp[ 5];
					m6 ^= dm & tp[ 6]; m7 ^= dm & tp[ 7];
					m8 ^= dm & tp[ 8]; m9 ^= dm & tp[ 9];
					mA ^= dm & tp[10]; mB ^= dm & tp[11];
					mC ^= dm & tp[12]; mD ^= dm & tp[13];
					mE ^= dm & tp[14]; mF ^= dm & tp[15];
					tp += 16;
				}
			}

			#pragma unroll
			for (int r = 0; r < 6; r++) {
				ROUND_BIG(r, d_alpha_n);
			}
			T_BIG;
		}

		#define INPUT_BIG { \
			m0 = 0; m1 = 0; m2 = 0; m3 = 0; m4 = 0; m5 = 0; m6 = 0; m7 = 0; \
			m8 = 0; m9 = 0; mA = 0; mB = 0; mC = 0; mD = 0; mE = 0; mF = 0; \
			tp = &d_T512[0][0]; \
			for (int u = 0; u < 8; u++) { \
				db = endtag[u]; \
				for (int v = 0; v < 8; v++, db >>= 1) { \
					dm = -(uint32_t)(db & 1); \
					m0 ^= dm & tp[ 0]; m1 ^= dm & tp[ 1]; \
					m2 ^= dm & tp[ 2]; m3 ^= dm & tp[ 3]; \
					m4 ^= dm & tp[ 4]; m5 ^= dm & tp[ 5]; \
					m6 ^= dm & tp[ 6]; m7 ^= dm & tp[ 7]; \
					m8 ^= dm & tp[ 8]; m9 ^= dm & tp[ 9]; \
					mA ^= dm & tp[10]; mB ^= dm & tp[11]; \
					mC ^= dm & tp[12]; mD ^= dm & tp[13]; \
					mE ^= dm & tp[14]; mF ^= dm & tp[15]; \
					tp += 16; \
				} \
			} \
		}

		// close
		uint8_t endtag[8] = { 0x80, 0x00, 0x00, 0x00,  0x00, 0x00, 0x00, 0x00 };
		INPUT_BIG;

		#pragma unroll
		for (int r = 0; r < 6; r++) {
			ROUND_BIG(r, d_alpha_n);
		}
		T_BIG;

		endtag[0] = endtag[1] = 0x00;
		endtag[6] = 0x02;
		endtag[7] = 0x80;
		INPUT_BIG;

		// PF_BIG
		#pragma unroll
		for(int r = 0; r < 12; r++) {
			ROUND_BIG(r, d_alpha_f);
		}
		T_BIG;

		uint64_t hashPosition = thread;
		uint32_t *Hash = (uint32_t*)&g_hash[hashPosition << 3];
		#pragma unroll 16
		for(int i = 0; i < 16; i++)
			Hash[i] = cuda_swab32(h[i]);

		#undef INPUT_BIG
	}
}

__host__
void x16_hamsi512_cuda_hash_80(int thr_id, const uint32_t threads, const uint32_t startNounce, uint32_t *d_hash)
{
	const uint32_t threadsperblock = 128;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	x16_hamsi512_gpu_hash_80 <<<grid, block>>> (threads, startNounce, (uint64_t*)d_hash);
}
