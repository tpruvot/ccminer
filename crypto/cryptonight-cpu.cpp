#include <miner.h>
#include <memory.h>
#include <x86intrin.h>

#include "oaes_lib.h"
#include "cryptonight.h"

extern "C" {
#include <sph/sph_blake.h>
#include <sph/sph_groestl.h>
#include <sph/sph_jh.h>
#include <sph/sph_skein.h>
#include "cpu/c_keccak.h"
}

struct cryptonight_ctx {
	uint8_t long_state[MEMORY];
	union cn_slow_hash_state state;
	uint8_t text[INIT_SIZE_BYTE];
	uint8_t a[AES_BLOCK_SIZE];
	uint8_t b[AES_BLOCK_SIZE];
	uint8_t c[AES_BLOCK_SIZE];
	oaes_ctx* aes_ctx;
};

static void do_blake_hash(const void* input, size_t len, void* output)
{
	uchar hash[32];
	sph_blake256_context ctx;
	sph_blake256_set_rounds(14);
	sph_blake256_init(&ctx);
	sph_blake256(&ctx, input, len);
	sph_blake256_close(&ctx, hash);
	memcpy(output, hash, 32);
}

static void do_groestl_hash(const void* input, size_t len, void* output)
{
	uchar hash[32];
	sph_groestl256_context ctx;
	sph_groestl256_init(&ctx);
	sph_groestl256(&ctx, input, len);
	sph_groestl256_close(&ctx, hash);
	memcpy(output, hash, 32);
}

static void do_jh_hash(const void* input, size_t len, void* output)
{
	uchar hash[64];
	sph_jh256_context ctx;
	sph_jh256_init(&ctx);
	sph_jh256(&ctx, input, len);
	sph_jh256_close(&ctx, hash);
	memcpy(output, hash, 32);
}

static void do_skein_hash(const void* input, size_t len, void* output)
{
	uchar hash[32];
	sph_skein256_context ctx;
	sph_skein256_init(&ctx);
	sph_skein256(&ctx, input, len);
	sph_skein256_close(&ctx, hash);
	memcpy(output, hash, 32);
}

// todo: use sph if possible
static void keccak_hash_permutation(union hash_state *state) {
	keccakf((uint64_t*)state, 24);
}

static void keccak_hash_process(union hash_state *state, const uint8_t *buf, size_t count) {
	keccak1600(buf, (int)count, (uint8_t*)state);
}

extern "C" int fast_aesb_single_round(const uint8_t *in, uint8_t*out, const uint8_t *expandedKey);
extern "C" int aesb_single_round(const uint8_t *in, uint8_t*out, const uint8_t *expandedKey);
extern "C" int aesb_pseudo_round_mut(uint8_t *val, uint8_t *expandedKey);
extern "C" int fast_aesb_pseudo_round_mut(uint8_t *val, uint8_t *expandedKey);

static void (* const extra_hashes[4])(const void*, size_t, void *) = {
	do_blake_hash, do_groestl_hash, do_jh_hash, do_skein_hash
};

static uint64_t mul128(uint64_t multiplier, uint64_t multiplicand, uint64_t* product_hi)
{
	// multiplier   = ab = a * 2^32 + b
	// multiplicand = cd = c * 2^32 + d
	// ab * cd = a * c * 2^64 + (a * d + b * c) * 2^32 + b * d
	uint64_t a = hi_dword(multiplier);
	uint64_t b = lo_dword(multiplier);
	uint64_t c = hi_dword(multiplicand);
	uint64_t d = lo_dword(multiplicand);

	uint64_t ac = a * c;
	uint64_t ad = a * d;
	uint64_t bc = b * c;
	uint64_t bd = b * d;

	uint64_t adbc = ad + bc;
	uint64_t adbc_carry = adbc < ad ? 1 : 0;

	// multiplier * multiplicand = product_hi * 2^64 + product_lo
	uint64_t product_lo = bd + (adbc << 32);
	uint64_t product_lo_carry = product_lo < bd ? 1 : 0;
	*product_hi = ac + (adbc >> 32) + (adbc_carry << 32) + product_lo_carry;

	return product_lo;
}

static size_t e2i(const uint8_t* a) {
	return (*((uint64_t*) a) / AES_BLOCK_SIZE) & (MEMORY / AES_BLOCK_SIZE - 1);
}

static void mul(const uint8_t* a, const uint8_t* b, uint8_t* res) {
	((uint64_t*) res)[1] = mul128(((uint64_t*) a)[0], ((uint64_t*) b)[0], (uint64_t*) res);
}

static void sum_half_blocks(uint8_t* a, const uint8_t* b) {
	((uint64_t*) a)[0] += ((uint64_t*) b)[0];
	((uint64_t*) a)[1] += ((uint64_t*) b)[1];
}

static void sum_half_blocks_dst(const uint8_t* a, const uint8_t* b, uint8_t* dst) {
	((uint64_t*) dst)[0] = ((uint64_t*) a)[0] + ((uint64_t*) b)[0];
	((uint64_t*) dst)[1] = ((uint64_t*) a)[1] + ((uint64_t*) b)[1];
}

static void mul_sum_dst(const uint8_t* a, const uint8_t* b, const uint8_t* c, uint8_t* dst) {
	((uint64_t*) dst)[1] = mul128(((uint64_t*) a)[0], ((uint64_t*) b)[0], (uint64_t*) dst) + ((uint64_t*) c)[1];
	((uint64_t*) dst)[0] += ((uint64_t*) c)[0];
}

static void mul_sum_xor_dst(const uint8_t* a, uint8_t* c, uint8_t* dst, const int variant, const uint64_t tweak) {
	uint64_t hi, lo = mul128(((uint64_t*) a)[0], ((uint64_t*) dst)[0], &hi) + ((uint64_t*) c)[1];
	hi += ((uint64_t*) c)[0];

	((uint64_t*) c)[0] = ((uint64_t*) dst)[0] ^ hi;
	((uint64_t*) c)[1] = ((uint64_t*) dst)[1] ^ lo;
	((uint64_t*) dst)[0] = hi;
	((uint64_t*) dst)[1] = variant ? lo ^ tweak : lo;
}

static inline void mul_sum_xor_dst1(const uint8_t *cb, uint8_t *a, uint8_t *dst, uint8_t *ptr,
				   const uint64_t offset, const __m128i *b1, const uint64_t *bs, const uint8_t *bb)
{
	uint64_t hi __attribute__ ((aligned(16)));
	uint64_t lo __attribute__ ((aligned(16)));

	lo = mul128(((uint64_t*)cb)[0], *bs, &hi);

	const __m128i chunk1 = _mm_xor_si128(_mm_loadu_si128((__m128i *)((ptr) + ((offset) ^ 0x10))), _mm_set_epi64x(lo, hi));

	hi ^= ((uint64_t *)((ptr) + ((offset) ^ 0x20)))[0];
	lo ^= ((uint64_t *)((ptr) + ((offset) ^ 0x20)))[1];

	const __m128i chunk2 = _mm_loadu_si128((__m128i *)((ptr) + ((offset) ^ 0x20)));
	const __m128i chunk3 = _mm_loadu_si128((__m128i *)((ptr) + ((offset) ^ 0x30)));

	const __m128i _b = _mm_loadu_si128((__m128i *)bb);
	const __m128i _a = _mm_loadu_si128((__m128i *)a);

	_mm_storeu_si128((__m128i *)((ptr) + ((offset) ^ 0x10)), _mm_add_epi64(chunk3, *b1));
	_mm_storeu_si128((__m128i *)((ptr) + ((offset) ^ 0x20)), _mm_add_epi64(chunk1, _b));
	_mm_storeu_si128((__m128i *)((ptr) + ((offset) ^ 0x30)), _mm_add_epi64(chunk2, _a));

	hi += ((uint64_t*)a)[0];
	lo += ((uint64_t*)a)[1];

    ((uint64_t*)a)[0] = *bs ^ hi;
    ((uint64_t*)a)[1] = ((uint64_t*) dst)[1] ^ lo;

    ((uint64_t *)dst)[0] = hi;
    ((uint64_t *)dst)[1] = lo;
}

static void copy_block(uint8_t* dst, const uint8_t* src) {
	((uint64_t*) dst)[0] = ((uint64_t*) src)[0];
	((uint64_t*) dst)[1] = ((uint64_t*) src)[1];
}

static void xor_blocks(uint8_t* a, const uint8_t* b) {
	((uint64_t*) a)[0] ^= ((uint64_t*) b)[0];
	((uint64_t*) a)[1] ^= ((uint64_t*) b)[1];
}

static void xor_blocks_dst(const uint8_t* a, const uint8_t* b, uint8_t* dst) {
	((uint64_t*) dst)[0] = ((uint64_t*) a)[0] ^ ((uint64_t*) b)[0];
	((uint64_t*) dst)[1] = ((uint64_t*) a)[1] ^ ((uint64_t*) b)[1];
}

static void cryptonight_store_variant(void* state, int variant) {
	if (variant == 1 || cryptonight_fork == 8) {
		// monero and graft
		const uint8_t tmp = ((const uint8_t*)(state))[11];
		const uint8_t index = (((tmp >> 3) & 6) | (tmp & 1)) << 1;
		((uint8_t*)(state))[11] = tmp ^ ((0x75310 >> index) & 0x30);
	} else if (variant == 2 && cryptonight_fork == 3) {
		// stellite
		const uint8_t tmp = ((const uint8_t*)(state))[11];
		const uint8_t index = (((tmp >> 4) & 6) | (tmp & 1)) << 1;
		((uint8_t*)(state))[11] = tmp ^ ((0x75312 >> index) & 0x30);
	}
}

static inline void shuffle_add(const uint8_t *ptr, const uint64_t offset, const uint8_t *a, const uint8_t *b, const __m128i *b1)
{
	const __m128i _b = _mm_loadu_si128((__m128i *)b);
	const __m128i _a = _mm_loadu_si128((__m128i *)a);

	const __m128i chunk1 = _mm_loadu_si128((__m128i *)((ptr) + ((offset) ^ 0x10)));
	const __m128i chunk2 = _mm_loadu_si128((__m128i *)((ptr) + ((offset) ^ 0x20)));
	const __m128i chunk3 = _mm_loadu_si128((__m128i *)((ptr) + ((offset) ^ 0x30)));

	_mm_storeu_si128((__m128i *)((ptr) + ((offset) ^ 0x10)), _mm_add_epi64(chunk3, *b1));
	_mm_storeu_si128((__m128i *)((ptr) + ((offset) ^ 0x20)), _mm_add_epi64(chunk1, _b));
	_mm_storeu_si128((__m128i *)((ptr) + ((offset) ^ 0x30)), _mm_add_epi64(chunk2, _a));
}

static inline void div_sq(const uint8_t *b, const uint8_t *c, uint64_t *division_result, uint64_t *sqrt_result, uint64_t *bs)
{
	uint64_t d_r = *division_result;
	uint64_t s_r = *sqrt_result;
	uint64_t b_s = ((uint64_t*)b)[0];

	b_s ^= d_r ^ (s_r << 32);
	*bs = b_s;
	const uint64_t dividend = ((uint64_t*)c)[1];
	const uint32_t divisor = (((uint64_t*)c)[0] + (uint32_t)(s_r << 1)) | 0x80000001UL;
	d_r = ((uint32_t)(dividend / divisor)) + (((uint64_t)(dividend % divisor)) << 32);
	*division_result = d_r;
	const uint64_t sqrt_input = ((uint64_t*)c)[0] + d_r;

	const __m128i exp_double_bias = _mm_set_epi64x(0, 1023ULL << 52);
	__m128d x = _mm_castsi128_pd(_mm_add_epi64(_mm_cvtsi64_si128(sqrt_input >> 12), exp_double_bias));
	x = _mm_sqrt_sd(_mm_setzero_pd(), x);
	s_r = (uint64_t)(_mm_cvtsi128_si64(_mm_sub_epi64(_mm_castpd_si128(x), exp_double_bias))) >> 19;

	const uint64_t s = s_r >> 1;
	const uint64_t b_ = s_r & 1;
	const uint64_t r2 = (uint64_t)(s) * (s + b_) + (s_r << 32);
	s_r += ((r2 + b_ > sqrt_input) ? -1 : 0) + ((r2 + (1ULL << 32) < sqrt_input - s) ? 1 : 0);
	*sqrt_result = s_r;
}

static void cryptonight_hash_ctx(void* output, const void* input, const size_t len, struct cryptonight_ctx* ctx, const int variant)
{
	size_t i, j;

	keccak_hash_process(&ctx->state.hs, (const uint8_t*) input, len);
	ctx->aes_ctx = (oaes_ctx*) oaes_alloc();
	memcpy(ctx->text, ctx->state.init, INIT_SIZE_BYTE);

	oaes_key_import_data(ctx->aes_ctx, ctx->state.hs.b, AES_KEY_SIZE);
	for (i = 0; likely(i < MEMORY); i += INIT_SIZE_BYTE) {
		#undef RND
			#define RND(p) aesb_pseudo_round_mut(&ctx->text[AES_BLOCK_SIZE * p], ctx->aes_ctx->key->exp_data);
		RND(0);
		RND(1);
		RND(2);
		RND(3);
		RND(4);
		RND(5);
		RND(6);
		RND(7);
		memcpy(&ctx->long_state[i], ctx->text, INIT_SIZE_BYTE);
	}

	xor_blocks_dst(&ctx->state.k[0], &ctx->state.k[32], ctx->a);
	xor_blocks_dst(&ctx->state.k[16], &ctx->state.k[48], ctx->b);

	if (cryptonight_fork == 7 || cryptonight_fork == 8) {
		uint64_t division_result = ctx->state.hs.w[12];
		uint64_t sqrt_result = ctx->state.hs.w[13];
		__m128i dv = _mm_set_epi64x(ctx->state.hs.w[9] ^ ctx->state.hs.w[11], ctx->state.hs.w[8] ^ ctx->state.hs.w[10]);

		for (i = 0; likely(i < ITER / 4); ++i) {
			uint64_t k, l, bs;
			l = ((uint64_t *)(ctx->a))[0] & 0x1FFFF0;
			aesb_single_round(&ctx->long_state[l], ctx->c, ctx->a);
			k = ((uint64_t *)(ctx->c))[0] & 0x1FFFF0;
			shuffle_add(ctx->long_state, l, ctx->a, ctx->b, &dv);
			xor_blocks_dst(ctx->c, ctx->b, &ctx->long_state[l]);
			div_sq(&ctx->long_state[k], ctx->c, &division_result, &sqrt_result, &bs);
			mul_sum_xor_dst1(ctx->c, ctx->a, &ctx->long_state[k], ctx->long_state, k, &dv, &bs, ctx->b);
			dv = _mm_loadu_si128((__m128i *)&ctx->b);

			l = ((uint64_t *)(ctx->a))[0] & 0x1FFFF0;
			aesb_single_round(&ctx->long_state[l], ctx->b, ctx->a);
			k = ((uint64_t *)(ctx->b))[0] & 0x1FFFF0;
			shuffle_add(ctx->long_state, l, ctx->a, ctx->c, &dv);
			xor_blocks_dst(ctx->b, ctx->c, &ctx->long_state[l]);
			div_sq(&ctx->long_state[k], ctx->b, &division_result, &sqrt_result, &bs);
			mul_sum_xor_dst1(ctx->b, ctx->a, &ctx->long_state[k], ctx->long_state, k, &dv, &bs, ctx->c);
			dv = _mm_loadu_si128((__m128i *)&ctx->c);
		}
	} else {
		const uint64_t tweak = variant ? *((uint64_t*) (((uint8_t*)input) + 35)) ^ ctx->state.hs.w[24] : 0;

		for (i = 0; likely(i < ITER / 4); ++i) {
			j = e2i(ctx->a) * AES_BLOCK_SIZE;
			aesb_single_round(&ctx->long_state[j], ctx->c, ctx->a);
			xor_blocks_dst(ctx->c, ctx->b, &ctx->long_state[j]);
			cryptonight_store_variant(&ctx->long_state[j], variant);
			mul_sum_xor_dst(ctx->c, ctx->a, &ctx->long_state[e2i(ctx->c) * AES_BLOCK_SIZE], variant, tweak);

			j = e2i(ctx->a) * AES_BLOCK_SIZE;
			aesb_single_round(&ctx->long_state[j], ctx->b, ctx->a);
			xor_blocks_dst(ctx->b, ctx->c, &ctx->long_state[j]);
			cryptonight_store_variant(&ctx->long_state[j], variant);
			mul_sum_xor_dst(ctx->b, ctx->a, &ctx->long_state[e2i(ctx->b) * AES_BLOCK_SIZE], variant, tweak);
		}
	}

	memcpy(ctx->text, ctx->state.init, INIT_SIZE_BYTE);
	oaes_key_import_data(ctx->aes_ctx, &ctx->state.hs.b[32], AES_KEY_SIZE);

	for (i = 0; likely(i < MEMORY); i += INIT_SIZE_BYTE) {
		#undef RND
		#define RND(p) xor_blocks(&ctx->text[p * AES_BLOCK_SIZE], &ctx->long_state[i + p * AES_BLOCK_SIZE]); \
			aesb_pseudo_round_mut(&ctx->text[p * AES_BLOCK_SIZE], ctx->aes_ctx->key->exp_data);
		RND(0);
		RND(1);
		RND(2);
		RND(3);
		RND(4);
		RND(5);
		RND(6);
		RND(7);
	}
	memcpy(ctx->state.init, ctx->text, INIT_SIZE_BYTE);
	keccak_hash_permutation(&ctx->state.hs);

	int extra_algo = ctx->state.hs.b[0] & 3;
	extra_hashes[extra_algo](&ctx->state, 200, output);
	if (opt_debug) applog(LOG_DEBUG, "extra algo=%d", extra_algo);

	oaes_free((OAES_CTX **) &ctx->aes_ctx);
}

void cryptonight_hash_variant(void* output, const void* input, size_t len, int variant)
{
	struct cryptonight_ctx *ctx = (struct cryptonight_ctx*)malloc(sizeof(struct cryptonight_ctx));
	cryptonight_hash_ctx(output, input, len, ctx, variant);
	free(ctx);
}

void cryptonight_hash(void* output, const void* input)
{
	cryptonight_fork = 1;
	cryptonight_hash_variant(output, input, 76, 0);
}

void graft_hash(void* output, const void* input)
{
	cryptonight_fork = 8;
	cryptonight_hash_variant(output, input, 76, 1);
}

void monero_hash(void* output, const void* input)
{
	cryptonight_fork = 7;
	cryptonight_hash_variant(output, input, 76, 1);
}

void stellite_hash(void* output, const void* input)
{
	cryptonight_fork = 3;
	cryptonight_hash_variant(output, input, 76, 2);
}

