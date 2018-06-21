#include <miner.h>
#include <memory.h>

#include "oaes_lib.h"
#include "cryptonight.h"

extern "C" {
#include <sph/sph_blake.h>
#include <sph/sph_groestl.h>
#include <sph/sph_jh.h>
#include <sph/sph_skein.h>
#include "cpu/c_keccak.h"
}

#define VARIANT1_1(p) \
  do if (variant > 0) \
  { \
		const uint8_t tmp = ((const uint8_t*)(p))[11]; \
    static const uint32_t table = 0x75310; \
    const uint8_t index = (((tmp >> 3) & 6) | (tmp & 1)) << 1; \
    ((uint8_t*)(p))[11] = tmp ^ ((table >> index) & 0x30); \
  } while(0)

#define VARIANT1_INIT() \
 if (variant > 0 && len < 43) \
  { \
    return 0; \
  } \
  const uint64_t tweak1_2 = variant > 0 ? *((const uint64_t*) (((const uint8_t*)input) + 35)) ^ ctx->state.hs.w[24] : 0
 

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

static void mul_sum_xor_dst(const uint8_t* a, uint8_t* c, uint8_t* dst, const int variant, const uint64_t tweak1_2) {
	uint64_t hi, lo = mul128(((uint64_t*) a)[0], ((uint64_t*) dst)[0], &hi) + ((uint64_t*) c)[1];
	hi += ((uint64_t*) c)[0];

	((uint64_t*) c)[0] = ((uint64_t*) dst)[0] ^ hi;
	((uint64_t*) c)[1] = ((uint64_t*) dst)[1] ^ lo;
	((uint64_t*) dst)[0] = hi;
	((uint64_t*) dst)[1] = variant > 0 ? lo ^ tweak1_2 : lo;
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

static int cryptonight_hash_ctx(void* output, const void* input, size_t len, struct cryptonight_ctx* ctx, int variant)
{
	size_t i, j;
	keccak_hash_process(&ctx->state.hs, (const uint8_t*) input, len);
	ctx->aes_ctx = (oaes_ctx*) oaes_alloc();
	memcpy(ctx->text, ctx->state.init, INIT_SIZE_BYTE);

	VARIANT1_INIT();

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

	for (i = 0; likely(i < ITER / 4); ++i) {
		j = e2i(ctx->a) * AES_BLOCK_SIZE;
		aesb_single_round(&ctx->long_state[j], ctx->c, ctx->a);
		xor_blocks_dst(ctx->c, ctx->b, &ctx->long_state[j]);
		VARIANT1_1(&ctx->long_state[j]);

		mul_sum_xor_dst(ctx->c, ctx->a, &ctx->long_state[e2i(ctx->c) * AES_BLOCK_SIZE], variant, tweak1_2);

		j = e2i(ctx->a) * AES_BLOCK_SIZE;
		aesb_single_round(&ctx->long_state[j], ctx->b, ctx->a);
		xor_blocks_dst(ctx->b, ctx->c, &ctx->long_state[j]);
		VARIANT1_1(&ctx->long_state[j]);

		mul_sum_xor_dst(ctx->b, ctx->a, &ctx->long_state[e2i(ctx->b) * AES_BLOCK_SIZE], variant, tweak1_2);
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
	return 1;
}

int cryptonight_hash(void* output, const void* input, size_t len, int variant)
{
	struct cryptonight_ctx *ctx = (struct cryptonight_ctx*)malloc(sizeof(struct cryptonight_ctx));
	int rc = cryptonight_hash_ctx(output, input, len, ctx, variant);
	free(ctx);
	return rc;
}
