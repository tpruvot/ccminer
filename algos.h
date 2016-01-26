#ifndef ALGOS_H
#define ALGOS_H

#include <string.h>
#include "compat.h"

enum sha_algos {
	ALGO_BLAKECOIN = 0,
	ALGO_BLAKE,
	ALGO_BMW,
	ALGO_C11,
	ALGO_DEEP,
	ALGO_DMD_GR,
	ALGO_FRESH,
	ALGO_FUGUE256,		/* Fugue256 */
	ALGO_GROESTL,
	ALGO_HEAVY,		/* Heavycoin hash */
	ALGO_KECCAK,
	ALGO_JACKPOT,
	ALGO_LUFFA,
	ALGO_LYRA2,
	ALGO_LYRA2v2,
	ALGO_MJOLLNIR,		/* Hefty hash */
	ALGO_MYR_GR,
	ALGO_NEOSCRYPT,
	ALGO_NIST5,
	ALGO_PENTABLAKE,
	ALGO_QUARK,
	ALGO_QUBIT,
	ALGO_SCRYPT,
	ALGO_SCRYPT_JANE,
	ALGO_SIB,
	ALGO_SKEIN,
	ALGO_SKEIN2,
	ALGO_S3,
	ALGO_X11,
	ALGO_X13,
	ALGO_X14,
	ALGO_X15,
	ALGO_X17,
	ALGO_VANILLA,
	ALGO_WHIRLCOIN,
	ALGO_WHIRLPOOL,
	ALGO_WHIRLPOOLX,
	ALGO_ZR5,
	ALGO_AUTO,
	ALGO_COUNT
};

extern volatile enum sha_algos opt_algo;

static const char *algo_names[] = {
	"blakecoin",
	"blake",
	"bmw",
	"c11",
	"deep",
	"dmd-gr",
	"fresh",
	"fugue256",
	"groestl",
	"heavy",
	"keccak",
	"jackpot",
	"luffa",
	"lyra2",
	"lyra2v2",
	"mjollnir",
	"myr-gr",
	"neoscrypt",
	"nist5",
	"penta",
	"quark",
	"qubit",
	"scrypt",
	"scrypt-jane",
	"sib",
	"skein",
	"skein2",
	"s3",
	"x11",
	"x13",
	"x14",
	"x15",
	"x17",
	"vanilla",
	"whirlcoin",
	"whirlpool",
	"whirlpoolx",
	"zr5",
	"auto", /* reserved for multi algo */
	""
};

// string to int/enum
static inline int algo_to_int(char* arg)
{
	int i;

	for (i = 0; i < ALGO_COUNT; i++) {
		if (algo_names[i] && !strcasecmp(arg, algo_names[i])) {
			return i;
		}
	}

	if (i == ALGO_COUNT) {
		// some aliases...
		if (!strcasecmp("all", arg))
			i = ALGO_AUTO;
		else if (!strcasecmp("flax", arg))
			i = ALGO_C11;
		else if (!strcasecmp("diamond", arg))
			i = ALGO_DMD_GR;
		else if (!strcasecmp("doom", arg))
			i = ALGO_LUFFA;
		else if (!strcasecmp("lyra2re", arg))
			i = ALGO_LYRA2;
		else if (!strcasecmp("lyra2rev2", arg))
			i = ALGO_LYRA2v2;
		else if (!strcasecmp("whirl", arg))
			i = ALGO_WHIRLPOOL;
		else if (!strcasecmp("ziftr", arg))
			i = ALGO_ZR5;
		else
			i = -1;
	}

	return i;
}

#endif
