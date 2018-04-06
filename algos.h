#ifndef ALGOS_H
#define ALGOS_H

#include <string.h>
#include "compat.h"

enum sha_algos {
	ALGO_BLAKECOIN = 0,
	ALGO_BLAKE,
	ALGO_BLAKE2S,
	ALGO_BMW,
	ALGO_BASTION,
	ALGO_C11,
	ALGO_CRYPTOLIGHT,
	ALGO_CRYPTONIGHT,
	ALGO_DEEP,
	ALGO_DECRED,
	ALGO_DMD_GR,
	ALGO_EQUIHASH,
	ALGO_FRESH,
	ALGO_FUGUE256,		/* Fugue256 */
	ALGO_GROESTL,
	ALGO_HEAVY,		/* Heavycoin hash */
	ALGO_HMQ1725,
	ALGO_HSR,
	ALGO_KECCAK,
	ALGO_KECCAKC,		/* refreshed Keccak with pool factor 256 */
	ALGO_JACKPOT,
	ALGO_JHA,
	ALGO_LBRY,
	ALGO_LUFFA,
	ALGO_LYRA2,
	ALGO_LYRA2v2,
	ALGO_LYRA2Z,
	ALGO_MJOLLNIR,		/* Hefty hash */
	ALGO_MYR_GR,
	ALGO_NEOSCRYPT,
	ALGO_NIST5,
	ALGO_PENTABLAKE,
	ALGO_PHI,
	ALGO_POLYTIMOS,
	ALGO_QUARK,
	ALGO_QUBIT,
	ALGO_SCRYPT,
	ALGO_SCRYPT_JANE,
	ALGO_SHA256D,
	ALGO_SHA256T,
	ALGO_SIA,
	ALGO_SIB,
	ALGO_SKEIN,
	ALGO_SKEIN2,
	ALGO_SKUNK,
	ALGO_S3,
	ALGO_TIMETRAVEL,
	ALGO_TRIBUS,
	ALGO_BITCORE,
	ALGO_X11EVO,
	ALGO_X11,
	ALGO_X12,
	ALGO_X13,
	ALGO_X14,
	ALGO_X15,
	ALGO_X16R,
	ALGO_X16S,
	ALGO_X17,
	ALGO_VANILLA,
	ALGO_VELTOR,
	ALGO_WHIRLCOIN,
	ALGO_WHIRLPOOL,
	ALGO_WHIRLPOOLX,
	ALGO_WILDKECCAK,
	ALGO_ZR5,
	ALGO_MONERO,
	ALGO_GRAFT,
	ALGO_STELLITE,
	ALGO_AUTO,
	ALGO_COUNT
};

extern volatile enum sha_algos opt_algo;

static const char *algo_names[] = {
	"blakecoin",
	"blake",
	"blake2s",
	"bmw",
	"bastion",
	"c11",
	"cryptolight",
	"cryptonight",
	"deep",
	"decred",
	"dmd-gr",
	"equihash",
	"fresh",
	"fugue256",
	"groestl",
	"heavy",
	"hmq1725",
	"hsr",
	"keccak",
	"keccakc",
	"jackpot",
	"jha",
	"lbry",
	"luffa",
	"lyra2",
	"lyra2v2",
	"lyra2z",
	"mjollnir",
	"myr-gr",
	"neoscrypt",
	"nist5",
	"penta",
	"phi",
	"polytimos",
	"quark",
	"qubit",
	"scrypt",
	"scrypt-jane",
	"sha256d",
	"sha256t",
	"sia",
	"sib",
	"skein",
	"skein2",
	"skunk",
	"s3",
	"timetravel",
	"tribus",
	"bitcore",
	"x11evo",
	"x11",
	"x12",
	"x13",
	"x14",
	"x15",
	"x16r",
	"x16s",
	"x17",
	"vanilla",
	"veltor",
	"whirlcoin",
	"whirlpool",
	"whirlpoolx",
	"wildkeccak",
	"zr5",
	"monero",
	"graft",
	"stellite",
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
		else if (!strcasecmp("cryptonight-light", arg))
			i = ALGO_CRYPTOLIGHT;
		else if (!strcasecmp("cryptonight-lite", arg))
			i = ALGO_CRYPTOLIGHT;
		else if (!strcasecmp("flax", arg))
			i = ALGO_C11;
		else if (!strcasecmp("diamond", arg))
			i = ALGO_DMD_GR;
		else if (!strcasecmp("equi", arg))
			i = ALGO_EQUIHASH;
		else if (!strcasecmp("doom", arg))
			i = ALGO_LUFFA;
		else if (!strcasecmp("hmq17", arg))
			i = ALGO_HMQ1725;
		else if (!strcasecmp("hshare", arg))
			i = ALGO_HSR;
		else if (!strcasecmp("lyra2re", arg))
			i = ALGO_LYRA2;
		else if (!strcasecmp("lyra2rev2", arg))
			i = ALGO_LYRA2v2;
		else if (!strcasecmp("phi1612", arg))
			i = ALGO_PHI;
		else if (!strcasecmp("bitcoin", arg))
			i = ALGO_SHA256D;
		else if (!strcasecmp("sha256", arg))
			i = ALGO_SHA256D;
		else if (!strcasecmp("thorsriddle", arg))
			i = ALGO_VELTOR;
		else if (!strcasecmp("timetravel10", arg))
			i = ALGO_BITCORE;
		else if (!strcasecmp("whirl", arg))
			i = ALGO_WHIRLPOOL;
		else if (!strcasecmp("ziftr", arg))
			i = ALGO_ZR5;
		else
			i = -1;
	}

	return i;
}

static inline int get_cryptonight_algo(int fork)
{
	int algo = ALGO_COUNT;

	switch (fork) {
		case 8:
			algo = ALGO_GRAFT;
			break;

		case 7:
			algo = ALGO_MONERO;
			break;

		case 3:
			algo = ALGO_STELLITE;
			break;

		default:
			algo = ALGO_CRYPTONIGHT;
			break;
	}

	return algo;
}

#endif
