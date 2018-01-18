// Ravencoin X16R hybrid CPU/CUDA Implementation, penfold 2017

extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"
#include "sph/sph_skein.h"
#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"
#include "sph/sph_hamsi.h"
#include "sph/sph_fugue.h"
#include "sph/sph_shabal.h"
#include "sph/sph_whirlpool.h"
#include "sph/sph_sha2.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include "cuda_x16r.h"

#include <stdio.h>
#include <unistd.h>
#include <memory.h>

#define X16R_BLAKE      0
#define X16R_BMW        1
#define X16R_GROESTL    2
#define X16R_JH         3
#define X16R_KECCAK     4
#define X16R_SKEIN      5
#define X16R_LUFFA      6
#define X16R_CUBEHASH   7
#define X16R_SHAVITE    8
#define X16R_SIMD       9
#define X16R_ECHO       10
#define X16R_HAMSI      11
#define X16R_FUGUE      12
#define X16R_SHABAL     13
#define X16R_WHIRLPOOL  14
#define X16R_SHA512     15

#define X16R_HASH_COUNT (X16R_SHA512 + 1)

static char *hash_names[X16R_HASH_COUNT] =
{
    "blake",
    "bmw",
    "groestl",
    "jh",
    "keccak",
    "skein",
    "luffa",
    "cubehash",
    "shavite",
    "simd",
    "echo",
    "hamsi",
    "fugue",
    "shabal",
    "whirlpool",
    "sha512",
};

static uint32_t *d_hash[MAX_GPUS];

// CPU HASH
inline int get_hash_selection(const uint8_t *prev_block_hash, int index)
{
    uint8_t nibble = prev_block_hash[7 - (index / 2)];
    if (index % 2 == 0)
    {
        return nibble >> 4;
    }
    else
    {
        return nibble & 0x0f;
    }
}

extern "C" void x16rhash(void *output, const void *input)
{
    sph_blake512_context     ctx_blake;      //0
    sph_bmw512_context       ctx_bmw;        //1
    sph_groestl512_context   ctx_groestl;    //2
    sph_jh512_context        ctx_jh;         //3
    sph_keccak512_context    ctx_keccak;     //4
    sph_skein512_context     ctx_skein;      //5
    sph_luffa512_context     ctx_luffa;      //6
    sph_cubehash512_context  ctx_cubehash;   //7
    sph_shavite512_context   ctx_shavite;    //8
    sph_simd512_context      ctx_simd;       //9
    sph_echo512_context      ctx_echo;       //A
    sph_hamsi512_context     ctx_hamsi;      //B
    sph_fugue512_context     ctx_fugue;      //C
    sph_shabal512_context    ctx_shabal;     //D
    sph_whirlpool_context    ctx_whirlpool;  //E
    sph_sha512_context       ctx_sha512;     //F

    uchar _ALIGN(64) hash[64];
    uint32_t *phash = (uint32_t *) hash;
    uint8_t *prev_block_hash = (uint8_t *) input + 4;

    for (int i = 0; i < X16R_HASH_COUNT; i++)
    {
        int length;
        if (i == 0)
        {
            length = 80;
        }
        else
        {
            input = phash;
            length = 64;
        }

        int hash_selection = get_hash_selection(prev_block_hash, i);
        switch (hash_selection)
        {
            case X16R_BLAKE:
                sph_blake512_init(&ctx_blake);
                sph_blake512(&ctx_blake, input, length);
                sph_blake512_close(&ctx_blake, phash);
                break;
            case X16R_BMW:
                sph_bmw512_init(&ctx_bmw);
                sph_bmw512(&ctx_bmw, input, length);
                sph_bmw512_close(&ctx_bmw, phash);
                break;
            case X16R_GROESTL:
                sph_groestl512_init(&ctx_groestl);
                sph_groestl512(&ctx_groestl, input, length);
                sph_groestl512_close(&ctx_groestl, phash);
                break;
            case X16R_JH:
                sph_jh512_init(&ctx_jh);
                sph_jh512(&ctx_jh, input, length);
                sph_jh512_close(&ctx_jh, phash);
                break;
            case X16R_KECCAK:
                sph_keccak512_init(&ctx_keccak);
                sph_keccak512(&ctx_keccak, input, length);
                sph_keccak512_close(&ctx_keccak, phash);
                break;
            case X16R_SKEIN:
                sph_skein512_init(&ctx_skein);
                sph_skein512(&ctx_skein, input, length);
                sph_skein512_close(&ctx_skein, phash);
                break;
            case X16R_LUFFA:
                sph_luffa512_init(&ctx_luffa);
                sph_luffa512(&ctx_luffa, input, length);
                sph_luffa512_close(&ctx_luffa, phash);
                break;
            case X16R_CUBEHASH:
                sph_cubehash512_init(&ctx_cubehash);
                sph_cubehash512(&ctx_cubehash, input, length);
                sph_cubehash512_close(&ctx_cubehash, phash);
                break;
            case X16R_SHAVITE:
                sph_shavite512_init(&ctx_shavite);
                sph_shavite512(&ctx_shavite, input, length);
                sph_shavite512_close(&ctx_shavite, phash);
                break;
            case X16R_SIMD:
                sph_simd512_init(&ctx_simd);
                sph_simd512(&ctx_simd, input, length);
                sph_simd512_close(&ctx_simd, phash);
                break;
            case X16R_ECHO:
                sph_echo512_init(&ctx_echo);
                sph_echo512(&ctx_echo, input, length);
                sph_echo512_close(&ctx_echo, phash);
                break;
            case X16R_HAMSI:
                sph_hamsi512_init(&ctx_hamsi);
                sph_hamsi512(&ctx_hamsi, input, length);
                sph_hamsi512_close(&ctx_hamsi, phash);
                break;
            case X16R_FUGUE:
                sph_fugue512_init(&ctx_fugue);
                sph_fugue512(&ctx_fugue, input, length);
                sph_fugue512_close(&ctx_fugue, phash);
                break;
            case X16R_SHABAL:
                sph_shabal512_init(&ctx_shabal);
                sph_shabal512(&ctx_shabal, input, length);
                sph_shabal512_close(&ctx_shabal, phash);
                break;
            case X16R_WHIRLPOOL:
                sph_whirlpool_init(&ctx_whirlpool);
                sph_whirlpool(&ctx_whirlpool, input, length);
                sph_whirlpool_close(&ctx_whirlpool, phash);
                break;
            case X16R_SHA512:
                sph_sha512_init(&ctx_sha512);
                sph_sha512(&ctx_sha512, input, length);
                sph_sha512_close(&ctx_sha512, phash);
                break;
            default:
                gpulog(LOG_ERR, -1, "Unknown hash selection: %d (this should never happen!)", hash_selection);
                break;
        }
    }
    memcpy(output, phash, 32);
}

//#define _DEBUG
#define _DEBUG_PREFIX "x16r"
#include "cuda_debug.cuh"

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_x16r(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t first_nonce = pdata[19];
	int intensity = (device_sm[device_map[thr_id]] >= 500 && !is_windows()) ? 20 : 19;
	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity); // 19=256*256*8;

    // FIXME: CPU fallback for unimplemented first round algorithms
    uint32_t _ALIGN(64) h_hash[16];
    sph_simd512_context      ctx_simd;       //9
    sph_echo512_context      ctx_echo;       //A
    sph_hamsi512_context     ctx_hamsi;      //B
    sph_fugue512_context     ctx_fugue;      //C
    sph_shabal512_context    ctx_shabal;     //D
    sph_whirlpool_context    ctx_whirlpool;  //E
    sph_sha512_context       ctx_sha512;     //F

	if (opt_benchmark)
	{
		ptarget[7] = 0x5;
    }

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1)
		{
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			CUDA_LOG_ERROR();
		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		quark_blake512_cpu_init(thr_id, throughput);
		quark_blake512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		qubit_luffa512_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput);
		x11_cubehash512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);
		x11_echo512_cpu_init(thr_id, throughput);
		x13_hamsi512_cpu_init(thr_id, throughput);
		x13_fugue512_cpu_init(thr_id, throughput);
		x14_shabal512_cpu_init(thr_id, throughput);
		x15_whirlpool_cpu_init(thr_id, throughput, 0);
		x17_sha512_cpu_init(thr_id, throughput);

		CUDA_CALL_OR_RET_X(cudaMalloc(&d_hash[thr_id], (size_t) 64 * throughput), 0);

		cuda_check_cpu_init(thr_id, throughput);

		init[thr_id] = true;
	}

    uint32_t endiandata[20];
    for (int i = 0; i < 20; i++)
    {
        be32enc(&endiandata[i], pdata[i]);
    }

    uint8_t *prev_block_hash = (uint8_t *) endiandata + 4;
    int hash_selection[X16R_HASH_COUNT];
    for (int i = 0; i < X16R_HASH_COUNT; i++)
    {
        hash_selection[i] = get_hash_selection(prev_block_hash, i);
    }

    if (opt_debug)
    {
        applog_hex((void *) endiandata, 80);
        gpulog(LOG_DEBUG, thr_id, "hash selection: %s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s-%s",
            hash_names[hash_selection[0]],
            hash_names[hash_selection[1]],
            hash_names[hash_selection[2]],
            hash_names[hash_selection[3]],
            hash_names[hash_selection[4]],
            hash_names[hash_selection[5]],
            hash_names[hash_selection[6]],
            hash_names[hash_selection[7]],
            hash_names[hash_selection[8]],
            hash_names[hash_selection[9]],
            hash_names[hash_selection[10]],
            hash_names[hash_selection[11]],
            hash_names[hash_selection[12]],
            hash_names[hash_selection[13]],
            hash_names[hash_selection[14]],
            hash_names[hash_selection[15]]
        );
    }

    switch (hash_selection[0])
    {
        case X16R_BLAKE:
	        quark_blake512_cpu_setBlock_80(thr_id, endiandata);
	        break;
        case X16R_BMW:
            quark_bmw512_cpu_setBlock_80(endiandata);
            break;
        case X16R_GROESTL:
            groestl512_setBlock_80(thr_id, endiandata);
            break;
        case X16R_JH:
	        jh512_setBlock_80(thr_id, endiandata);
	        break;
        case X16R_KECCAK:
            keccak512_setBlock_80(thr_id, endiandata);
            break;
        case X16R_SKEIN:
            skein512_cpu_setBlock_80(endiandata);
            break;
        case X16R_LUFFA:
            qubit_luffa512_cpu_setBlock_80(endiandata);
            break;
        case X16R_CUBEHASH:
            cubehash512_setBlock_80(thr_id, endiandata);
            break;
        case X16R_SHAVITE:
            x11_shavite512_setBlock_80(endiandata);
            break;
        case X16R_SIMD:
            if (opt_debug)
            {
                gpulog(LOG_DEBUG, thr_id, "Not yet implemented: simd512/80 (falling back on CPU for first round)");
            }
            break;
        case X16R_ECHO:
            if (opt_debug)
            {
                gpulog(LOG_DEBUG, thr_id, "Not yet implemented: echo512/80 (falling back on CPU for first round)");
            }
            break;
        case X16R_HAMSI:
            if (opt_debug)
            {
                gpulog(LOG_DEBUG, thr_id, "Not yet implemented: hamsi512/80 (falling back on CPU for first round)");
            }
            break;
        case X16R_FUGUE:
            if (opt_debug)
            {
                gpulog(LOG_DEBUG, thr_id, "Not yet implemented: fugue512/80 (falling back on CPU for first round)");
            }
            break;
        case X16R_SHABAL:
            if (opt_debug)
            {
                gpulog(LOG_DEBUG, thr_id, "Not yet implemented: shabal512/80 (falling back on CPU for first round)");
            }
            break;
        case X16R_WHIRLPOOL:
            if (opt_debug)
            {
                gpulog(LOG_DEBUG, thr_id, "Not yet implemented: whirlpool512/80 (falling back on CPU for first round)");
            }
            break;
        case X16R_SHA512:
            if (opt_debug)
            {
                gpulog(LOG_DEBUG, thr_id, "Not yet implemented: sha512/80 (falling back on CPU for first round)");
            }
            break;
        default:
            gpulog(LOG_ERR, thr_id, "Unknown hash selection: %d (this should never happen!)", hash_selection[0]);
            break;
    }

	cuda_check_cpu_setTarget(ptarget);

	do
	{
        int order = 0;

		switch (hash_selection[0])
		{
            case X16R_BLAKE:
                quark_blake512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                break;
            case X16R_BMW:
                quark_bmw512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
                break;
            case X16R_GROESTL:
                groestl512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                break;
            case X16R_JH:
                jh512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                break;
            case X16R_KECCAK:
                keccak512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                break;
            case X16R_SKEIN:
                skein512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], 1); order++;
                break;
            case X16R_LUFFA:
                qubit_luffa512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
                break;
            case X16R_CUBEHASH:
                cubehash512_cuda_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                break;
            case X16R_SHAVITE:
                x11_shavite512_cpu_hash_80(thr_id, throughput, pdata[19], d_hash[thr_id], order++);
                break;
            case X16R_SIMD:
                sph_simd512_init(&ctx_simd);
                sph_simd512(&ctx_simd, endiandata, 80);
                sph_simd512_close(&ctx_simd, h_hash);
                cudaMemcpy(d_hash[thr_id], h_hash, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
                break;
            case X16R_ECHO:
                sph_echo512_init(&ctx_echo);
                sph_echo512(&ctx_echo, endiandata, 80);
                sph_echo512_close(&ctx_echo, h_hash);
                cudaMemcpy(d_hash[thr_id], h_hash, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
                break;
            case X16R_HAMSI:
                sph_hamsi512_init(&ctx_hamsi);
                sph_hamsi512(&ctx_hamsi, endiandata, 80);
                sph_hamsi512_close(&ctx_hamsi, h_hash);
                cudaMemcpy(d_hash[thr_id], h_hash, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
                break;
            case X16R_FUGUE:
                sph_fugue512_init(&ctx_fugue);
                sph_fugue512(&ctx_fugue, endiandata, 80);
                sph_fugue512_close(&ctx_fugue, h_hash);
                cudaMemcpy(d_hash[thr_id], h_hash, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
                break;
            case X16R_SHABAL:
                sph_shabal512_init(&ctx_shabal);
                sph_shabal512(&ctx_shabal, endiandata, 80);
                sph_shabal512_close(&ctx_shabal, h_hash);
                cudaMemcpy(d_hash[thr_id], h_hash, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
                break;
            case X16R_WHIRLPOOL:
                // FIXME: wrong hash?
		        //whirlpool512_hash_80_sm3(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                sph_whirlpool_init(&ctx_whirlpool);
                sph_whirlpool(&ctx_whirlpool, endiandata, 80);
                sph_whirlpool_close(&ctx_whirlpool, h_hash);
                cudaMemcpy(d_hash[thr_id], h_hash, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
                break;
            case X16R_SHA512:
                sph_sha512_init(&ctx_sha512);
                sph_sha512(&ctx_sha512, endiandata, 80);
                sph_sha512_close(&ctx_sha512, h_hash);
                cudaMemcpy(d_hash[thr_id], h_hash, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
		        break;
            default:
                gpulog(LOG_ERR, thr_id, "Round %d unknown hash selection: %d (this should never happen!)", 0, hash_selection[0]);
                break;
		}

        for (int i = 1; i < X16R_HASH_COUNT; i++)
		{
            switch (hash_selection[i])
            {
                case X16R_BLAKE:
                    quark_blake512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_BMW:
                    quark_bmw512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_GROESTL:
                    quark_groestl512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_JH:
                    quark_jh512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_KECCAK:
                    quark_keccak512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_SKEIN:
                    quark_skein512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_LUFFA:
                    x11_luffa512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_CUBEHASH:
                    x11_cubehash512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_SHAVITE:
                    x11_shavite512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_SIMD:
                    x11_simd512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_ECHO:
                    x11_echo512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_HAMSI:
                    x13_hamsi512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_FUGUE:
                    x13_fugue512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_SHABAL:
                    x14_shabal512_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_WHIRLPOOL:
                    x15_whirlpool_cpu_hash_64(thr_id, throughput, pdata[19], NULL, d_hash[thr_id], order++);
                    break;
                case X16R_SHA512:
                    x17_sha512_cpu_hash_64(thr_id, throughput, pdata[19], d_hash[thr_id]); order++;
                    break;
                default:
                    gpulog(LOG_ERR, thr_id, "Round %d unknown hash selection: %d (this should never happen!)", i, hash_selection[i]);
                    break;
            }
        }

        *hashes_done = pdata[19] - first_nonce + throughput;

        work->nonces[0] = cuda_check_hash(thr_id, throughput, pdata[19], d_hash[thr_id]);
        if (work->nonces[0] != UINT32_MAX)
        {
            const uint32_t Htarg = ptarget[7];
            uint32_t _ALIGN(64) vhash[8];
            be32enc(&endiandata[19], work->nonces[0]);
            x16rhash(vhash, endiandata);
            if (vhash[7] <= Htarg && fulltest(vhash, ptarget))
            {
                work->valid_nonces = 1;
                work_set_target_ratio(work, vhash);
                work->nonces[1] = cuda_check_hash_suppl(thr_id, throughput, pdata[19], d_hash[thr_id], 1);
                if (work->nonces[1] != 0)
                {
                    be32enc(&endiandata[19], work->nonces[1]);
                    x16rhash(vhash, endiandata);
                    bn_set_target_ratio(work, vhash, 1);
                    work->valid_nonces++;
                    pdata[19] = max(work->nonces[0], work->nonces[1]) + 1;
                }
                else
                {
                    pdata[19] = work->nonces[0] + 1;
                }
                return work->valid_nonces;
            }
            else
            {
                gpu_increment_reject(thr_id);
                if (!opt_quiet)
                {
                    gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
                }
                pdata[19] = work->nonces[0] + 1;
                continue;
            }
        }

		if ((uint64_t) throughput + pdata[19] >= max_nonce)
		{
			pdata[19] = max_nonce;
			break;
		}
		pdata[19] += throughput;

	}
	while (!work_restart[thr_id].restart);

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}

// cleanup
extern "C" void free_x16r(int thr_id)
{
	if (!init[thr_id])
	{
		return;
    }

	cudaThreadSynchronize();

	cudaFree(d_hash[thr_id]);

	quark_blake512_cpu_free(thr_id);
    quark_groestl512_cpu_free(thr_id);
    x11_simd512_cpu_free(thr_id);
    x13_fugue512_cpu_free(thr_id);
    x15_whirlpool_cpu_free(thr_id);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}