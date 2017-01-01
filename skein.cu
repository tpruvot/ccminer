/**
* SKEIN512 80 + SHA256 64
* by tpruvot@github - 2015
* Optimized by sp-hash@github - 2015
*/

extern "C" {
#include "sph/sph_skein.h"
}

#include "miner.h"
#include "cuda_helper.h"
#include <openssl/sha.h>
 
static uint32_t foundnonces[MAX_GPUS][2];

extern void skein512_cpu_setBlock_80(uint32_t thr_id,void *pdata);
extern void skein512_cpu_hash_80_50(int thr_id, uint32_t threads, uint32_t startNounce, int swapu, uint2 target, uint32_t *h_found);
extern void skein512_cpu_hash_80_52(int thr_id, uint32_t threads, uint32_t startNounce, int swapu, uint2 target, uint32_t *h_found);

extern "C" void skeincoinhash(void *output, const void *input)
{
	sph_skein512_context ctx_skein;
	SHA256_CTX sha256;

	uint32_t hash[16];

	sph_skein512_init(&ctx_skein);
	sph_skein512(&ctx_skein, input, 80);
	sph_skein512_close(&ctx_skein, hash);

	SHA256_Init(&sha256);
	SHA256_Update(&sha256, (unsigned char *)hash, 64);
	SHA256_Final((unsigned char *)hash, &sha256);

	memcpy(output, hash, 32);
}

static __inline uint32_t swab32_if(uint32_t val, bool iftrue)
{
	return iftrue ? swab32(val) : val;
}

bool init[MAX_GPUS] = { 0 };

int scanhash_skeincoin(int thr_id, uint32_t *pdata,
								  const uint32_t *ptarget, uint32_t max_nonce,
								  uint32_t *hashes_done)
{
	const uint32_t first_nonce = pdata[19];
	const bool swap = true;

	uint32_t intensity = (device_sm[device_map[thr_id]] > 500) ? 1 << 26 : 1 << 25;
	uint32_t throughput = device_intensity(device_map[thr_id], __func__, intensity); // 256*4096
	throughput = min(throughput, max_nonce - first_nonce);

	if (opt_benchmark)
	{
		((uint64_t*)ptarget)[3] = 0x3000f0000;
	}
//	uint64_t target = ((uint64_t*)ptarget)[3];
	if (!init[thr_id])
	{
		CUDA_SAFE_CALL(cudaSetDevice(device_map[thr_id]));
		if (!opt_cpumining) cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		if (opt_n_gputhreads == 1)
		{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		}

		cuda_check_cpu_init(thr_id, throughput);
		init[thr_id] = true;
	}

	uint32_t endiandata[20];
	for (int k = 0; k < 20; k++)
		be32enc(&endiandata[k], pdata[k]);

	skein512_cpu_setBlock_80(thr_id, (void*)endiandata);
	do
	{
//		if(scan_abort_flag || work_restart[thr_id].restart) return 0;
		if (device_sm[device_map[thr_id]] > 500)
			skein512_cpu_hash_80_52(thr_id, throughput, pdata[19], swap, ((uint2*)ptarget)[3], foundnonces[thr_id]);
		else
			skein512_cpu_hash_80_50(thr_id, throughput, pdata[19], swap, ((uint2*)ptarget)[3], foundnonces[thr_id]);
		
		if (foundnonces[thr_id][0] != 0xffffffff)
		{
			uint32_t vhash64[8];

			endiandata[19] = swab32_if(foundnonces[thr_id][0], swap);
			
			skeincoinhash(vhash64, endiandata);

			uint64_t test = ((uint64_t*)vhash64)[3];
			if (test <= ((uint64_t*)ptarget)[3] && fulltest(vhash64, ptarget))
			{
				int res = 1;

				if (opt_debug || opt_benchmark)
					applog(LOG_INFO, "GPU #%d: found nonce $%08X", thr_id, foundnonces[thr_id][0]);
				if (foundnonces[thr_id][1] != 0xffffffff)
				{
					if (foundnonces[thr_id][1] == foundnonces[thr_id][0])
					{
						//applog(LOG_WARNING, "Duplicate nonce: #%d", test);
					}
					else
					{
						if (opt_debug || opt_benchmark)
							applog(LOG_INFO, "GPU #%d: found second nonce $%08X", thr_id, foundnonces[thr_id][1]);
						pdata[21] = swab32_if(foundnonces[thr_id][1], !swap);
						res++;
					}
				}
				*hashes_done = pdata[19] - first_nonce + throughput;
				pdata[19] = swab32_if(foundnonces[thr_id][0], !swap);
				return res;
			}
			else 
			{
				if (test != ((uint64_t*)ptarget)[3])
					applog(LOG_WARNING, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundnonces[thr_id][0]);
				else
					applog(LOG_WARNING, "Lost work: #%d", test);

			}
		}
		pdata[19] += throughput;
	} while (!scan_abort_flag && !work_restart[thr_id].restart && ((uint64_t)max_nonce > ((uint64_t)(pdata[19]) + (uint64_t)throughput)));

	*hashes_done = pdata[19] - first_nonce;
	return 0;
}
