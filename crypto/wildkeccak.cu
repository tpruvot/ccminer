// WildKeccak CUDA Kernel, Code based on Linux Wolf0 bbr-miner implementation from 2014
// Adapted to ccminer 2.0 - tpruvot 2016-2017
//
// NOTE FOR SP: this ccminer version is licensed under GPLv3 Licence

extern "C" {
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
}

#include <miner.h>
#include <cuda_helper.h>
#include <cuda_vector_uint2x4.h> // todo

#include "wildkeccak.h"

extern char *device_config[MAX_GPUS]; // -l
extern uint64_t* pscratchpad_buff;

static uint64_t*    d_input[MAX_GPUS];
static uint32_t*    d_retnonce[MAX_GPUS];
static ulonglong4*  d_scratchpad[MAX_GPUS];

static uint64_t*    h_scratchpad[MAX_GPUS] = { 0 };
static cudaStream_t bufpad_stream[MAX_GPUS] = { 0 };
static cudaStream_t kernel_stream[MAX_GPUS] = { 0 };

uint64_t scratchpad_size = 0;

uint32_t WK_CUDABlocks   = 64;
uint32_t WK_CUDAThreads  = 256;

#define st0 	vst0.x
#define st1 	vst0.y
#define st2 	vst0.z
#define st3 	vst0.w

#define st4 	vst4.x
#define st5 	vst4.y
#define st6 	vst4.z
#define st7 	vst4.w

#define st8 	vst8.x
#define st9 	vst8.y
#define st10	vst8.z
#define st11	vst8.w

#define st12	vst12.x
#define st13	vst12.y
#define st14	vst12.z
#define st15	vst12.w

#define st16	vst16.x
#define st17	vst16.y
#define st18	vst16.z
#define st19	vst16.w

#define st20	vst20.x
#define st21	vst20.y
#define st22	vst20.z
#define st23	vst20.w

#if __CUDA_ARCH__ >= 320

__device__ __forceinline__ uint64_t cuda_rotl641(const uint64_t value)
{
	uint2 result;
	asm("shf.l.wrap.b32 %0, %1, %2, 1U;" : "=r"(result.x)
		: "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))));
	asm("shf.l.wrap.b32 %0, %1, %2, 1U;" : "=r"(result.y)
		: "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))));
	return __double_as_longlong(__hiloint2double(result.y, result.x));
}

#else
__noinline__ __device__ uint64_t cuda_rotl641(const uint64_t x) { return((x << 1) | (x >> 63)); }
#endif

__noinline__ __device__ uint64_t bitselect(const uint64_t a, const uint64_t b, const uint64_t c) { return(a ^ (c & (b ^ a))); }

#define ROTL641(x) (cuda_rotl641(x))

#define RND() \
	bc[0] = st0 ^ st5 ^ st10 * st15 * st20 ^ ROTL641(st2 ^ st7 ^ st12 * st17 * st22); \
	bc[1] = st1 ^ st6 ^ st11 * st16 * st21 ^ ROTL641(st3 ^ st8 ^ st13 * st18 * st23); \
	bc[2] = st2 ^ st7 ^ st12 * st17 * st22 ^ ROTL641(st4 ^ st9 ^ st14 * st19 * st24); \
	bc[3] = st3 ^ st8 ^ st13 * st18 * st23 ^ ROTL641(st0 ^ st5 ^ st10 * st15 * st20); \
	bc[4] = st4 ^ st9 ^ st14 * st19 * st24 ^ ROTL641(st1 ^ st6 ^ st11 * st16 * st21); \
	tmp1 = st1 ^ bc[0]; \
	\
	st0  ^= bc[4]; \
	st1  = ROTL64(st6  ^ bc[0], 44); \
	st6  = ROTL64(st9  ^ bc[3], 20); \
	st9  = ROTL64(st22 ^ bc[1], 61); \
	st22 = ROTL64(st14 ^ bc[3], 39); \
	st14 = ROTL64(st20 ^ bc[4], 18); \
	st20 = ROTL64(st2  ^ bc[1], 62); \
	st2  = ROTL64(st12 ^ bc[1], 43); \
	st12 = ROTL64(st13 ^ bc[2], 25); \
	st13 = ROTL64(st19 ^ bc[3], 8); \
	st19 = ROTL64(st23 ^ bc[2], 56); \
	st23 = ROTL64(st15 ^ bc[4], 41); \
	st15 = ROTL64(st4  ^ bc[3], 27); \
	st4  = ROTL64(st24 ^ bc[3], 14); \
	st24 = ROTL64(st21 ^ bc[0], 2); \
	st21 = ROTL64(st8  ^ bc[2], 55); \
	st8  = ROTL64(st16 ^ bc[0], 45); \
	st16 = ROTL64(st5  ^ bc[4], 36); \
	st5  = ROTL64(st3  ^ bc[2], 28); \
	st3  = ROTL64(st18 ^ bc[2], 21); \
	st18 = ROTL64(st17 ^ bc[1], 15); \
	st17 = ROTL64(st11 ^ bc[0], 10); \
	st11 = ROTL64(st7  ^ bc[1], 6); \
	st7  = ROTL64(st10 ^ bc[4], 3); \
	st10 = ROTL641(tmp1); \
	\
	tmp1 = st0; tmp2 = st1; st0 = bitselect(st0 ^ st2, st0, st1); st1 = bitselect(st1 ^ st3, st1, st2); \
	 st2 = bitselect(st2 ^ st4, st2, st3); st3 = bitselect(st3 ^ tmp1, st3, st4); st4 = bitselect(st4 ^ tmp2, st4, tmp1); \
	tmp1 = st5; tmp2 = st6; st5 = bitselect(st5 ^ st7, st5, st6); st6 = bitselect(st6 ^ st8, st6, st7); \
	 st7 = bitselect(st7 ^ st9, st7, st8); st8 = bitselect(st8 ^ tmp1, st8, st9); st9 = bitselect(st9 ^ tmp2, st9, tmp1); \
	tmp1 = st10; tmp2 = st11; st10 = bitselect(st10 ^ st12, st10, st11); st11 = bitselect(st11 ^ st13, st11, st12); \
	st12 = bitselect(st12 ^ st14, st12, st13); st13 = bitselect(st13 ^ tmp1, st13, st14); st14 = bitselect(st14 ^ tmp2, st14, tmp1); \
	tmp1 = st15; tmp2 = st16; st15 = bitselect(st15 ^ st17, st15, st16); st16 = bitselect(st16 ^ st18, st16, st17); \
	st17 = bitselect(st17 ^ st19, st17, st18); st18 = bitselect(st18 ^ tmp1, st18, st19); st19 = bitselect(st19 ^ tmp2, st19, tmp1); \
	tmp1 = st20; tmp2 = st21; st20 = bitselect(st20 ^ st22, st20, st21); st21 = bitselect(st21 ^ st23, st21, st22); \
	st22 = bitselect(st22 ^ st24, st22, st23); st23 = bitselect(st23 ^ tmp1, st23, st24); st24 = bitselect(st24 ^ tmp2, st24, tmp1); \
	st0 ^= 1;

#define LASTRND1() \
	bc[0] = st0 ^ st5 ^ st10 * st15 * st20 ^ ROTL64(st2 ^ st7 ^ st12 * st17 * st22, 1); \
	bc[1] = st1 ^ st6 ^ st11 * st16 * st21 ^ ROTL64(st3 ^ st8 ^ st13 * st18 * st23, 1); \
	bc[2] = st2 ^ st7 ^ st12 * st17 * st22 ^ ROTL64(st4 ^ st9 ^ st14 * st19 * st24, 1); \
	bc[3] = st3 ^ st8 ^ st13 * st18 * st23 ^ ROTL64(st0 ^ st5 ^ st10 * st15 * st20, 1); \
	bc[4] = st4 ^ st9 ^ st14 * st19 * st24 ^ ROTL64(st1 ^ st6 ^ st11 * st16 * st21, 1); \
	\
	st0 ^= bc[4]; \
	st1 = ROTL64(st6 ^ bc[0], 44); \
	st2 = ROTL64(st12 ^ bc[1], 43); \
	st4 = ROTL64(st24 ^ bc[3], 14); \
	st3 = ROTL64(st18 ^ bc[2], 21); \
	\
	tmp1 = st0; st0 = bitselect(st0 ^ st2, st0, st1); st1 = bitselect(st1 ^ st3, st1, st2); st2 = bitselect(st2 ^ st4, st2, st3); st3 = bitselect(st3 ^ tmp1, st3, st4); \
	st0 ^= 1;

#define LASTRND2() \
	bc[2] = st2 ^ st7 ^ st12 * st17 * st22 ^ ROTL64(st4 ^ st9 ^ st14 * st19 * st24, 1); \
	bc[3] = st3 ^ st8 ^ st13 * st18 * st23 ^ ROTL64(st0 ^ st5 ^ st10 * st15 * st20, 1); \
	bc[4] = st4 ^ st9 ^ st14 * st19 * st24 ^ ROTL64(st1 ^ st6 ^ st11 * st16 * st21, 1); \
	\
	st0 ^= bc[4]; \
	st4 = ROTL64(st24 ^ bc[3], 14); \
	st3 = ROTL64(st18 ^ bc[2], 21); \
	st3 = bitselect(st3 ^ st0, st3, st4);

__device__ ulonglong4 operator^(const ulonglong4 &a, const ulonglong4 &b)
{
	return(make_ulonglong4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w));
}

#define MIX(vst) vst = vst ^ scratchpad[vst.x % scr_size] ^ scratchpad[vst.y % scr_size] ^ scratchpad[vst.z % scr_size] ^ scratchpad[vst.w % scr_size];

#define MIX_ALL MIX(vst0); MIX(vst4); MIX(vst8); MIX(vst12); MIX(vst16); MIX(vst20);

__global__
void wk(uint32_t* __restrict__ retnonce, const uint64_t* __restrict__ input, const ulonglong4* __restrict__ scratchpad,
	const uint32_t scr_size, const uint32_t target, uint64_t startNonce)
{
	ulonglong4 vst0, vst4, vst8, vst12, vst16, vst20;
	uint64_t bc[5];
	uint64_t st24, tmp1, tmp2;

	const uint64_t nonce = startNonce + (blockDim.x * blockIdx.x) + threadIdx.x;
	vst0  = make_ulonglong4((nonce << 8) + (input[0] & 0xFF), input[1] & 0xFFFFFFFFFFFFFF00ULL, input[2], input[3]);
	vst4  = make_ulonglong4(input[4], input[5], input[6], input[7]);
	vst8  = make_ulonglong4(input[8], input[9], (input[10] & 0xFF) | 0x100, 0);
	vst12 = make_ulonglong4(0, 0, 0, 0);
	vst16 = make_ulonglong4(0x8000000000000000ULL, 0, 0, 0);
	vst20 = make_ulonglong4(0, 0, 0, 0);
	st24  = 0;

	RND();
	MIX_ALL;

	for(int i = 0; i < 22; i++) {
		RND();
		MIX_ALL;
	}

	LASTRND1();

	vst4  = make_ulonglong4(1, 0, 0, 0);
	vst8  = make_ulonglong4(0, 0, 0, 0);
	vst12 = make_ulonglong4(0, 0, 0, 0);
	vst16 = make_ulonglong4(0x8000000000000000ULL, 0, 0, 0);
	vst20 = make_ulonglong4(0, 0, 0, 0);
	st24  = 0;

	RND();
	MIX_ALL;

	#pragma unroll
	for(int i = 0; i < 22; i++) {
		RND();
		MIX_ALL;
	}

	LASTRND2();

	if((st3 >> 32) <= target) {
		retnonce[0] = (uint32_t) nonce;
		retnonce[1] = retnonce[0];
	}
}

__host__
void wildkeccak_kernel(const int thr_id, const uint32_t threads, const uint32_t startNounce, const uint2 target, uint32_t *resNonces)
{
	CUDA_SAFE_CALL(cudaMemsetAsync(d_retnonce[thr_id], 0xff, 2 * sizeof(uint32_t), kernel_stream[thr_id]));

	const uint32_t threadsperblock = WK_CUDAThreads;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	wk <<<grid, block, 0, kernel_stream[thr_id]>>> (d_retnonce[thr_id], d_input[thr_id], d_scratchpad[thr_id],
		(uint32_t)(scratchpad_size >> 2), target.y, startNounce);

	cudaMemcpyAsync(resNonces, d_retnonce[thr_id], 2*sizeof(uint32_t), cudaMemcpyDeviceToHost, kernel_stream[thr_id]);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_wildkeccak(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t *ptarget = work->target;
	uint32_t throughput = 0;
	uint64_t n, nonce, first;
	uint8_t *pdata = (uint8_t*) work->data;
	memcpy(&first, &pdata[1], 8);
	n = nonce = first;

	if (!scratchpad_size || !h_scratchpad[thr_id]) {
		if (h_scratchpad[thr_id])
			applog(LOG_ERR, "Scratchpad size is not set!");
		work->data[0] = 0; // invalidate
		sleep(1);
		return -EBUSY;
	}

	if (!init[thr_id]) {

		if (device_config[thr_id]) {
			sscanf(device_config[thr_id], "%ux%u", &WK_CUDABlocks, &WK_CUDAThreads);
			gpulog(LOG_INFO, thr_id, "Using %u x %u kernel launch config, %u threads",
				WK_CUDABlocks, WK_CUDAThreads, throughput);
		} else {
			throughput = cuda_default_throughput(thr_id, WK_CUDABlocks*WK_CUDAThreads);
			gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);
		}

		cudaSetDevice(device_map[thr_id]);
		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage (linux)
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
		}

		CUDA_SAFE_CALL(cudaMalloc(&d_input[thr_id], 88));
		CUDA_SAFE_CALL(cudaMalloc(&d_retnonce[thr_id], 2*sizeof(uint32_t)));

		int status = (int) cudaMalloc(&d_scratchpad[thr_id], WILD_KECCAK_SCRATCHPAD_BUFFSIZE);
		if (status != cudaSuccess) {
			gpulog(LOG_ERR, thr_id, "Unable to allocate device memory, %u MB, err %d",
				(uint32_t) (WILD_KECCAK_SCRATCHPAD_BUFFSIZE/(1024*1024)), status);
			exit(-ENOMEM);
		}

		cudaStreamCreate(&bufpad_stream[thr_id]);
		cudaStreamCreate(&kernel_stream[thr_id]);

		CUDA_SAFE_CALL(cudaMemcpyAsync(d_scratchpad[thr_id], h_scratchpad[thr_id], scratchpad_size << 3, cudaMemcpyHostToDevice, bufpad_stream[thr_id]));

		init[thr_id] = true;
	}

	throughput = WK_CUDABlocks * WK_CUDAThreads;

	cudaMemcpy(d_input[thr_id], pdata, 88, cudaMemcpyHostToDevice);
//	cudaMemset(d_retnonce[thr_id], 0xFF, 2*sizeof(uint32_t));

	if (h_scratchpad[thr_id]) {
		cudaStreamSynchronize(bufpad_stream[thr_id]);
	}

	do {
//		const uint32_t blocks = WK_CUDABlocks, threads = WK_CUDAThreads;
//		const dim3 block(blocks);
//		const dim3 thread(threads);
		uint32_t h_retnonce[2] = { UINT32_MAX, UINT32_MAX };
		uint2 target = make_uint2(ptarget[6], ptarget[7]);

		wildkeccak_kernel(thr_id, throughput, (uint32_t) nonce, target, h_retnonce);
		/*
		wk <<<block, thread, 0, kernel_stream[thr_id]>>> (d_retnonce[thr_id], d_input[thr_id], d_scratchpad[thr_id],
			(uint32_t)(scratchpad_size >> 2), nonce, ptarget[7]);
		*/

		*hashes_done = (unsigned long) (n - first + throughput);

		cudaStreamSynchronize(kernel_stream[thr_id]);
		if(h_retnonce[0] != UINT32_MAX) {
			uint8_t _ALIGN(64) cpuhash[32];
			uint32_t* vhash = (uint32_t*) cpuhash;
			uint64_t nonce64;
			memcpy(&pdata[1], &h_retnonce[0], sizeof(uint32_t));
			memcpy(&nonce64, &pdata[1], 8);
			wildkeccak_hash(cpuhash, pdata, pscratchpad_buff, scratchpad_size);
			if (!cpuhash[31] && vhash[7] <= ptarget[7] && fulltest(vhash, ptarget)) {
				work_set_target_ratio(work, vhash);
				//applog_hex(pdata,   84);
				//applog_hex(cpuhash, 32);
				//applog_hex(ptarget, 32);
				memcpy(work->nonces, &nonce64, 8);
				if (n + throughput > max_nonce) {
					*hashes_done = (unsigned long) (max_nonce - first);
				}
				work->valid_nonces = 1;
				return 1;
			} else if (vhash[7] > ptarget[7]) {
				gpu_increment_reject(thr_id);
				if (!opt_quiet)
				gpulog(LOG_WARNING, thr_id, "result for nonce %08x does not validate on CPU!", h_retnonce[0]);
			}
		}

		n += throughput;
		nonce += throughput;

	} while(!work_restart[thr_id].restart && (uint64_t) max_nonce > (uint64_t) nonce + throughput);

	*hashes_done = (unsigned long) (n - first + 1);
	return 0;
}

void wildkeccak_scratchpad_need_update(uint64_t* pscratchpad_buff)
{
	for(int i = 0; i < opt_n_threads; i++) {
		h_scratchpad[i] = pscratchpad_buff;
		if (init[i]) {
			gpulog(LOG_DEBUG, i, "Starting scratchpad update...");
			cudaMemcpyAsync(d_scratchpad[i], h_scratchpad[i], scratchpad_size << 3, cudaMemcpyHostToDevice, bufpad_stream[i]);
			work_restart[i].restart = true;
		}
	}
}

void free_wildkeccak(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaThreadSynchronize();

	cudaFree(d_scratchpad[thr_id]);
	cudaFree(d_input[thr_id]);
	cudaFree(d_retnonce[thr_id]);

	cudaStreamDestroy(bufpad_stream[thr_id]);
	cudaStreamDestroy(kernel_stream[thr_id]);

	cudaDeviceSynchronize();

	init[thr_id] = false;
}
