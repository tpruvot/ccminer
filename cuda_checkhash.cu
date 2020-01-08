/**
 * This code compares final hash against target
 */
#include <stdio.h>
#include <memory.h>

#include "miner.h"

#include "cuda_helper.h"

__constant__ uint32_t pTarget[8]; // 32 bytes

// store MAX_GPUS device arrays of 8 nonces
static uint32_t* h_resNonces[MAX_GPUS] = { NULL };
static uint32_t* d_resNonces[MAX_GPUS] = { NULL };
static __thread bool init_done = false;

__host__
void cuda_check_cpu_init(int thr_id, uint32_t threads)
{
    CUDA_CALL_OR_RET(cudaMalloc(&d_resNonces[thr_id], 32));
    CUDA_SAFE_CALL(cudaMallocHost(&h_resNonces[thr_id], 32));
    init_done = true;
}

__host__
void cuda_check_cpu_free(int thr_id)
{
	if (!init_done) return;
	cudaFree(d_resNonces[thr_id]);
	cudaFreeHost(h_resNonces[thr_id]);
	d_resNonces[thr_id] = NULL;
	h_resNonces[thr_id] = NULL;
	init_done = false;
}

// Target Difficulty
__host__
void cuda_check_cpu_setTarget(const void *ptarget)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(pTarget, ptarget, 32, 0, cudaMemcpyHostToDevice));
}

/* --------------------------------------------------------------------------------------------- */

__device__ __forceinline__
static bool hashbelowtarget(const uint32_t *const __restrict__ hash, const uint32_t *const __restrict__ target)
{
	if (hash[7] > target[7])
		return false;
	if (hash[7] < target[7])
		return true;
	if (hash[6] > target[6])
		return false;
	if (hash[6] < target[6])
		return true;

	if (hash[5] > target[5])
		return false;
	if (hash[5] < target[5])
		return true;
	if (hash[4] > target[4])
		return false;
	if (hash[4] < target[4])
		return true;

	if (hash[3] > target[3])
		return false;
	if (hash[3] < target[3])
		return true;
	if (hash[2] > target[2])
		return false;
	if (hash[2] < target[2])
		return true;

	if (hash[1] > target[1])
		return false;
	if (hash[1] < target[1])
		return true;
	if (hash[0] > target[0])
		return false;

	return true;
}

__global__ __launch_bounds__(512, 4)
void cuda_checkhash_64(uint32_t threads, uint32_t startNounce, uint32_t *hash, uint32_t *resNonces)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		// shl 4 = *16 x 4 (uint32) = 64 bytes
		// todo: use only 32 bytes * threads if possible
		uint32_t *inpHash = &hash[thread << 4];

		if (resNonces[0] == UINT32_MAX) {
			if (hashbelowtarget(inpHash, pTarget))
				resNonces[0] = (startNounce + thread);
		}
	}
}

__global__ __launch_bounds__(512, 4)
void cuda_checkhash_32(uint32_t threads, uint32_t startNounce, uint32_t *hash, uint32_t *resNonces)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t *inpHash = &hash[thread << 3];

		if (resNonces[0] == UINT32_MAX) {
			if (hashbelowtarget(inpHash, pTarget))
				resNonces[0] = (startNounce + thread);
		}
	}
}

__host__
uint32_t cuda_check_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash)
{
	cudaMemset(d_resNonces[thr_id], 0xff, sizeof(uint32_t));

	const uint32_t threadsperblock = 512;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	if (bench_algo >= 0) // dont interrupt the global benchmark
		return UINT32_MAX;

	if (!init_done) {
		applog(LOG_ERR, "missing call to cuda_check_cpu_init");
		return UINT32_MAX;
	}

	cuda_checkhash_64 <<<grid, block>>> (threads, startNounce, d_inputHash, d_resNonces[thr_id]);
	cudaDeviceSynchronize();

	cudaMemcpy(h_resNonces[thr_id], d_resNonces[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return h_resNonces[thr_id][0];
}

__host__
uint32_t cuda_check_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash)
{
	cudaMemset(d_resNonces[thr_id], 0xff, sizeof(uint32_t));

	const uint32_t threadsperblock = 512;

	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	if (bench_algo >= 0) // dont interrupt the global benchmark
		return UINT32_MAX;

	if (!init_done) {
		applog(LOG_ERR, "missing call to cuda_check_cpu_init");
		return UINT32_MAX;
	}

	cuda_checkhash_32 <<<grid, block>>> (threads, startNounce, d_inputHash, d_resNonces[thr_id]);
	cudaDeviceSynchronize();

	cudaMemcpy(h_resNonces[thr_id], d_resNonces[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return h_resNonces[thr_id][0];
}

/* --------------------------------------------------------------------------------------------- */

__global__ __launch_bounds__(512, 4)
void cuda_checkhash_64_suppl(uint32_t startNounce, uint32_t *hash, uint32_t *resNonces)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	uint32_t *inpHash = &hash[thread << 4];

	if (hashbelowtarget(inpHash, pTarget)) {
		int resNum = ++resNonces[0];
		__threadfence();
		if (resNum < 8)
			resNonces[resNum] = (startNounce + thread);
	}
}

__host__
uint32_t cuda_check_hash_suppl(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash, uint8_t numNonce)
{
	uint32_t rescnt, result = 0;

	const uint32_t threadsperblock = 512;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);

	if (!init_done) {
		applog(LOG_ERR, "missing call to cuda_check_cpu_init");
		return 0;
	}

	// first element stores the count of found nonces
	cudaMemset(d_resNonces[thr_id], 0, sizeof(uint32_t));

	cuda_checkhash_64_suppl <<<grid, block>>> (startNounce, d_inputHash, d_resNonces[thr_id]);
	cudaDeviceSynchronize();

	cudaMemcpy(h_resNonces[thr_id], d_resNonces[thr_id], 32, cudaMemcpyDeviceToHost);
	rescnt = h_resNonces[thr_id][0];
	if (rescnt > numNonce) {
		if (numNonce <= rescnt) {
			result = h_resNonces[thr_id][numNonce+1];
		}
		if (opt_debug)
			applog(LOG_WARNING, "Found %d nonces: %x + %x", rescnt, h_resNonces[thr_id][1], result);
	}

	return result;
}

/* --------------------------------------------------------------------------------------------- */

__global__
void cuda_check_hash_branch_64(uint32_t threads, uint32_t startNounce, uint32_t *g_nonceVector, uint32_t *g_hash, uint32_t *resNounce)
{
	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		uint32_t nounce = g_nonceVector[thread];
		uint32_t hashPosition = (nounce - startNounce) << 4;
		uint32_t *inpHash = &g_hash[hashPosition];

		for (int i = 7; i >= 0; i--) {
			if (inpHash[i] > pTarget[i]) {
				return;
			}
			if (inpHash[i] < pTarget[i]) {
				break;
			}
		}
		if (resNounce[0] > nounce)
			resNounce[0] = nounce;
	}
}

__host__
uint32_t cuda_check_hash_branch(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_inputHash, int order)
{
	const uint32_t threadsperblock = 256;

	uint32_t result = UINT32_MAX;

	if (bench_algo >= 0) // dont interrupt the global benchmark
		return result;

	if (!init_done) {
		applog(LOG_ERR, "missing call to cuda_check_cpu_init");
		return result;
	}

	cudaMemset(d_resNonces[thr_id], 0xff, sizeof(uint32_t));

	dim3 grid((threads + threadsperblock-1)/threadsperblock);
	dim3 block(threadsperblock);

	cuda_check_hash_branch_64 <<<grid, block>>> (threads, startNounce, d_nonceVector, d_inputHash, d_resNonces[thr_id]);

	MyStreamSynchronize(NULL, order, thr_id);

	cudaMemcpy(h_resNonces[thr_id], d_resNonces[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	result = *h_resNonces[thr_id];

	return result;
}

/* Function to get the compiled Shader Model version */
int cuda_arch[MAX_GPUS] = { 0 };
__global__ void nvcc_get_arch(int *d_version)
{
	*d_version = 0;
#ifdef __CUDA_ARCH__
	*d_version = __CUDA_ARCH__;
#endif
}

__host__
int cuda_get_arch(int thr_id)
{
	int *d_version;
	int dev_id = device_map[thr_id];
	if (cuda_arch[dev_id] == 0) {
		// only do it once...
		cudaMalloc(&d_version, sizeof(int));
		nvcc_get_arch <<< 1, 1 >>> (d_version);
		cudaMemcpy(&cuda_arch[dev_id], d_version, sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(d_version);
	}
	return cuda_arch[dev_id];
}
