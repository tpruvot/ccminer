/**
 * __shfl_up require SM 3.0 arch!
 *
 * SM 2 alternative method by tpruvot@github 2015
 */
#include <stdio.h>
#include <memory.h>

#include "cuda_helper.h"

#ifdef __INTELLISENSE__
/* just for vstudio code colors */
#define __CUDA_ARCH__ 300
#define __shfl_up(var, delta, width) (0)
#endif

#if CUDA_VERSION >= 9000
#undef __shfl_up
#define __shfl_up(var, delta, width) __shfl_up_sync(0xFFFFFFFF, var, delta, width)
#endif

static uint32_t *h_numValid[MAX_GPUS];
static uint32_t *d_tempBranch1Nonces[MAX_GPUS];
static uint32_t *d_partSum[2][MAX_GPUS]; // für bis zu vier partielle Summen

// True/False tester
typedef uint32_t(*cuda_compactTestFunction_t)(uint32_t *inpHash);

__device__ uint32_t QuarkTrueTest(uint32_t *inpHash)
{
	return ((inpHash[0] & 0x08) == 0x08);
}

__device__ uint32_t QuarkFalseTest(uint32_t *inpHash)
{
	return ((inpHash[0] & 0x08) == 0);
}

__device__ cuda_compactTestFunction_t d_QuarkTrueFunction = QuarkTrueTest, d_QuarkFalseFunction = QuarkFalseTest;

cuda_compactTestFunction_t h_QuarkTrueFunction[MAX_GPUS], h_QuarkFalseFunction[MAX_GPUS];

// Setup/Alloc Function
__host__
void quark_compactTest_cpu_init(int thr_id, uint32_t threads)
{
	int dev_id = device_map[thr_id];
	cuda_get_arch(thr_id);

	cudaMemcpyFromSymbol(&h_QuarkTrueFunction[thr_id], d_QuarkTrueFunction, sizeof(cuda_compactTestFunction_t));
	cudaMemcpyFromSymbol(&h_QuarkFalseFunction[thr_id], d_QuarkFalseFunction, sizeof(cuda_compactTestFunction_t));

	if (cuda_arch[dev_id] >= 300) {
		uint32_t s1 = (threads / 256) * 2;
		CUDA_SAFE_CALL(cudaMalloc(&d_tempBranch1Nonces[thr_id], sizeof(uint32_t) * threads * 2));
		CUDA_SAFE_CALL(cudaMalloc(&d_partSum[0][thr_id], sizeof(uint32_t) * s1)); // BLOCKSIZE (Threads/Block)
		CUDA_SAFE_CALL(cudaMalloc(&d_partSum[1][thr_id], sizeof(uint32_t) * s1)); // BLOCKSIZE (Threads/Block)
	} else {
		CUDA_SAFE_CALL(cudaMalloc(&d_tempBranch1Nonces[thr_id], sizeof(uint32_t) * threads));
	}

	cudaMallocHost(&h_numValid[thr_id], 2*sizeof(uint32_t));
}

// Because all alloc should have a free...
__host__
void quark_compactTest_cpu_free(int thr_id)
{
	int dev_id = device_map[thr_id];

	cudaFreeHost(h_numValid[thr_id]);

	if (cuda_arch[dev_id] >= 300) {
		cudaFree(d_tempBranch1Nonces[thr_id]);
		cudaFree(d_partSum[0][thr_id]);
		cudaFree(d_partSum[1][thr_id]);
	} else {
		cudaFree(d_tempBranch1Nonces[thr_id]);
	}
}

__global__
void quark_compactTest_gpu_SCAN(uint32_t *data, const int width, uint32_t *partial_sums=NULL, cuda_compactTestFunction_t testFunc=NULL,
	uint32_t threads=0, uint32_t startNounce=0, uint32_t *inpHashes=NULL, uint32_t *d_validNonceTable=NULL)
{
#if __CUDA_ARCH__ >= 300
	__shared__ uint32_t sums[32];
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	//int lane_id = id % warpSize;
	int lane_id = id % width;
	// determine a warp_id within a block
	 //int warp_id = threadIdx.x / warpSize;
	int warp_id = threadIdx.x / width;

	sums[lane_id] = 0;

	// Below is the basic structure of using a shfl instruction
	// for a scan.
	// Record "value" as a variable - we accumulate it along the way
	uint32_t value;
	if(testFunc != NULL)
	{
		if (id < threads)
		{
			uint32_t *inpHash;
			if(d_validNonceTable == NULL)
			{
				// keine Nonce-Liste
				inpHash = &inpHashes[id<<4];
			} else {
				// Nonce-Liste verfügbar
				int nonce = d_validNonceTable[id] - startNounce;
				inpHash = &inpHashes[nonce<<4];
			}			
			value = (*testFunc)(inpHash);
		} else {
			value = 0;
		}
	} else {
		value = data[id];
	}

	__syncthreads();

	// Now accumulate in log steps up the chain
	// compute sums, with another thread's value who is
	// distance delta away (i).  Note
	// those threads where the thread 'i' away would have
	// been out of bounds of the warp are unaffected.  This
	// creates the scan sum.

	#pragma unroll
	for (int i=1; i<=width; i*=2)
	{
		uint32_t n = __shfl_up((int)value, i, width);
		if (lane_id >= i) value += n;
	}

	// value now holds the scan value for the individual thread
	// next sum the largest values for each warp

	// write the sum of the warp to smem
	//if (threadIdx.x % warpSize == warpSize-1)
	if (threadIdx.x % width == width-1)
	{
		sums[warp_id] = value;
	}

	__syncthreads();

	//
	// scan sum the warp sums
	// the same shfl scan operation, but performed on warp sums
	//
	if (warp_id == 0)
	{
		uint32_t warp_sum = sums[lane_id];

		for (int i=1; i<=width; i*=2)
		{
			uint32_t n = __shfl_up((int)warp_sum, i, width);
			if (lane_id >= i) warp_sum += n;
		}

		sums[lane_id] = warp_sum;
	}

	__syncthreads();

	// perform a uniform add across warps in the block
	// read neighbouring warp's sum and add it to threads value
	uint32_t blockSum = 0;

	if (warp_id > 0)
	{
		blockSum = sums[warp_id-1];
	}

	value += blockSum;

	// Now write out our result
	data[id] = value;

	// last thread has sum, write write out the block's sum
	if (partial_sums != NULL && threadIdx.x == blockDim.x-1)
	{
		partial_sums[blockIdx.x] = value;
	}
#endif // SM3+
}

// Uniform add: add partial sums array
__global__
void quark_compactTest_gpu_ADD(uint32_t *data, uint32_t *partial_sums, int len)
{
	__shared__ uint32_t buf;
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if (id > len) return;

	if (threadIdx.x == 0)
	{
		buf = partial_sums[blockIdx.x];
	}

	__syncthreads();
	data[id] += buf;
}

__global__
void quark_compactTest_gpu_SCATTER(uint32_t *sum, uint32_t *outp, cuda_compactTestFunction_t testFunc,
	uint32_t threads=0, uint32_t startNounce=0, uint32_t *inpHashes=NULL, uint32_t *d_validNonceTable=NULL)
{
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32_t actNounce = id;
	uint32_t value;
	if (id < threads)
	{
		uint32_t *inpHash;
		if(d_validNonceTable == NULL)
		{
			// keine Nonce-Liste
			inpHash = &inpHashes[id<<4];
		} else {
			// Nonce-Liste verfügbar
			int nonce = d_validNonceTable[id] - startNounce;
			actNounce = nonce;
			inpHash = &inpHashes[nonce<<4];
		}

		value = (*testFunc)(inpHash);
	} else {
		value = 0;
	}

	if (value) {
		int idx = sum[id];
		if(idx > 0)
			outp[idx-1] = startNounce + actNounce;
	}
}

__host__ static uint32_t quark_compactTest_roundUpExp(uint32_t val)
{
	if(val == 0)
		return 0;

	uint32_t mask = 0x80000000;
	while( (val & mask) == 0 ) mask = mask >> 1;

	if( (val & (~mask)) != 0 )
		return mask << 1;

	return mask;
}

__host__
void quark_compactTest_cpu_singleCompaction(int thr_id, uint32_t threads, uint32_t *nrm,uint32_t *d_nonces1,
	cuda_compactTestFunction_t function, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable)
{
	int orgThreads = threads;
	threads = quark_compactTest_roundUpExp(threads);
	// threadsPerBlock ausrechnen
	int blockSize = 256;
	int nSummen = threads / blockSize;

	int thr1 = (threads+blockSize-1) / blockSize;
	int thr2 = threads / (blockSize*blockSize);
	int blockSize2 = (nSummen < blockSize) ? nSummen : blockSize;
	int thr3 = (nSummen + blockSize2-1) / blockSize2;

	bool callThrid = (thr2 > 0) ? true : false;

	// Erster Initialscan
	quark_compactTest_gpu_SCAN <<<thr1,blockSize>>>(
		d_tempBranch1Nonces[thr_id], 32, d_partSum[0][thr_id], function, orgThreads, startNounce, inpHashes, d_validNonceTable);	

	// weitere Scans
	if(callThrid) {
		quark_compactTest_gpu_SCAN<<<thr2,blockSize>>>(d_partSum[0][thr_id], 32, d_partSum[1][thr_id]);
		quark_compactTest_gpu_SCAN<<<1, thr2>>>(d_partSum[1][thr_id], (thr2>32) ? 32 : thr2);
	} else {
		quark_compactTest_gpu_SCAN<<<thr3,blockSize2>>>(d_partSum[0][thr_id], (blockSize2>32) ? 32 : blockSize2);
	}

	// Sync + Anzahl merken
	cudaStreamSynchronize(NULL);

	if(callThrid)
		cudaMemcpy(nrm, &(d_partSum[1][thr_id])[thr2-1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
	else
		cudaMemcpy(nrm, &(d_partSum[0][thr_id])[nSummen-1], sizeof(uint32_t), cudaMemcpyDeviceToHost);

	if(callThrid) {
		quark_compactTest_gpu_ADD<<<thr2-1, blockSize>>>(d_partSum[0][thr_id]+blockSize, d_partSum[1][thr_id], blockSize*thr2);
	}
	quark_compactTest_gpu_ADD<<<thr1-1, blockSize>>>(d_tempBranch1Nonces[thr_id]+blockSize, d_partSum[0][thr_id], threads);

	quark_compactTest_gpu_SCATTER<<<thr1,blockSize,0>>>(d_tempBranch1Nonces[thr_id], d_nonces1, 
		function, orgThreads, startNounce, inpHashes, d_validNonceTable);

	// Sync
	cudaStreamSynchronize(NULL);
}

#if __CUDA_ARCH__ < 300
__global__ __launch_bounds__(128, 8)
void quark_filter_gpu_sm2(const uint32_t threads, const uint32_t* d_hash, uint32_t* d_branch2, uint32_t* d_NonceBranch)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads)
	{
		const uint32_t offset = thread * 16U; // 64U / sizeof(uint32_t);
		uint4 *psrc = (uint4*) (&d_hash[offset]);
		d_NonceBranch[thread] = ((uint8_t*)psrc)[0] & 0x8;
		if (d_NonceBranch[thread]) return;
		// uint4 = 4x uint32_t = 16 bytes
		uint4 *pdst = (uint4*) (&d_branch2[offset]);
		pdst[0] = psrc[0];
		pdst[1] = psrc[1];
		pdst[2] = psrc[2];
		pdst[3] = psrc[3];
	}
}

__global__ __launch_bounds__(128, 8)
void quark_merge_gpu_sm2(const uint32_t threads, uint32_t* d_hash, uint32_t* d_branch2, uint32_t* const d_NonceBranch)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
	if (thread < threads && !d_NonceBranch[thread])
	{
		const uint32_t offset = thread * 16U;
		uint4 *pdst = (uint4*) (&d_hash[offset]);
		uint4 *psrc = (uint4*) (&d_branch2[offset]);
		pdst[0] = psrc[0];
		pdst[1] = psrc[1];
		pdst[2] = psrc[2];
		pdst[3] = psrc[3];
	}
}
#else
__global__ void quark_filter_gpu_sm2(const uint32_t threads, const uint32_t* d_hash, uint32_t* d_branch2, uint32_t* d_NonceBranch) {}
__global__ void quark_merge_gpu_sm2(const uint32_t threads, uint32_t* d_hash, uint32_t* d_branch2, uint32_t* const d_NonceBranch) {}
#endif

__host__
uint32_t quark_filter_cpu_sm2(const int thr_id, const uint32_t threads, const uint32_t *inpHashes, uint32_t* d_branch2)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	// extract algo permution hashes to a second branch buffer
	quark_filter_gpu_sm2 <<<grid, block>>> (threads, inpHashes, d_branch2, d_tempBranch1Nonces[thr_id]);
	return threads;
}

__host__
void quark_merge_cpu_sm2(const int thr_id, const uint32_t threads, uint32_t *outpHashes, uint32_t* d_branch2)
{
	const uint32_t threadsperblock = 128;
	dim3 grid((threads + threadsperblock - 1) / threadsperblock);
	dim3 block(threadsperblock);
	// put back second branch hashes to the common buffer d_hash
	quark_merge_gpu_sm2 <<<grid, block>>> (threads, outpHashes, d_branch2, d_tempBranch1Nonces[thr_id]);
}

////// ACHTUNG: Diese funktion geht aktuell nur mit threads > 65536 (Am besten 256 * 1024 oder 256*2048)
__host__
void quark_compactTest_cpu_dualCompaction(int thr_id, uint32_t threads, uint32_t *nrm, uint32_t *d_nonces1,
	 uint32_t *d_nonces2, uint32_t startNounce, uint32_t *inpHashes, uint32_t *d_validNonceTable)
{
	quark_compactTest_cpu_singleCompaction(thr_id, threads, &nrm[0], d_nonces1, h_QuarkTrueFunction[thr_id], startNounce, inpHashes, d_validNonceTable);
	quark_compactTest_cpu_singleCompaction(thr_id, threads, &nrm[1], d_nonces2, h_QuarkFalseFunction[thr_id], startNounce, inpHashes, d_validNonceTable);
}

__host__
void quark_compactTest_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes,
	uint32_t *d_validNonceTable, uint32_t *d_nonces1, uint32_t *nrm1, uint32_t *d_nonces2, uint32_t *nrm2, int order)
{
	// Wenn validNonceTable genutzt wird, dann werden auch nur die Nonces betrachtet, die dort enthalten sind
	// "threads" ist in diesem Fall auf die Länge dieses Array's zu setzen!
	
	quark_compactTest_cpu_dualCompaction(thr_id, threads,
		h_numValid[thr_id], d_nonces1, d_nonces2,
		startNounce, inpHashes, d_validNonceTable);

	cudaStreamSynchronize(NULL); // Das original braucht zwar etwas CPU-Last, ist an dieser Stelle aber evtl besser
	*nrm1 = h_numValid[thr_id][0];
	*nrm2 = h_numValid[thr_id][1];
}

__host__
void quark_compactTest_single_false_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *inpHashes,
	uint32_t *d_validNonceTable, uint32_t *d_nonces1, uint32_t *nrm1, int order)
{
	// Wenn validNonceTable genutzt wird, dann werden auch nur die Nonces betrachtet, die dort enthalten sind
	// "threads" ist in diesem Fall auf die Länge dieses Array's zu setzen!

	quark_compactTest_cpu_singleCompaction(thr_id, threads, h_numValid[thr_id], d_nonces1, h_QuarkFalseFunction[thr_id], startNounce, inpHashes, d_validNonceTable);

	cudaStreamSynchronize(NULL);
	*nrm1 = h_numValid[thr_id][0];
}
