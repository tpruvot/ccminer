#pragma once

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdexcept>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef WIN32
#define _SNPRINTF _snprintf
#else
#define _SNPRINTF snprintf
#endif

#ifndef nullptr
#define nullptr NULL
#endif

#ifdef WIN32
#define rt_error std::runtime_error
#else
class rt_error : public std::runtime_error
{
public:
	explicit rt_error(const std::string& str) : std::runtime_error(str) {}
};
#endif

#define checkCudaErrors(call)								\
do {														\
	cudaError_t err = call;									\
	if (cudaSuccess != err) {								\
		char errorBuff[512];								\
		_SNPRINTF(errorBuff, sizeof(errorBuff) - 1,			\
			"CUDA error '%s' in func '%s' line %d",			\
			cudaGetErrorString(err), __FUNCTION__, __LINE__); \
		throw rt_error(errorBuff);				\
		}													\
} while (0)

#define checkCudaDriverErrors(call)							\
do {														\
	CUresult err = call;									\
	if (CUDA_SUCCESS != err) {								\
		char errorBuff[512];								\
		_SNPRINTF(errorBuff, sizeof(errorBuff) - 1,			\
			"CUDA error DRIVER: '%d' in func '%s' line %d", \
			err, __FUNCTION__, __LINE__);	\
		throw rt_error(errorBuff);				\
				}											\
} while (0)

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef unsigned char uchar;

struct packer_default;
struct packer_cantor;

#define MAXREALSOLS 9

struct scontainerreal {
	u32 sols[MAXREALSOLS][512];
	u32 nsols;
};

#if 0
#include <functional>
#define fn_solution std::function<void(int thr_id, const std::vector<uint32_t>&, size_t, const unsigned char*)>
#define fn_hashdone std::function<void(int thr_id)>
#define fn_cancel   std::function<bool(int thr_id)>
#else
typedef void (*fn_solution)(int thr_id, const std::vector<uint32_t>&, size_t, const unsigned char*);
typedef void (*fn_hashdone)(int thr_id);
typedef bool (*fn_cancel)(int thr_id);
#endif

template <u32 RB, u32 SM> struct equi;

// ---------------------------------------------------------------------------------------------------

struct eq_cuda_context_interface
{
	//virtual ~eq_cuda_context_interface();

	virtual void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		fn_cancel cancelf,
		fn_solution solutionf,
		fn_hashdone hashdonef);

public:
	int thread_id;
	int device_id;
	int throughput;
	int totalblocks;
	int threadsperblock;
	int threadsperblock_digits;
	size_t equi_mem_sz;
};

// ---------------------------------------------------------------------------------------------------

template <u32 RB, u32 SM, u32 SSM, u32 THREADS, typename PACKER>
class eq_cuda_context : public eq_cuda_context_interface
{
	equi<RB, SM>* device_eq;
	scontainerreal* solutions;
	CUcontext pctx;

	void solve(const char *tequihash_header,
		unsigned int tequihash_header_len,
		const char* nonce,
		unsigned int nonce_len,
		fn_cancel cancelf,
		fn_solution solutionf,
		fn_hashdone hashdonef);
public:
	eq_cuda_context(int thr_id, int dev_id);
	void freemem();
	~eq_cuda_context();
};

// RB, SM, SSM, TPB, PACKER... but any change only here will fail..
#define CONFIG_MODE_1	9, 1248, 12, 640, packer_cantor
//#define CONFIG_MODE_2	8, 640, 12, 512, packer_default
