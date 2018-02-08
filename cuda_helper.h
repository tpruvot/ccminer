#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
/* reduce vstudio warnings (__byteperm, blockIdx...) */
#include <device_functions.h>
#include <device_launch_parameters.h>
#define __launch_bounds__(max_tpb, min_blocks)
#endif

#include <stdbool.h>
#include <stdint.h>

#ifndef UINT32_MAX
/* slackware need that */
#define UINT32_MAX UINT_MAX
#endif

#ifndef MAX_GPUS
#define MAX_GPUS 20
#endif

extern "C" short device_map[MAX_GPUS];
extern "C"  long device_sm[MAX_GPUS];
extern "C" short device_mpcount[MAX_GPUS];
extern int cuda_arch[MAX_GPUS];

// common functions
extern int cuda_get_arch(int thr_id);
extern void cuda_check_cpu_init(int thr_id, uint32_t threads);
extern void cuda_check_cpu_free(int thr_id);
extern void cuda_check_cpu_setTarget(const void *ptarget);
extern uint32_t cuda_check_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash);
extern uint32_t cuda_check_hash_suppl(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash, uint8_t numNonce);
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);
extern void cudaReportHardwareFailure(int thr_id, cudaError_t error, const char* func);
extern __device__ __device_builtin__ void __syncthreads(void);
extern __device__ __device_builtin__ void __threadfence(void);

#ifndef __CUDA_ARCH__
// define blockDim and threadIdx for host
extern const dim3 blockDim;
extern const uint3 threadIdx;
#endif

#ifndef SPH_C32
#define SPH_C32(x) (x)
// #define SPH_C32(x) ((uint32_t)(x ## U))
#endif

#ifndef SPH_C64
#define SPH_C64(x) (x)
// #define SPH_C64(x) ((uint64_t)(x ## ULL))
#endif

#ifndef SPH_T32
#define SPH_T32(x) (x)
// #define SPH_T32(x) ((x) & SPH_C32(0xFFFFFFFF))
#endif

#ifndef SPH_T64
#define SPH_T64(x) (x)
// #define SPH_T64(x) ((x) & SPH_C64(0xFFFFFFFFFFFFFFFF))
#endif

#if __CUDA_ARCH__ < 320
// Host and Compute 3.0
#define ROTL32(x, n) SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))
#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define __ldg(x) (*(x))
#else
// Compute 3.2+
#define ROTL32(x, n) __funnelshift_l( (x), (x), (n) )
#define ROTR32(x, n) __funnelshift_r( (x), (x), (n) )
#endif

#define AS_U32(addr)   *((uint32_t*)(addr))
#define AS_U64(addr)   *((uint64_t*)(addr))
#define AS_UINT2(addr) *((uint2*)(addr))
#define AS_UINT4(addr) *((uint4*)(addr))
#define AS_UL2(addr)   *((ulonglong2*)(addr))

__device__ __forceinline__ uint64_t MAKE_ULONGLONG(uint32_t LO, uint32_t HI)
{
#if __CUDA_ARCH__ >= 130
	return __double_as_longlong(__hiloint2double(HI, LO));
#else
	return (uint64_t)LO | (((uint64_t)HI) << 32);
#endif
}

// das Hi Word in einem 64 Bit Typen ersetzen
__device__ __forceinline__ uint64_t REPLACE_HIDWORD(const uint64_t &x, const uint32_t &y) {
	return (x & 0xFFFFFFFFULL) | (((uint64_t)y) << 32U);
}

// das Lo Word in einem 64 Bit Typen ersetzen
__device__ __forceinline__ uint64_t REPLACE_LODWORD(const uint64_t &x, const uint32_t &y) {
	return (x & 0xFFFFFFFF00000000ULL) | ((uint64_t)y);
}

// Endian Drehung für 32 Bit Typen
#ifdef __CUDA_ARCH__
__device__ __forceinline__ uint32_t cuda_swab32(uint32_t x)
{
	/* device */
	return __byte_perm(x, x, 0x0123);
}
#else
	/* host */
	#define cuda_swab32(x) \
	((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) | \
		(((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#endif

// das Lo Word aus einem 64 Bit Typen extrahieren
__device__ __forceinline__ uint32_t _LODWORD(const uint64_t &x) {
#if __CUDA_ARCH__ >= 130
	return (uint32_t)__double2loint(__longlong_as_double(x));
#else
	return (uint32_t)(x & 0xFFFFFFFFULL);
#endif
}

// das Hi Word aus einem 64 Bit Typen extrahieren
__device__ __forceinline__ uint32_t _HIDWORD(const uint64_t &x) {
#if __CUDA_ARCH__ >= 130
	return (uint32_t)__double2hiint(__longlong_as_double(x));
#else
	return (uint32_t)(x >> 32);
#endif
}

#ifdef __CUDA_ARCH__
__device__ __forceinline__ uint64_t cuda_swab64(uint64_t x)
{
	// Input:       77665544 33221100
	// Output:      00112233 44556677
	uint64_t result;
	//result = __byte_perm((uint32_t) x, 0, 0x0123);
	//return (result << 32) + __byte_perm(_HIDWORD(x), 0, 0x0123);
	asm("{ .reg .b32 x, y; // swab64\n\t"
		"mov.b64 {x,y}, %1;\n\t"
		"prmt.b32 x, x, 0, 0x0123;\n\t"
		"prmt.b32 y, y, 0, 0x0123;\n\t"
		"mov.b64 %0, {y,x};\n\t"
	"}\n" : "=l"(result): "l"(x));
	return result;
}
#else
	/* host */
	#define cuda_swab64(x) \
		((uint64_t)((((uint64_t)(x) & 0xff00000000000000ULL) >> 56) | \
			(((uint64_t)(x) & 0x00ff000000000000ULL) >> 40) | \
			(((uint64_t)(x) & 0x0000ff0000000000ULL) >> 24) | \
			(((uint64_t)(x) & 0x000000ff00000000ULL) >>  8) | \
			(((uint64_t)(x) & 0x00000000ff000000ULL) <<  8) | \
			(((uint64_t)(x) & 0x0000000000ff0000ULL) << 24) | \
			(((uint64_t)(x) & 0x000000000000ff00ULL) << 40) | \
			(((uint64_t)(x) & 0x00000000000000ffULL) << 56)))
#endif

// swap two uint32_t without extra registers
__device__ __host__ __forceinline__ void xchg(uint32_t &x, uint32_t &y) {
	x ^= y; y = x ^ y; x ^= y;
}
// for other types...
#define XCHG(x, y) { x ^= y; y = x ^ y; x ^= y; }

/*********************************************************************/
// Macros to catch CUDA errors in CUDA runtime calls

#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		fprintf(stderr, "Cuda error in func '%s' at line %i : %s.\n", \
		         __FUNCTION__, __LINE__, cudaGetErrorString(err) );   \
		exit(EXIT_FAILURE);                                           \
	}                                                                 \
} while (0)

#define CUDA_CALL_OR_RET(call) do {                                   \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		cudaReportHardwareFailure(thr_id, err, __FUNCTION__);         \
		return;                                                       \
	}                                                                 \
} while (0)

#define CUDA_CALL_OR_RET_X(call, ret) do {                            \
	cudaError_t err = call;                                           \
	if (cudaSuccess != err) {                                         \
		cudaReportHardwareFailure(thr_id, err, __FUNCTION__);         \
		return ret;                                                   \
	}                                                                 \
} while (0)

/*********************************************************************/
#if !defined(__CUDA_ARCH__) || defined(_WIN64)
#define USE_XOR_ASM_OPTS 0
#else
#define USE_XOR_ASM_OPTS 1
#endif

#if USE_XOR_ASM_OPTS
// device asm for whirpool
__device__ __forceinline__
uint64_t xor1(uint64_t a, uint64_t b)
{
	uint64_t result;
	asm("xor.b64 %0, %1, %2; // xor1" : "=l"(result) : "l"(a), "l"(b));
	return result;
}
#else
#define xor1(a,b) (a ^ b)
#endif

#if USE_XOR_ASM_OPTS
// device asm for whirpool
__device__ __forceinline__
uint64_t xor3(uint64_t a, uint64_t b, uint64_t c)
{
	uint64_t result;
	asm("xor.b64 %0, %2, %3; // xor3\n\t"
	    "xor.b64 %0, %0, %1;\n\t"
		/* output : input registers */
		: "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;
}
#else
#define xor3(a,b,c) (a ^ b ^ c)
#endif

#if USE_XOR_ASM_OPTS
// device asm for whirpool
__device__ __forceinline__
uint64_t xor8(uint64_t a, uint64_t b, uint64_t c, uint64_t d,uint64_t e,uint64_t f,uint64_t g, uint64_t h)
{
	uint64_t result;
	asm("xor.b64 %0, %1, %2;" : "=l"(result) : "l"(g) ,"l"(h));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(f));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(e));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(d));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(c));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(b));
	asm("xor.b64 %0, %0, %1;" : "+l"(result) : "l"(a));
	return result;
}
#else
#define xor8(a,b,c,d,e,f,g,h) ((a^b)^(c^d)^(e^f)^(g^h))
#endif

// device asm for x17
__device__ __forceinline__
uint64_t xandx(uint64_t a, uint64_t b, uint64_t c)
{
#ifdef __CUDA_ARCH__
	uint64_t result;
	asm("{ // xandx \n\t"
		".reg .u64 n;\n\t"
		"xor.b64 %0, %2, %3;\n\t"
		"and.b64 n, %0, %1;\n\t"
		"xor.b64 %0, n, %3;\n\t"
	"}\n" : "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;
#else
	return ((b^c) & a) ^ c;
#endif
}

// device asm for x17
__device__ __forceinline__
uint64_t andor(uint64_t a, uint64_t b, uint64_t c)
{
#ifdef __CUDA_ARCH__
	uint64_t result;
	asm("{ // andor\n\t"
		".reg .u64 m,n;\n\t"
		"and.b64 m,  %1, %2;\n\t"
		" or.b64 n,  %1, %2;\n\t"
		"and.b64 %0, n,  %3;\n\t"
		" or.b64 %0, %0, m;\n\t"
	"}\n" : "=l"(result) : "l"(a), "l"(b), "l"(c));
	return result;
#else
	return ((a | b) & c) | (a & b);
#endif
}

// device asm for x17
__device__ __forceinline__
uint64_t shr_t64(uint64_t x, uint32_t n)
{
#ifdef __CUDA_ARCH__
	uint64_t result;
	asm("shr.b64 %0,%1,%2;\n\t"
	: "=l"(result) : "l"(x), "r"(n));
	return result;
#else
	return x >> n;
#endif
}

__device__ __forceinline__
uint64_t shl_t64(uint64_t x, uint32_t n)
{
#ifdef __CUDA_ARCH__
	uint64_t result;
	asm("shl.b64 %0,%1,%2;\n\t"
	: "=l"(result) : "l"(x), "r"(n));
	return result;
#else
	return x << n;
#endif
}

__device__ __forceinline__
uint32_t shr_t32(uint32_t x,uint32_t n) {
#ifdef __CUDA_ARCH__
	uint32_t result;
	asm("shr.b32 %0,%1,%2;"	: "=r"(result) : "r"(x), "r"(n));
	return result;
#else
	return x >> n;
#endif
}

__device__ __forceinline__
uint32_t shl_t32(uint32_t x,uint32_t n) {
#ifdef __CUDA_ARCH__
	uint32_t result;
	asm("shl.b32 %0,%1,%2;" : "=r"(result) : "r"(x), "r"(n));
	return result;
#else
	return x << n;
#endif
}

#ifndef USE_ROT_ASM_OPT
#define USE_ROT_ASM_OPT 1
#endif

// 64-bit ROTATE RIGHT
#if __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 1
/* complicated sm >= 3.5 one (with Funnel Shifter beschleunigt), to bench */
__device__ __forceinline__
uint64_t ROTR64(const uint64_t value, const int offset) {
	uint2 result;
	if(offset < 32) {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
	} else {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
	}
	return __double_as_longlong(__hiloint2double(result.y, result.x));
}
#elif __CUDA_ARCH__ >= 120 && USE_ROT_ASM_OPT == 2
__device__ __forceinline__
uint64_t ROTR64(const uint64_t x, const int offset)
{
	uint64_t result;
	asm("{ // ROTR64 \n\t"
		".reg .b64 lhs;\n\t"
		".reg .u32 roff;\n\t"
		"shr.b64 lhs, %1, %2;\n\t"
		"sub.u32 roff, 64, %2;\n\t"
		"shl.b64 %0, %1, roff;\n\t"
		"add.u64 %0, %0, lhs;\n\t"
	"}\n" : "=l"(result) : "l"(x), "r"(offset));
	return result;
}
#else
/* host */
#define ROTR64(x, n)  (((x) >> (n)) | ((x) << (64 - (n))))
#endif

// 64-bit ROTATE LEFT
#if __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 1
__device__ __forceinline__
uint64_t ROTL64(const uint64_t value, const int offset) {
	uint2 result;
	if(offset >= 32) {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
	} else {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(__double2hiint(__longlong_as_double(value))), "r"(__double2loint(__longlong_as_double(value))), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(__double2loint(__longlong_as_double(value))), "r"(__double2hiint(__longlong_as_double(value))), "r"(offset));
	}
	return  __double_as_longlong(__hiloint2double(result.y, result.x));
}
#elif __CUDA_ARCH__ >= 120 && USE_ROT_ASM_OPT == 2
__device__ __forceinline__
uint64_t ROTL64(const uint64_t x, const int offset)
{
	uint64_t result;
	asm("{ // ROTL64 \n\t"
		".reg .b64 lhs;\n\t"
		".reg .u32 roff;\n\t"
		"shl.b64 lhs, %1, %2;\n\t"
		"sub.u32 roff, 64, %2;\n\t"
		"shr.b64 %0, %1, roff;\n\t"
		"add.u64 %0, lhs, %0;\n\t"
	"}\n" : "=l"(result) : "l"(x), "r"(offset));
	return result;
}
#elif __CUDA_ARCH__ >= 320 && USE_ROT_ASM_OPT == 3
__device__
uint64_t ROTL64(const uint64_t x, const int offset)
{
	uint64_t res;
	asm("{ // ROTL64 \n\t"
		".reg .u32 tl,th,vl,vh;\n\t"
		".reg .pred p;\n\t"
		"mov.b64 {tl,th}, %1;\n\t"
		"shf.l.wrap.b32 vl, tl, th, %2;\n\t"
		"shf.l.wrap.b32 vh, th, tl, %2;\n\t"
		"setp.lt.u32 p, %2, 32;\n\t"
		"@!p mov.b64 %0, {vl,vh};\n\t"
		"@p  mov.b64 %0, {vh,vl};\n\t"
	"}\n" : "=l"(res) : "l"(x) , "r"(offset)
	);
	return res;
}
#else
/* host */
#define ROTL64(x, n)  (((x) << (n)) | ((x) >> (64 - (n))))
#endif

__device__ __forceinline__
uint64_t SWAPDWORDS(uint64_t value)
{
#if __CUDA_ARCH__ >= 320
	uint2 temp;
	asm("mov.b64 {%0, %1}, %2; ": "=r"(temp.x), "=r"(temp.y) : "l"(value));
	asm("mov.b64 %0, {%1, %2}; ": "=l"(value) : "r"(temp.y), "r"(temp.x));
	return value;
#else
	return ROTL64(value, 32);
#endif
}

/* lyra2/bmw - uint2 vector's operators */

__device__ __forceinline__
void LOHI(uint32_t &lo, uint32_t &hi, uint64_t x) {
#ifdef __CUDA_ARCH__
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(lo), "=r"(hi) : "l"(x));
#else
	lo = (uint32_t)(x);
	hi = (uint32_t)(x >> 32);
#endif
}

static __host__ __device__ __forceinline__ uint2 vectorize(uint64_t v) {
	uint2 result;
#ifdef __CUDA_ARCH__
	asm("mov.b64 {%0,%1},%2; \n\t"
		: "=r"(result.x), "=r"(result.y) : "l"(v));
#else
	result.x = (uint32_t)(v);
	result.y = (uint32_t)(v >> 32);
#endif
	return result;
}

static __host__ __device__ __forceinline__ uint64_t devectorize(uint2 v) {
#ifdef __CUDA_ARCH__
	return MAKE_ULONGLONG(v.x, v.y);
#else
	return (((uint64_t)v.y) << 32) + v.x;
#endif
}

/**
 * uint2 direct ops by c++ operator definitions
 */
static __device__ __forceinline__ uint2 operator^ (uint2 a, uint2 b) { return make_uint2(a.x ^ b.x, a.y ^ b.y); }
static __device__ __forceinline__ uint2 operator& (uint2 a, uint2 b) { return make_uint2(a.x & b.x, a.y & b.y); }
static __device__ __forceinline__ uint2 operator| (uint2 a, uint2 b) { return make_uint2(a.x | b.x, a.y | b.y); }
static __device__ __forceinline__ uint2 operator~ (uint2 a) { return make_uint2(~a.x, ~a.y); }
static __device__ __forceinline__ void operator^= (uint2 &a, uint2 b) { a = a ^ b; }

static __device__ __forceinline__ uint2 operator+ (uint2 a, uint2 b) {
	return vectorize(devectorize(a) + devectorize(b));
}
static __device__ __forceinline__ void operator+= (uint2 &a, uint2 b) { a = a + b; }

static __device__ __forceinline__ uint2 operator- (uint2 a, uint2 b) {
	return vectorize(devectorize(a) - devectorize(b));
}
static __device__ __forceinline__ void operator-= (uint2 &a, uint2 b) { a = a - b; }

/**
 * basic multiplication between 64bit no carry outside that range (ie mul.lo.b64(a*b))
 * (what does uint64 "*" operator)
 */
static __device__ __forceinline__ uint2 operator* (uint2 a, uint2 b)
{
#ifdef __CUDA_ARCH__
	uint2 result;
	asm("{ // uint2 a*b \n\t"
		"mul.lo.u32       %0, %2, %4;  \n\t"
		"mul.hi.u32       %1, %2, %4;  \n\t"
		"mad.lo.cc.u32    %1, %3, %4, %1; \n\t"
		"madc.lo.u32      %1, %3, %5, %1; \n\t"
	"}\n" : "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
#else
	// incorrect but unused host equiv
	return make_uint2(a.x * b.x, a.y * b.y);
#endif
}

// uint2 ROR/ROL methods
__device__ __forceinline__
uint2 ROR2(const uint2 a, const int offset)
{
	uint2 result;
#if __CUDA_ARCH__ > 300
	if (offset < 32) {
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	} else /* if (offset < 64) */ {
		/* offset SHOULD BE < 64 ! */
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
#else
	if (!offset)
		result = a;
	else if (offset < 32) {
		result.y = ((a.y >> offset) | (a.x << (32 - offset)));
		result.x = ((a.x >> offset) | (a.y << (32 - offset)));
	} else if (offset == 32) {
		result.y = a.x;
		result.x = a.y;
	} else {
		result.y = ((a.x >> (offset - 32)) | (a.y << (64 - offset)));
		result.x = ((a.y >> (offset - 32)) | (a.x << (64 - offset)));
	}
#endif
	return result;
}

__device__ __forceinline__
uint2 ROL2(const uint2 a, const int offset)
{
	uint2 result;
#if __CUDA_ARCH__ > 300
	if (offset >= 32) {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	else {
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
#else
	if (!offset)
		result = a;
	else
		result = ROR2(a, 64 - offset);
#endif
	return result;
}

__device__ __forceinline__
uint2 SWAPUINT2(uint2 value)
{
	return make_uint2(value.y, value.x);
}

/* Byte aligned Rotations (lyra2) */
#ifdef __CUDA_ARCH__
__device__ __inline__ uint2 ROL8(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x6543);
	result.y = __byte_perm(a.y, a.x, 0x2107);
	return result;
}

__device__ __inline__ uint2 ROR16(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x1076);
	result.y = __byte_perm(a.y, a.x, 0x5432);
	return result;
}

__device__ __inline__ uint2 ROR24(const uint2 a)
{
	uint2 result;
	result.x = __byte_perm(a.y, a.x, 0x2107);
	result.y = __byte_perm(a.y, a.x, 0x6543);
	return result;
}
#else
#define ROL8(u)  ROL2(u, 8)
#define ROR16(u) ROR2(u,16)
#define ROR24(u) ROR2(u,24)
#endif

/* uint2 for bmw512 - to double check later */

__device__ __forceinline__
static uint2 SHL2(uint2 a, int offset)
{
#if __CUDA_ARCH__ > 300
	uint2 result;
	if (offset < 32)  {
		asm("{ // SHL2 (l) \n\t"
			"shf.l.clamp.b32 %1, %2, %3, %4; \n\t"
			"shl.b32         %0, %2, %4;     \n\t"
		"}\n" : "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	} else {
		asm("{ // SHL2 (h) \n\t"
			"shf.l.clamp.b32 %1, %2, %3, %4; \n\t"
			"shl.b32         %0, %2, %4;     \n\t"
		"}\n" : "=r"(result.x), "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	return result;
#else
	if (offset <= 32) {
		a.y = (a.y << offset) | (a.x >> (32 - offset));
		a.x = (a.x << offset);
	} else {
		a.y = (a.x << (offset-32));
		a.x = 0;
	}
	return a;
#endif
}

__device__ __forceinline__
static uint2 SHR2(uint2 a, int offset)
{
#if __CUDA_ARCH__ > 300
	uint2 result;
	if (offset<32) {
		asm("{\n\t"
			"shf.r.clamp.b32 %0,%2,%3,%4; \n\t"
			"shr.b32 %1,%3,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	} else {
		asm("{\n\t"
			"shf.l.clamp.b32 %0,%2,%3,%4; \n\t"
			"shl.b32 %1,%3,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	return result;
#else
	if (offset <= 32) {
		a.x = (a.x >> offset) | (a.y << (32 - offset));
		a.y = (a.y >> offset);
	} else {
		a.x = (a.y >> (offset - 32));
		a.y = 0;
	}
	return a;
#endif
}

// CUDA 9+ deprecated functions warnings (new mask param)
#if CUDA_VERSION >= 9000 && __CUDA_ARCH__ >= 300
#undef __shfl
#define __shfl(var, srcLane, width)  __shfl_sync(0xFFFFFFFFu, var, srcLane, width)
#undef __shfl_up
#define __shfl_up(var, delta, width) __shfl_up_sync(0xFFFFFFFF, var, delta, width)
#undef __any
#define __any(p) __any_sync(0xFFFFFFFFu, p)
#endif

#endif // #ifndef CUDA_HELPER_H
