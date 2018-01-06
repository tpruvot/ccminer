/* DJM CRAP to strip (again) made for SM 3.2+ */

#ifndef CUDA_LYRA_VECTOR_H
#define CUDA_LYRA_VECTOR_H

///////////////////////////////////////////////////////////////////////////////////
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __LDG_PTR   "l"
#else
#define __LDG_PTR   "r"
#endif

#include "cuda_helper.h"

#if __CUDA_ARCH__ < 300
#define __shfl(x, y, z) (x)
#endif

#if __CUDA_ARCH__ < 320 && !defined(__ldg4)
#define __ldg4(x) (*(x))
#endif

#if CUDA_VERSION >= 9000
#undef __shfl
#define __shfl(a, b) __shfl_sync(0xFFFFFFFF, a, b)
#endif

typedef struct __align__(32) uint8 {
	unsigned int s0, s1, s2, s3, s4, s5, s6, s7;
} uint8;

typedef struct __align__(64) uint2_8 {
	uint2 s0, s1, s2, s3, s4, s5, s6, s7;
} uint2_8;

typedef struct __align__(64) ulonglong2to8 {
	ulonglong2 l0,l1,l2,l3;
} ulonglong2to8;

typedef struct __align__(128) ulonglong8to16 {
	ulonglong2to8 lo, hi;
} ulonglong8to16;

typedef struct __align__(128) ulonglong16to32{
	ulonglong8to16 lo, hi;
} ulonglong16to32;

typedef struct __align__(128) ulonglong32to64{
	ulonglong16to32 lo, hi;
} ulonglong32to64;

typedef struct __align__(128) ulonglonglong {
	ulonglong2 s0,s1,s2,s3,s4,s5,s6,s7;
} ulonglonglong;

typedef struct __align__(64) uint16 {
	union {
		struct {unsigned int  s0, s1, s2, s3, s4, s5, s6, s7;};
		uint8 lo;
	};
	union {
		struct {unsigned int s8, s9, sa, sb, sc, sd, se, sf;};
		uint8 hi;
	};
} uint16;

typedef struct __align__(128) uint2_16 {
	union {
		struct { uint2  s0, s1, s2, s3, s4, s5, s6, s7; };
		uint2_8 lo;
	};
	union {
		struct { uint2 s8, s9, sa, sb, sc, sd, se, sf; };
		uint2_8 hi;
	};
} uint2_16;

typedef struct __align__(128) uint32 {
	uint16 lo,hi;
} uint32;

struct __align__(128) ulong8 {
	ulonglong4 s0, s1, s2, s3;
};
typedef __device_builtin__ struct ulong8 ulong8;

typedef struct __align__(128) ulonglong16{
	ulonglong4 s0, s1, s2, s3, s4, s5, s6, s7;
} ulonglong16;

typedef struct __align__(16) uint28 {
	uint2 x, y, z, w;
} uint2x4;
typedef uint2x4 uint28; /* name deprecated */

typedef struct __builtin_align__(32) uint48 {
		uint4 s0,s1;
} uint48;

typedef struct __builtin_align__(128) uint4x16{
	uint4 s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
} uint4x16;

static __inline__ __device__ ulonglong2to8 make_ulonglong2to8(ulonglong2 s0, ulonglong2 s1, ulonglong2 s2, ulonglong2 s3)
{
	ulonglong2to8 t; t.l0=s0; t.l1=s1; t.l2=s2; t.l3=s3;
	return t;
}

static __inline__ __device__ ulonglong8to16 make_ulonglong8to16(const ulonglong2to8 &s0, const ulonglong2to8 &s1)
{
	ulonglong8to16 t; t.lo = s0; t.hi = s1;
	return t;
}

static __inline__ __device__ ulonglong16to32 make_ulonglong16to32(const ulonglong8to16 &s0, const ulonglong8to16 &s1)
{
	ulonglong16to32 t; t.lo = s0; t.hi = s1;
	return t;
}

static __inline__ __device__ ulonglong32to64 make_ulonglong32to64(const ulonglong16to32 &s0, const ulonglong16to32 &s1)
{
	ulonglong32to64 t; t.lo = s0; t.hi = s1;
	return t;
}

static __inline__ __host__ __device__ ulonglonglong make_ulonglonglong(
	const ulonglong2 &s0, const ulonglong2 &s1, const ulonglong2 &s2, const ulonglong2 &s3,
	const ulonglong2 &s4, const ulonglong2 &s5)
{
	ulonglonglong t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5;
	return t;
}

static __inline__ __device__ uint48 make_uint48(uint4 s0, uint4 s1)
{
	uint48 t; t.s0 = s0; t.s1 = s1;
	return t;
}

static __inline__ __device__ uint28 make_uint28(uint2 s0, uint2 s1, uint2 s2, uint2 s3)
{
	uint28 t; t.x = s0; t.y = s1; t.z = s2; t.w = s3;
	return t;
}

static __inline__ __host__ __device__ uint4x16 make_uint4x16(
	uint4 s0, uint4 s1, uint4 s2, uint4 s3, uint4 s4, uint4 s5, uint4 s6, uint4 s7,
	uint4 s8, uint4 s9, uint4 sa, uint4 sb, uint4 sc, uint4 sd, uint4 se, uint4 sf)
{
	uint4x16 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	t.s8 = s8; t.s9 = s9; t.s10 = sa; t.s11 = sb; t.s12 = sc; t.s13 = sd; t.s14 = se; t.s15 = sf;
	return t;
}

static __inline__  __device__ uint2_16 make_uint2_16(
	uint2 s0, uint2 s1, uint2 s2, uint2 s3, uint2 s4, uint2 s5, uint2 s6, uint2 s7,
	uint2 s8, uint2 s9, uint2 sa, uint2 sb, uint2 sc, uint2 sd, uint2 se, uint2 sf)
{
	uint2_16 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	t.s8 = s8; t.s9 = s9; t.sa = sa; t.sb = sb; t.sc = sc; t.sd = sd; t.se = se; t.sf = sf;
	return t;
}

static __inline__ __host__ __device__ uint16 make_uint16(
	unsigned int s0, unsigned int s1, unsigned int s2, unsigned int s3, unsigned int s4, unsigned int s5, unsigned int s6, unsigned int s7,
	unsigned int s8, unsigned int s9, unsigned int sa, unsigned int sb, unsigned int sc, unsigned int sd, unsigned int se, unsigned int sf)
{
	uint16 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	t.s8 = s8; t.s9 = s9; t.sa = sa; t.sb = sb; t.sc = sc; t.sd = sd; t.se = se; t.sf = sf;
	return t;
}

static __inline__ __host__ __device__ uint16 make_uint16(const uint8 &a, const uint8 &b)
{
	uint16 t; t.lo=a; t.hi=b; return t;
}

static __inline__ __host__ __device__ uint32 make_uint32(const uint16 &a, const uint16 &b)
{
	uint32 t; t.lo = a; t.hi = b; return t;
}


static __inline__ __host__ __device__ uint8 make_uint8(
	unsigned int s0, unsigned int s1, unsigned int s2, unsigned int s3, unsigned int s4, unsigned int s5, unsigned int s6, unsigned int s7)
{
	uint8 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	return t;
}

static __inline__ __host__ __device__ uint2_8 make_uint2_8(
	uint2 s0, uint2 s1, uint2 s2, uint2 s3, uint2 s4, uint2 s5, uint2 s6, uint2 s7)
{
	uint2_8 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	return t;
}

static __inline__ __host__ __device__ ulonglong16 make_ulonglong16(const ulonglong4 &s0, const ulonglong4 &s1,
	const ulonglong4 &s2, const ulonglong4 &s3, const ulonglong4 &s4, const ulonglong4 &s5, const ulonglong4 &s6, const ulonglong4 &s7)
{
	ulonglong16 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3; t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	return t;
}

static __inline__ __host__ __device__ ulong8 make_ulong8(
	ulonglong4 s0, ulonglong4 s1, ulonglong4 s2, ulonglong4 s3)
{
	ulong8 t; t.s0 = s0; t.s1 = s1; t.s2 = s2; t.s3 = s3;// t.s4 = s4; t.s5 = s5; t.s6 = s6; t.s7 = s7;
	return t;
}


static __forceinline__ __device__ uchar4 operator^ (uchar4 a, uchar4 b) { return make_uchar4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w); }
static __forceinline__ __device__ uchar4 operator+ (uchar4 a, uchar4 b) { return make_uchar4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

static __forceinline__ __device__ uint4 operator+ (uint4 a, uint4 b) { return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

static __forceinline__ __device__ ulonglong4 operator^ (ulonglong4 a, ulonglong4 b) { return make_ulonglong4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w); }
static __forceinline__ __device__ ulonglong4 operator+ (ulonglong4 a, ulonglong4 b) { return make_ulonglong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __forceinline__ __device__ ulonglong2 operator^ (ulonglong2 a, ulonglong2 b) { return make_ulonglong2(a.x ^ b.x, a.y ^ b.y); }
static __forceinline__ __device__ ulonglong2 operator+ (ulonglong2 a, ulonglong2 b) { return make_ulonglong2(a.x + b.x, a.y + b.y); }

static __forceinline__ __device__ ulong8 operator^ (const ulong8 &a, const ulong8 &b) {
	return make_ulong8(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3);
}

static __forceinline__ __device__ ulong8 operator+ (const ulong8 &a, const ulong8 &b) {
	return make_ulong8(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3);
}

static __forceinline__ __device__  __host__ uint8 operator^ (const uint8 &a, const uint8 &b) { return make_uint8(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3, a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7); }

static __forceinline__ __device__  __host__ uint8 operator+ (const uint8 &a, const uint8 &b) { return make_uint8(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7); }

static __forceinline__ __device__   uint2_8 operator^ (const uint2_8 &a, const uint2_8 &b) { return make_uint2_8(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3, a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7); }

static __forceinline__ __device__   uint2_8 operator+ (const uint2_8 &a, const uint2_8 &b) { return make_uint2_8(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7); }


////////////// mess++ //////

static __forceinline__ __device__  uint28 operator^ (const uint28 &a, const uint28 &b) {
	return make_uint28(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

static __forceinline__ __device__  uint28 operator+ (const uint28 &a, const uint28 &b) {
	return make_uint28(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

static __forceinline__ __device__  uint48 operator+ (const uint48 &a, const uint48 &b) {
	return make_uint48(a.s0 + b.s0, a.s1 + b.s1);
}

/////////////////////////

static __forceinline__ __device__ __host__ uint16 operator^ (const uint16 &a, const uint16 &b) {
	return make_uint16(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3, a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7,
		a.s8 ^ b.s8, a.s9 ^ b.s9, a.sa ^ b.sa, a.sb ^ b.sb, a.sc ^ b.sc, a.sd ^ b.sd, a.se ^ b.se, a.sf ^ b.sf);
}

static __forceinline__ __device__  __host__ uint16 operator+ (const uint16 &a, const uint16 &b) {
	return make_uint16(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7,
		a.s8 + b.s8, a.s9 + b.s9, a.sa + b.sa, a.sb + b.sb, a.sc + b.sc, a.sd + b.sd, a.se + b.se, a.sf + b.sf);
}

static __forceinline__ __device__  uint2_16 operator^ (const uint2_16 &a, const uint2_16 &b) {
	return make_uint2_16(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3, a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7,
		a.s8 ^ b.s8, a.s9 ^ b.s9, a.sa ^ b.sa, a.sb ^ b.sb, a.sc ^ b.sc, a.sd ^ b.sd, a.se ^ b.se, a.sf ^ b.sf);
}

static __forceinline__ __device__  uint2_16 operator+ (const uint2_16 &a, const uint2_16 &b) {
	return make_uint2_16(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7,
		a.s8 + b.s8, a.s9 + b.s9, a.sa + b.sa, a.sb + b.sb, a.sc + b.sc, a.sd + b.sd, a.se + b.se, a.sf + b.sf);
}

static __forceinline__ __device__  uint32 operator^ (const uint32 &a, const uint32 &b) {
	return make_uint32(a.lo ^ b.lo, a.hi ^ b.hi);
}

static __forceinline__ __device__  uint32 operator+ (const uint32 &a, const uint32 &b) {
	return make_uint32(a.lo + b.lo, a.hi + b.hi);
}

static __forceinline__ __device__ ulonglong16 operator^ (const ulonglong16 &a, const ulonglong16 &b) {
	return make_ulonglong16(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3, a.s4 ^ b.s4, a.s5 ^ b.s5, a.s6 ^ b.s6, a.s7 ^ b.s7);
}

static __forceinline__ __device__ ulonglong16 operator+ (const ulonglong16 &a, const ulonglong16 &b) {
	return make_ulonglong16(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5, a.s6 + b.s6, a.s7 + b.s7);
}

static __forceinline__ __device__ void operator^= (ulong8 &a, const ulong8 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator^= (uint28 &a, const uint28 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator+= (uint28 &a, const uint28 &b) { a = a + b; }

static __forceinline__ __device__ void operator^= (uint2_8 &a, const uint2_8 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator+= (uint2_8 &a, const uint2_8 &b) { a = a + b; }

static __forceinline__ __device__ void operator^= (uint32 &a, const uint32 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator+= (uint32 &a, const uint32 &b) { a = a + b; }

static __forceinline__ __device__ void operator^= (uchar4 &a, uchar4 b) { a = a ^ b; }

static __forceinline__ __device__  __host__ void operator^= (uint8 &a, const uint8 &b) { a = a ^ b; }
static __forceinline__ __device__  __host__ void operator^= (uint16 &a, const uint16 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator^= (ulonglong16 &a, const ulonglong16 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator^= (ulonglong4 &a, const ulonglong4 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator+= (ulonglong4 &a, const ulonglong4 &b) { a = a + b; }

static __forceinline__ __device__ void operator^= (ulonglong2 &a, const ulonglong2 &b) { a = a ^ b; }
static __forceinline__ __device__ void operator+= (ulonglong2 &a, const ulonglong2 &b) { a = a + b; }

static __forceinline__ __device__
ulonglong2to8 operator^ (const ulonglong2to8 &a, const ulonglong2to8 &b)
{
	return make_ulonglong2to8(a.l0 ^ b.l0, a.l1 ^ b.l1, a.l2 ^ b.l2, a.l3 ^ b.l3);
}
static __forceinline__ __device__
ulonglong2to8 operator+ (const ulonglong2to8 &a, const ulonglong2to8 &b)
{
	return make_ulonglong2to8(a.l0 + b.l0, a.l1 + b.l1, a.l2 + b.l2, a.l3 + b.l3);
}

static __forceinline__ __device__
ulonglong8to16 operator^ (const ulonglong8to16 &a, const ulonglong8to16 &b)
{
	return make_ulonglong8to16(a.lo ^ b.lo, a.hi ^ b.hi);
}

static __forceinline__ __device__
ulonglong8to16 operator+ (const ulonglong8to16 &a, const ulonglong8to16 &b)
{
	return make_ulonglong8to16(a.lo + b.lo, a.hi + b.hi);
}

static __forceinline__ __device__
ulonglong16to32 operator^ (const ulonglong16to32 &a, const ulonglong16to32 &b)
{
	return make_ulonglong16to32(a.lo ^ b.lo, a.hi ^ b.hi);
}

static __forceinline__ __device__
ulonglong16to32 operator+ (const ulonglong16to32 &a, const ulonglong16to32 &b)
{
	return make_ulonglong16to32(a.lo + b.lo, a.hi + b.hi);
}

static __forceinline__ __device__
ulonglong32to64 operator^ (const ulonglong32to64 &a, const ulonglong32to64 &b)
{
	return make_ulonglong32to64(a.lo ^ b.lo, a.hi ^ b.hi);
}

static __forceinline__ __device__
ulonglong32to64 operator+ (const ulonglong32to64 &a, const ulonglong32to64 &b)
{
	return make_ulonglong32to64(a.lo + b.lo, a.hi + b.hi);
}

static __forceinline__ __device__ ulonglonglong operator^ (const ulonglonglong &a, const ulonglonglong &b) {
	return make_ulonglonglong(a.s0 ^ b.s0, a.s1 ^ b.s1, a.s2 ^ b.s2, a.s3 ^ b.s3, a.s4 ^ b.s4, a.s5 ^ b.s5);
}

static __forceinline__ __device__ ulonglonglong operator+ (const ulonglonglong &a, const ulonglonglong &b) {
	return make_ulonglonglong(a.s0 + b.s0, a.s1 + b.s1, a.s2 + b.s2, a.s3 + b.s3, a.s4 + b.s4, a.s5 + b.s5);
}

static __forceinline__ __device__ void operator^= (ulonglong2to8 &a, const ulonglong2to8 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator+= (uint4 &a, uint4 b) { a = a + b; }
static __forceinline__ __device__ void operator+= (uchar4 &a, uchar4 b) { a = a + b; }
static __forceinline__ __device__  __host__ void operator+= (uint8 &a, const uint8 &b) { a = a + b; }
static __forceinline__ __device__  __host__ void operator+= (uint16 &a, const uint16 &b) { a = a + b; }
static __forceinline__ __device__   void operator+= (uint2_16 &a, const uint2_16 &b) { a = a + b; }
static __forceinline__ __device__   void operator^= (uint2_16 &a, const uint2_16 &b) { a = a + b; }

static __forceinline__ __device__ void operator+= (ulong8 &a, const ulong8 &b) { a = a + b; }
static __forceinline__ __device__ void operator+= (ulonglong16 &a, const ulonglong16 &b) { a = a + b; }
static __forceinline__ __device__ void operator+= (ulonglong8to16 &a, const ulonglong8to16 &b) { a = a + b; }
static __forceinline__ __device__ void operator^= (ulonglong8to16 &a, const ulonglong8to16 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator+= (ulonglong16to32 &a, const ulonglong16to32 &b) { a = a + b; }
static __forceinline__ __device__ void operator^= (ulonglong16to32 &a, const ulonglong16to32 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator+= (ulonglong32to64 &a, const ulonglong32to64 &b) { a = a + b; }
static __forceinline__ __device__ void operator^= (ulonglong32to64 &a, const ulonglong32to64 &b) { a = a ^ b; }

static __forceinline__ __device__ void operator+= (ulonglonglong &a, const ulonglonglong &b) { a = a + b; }
static __forceinline__ __device__ void operator^= (ulonglonglong &a, const ulonglonglong &b) { a = a ^ b; }

#if __CUDA_ARCH__ < 320

#define rotate ROTL32
#define rotateR ROTR32

#else

static __forceinline__ __device__ uint4 rotate4(uint4 vec4, uint32_t shift)
{
	uint4 ret;
	asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(ret.x) : "r"(vec4.x), "r"(vec4.x), "r"(shift));
	asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(ret.y) : "r"(vec4.y), "r"(vec4.y), "r"(shift));
	asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(ret.z) : "r"(vec4.z), "r"(vec4.z), "r"(shift));
	asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(ret.w) : "r"(vec4.w), "r"(vec4.w), "r"(shift));
	return ret;
}

static __forceinline__ __device__ uint4 rotate4R(uint4 vec4, uint32_t shift)
{
	uint4 ret;
	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(ret.x) : "r"(vec4.x), "r"(vec4.x), "r"(shift));
	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(ret.y) : "r"(vec4.y), "r"(vec4.y), "r"(shift));
	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(ret.z) : "r"(vec4.z), "r"(vec4.z), "r"(shift));
	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(ret.w) : "r"(vec4.w), "r"(vec4.w), "r"(shift));
	return ret;
}

static __forceinline__ __device__ uint32_t rotate(uint32_t vec4, uint32_t shift)
{
	uint32_t ret;
	asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(vec4), "r"(vec4), "r"(shift));
	return ret;
}

static __forceinline__ __device__ uint32_t rotateR(uint32_t vec4, uint32_t shift)
{
	uint32_t ret;
	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(ret) : "r"(vec4), "r"(vec4), "r"(shift));
	return ret;
}

static __device__ __inline__ ulonglong4 __ldg4(const ulonglong4 *ptr)
{
	ulonglong4 ret;
	asm("ld.global.nc.v2.u64 {%0,%1}, [%2];"  : "=l"(ret.x), "=l"(ret.y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v2.u64 {%0,%1}, [%2+16];"  : "=l"(ret.z), "=l"(ret.w) : __LDG_PTR(ptr));
	return ret;
}

static __device__ __inline__ void ldg4(const ulonglong4 *ptr,ulonglong4 *ret)
{
	asm("ld.global.nc.v2.u64 {%0,%1}, [%2];"     : "=l"(ret[0].x), "=l"(ret[0].y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v2.u64 {%0,%1}, [%2+16];"  : "=l"(ret[0].z), "=l"(ret[0].w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v2.u64 {%0,%1}, [%2+32];"  : "=l"(ret[1].x), "=l"(ret[1].y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v2.u64 {%0,%1}, [%2+48];"  : "=l"(ret[1].z), "=l"(ret[1].w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v2.u64 {%0,%1}, [%2+64];"  : "=l"(ret[2].x), "=l"(ret[2].y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v2.u64 {%0,%1}, [%2+80];"  : "=l"(ret[2].z), "=l"(ret[2].w) : __LDG_PTR(ptr));
}

static __device__ __inline__ uint28 __ldg4(const uint28 *ptr)
{
	uint28 ret;
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.x.x), "=r"(ret.x.y), "=r"(ret.y.x), "=r"(ret.y.y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];" : "=r"(ret.z.x), "=r"(ret.z.y), "=r"(ret.w.x), "=r"(ret.w.y) : __LDG_PTR(ptr));
	return ret;
}

static __device__ __inline__ uint48 __ldg4(const uint48 *ptr)
{
	uint48 ret;
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret.s0.x), "=r"(ret.s0.y), "=r"(ret.s0.z), "=r"(ret.s0.w) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];" : "=r"(ret.s1.x), "=r"(ret.s1.y), "=r"(ret.s1.z), "=r"(ret.s1.w) : __LDG_PTR(ptr));
	return ret;
}

static __device__ __inline__ void ldg4(const uint28 *ptr, uint28 *ret)
{
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"  : "=r"(ret[0].x.x), "=r"(ret[0].x.y), "=r"(ret[0].y.x), "=r"(ret[0].y.y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+16];" : "=r"(ret[0].z.x), "=r"(ret[0].z.y), "=r"(ret[0].w.x), "=r"(ret[0].w.y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+32];"  : "=r"(ret[1].x.x), "=r"(ret[1].x.y), "=r"(ret[1].y.x), "=r"(ret[1].y.y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+48];" : "=r"(ret[1].z.x), "=r"(ret[1].z.y), "=r"(ret[1].w.x), "=r"(ret[1].w.y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+64];"  : "=r"(ret[2].x.x), "=r"(ret[2].x.y), "=r"(ret[2].y.x), "=r"(ret[2].y.y) : __LDG_PTR(ptr));
	asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4+80];" : "=r"(ret[2].z.x), "=r"(ret[2].z.y), "=r"(ret[2].w.x), "=r"(ret[2].w.y) : __LDG_PTR(ptr));
}

#endif /* __CUDA_ARCH__ < 320 */


static __forceinline__ __device__ uint8 swapvec(const uint8 &buf)
{
	uint8 vec;
	vec.s0 = cuda_swab32(buf.s0);
	vec.s1 = cuda_swab32(buf.s1);
	vec.s2 = cuda_swab32(buf.s2);
	vec.s3 = cuda_swab32(buf.s3);
	vec.s4 = cuda_swab32(buf.s4);
	vec.s5 = cuda_swab32(buf.s5);
	vec.s6 = cuda_swab32(buf.s6);
	vec.s7 = cuda_swab32(buf.s7);
	return vec;
}

static __forceinline__ __device__ uint8 swapvec(const uint8 *buf)
{
	uint8 vec;
	vec.s0 = cuda_swab32(buf[0].s0);
	vec.s1 = cuda_swab32(buf[0].s1);
	vec.s2 = cuda_swab32(buf[0].s2);
	vec.s3 = cuda_swab32(buf[0].s3);
	vec.s4 = cuda_swab32(buf[0].s4);
	vec.s5 = cuda_swab32(buf[0].s5);
	vec.s6 = cuda_swab32(buf[0].s6);
	vec.s7 = cuda_swab32(buf[0].s7);
	return vec;
}

static __forceinline__ __device__ uint16 swapvec(const uint16 *buf)
{
	uint16 vec;
	vec.s0 = cuda_swab32(buf[0].s0);
	vec.s1 = cuda_swab32(buf[0].s1);
	vec.s2 = cuda_swab32(buf[0].s2);
	vec.s3 = cuda_swab32(buf[0].s3);
	vec.s4 = cuda_swab32(buf[0].s4);
	vec.s5 = cuda_swab32(buf[0].s5);
	vec.s6 = cuda_swab32(buf[0].s6);
	vec.s7 = cuda_swab32(buf[0].s7);
	vec.s8 = cuda_swab32(buf[0].s8);
	vec.s9 = cuda_swab32(buf[0].s9);
	vec.sa = cuda_swab32(buf[0].sa);
	vec.sb = cuda_swab32(buf[0].sb);
	vec.sc = cuda_swab32(buf[0].sc);
	vec.sd = cuda_swab32(buf[0].sd);
	vec.se = cuda_swab32(buf[0].se);
	vec.sf = cuda_swab32(buf[0].sf);
	return vec;
}

static __forceinline__ __device__ uint16 swapvec(const uint16 &buf)
{
	uint16 vec;
	vec.s0 = cuda_swab32(buf.s0);
	vec.s1 = cuda_swab32(buf.s1);
	vec.s2 = cuda_swab32(buf.s2);
	vec.s3 = cuda_swab32(buf.s3);
	vec.s4 = cuda_swab32(buf.s4);
	vec.s5 = cuda_swab32(buf.s5);
	vec.s6 = cuda_swab32(buf.s6);
	vec.s7 = cuda_swab32(buf.s7);
	vec.s8 = cuda_swab32(buf.s8);
	vec.s9 = cuda_swab32(buf.s9);
	vec.sa = cuda_swab32(buf.sa);
	vec.sb = cuda_swab32(buf.sb);
	vec.sc = cuda_swab32(buf.sc);
	vec.sd = cuda_swab32(buf.sd);
	vec.se = cuda_swab32(buf.se);
	vec.sf = cuda_swab32(buf.sf);
	return vec;
}

static __device__ __forceinline__ uint28 shuffle4(const uint28 &var, int lane)
{
#if __CUDA_ARCH__ >= 300
	uint28 res;
	res.x.x = __shfl(var.x.x, lane);
	res.x.y = __shfl(var.x.y, lane);
	res.y.x = __shfl(var.y.x, lane);
	res.y.y = __shfl(var.y.y, lane);
	res.z.x = __shfl(var.z.x, lane);
	res.z.y = __shfl(var.z.y, lane);
	res.w.x = __shfl(var.w.x, lane);
	res.w.y = __shfl(var.w.y, lane);
	return res;
#else
	return var;
#endif
}

static __device__ __forceinline__ ulonglong4 shuffle4(ulonglong4 var, int lane)
{
#if __CUDA_ARCH__ >= 300
	ulonglong4 res;
    uint2 temp;
	temp = vectorize(var.x);
	temp.x = __shfl(temp.x, lane);
	temp.y = __shfl(temp.y, lane);
	res.x = devectorize(temp);
	temp = vectorize(var.y);
	temp.x = __shfl(temp.x, lane);
	temp.y = __shfl(temp.y, lane);
	res.y = devectorize(temp);
	temp = vectorize(var.z);
	temp.x = __shfl(temp.x, lane);
	temp.y = __shfl(temp.y, lane);
	res.z = devectorize(temp);
	temp = vectorize(var.w);
	temp.x = __shfl(temp.x, lane);
	temp.y = __shfl(temp.y, lane);
	res.w = devectorize(temp);
	return res;
#else
	return var;
#endif
}

#endif // #ifndef CUDA_LYRA_VECTOR_H
