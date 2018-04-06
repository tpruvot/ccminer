#ifndef __MINER_H__
#define __MINER_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <ccminer-config.h>

#include <stdbool.h>
#include <inttypes.h>
#include <sys/time.h>
#include <pthread.h>
#include <jansson.h>
#include <curl/curl.h>

#ifdef _MSC_VER
#undef HAVE_ALLOCA_H
#undef HAVE_SYSLOG_H
#endif

#ifdef STDC_HEADERS
# include <stdlib.h>
# include <stddef.h>
#else
# ifdef HAVE_STDLIB_H
#  include <stdlib.h>
# endif
#endif

#ifdef HAVE_ALLOCA_H
# include <alloca.h>
#elif !defined alloca
# ifdef __GNUC__
#  define alloca __builtin_alloca
# elif defined _AIX
#  define alloca __alloca
# elif defined _MSC_VER
#  include <malloc.h>
#  define alloca _alloca
# elif !defined HAVE_ALLOCA
void *alloca (size_t);
# endif
#endif

#include "compat.h"

#ifdef __INTELLISENSE__
/* should be in stdint.h but... */
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
typedef __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef __int16 int16_t;
typedef unsigned __int16 uint16_t;
typedef __int16 int8_t;
typedef unsigned __int16 uint8_t;

typedef unsigned __int32 time_t;
typedef char *  va_list;
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
# undef _ALIGN
# define _ALIGN(x) __align__(x)
#endif

#ifdef HAVE_SYSLOG_H
#include <syslog.h>
#define LOG_BLUE 0x10
#define LOG_RAW  0x99
#else
enum {
	LOG_ERR,
	LOG_WARNING,
	LOG_NOTICE,
	LOG_INFO,
	LOG_DEBUG,
	/* custom notices */
	LOG_BLUE = 0x10,
	LOG_RAW  = 0x99
};
#endif

typedef unsigned char uchar;

#undef unlikely
#undef likely
#if defined(__GNUC__) && (__GNUC__ > 2) && defined(__OPTIMIZE__)
#define unlikely(expr) (__builtin_expect(!!(expr), 0))
#define likely(expr) (__builtin_expect(!!(expr), 1))
#else
#define unlikely(expr) (expr)
#define likely(expr) (expr)
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

#ifndef max
# define max(a, b)  ((a) > (b) ? (a) : (b))
#endif
#ifndef min
# define min(a, b)  ((a) < (b) ? (a) : (b))
#endif

#ifndef UINT32_MAX
/* for gcc 4.4 */
#define UINT32_MAX UINT_MAX
#endif

static inline bool is_windows(void) {
#ifdef WIN32
        return 1;
#else
        return 0;
#endif
}

static inline bool is_x64(void) {
#if defined(__x86_64__) || defined(_WIN64) || defined(__aarch64__)
	return 1;
#elif defined(__amd64__) || defined(__amd64) || defined(_M_X64) || defined(_M_IA64)
	return 1;
#else
	return 0;
#endif
}

#if ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
#define WANT_BUILTIN_BSWAP
#else
#define bswap_32(x) ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) \
                   | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#define bswap_64(x) (((uint64_t) bswap_32((uint32_t)((x) & 0xffffffffu)) << 32) \
                   | (uint64_t) bswap_32((uint32_t)((x) >> 32)))
#endif

static inline uint32_t swab32(uint32_t v)
{
#ifdef WANT_BUILTIN_BSWAP
	return __builtin_bswap32(v);
#else
	return bswap_32(v);
#endif
}

static inline uint64_t swab64(uint64_t v)
{
#ifdef WANT_BUILTIN_BSWAP
	return __builtin_bswap64(v);
#else
	return bswap_64(v);
#endif
}

static inline void swab256(void *dest_p, const void *src_p)
{
	uint32_t *dest = (uint32_t *) dest_p;
	const uint32_t *src = (const uint32_t *) src_p;

	dest[0] = swab32(src[7]);
	dest[1] = swab32(src[6]);
	dest[2] = swab32(src[5]);
	dest[3] = swab32(src[4]);
	dest[4] = swab32(src[3]);
	dest[5] = swab32(src[2]);
	dest[6] = swab32(src[1]);
	dest[7] = swab32(src[0]);
}

#ifdef HAVE_SYS_ENDIAN_H
#include <sys/endian.h>
#endif

#if !HAVE_DECL_BE32DEC
static inline uint32_t be32dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint32_t)(p[3]) + ((uint32_t)(p[2]) << 8) +
	    ((uint32_t)(p[1]) << 16) + ((uint32_t)(p[0]) << 24));
}
#endif

#if !HAVE_DECL_LE32DEC
static inline uint32_t le32dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint32_t)(p[0]) + ((uint32_t)(p[1]) << 8) +
	    ((uint32_t)(p[2]) << 16) + ((uint32_t)(p[3]) << 24));
}
#endif

#if !HAVE_DECL_BE32ENC
static inline void be32enc(void *pp, uint32_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[3] = x & 0xff;
	p[2] = (x >> 8) & 0xff;
	p[1] = (x >> 16) & 0xff;
	p[0] = (x >> 24) & 0xff;
}
#endif

#if !HAVE_DECL_LE32ENC
static inline void le32enc(void *pp, uint32_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[0] = x & 0xff;
	p[1] = (x >> 8) & 0xff;
	p[2] = (x >> 16) & 0xff;
	p[3] = (x >> 24) & 0xff;
}
#endif

#if !HAVE_DECL_BE16DEC
static inline uint16_t be16dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint16_t)(p[1]) + ((uint16_t)(p[0]) << 8));
}
#endif

#if !HAVE_DECL_BE16ENC
static inline void be16enc(void *pp, uint16_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[1] = x & 0xff;
	p[0] = (x >> 8) & 0xff;
}
#endif

#if !HAVE_DECL_LE16DEC
static inline uint16_t le16dec(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;
	return ((uint16_t)(p[0]) + ((uint16_t)(p[1]) << 8));
}
#endif

#if !HAVE_DECL_LE16ENC
static inline void le16enc(void *pp, uint16_t x)
{
	uint8_t *p = (uint8_t *)pp;
	p[0] = x & 0xff;
	p[1] = (x >> 8) & 0xff;
}
#endif

/* used for struct work */
void *aligned_calloc(int size);
void aligned_free(void *ptr);

#if JANSSON_MAJOR_VERSION >= 2
#define JSON_LOADS(str, err_ptr) json_loads((str), 0, (err_ptr))
#define JSON_LOADF(str, err_ptr) json_load_file((str), 0, (err_ptr))
#else
#define JSON_LOADS(str, err_ptr) json_loads((str), (err_ptr))
#define JSON_LOADF(str, err_ptr) json_load_file((str), (err_ptr))
#endif

json_t * json_load_url(char* cfg_url, json_error_t *err);

#define USER_AGENT PACKAGE_NAME "/" PACKAGE_VERSION

void sha256_init(uint32_t *state);
void sha256_transform(uint32_t *state, const uint32_t *block, int swap);
void sha256d(unsigned char *hash, const unsigned char *data, int len);

#define HAVE_SHA256_4WAY 0
#define HAVE_SHA256_8WAY 0

struct work;

extern int scanhash_bastion(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_blake256(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done, int8_t blakerounds);
extern int scanhash_blake2s(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_bmw(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_c11(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_cryptolight(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_cryptonight(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_decred(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_deep(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_equihash(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_keccak256(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_fresh(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_fugue256(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_groestlcoin(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_hmq17(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_heavy(int thr_id,struct work *work, uint32_t max_nonce, unsigned long *hashes_done, uint32_t maxvote, int blocklen);
extern int scanhash_hsr(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_jha(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_jackpot(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done); // quark method
extern int scanhash_lbry(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_luffa(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_lyra2(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_lyra2v2(int thr_id,struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_lyra2Z(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_myriad(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_neoscrypt(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_nist5(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_pentablake(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_phi(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_polytimos(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_quark(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_qubit(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_sha256d(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_sha256t(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_sia(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_sib(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_skeincoin(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_skein2(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_skunk(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_s3(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_timetravel(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_tribus(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_bitcore(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_vanilla(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done, int8_t blake_rounds);
extern int scanhash_veltor(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_whirl(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_wildkeccak(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_x11evo(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_x11(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_x12(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_x13(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_x14(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_x15(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_x16r(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_x16s(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_x17(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done);
extern int scanhash_zr5(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done);

extern int scanhash_scrypt(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done,
	unsigned char *scratchbuf, struct timeval *tv_start, struct timeval *tv_end);
extern int scanhash_scrypt_jane(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done,
	unsigned char *scratchbuf, struct timeval *tv_start, struct timeval *tv_end);

/* free device allocated memory per algo */
void algo_free_all(int thr_id);

extern void free_bastion(int thr_id);
extern void free_bitcore(int thr_id);
extern void free_blake256(int thr_id);
extern void free_blake2s(int thr_id);
extern void free_bmw(int thr_id);
extern void free_c11(int thr_id);
extern void free_cryptolight(int thr_id);
extern void free_cryptonight(int thr_id);
extern void free_decred(int thr_id);
extern void free_deep(int thr_id);
extern void free_equihash(int thr_id);
extern void free_keccak256(int thr_id);
extern void free_fresh(int thr_id);
extern void free_fugue256(int thr_id);
extern void free_groestlcoin(int thr_id);
extern void free_heavy(int thr_id);
extern void free_hmq17(int thr_id);
extern void free_hsr(int thr_id);
extern void free_jackpot(int thr_id);
extern void free_jha(int thr_id);
extern void free_lbry(int thr_id);
extern void free_luffa(int thr_id);
extern void free_lyra2(int thr_id);
extern void free_lyra2v2(int thr_id);
extern void free_lyra2Z(int thr_id);
extern void free_myriad(int thr_id);
extern void free_neoscrypt(int thr_id);
extern void free_nist5(int thr_id);
extern void free_pentablake(int thr_id);
extern void free_phi(int thr_id);
extern void free_polytimos(int thr_id);
extern void free_quark(int thr_id);
extern void free_qubit(int thr_id);
extern void free_sha256d(int thr_id);
extern void free_sha256t(int thr_id);
extern void free_sia(int thr_id);
extern void free_sib(int thr_id);
extern void free_skeincoin(int thr_id);
extern void free_skein2(int thr_id);
extern void free_skunk(int thr_id);
extern void free_s3(int thr_id);
extern void free_timetravel(int thr_id);
extern void free_tribus(int thr_id);
extern void free_bitcore(int thr_id);
extern void free_vanilla(int thr_id);
extern void free_veltor(int thr_id);
extern void free_whirl(int thr_id);
extern void free_wildkeccak(int thr_id);
extern void free_x11evo(int thr_id);
extern void free_x11(int thr_id);
extern void free_x12(int thr_id);
extern void free_x13(int thr_id);
extern void free_x14(int thr_id);
extern void free_x15(int thr_id);
extern void free_x16r(int thr_id);
extern void free_x16s(int thr_id);
extern void free_x17(int thr_id);
extern void free_zr5(int thr_id);
//extern void free_sha256d(int thr_id);
extern void free_scrypt(int thr_id);
extern void free_scrypt_jane(int thr_id);

/* api related */
void *api_thread(void *userdata);
void api_set_throughput(int thr_id, uint32_t throughput);
void gpu_increment_reject(int thr_id);

struct monitor_info {
	uint32_t gpu_temp;
	uint32_t gpu_fan;
	uint32_t gpu_clock;
	uint32_t gpu_memclock;
	uint32_t gpu_power;

	pthread_mutex_t lock;
	pthread_cond_t sampling_signal;
	volatile bool sampling_flag;
	uint32_t tm_displayed;
};

struct cgpu_info {
	uint8_t gpu_id;
	uint8_t thr_id;
	uint16_t hw_errors;
	unsigned accepted;
	uint32_t rejected;
	double khashes;
	int has_monitoring;
	float gpu_temp;
	uint16_t gpu_fan;
	uint16_t gpu_fan_rpm;
	uint16_t gpu_arch;
	uint32_t gpu_clock;
	uint32_t gpu_memclock;
	uint64_t gpu_mem;
	uint64_t gpu_memfree;
	uint32_t gpu_power;
	uint32_t gpu_plimit;
	double gpu_vddc;
	int16_t gpu_pstate;
	int16_t gpu_bus;
	uint16_t gpu_vid;
	uint16_t gpu_pid;

	int8_t nvml_id;
	int8_t nvapi_id;

	char gpu_sn[64];
	char gpu_desc[64];
	double intensity;
	uint32_t throughput;

	struct monitor_info monitor;
};

struct thr_api {
	int id;
	pthread_t pth;
	struct thread_q	*q;
};

struct stats_data {
	uint32_t uid;
	uint32_t tm_stat;
	uint32_t hashcount;
	uint32_t height;

	double difficulty;
	double hashrate;

	uint8_t thr_id;
	uint8_t gpu_id;
	uint8_t hashfound;
	uint8_t ignored;

	uint8_t npool;
	uint8_t pool_type;
	uint16_t align;
};

struct hashlog_data {
	uint8_t npool;
	uint8_t pool_type;
	uint8_t nonce_id;
	uint8_t job_nonce_id;

	uint32_t height;
	double sharediff;

	uint32_t njobid;
	uint32_t nonce;
	uint32_t scanned_from;
	uint32_t scanned_to;
	uint32_t last_from;
	uint32_t tm_add;
	uint32_t tm_upd;
	uint32_t tm_sent;
};

/* end of api */

struct thr_info {
	int		id;
	pthread_t	pth;
	struct thread_q	*q;
	struct cgpu_info gpu;
};

struct work_restart {
	/* volatile to modify accross threads (vstudio thing) */
	volatile uint32_t restart;
	char padding[128 - sizeof(uint32_t)];
};

#ifdef HAVE_GETOPT_LONG
#include <getopt.h>
#else
struct option {
	const char *name;
	int has_arg;
	int *flag;
	int val;
};
#endif
extern int options_count();

extern bool opt_benchmark;
extern bool opt_debug;
extern bool opt_quiet;
extern bool opt_protocol;
extern bool opt_showdiff;
extern bool opt_tracegpu;
extern int opt_n_threads;
extern int active_gpus;
extern int gpu_threads;
extern int opt_timeout;
extern bool want_longpoll;
extern bool have_longpoll;
extern bool want_stratum;
extern bool have_stratum;
extern bool opt_stratum_stats;
extern char *opt_cert;
extern char *opt_proxy;
extern long opt_proxy_type;
extern bool use_syslog;
extern bool use_colors;
extern int use_pok;
extern pthread_mutex_t applog_lock;
extern struct thr_info *thr_info;
extern int longpoll_thr_id;
extern int stratum_thr_id;
extern int api_thr_id;
extern volatile bool abort_flag;
extern struct work_restart *work_restart;
extern bool opt_trust_pool;
extern uint16_t opt_vote;

extern uint64_t global_hashrate;
extern uint64_t net_hashrate;
extern double net_diff;
extern double stratum_diff;

#define MAX_GPUS 16
//#define MAX_THREADS 32 todo
extern char* device_name[MAX_GPUS];
extern short device_map[MAX_GPUS];
extern short device_mpcount[MAX_GPUS];
extern long  device_sm[MAX_GPUS];
extern uint32_t device_plimit[MAX_GPUS];
extern uint32_t gpus_intensity[MAX_GPUS];
extern int opt_cudaschedule;

// cuda.cpp
int cuda_num_devices();
void cuda_devicenames();
void cuda_reset_device(int thr_id, bool *init);
void cuda_shutdown();
int cuda_finddevice(char *name);
int cuda_version();
void cuda_print_devices();
int cuda_gpu_info(struct cgpu_info *gpu);
int cuda_available_memory(int thr_id);

uint32_t cuda_default_throughput(int thr_id, uint32_t defcount);
#define device_intensity(t,f,d) cuda_default_throughput(t,d)
double throughput2intensity(uint32_t throughput);

void cuda_log_lasterror(int thr_id, const char* func, int line);
void cuda_clear_lasterror();
#define CUDA_LOG_ERROR() cuda_log_lasterror(thr_id, __func__, __LINE__)

#define CL_N    "\x1B[0m"
#define CL_RED  "\x1B[31m"
#define CL_GRN  "\x1B[32m"
#define CL_YLW  "\x1B[33m"
#define CL_BLU  "\x1B[34m"
#define CL_MAG  "\x1B[35m"
#define CL_CYN  "\x1B[36m"

#define CL_BLK  "\x1B[22;30m" /* black */
#define CL_RD2  "\x1B[22;31m" /* red */
#define CL_GR2  "\x1B[22;32m" /* green */
#define CL_YL2  "\x1B[22;33m" /* dark yellow */
#define CL_BL2  "\x1B[22;34m" /* blue */
#define CL_MA2  "\x1B[22;35m" /* magenta */
#define CL_CY2  "\x1B[22;36m" /* cyan */
#define CL_SIL  "\x1B[22;37m" /* gray */

#ifdef WIN32
#define CL_GRY  "\x1B[01;30m" /* dark gray */
#else
#define CL_GRY  "\x1B[90m"    /* dark gray selectable in putty */
#endif
#define CL_LRD  "\x1B[01;31m" /* light red */
#define CL_LGR  "\x1B[01;32m" /* light green */
#define CL_LYL  "\x1B[01;33m" /* tooltips */
#define CL_LBL  "\x1B[01;34m" /* light blue */
#define CL_LMA  "\x1B[01;35m" /* light magenta */
#define CL_LCY  "\x1B[01;36m" /* light cyan */

#define CL_WHT  "\x1B[01;37m" /* white */

extern void format_hashrate(double hashrate, char *output);
extern void format_hashrate_unit(double hashrate, char *output, const char* unit);
extern void applog(int prio, const char *fmt, ...);
extern void gpulog(int prio, int thr_id, const char *fmt, ...);

void get_defconfig_path(char *out, size_t bufsize, char *argv0);
extern void cbin2hex(char *out, const char *in, size_t len);
extern char *bin2hex(const unsigned char *in, size_t len);
extern bool hex2bin(void *output, const char *hexstr, size_t len);
extern int timeval_subtract(struct timeval *result, struct timeval *x,
	struct timeval *y);
extern bool fulltest(const uint32_t *hash, const uint32_t *target);
void diff_to_target(uint32_t* target, double diff);
void work_set_target(struct work* work, double diff);
double target_to_diff(uint32_t* target);
extern void get_currentalgo(char* buf, int sz);

// bignum
double bn_convert_nbits(const uint32_t nbits);
void bn_nbits_to_uchar(const uint32_t nBits, uchar *target);
double bn_hash_target_ratio(uint32_t* hash, uint32_t* target);
void bn_store_hash_target_ratio(uint32_t* hash, uint32_t* target, struct work* work, int nonce);
void bn_set_target_ratio(struct work* work, uint32_t* hash, int nonce);
void work_set_target_ratio(struct work* work, uint32_t* hash);

// bench
extern int bench_algo;
void bench_init(int threads);
void bench_free();
bool bench_algo_switch_next(int thr_id);
void bench_set_throughput(int thr_id, uint32_t throughput);
void bench_display_results();

struct stratum_job {
	char *job_id;
	unsigned char prevhash[32];
	size_t coinbase_size;
	unsigned char *coinbase;
	unsigned char *xnonce2;
	int merkle_count;
	unsigned char **merkle;
	unsigned char version[4];
	unsigned char nbits[4];
	unsigned char ntime[4];
	unsigned char claim[32]; // lbry
	bool clean;
	unsigned char nreward[2];
	uint32_t height;
	uint32_t shares_count;
	double diff;
};

struct stratum_ctx {
	char *url;

	CURL *curl;
	char *curl_url;
	char curl_err_str[CURL_ERROR_SIZE];
	curl_socket_t sock;
	size_t sockbuf_size;
	char *sockbuf;

	double next_diff;
	double sharediff;

	char *session_id;
	size_t xnonce1_size;
	unsigned char *xnonce1;
	size_t xnonce2_size;
	struct stratum_job job;

	struct timeval tv_submit;
	uint32_t answer_msec;
	int pooln;
	time_t tm_connected;

	int rpc2;
	int is_equihash;
	int srvtime_diff;
};

#define POK_MAX_TXS   4
#define POK_MAX_TX_SZ 16384U
struct tx {
	uint8_t data[POK_MAX_TX_SZ];
	uint32_t len;
};

#define MAX_NONCES 2
struct work {
	uint32_t data[48];
	uint32_t target[8];
	uint32_t maxvote;

	char job_id[128];
	size_t xnonce2_len;
	uchar xnonce2[32];

	union {
		uint32_t u32[2];
		uint64_t u64[1];
	} noncerange;

	uint8_t pooln;
	uint8_t valid_nonces;
	uint8_t submit_nonce_id;
	uint8_t job_nonce_id;

	uint32_t nonces[MAX_NONCES];
	double sharediff[MAX_NONCES];
	double shareratio[MAX_NONCES];
	double targetdiff;

	uint32_t height;

	uint32_t scanned_from;
	uint32_t scanned_to;

	/* pok getwork txs */
	uint32_t tx_count;
	struct tx txs[POK_MAX_TXS];
	// zec solution
	uint8_t extra[1388];
};

#define POK_BOOL_MASK 0x00008000
#define POK_DATA_MASK 0xFFFF0000

#define MAX_POOLS 8
struct pool_infos {
	uint8_t id;
#define POOL_UNUSED   0
#define POOL_GETWORK  1
#define POOL_STRATUM  2
#define POOL_LONGPOLL 4
	uint8_t type;
#define POOL_ST_DEFINED 1
#define POOL_ST_VALID 2
#define POOL_ST_DISABLED 4
#define POOL_ST_REMOVED 8
	uint16_t status;
	int algo;
	char name[64];
	// credentials
	char url[512];
	char short_url[64];
	char user[192];
	char pass[384];
	// config options
	double max_diff;
	double max_rate;
	int shares_limit;
	int time_limit;
	int scantime;
	// connection
	struct stratum_ctx stratum;
	uint8_t allow_gbt;
	uint8_t allow_mininginfo;
	uint16_t check_dups; // 16_t for align
	int retries;
	int fail_pause;
	int timeout;
	// stats
	uint32_t work_time;
	uint32_t wait_time;
	uint32_t accepted_count;
	uint32_t rejected_count;
	uint32_t solved_count;
	uint32_t stales_count;
	time_t last_share_time;
	double best_share;
	uint32_t disconnects;
};

extern struct pool_infos pools[MAX_POOLS];
extern int num_pools;
extern volatile int cur_pooln;

void pool_init_defaults(void);
void pool_set_creds(int pooln);
void pool_set_attr(int pooln, const char* key, char* arg);
bool pool_switch_url(char *params);
bool pool_switch(int thr_id, int pooln);
bool pool_switch_next(int thr_id);
int pool_get_first_valid(int startfrom);
bool parse_pool_array(json_t *obj);
void pool_dump_infos(void);

json_t * json_rpc_call_pool(CURL *curl, struct pool_infos*,
	const char *req, bool lp_scan, bool lp, int *err);
json_t * json_rpc_longpoll(CURL *curl, char *lp_url, struct pool_infos*,
	const char *req, int *err);

bool stratum_socket_full(struct stratum_ctx *sctx, int timeout);
bool stratum_send_line(struct stratum_ctx *sctx, char *s);
char *stratum_recv_line(struct stratum_ctx *sctx);
bool stratum_connect(struct stratum_ctx *sctx, const char *url);
void stratum_disconnect(struct stratum_ctx *sctx);
bool stratum_subscribe(struct stratum_ctx *sctx);
bool stratum_authorize(struct stratum_ctx *sctx, const char *user, const char *pass);
bool stratum_handle_method(struct stratum_ctx *sctx, const char *s);
void stratum_free_job(struct stratum_ctx *sctx);

bool rpc2_stratum_authorize(struct stratum_ctx *sctx, const char *user, const char *pass);

bool equi_stratum_notify(struct stratum_ctx *sctx, json_t *params);
bool equi_stratum_set_target(struct stratum_ctx *sctx, json_t *params);
bool equi_stratum_submit(struct pool_infos *pool, struct work *work);
bool equi_stratum_show_message(struct stratum_ctx *sctx, json_t *id, json_t *params);
void equi_work_set_target(struct work* work, double diff);
void equi_store_work_solution(struct work* work, uint32_t* hash, void* sol_data);
int equi_verify_sol(void * const hdr, void * const sol);
double equi_network_diff(struct work *work);

void hashlog_remember_submit(struct work* work, uint32_t nonce);
void hashlog_remember_scan_range(struct work* work);
double hashlog_get_sharediff(char* jobid, int idnonce, double defvalue);
uint32_t hashlog_already_submittted(char* jobid, uint32_t nounce);
uint32_t hashlog_get_last_sent(char* jobid);
uint64_t hashlog_get_scan_range(char* jobid);
int  hashlog_get_history(struct hashlog_data *data, int max_records);
void hashlog_purge_old(void);
void hashlog_purge_job(char* jobid);
void hashlog_purge_all(void);
void hashlog_dump_job(char* jobid);
void hashlog_getmeminfo(uint64_t *mem, uint32_t *records);

void stats_remember_speed(int thr_id, uint32_t hashcount, double hashrate, uint8_t found, uint32_t height);
double stats_get_speed(int thr_id, double def_speed);
double stats_get_gpu_speed(int gpu_id);
int  stats_get_history(int thr_id, struct stats_data *data, int max_records);
void stats_purge_old(void);
void stats_purge_all(void);
void stats_getmeminfo(uint64_t *mem, uint32_t *records);

struct thread_q;

extern struct thread_q *tq_new(void);
extern void tq_free(struct thread_q *tq);
extern bool tq_push(struct thread_q *tq, void *data);
extern void *tq_pop(struct thread_q *tq, const struct timespec *abstime);
extern void tq_freeze(struct thread_q *tq);
extern void tq_thaw(struct thread_q *tq);

#define EXIT_CODE_OK            0
#define EXIT_CODE_USAGE         1
#define EXIT_CODE_POOL_TIMEOUT  2
#define EXIT_CODE_SW_INIT_ERROR 3
#define EXIT_CODE_CUDA_NODEVICE 4
#define EXIT_CODE_CUDA_ERROR    5
#define EXIT_CODE_TIME_LIMIT    0
#define EXIT_CODE_KILLED        7

void parse_arg(int key, char *arg);
void proper_exit(int reason);
void restart_threads(void);

size_t time2str(char* buf, time_t timer);
char* atime2str(time_t timer);

void applog_hex(void *data, int len);
void applog_hash(void *hash);
void applog_hash64(void *hash);
void applog_compare_hash(void *hash, void *hash_ref);

void print_hash_tests(void);
void bastionhash(void* output, const unsigned char* input);
void blake256hash(void *output, const void *input, int8_t rounds);
void blake2b_hash(void *output, const void *input);
void blake2s_hash(void *output, const void *input);
void bmw_hash(void *state, const void *input);
void c11hash(void *output, const void *input);
void cryptolight_hash(void* output, const void* input, int len);
int cryptonight_hash(void* output, const void* input, size_t len, int variant);
void decred_hash(void *state, const void *input);
void deephash(void *state, const void *input);
void luffa_hash(void *state, const void *input);
void fresh_hash(void *state, const void *input);
void fugue256_hash(unsigned char* output, const unsigned char* input, int len);
void heavycoin_hash(unsigned char* output, const unsigned char* input, int len);
void hmq17hash(void *output, const void *input);
void hsr_hash(void *output, const void *input);
void keccak256_hash(void *state, const void *input);
void jackpothash(void *state, const void *input);
void groestlhash(void *state, const void *input);
void jha_hash(void *output, const void *input);
void lbry_hash(void *output, const void *input);
void lyra2re_hash(void *state, const void *input);
void lyra2v2_hash(void *state, const void *input);
void lyra2Z_hash(void *state, const void *input);
void myriadhash(void *state, const void *input);
void neoscrypt(uchar *output, const uchar *input, uint32_t profile);
void nist5hash(void *state, const void *input);
void pentablakehash(void *output, const void *input);
void phihash(void *output, const void *input);
void polytimos_hash(void *output, const void *input);
void quarkhash(void *state, const void *input);
void qubithash(void *state, const void *input);
void scrypthash(void* output, const void* input);
void scryptjane_hash(void* output, const void* input);
void sha256d_hash(void *output, const void *input);
void sha256t_hash(void *output, const void *input);
void sibhash(void *output, const void *input);
void skeincoinhash(void *output, const void *input);
void skein2hash(void *output, const void *input);
void skunk_hash(void *state, const void *input);
void s3hash(void *output, const void *input);
void timetravel_hash(void *output, const void *input);
void bitcore_hash(void *output, const void *input);
void tribus_hash(void *output, const void *input);
void veltorhash(void *output, const void *input);
void wcoinhash(void *state, const void *input);
void whirlxHash(void *state, const void *input);
void x11evo_hash(void *output, const void *input);
void x11hash(void *output, const void *input);
void x12hash(void *output, const void *input);
void x13hash(void *output, const void *input);
void x14hash(void *output, const void *input);
void x15hash(void *output, const void *input);
void x16r_hash(void *output, const void *input);
void x16s_hash(void *output, const void *input);
void x17hash(void *output, const void *input);
void wildkeccak_hash(void *output, const void *input, uint64_t* scratchpad, uint64_t ssize);
void zr5hash(void *output, const void *input);
void zr5hash_pok(void *output, uint32_t *pdata);

#ifdef __cplusplus
}
#endif

#endif /* __MINER_H__ */
