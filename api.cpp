/*
 * Copyright 2014 ccminer team
 *
 * Implementation by tpruvot (based on cgminer)
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */
#define APIVERSION "1.9"

#ifdef WIN32
# define  _WINSOCK_DEPRECATED_NO_WARNINGS
# include <WS2tcpip.h>
#endif

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <inttypes.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <stdarg.h>
#include <assert.h>

#include <sys/stat.h>
#include <sys/types.h>

#include "miner.h"
#include "nvml.h"
#include "algos.h"

#ifndef WIN32
# include <errno.h>
# include <sys/socket.h>
# include <netinet/in.h>
# include <arpa/inet.h>
# include <netdb.h>
# define SOCKETTYPE long
# define SOCKETFAIL(a) ((a) < 0)
# define INVSOCK -1 /* INVALID_SOCKET */
# define INVINETADDR -1 /* INADDR_NONE */
# define CLOSESOCKET close
# define SOCKETINIT {}
# define SOCKERRMSG strerror(errno)
#else
# define SOCKETTYPE SOCKET
# define SOCKETFAIL(a) ((a) == SOCKET_ERROR)
# define INVSOCK INVALID_SOCKET
# define INVINETADDR INADDR_NONE
# define CLOSESOCKET closesocket
# define in_addr_t uint32_t
#endif

#define GROUP(g) (toupper(g))
#define PRIVGROUP GROUP('W')
#define NOPRIVGROUP GROUP('R')
#define ISPRIVGROUP(g) (GROUP(g) == PRIVGROUP)
#define GROUPOFFSET(g) (GROUP(g) - GROUP('A'))
#define VALIDGROUP(g) (GROUP(g) >= GROUP('A') && GROUP(g) <= GROUP('Z'))
#define COMMANDS(g) (apigroups[GROUPOFFSET(g)].commands)
#define DEFINEDGROUP(g) (ISPRIVGROUP(g) || COMMANDS(g) != NULL)
struct APIGROUPS {
	// This becomes a string like: "|cmd1|cmd2|cmd3|" so it's quick to search
	char *commands;
} apigroups['Z' - 'A' + 1]; // only A=0 to Z=25 (R: noprivs, W: allprivs)

struct IP4ACCESS {
	in_addr_t ip;
	in_addr_t mask;
	char group;
};

static int ips = 1;
static struct IP4ACCESS *ipaccess = NULL;

#define MYBUFSIZ       16384
#define SOCK_REC_BUFSZ 1024
#define QUEUE          10

#define ALLIP4         "0.0.0.0"
static const char *localaddr = "127.0.0.1";
static const char *UNAVAILABLE = " - API will not be available";
static const char *MUNAVAILABLE = " - API multicast listener will not be available";
static char *buffer = NULL;
static time_t startup = 0;
static int bye = 0;

extern char *opt_api_bind;
extern int opt_api_port;
extern char *opt_api_allow;
extern char *opt_api_groups;
extern bool opt_api_mcast;
extern char *opt_api_mcast_addr;
extern char *opt_api_mcast_code;
extern char *opt_api_mcast_des;
extern int opt_api_mcast_port;

// current stratum...
extern struct stratum_ctx stratum;

// sysinfos.cpp
extern int num_cpus;
extern float cpu_temp(int);
extern uint32_t cpu_clock(int);

char driver_version[32] = { 0 };

/***************************************************************/

static void gpustatus(int thr_id)
{
	struct pool_infos *p = &pools[cur_pooln];

	if (thr_id >= 0 && thr_id < opt_n_threads) {
		struct cgpu_info *cgpu = &thr_info[thr_id].gpu;
		double khashes_per_watt = 0;
		int gpuid = cgpu->gpu_id;
		char buf[512]; *buf = '\0';
		char* card;

		cuda_gpu_info(cgpu);
		cgpu->gpu_plimit = device_plimit[cgpu->gpu_id];

#ifdef USE_WRAPNVML
		cgpu->has_monitoring = true;
		cgpu->gpu_bus = gpu_busid(cgpu);
		cgpu->gpu_temp = gpu_temp(cgpu);
		cgpu->gpu_fan = (uint16_t) gpu_fanpercent(cgpu);
		cgpu->gpu_fan_rpm = (uint16_t) gpu_fanrpm(cgpu);
		cgpu->gpu_power = gpu_power(cgpu); // mWatts
		cgpu->gpu_plimit = gpu_plimit(cgpu); // mW or %
#endif
		cgpu->khashes = stats_get_speed(thr_id, 0.0) / 1000.0;
		if (cgpu->monitor.gpu_power) {
			cgpu->gpu_power = cgpu->monitor.gpu_power;
			khashes_per_watt = (double)cgpu->khashes / cgpu->monitor.gpu_power;
			khashes_per_watt *= 1000; // power in mW
			//gpulog(LOG_BLUE, thr_id, "KHW: %g", khashes_per_watt);
		}

		card = device_name[gpuid];

		snprintf(buf, sizeof(buf), "GPU=%d;BUS=%hd;CARD=%s;TEMP=%.1f;"
			"POWER=%u;FAN=%hu;RPM=%hu;"
			"FREQ=%u;MEMFREQ=%u;GPUF=%u;MEMF=%u;"
			"KHS=%.2f;KHW=%.5f;PLIM=%u;"
			"ACC=%u;REJ=%u;HWF=%u;I=%.1f;THR=%u|",
			gpuid, cgpu->gpu_bus, card, cgpu->gpu_temp,
			cgpu->gpu_power, cgpu->gpu_fan, cgpu->gpu_fan_rpm,
			cgpu->gpu_clock/1000, cgpu->gpu_memclock/1000, // base freqs in MHz
			cgpu->monitor.gpu_clock, cgpu->monitor.gpu_memclock, // current
			cgpu->khashes, khashes_per_watt, cgpu->gpu_plimit,
			cgpu->accepted, (unsigned) cgpu->rejected, (unsigned) cgpu->hw_errors,
			cgpu->intensity, cgpu->throughput);

		// append to buffer for multi gpus
		strcat(buffer, buf);
	}
}

/**
* Returns gpu/thread specific stats
*/
static char *getthreads(char *params)
{
	*buffer = '\0';
	for (int i = 0; i < opt_n_threads; i++)
		gpustatus(i);
	return buffer;
}

/*****************************************************************************/

/**
* Returns miner global infos
*/
static char *getsummary(char *params)
{
	char algo[64] = { 0 };
	time_t ts = time(NULL);
	double accps, uptime = difftime(ts, startup);
	uint32_t wait_time = 0, solved_count = 0;
	uint32_t accepted_count = 0, rejected_count = 0;
	for (int p = 0; p < num_pools; p++) {
		wait_time += pools[p].wait_time;
		accepted_count += pools[p].accepted_count;
		rejected_count += pools[p].rejected_count;
		solved_count += pools[p].solved_count;
	}
	accps = (60.0 * accepted_count) / (uptime ? uptime : 1.0);

	get_currentalgo(algo, sizeof(algo));

	*buffer = '\0';
	sprintf(buffer, "NAME=%s;VER=%s;API=%s;"
		"ALGO=%s;GPUS=%d;KHS=%.2f;SOLV=%d;ACC=%d;REJ=%d;"
		"ACCMN=%.3f;DIFF=%.6f;NETKHS=%.0f;"
		"POOLS=%u;WAIT=%u;UPTIME=%.0f;TS=%u|",
		PACKAGE_NAME, PACKAGE_VERSION, APIVERSION,
		algo, active_gpus, (double)global_hashrate / 1000.,
		solved_count, accepted_count, rejected_count,
		accps, net_diff > 1e-6 ? net_diff : stratum_diff, (double)net_hashrate / 1000.,
		num_pools, wait_time, uptime, (uint32_t) ts);
	return buffer;
}

/**
 * Returns some infos about current pool
 */
static char *getpoolnfo(char *params)
{
	char *s = buffer;
	char jobid[128] = { 0 };
	char extra[96] = { 0 };
	int pooln = params ? atoi(params) % num_pools : cur_pooln;
	struct pool_infos *p = &pools[pooln];
	uint32_t last_share = 0;
	if (p->last_share_time)
		last_share = (uint32_t) (time(NULL) - p->last_share_time);

	*s = '\0';

	if (stratum.job.job_id)
		strncpy(jobid, stratum.job.job_id, sizeof(stratum.job.job_id));
	if (stratum.job.xnonce2) {
		/* used temporary to be sure all is ok */
		sprintf(extra, "0x");
		if (p->algo == ALGO_DECRED) {
			char compat[32] = { 0 };
			cbin2hex(&extra[2], (const char*) stratum.xnonce1, min(36, stratum.xnonce2_size));
			cbin2hex(compat, (const char*) stratum.job.xnonce2, 4);
			memcpy(&extra[2], compat, 8); // compat extranonce
		} else {
			cbin2hex(&extra[2], (const char*) stratum.job.xnonce2, stratum.xnonce2_size);
		}
	}

	snprintf(s, MYBUFSIZ, "POOL=%s;ALGO=%s;URL=%s;USER=%s;SOLV=%d;ACC=%d;REJ=%d;STALE=%u;H=%u;JOB=%s;DIFF=%.6f;"
		"BEST=%.6f;N2SZ=%d;N2=%s;PING=%u;DISCO=%u;WAIT=%u;UPTIME=%u;LAST=%u|",
		strlen(p->name) ? p->name : p->short_url, algo_names[p->algo],
		p->url, p->type & POOL_STRATUM ? p->user : "",
		p->solved_count, p->accepted_count, p->rejected_count, p->stales_count,
		stratum.job.height, jobid, stratum_diff, p->best_share,
		(int) stratum.xnonce2_size, extra, stratum.answer_msec,
		p->disconnects, p->wait_time, p->work_time, last_share);

	return s;
}

/*****************************************************************************/

static void gpuhwinfos(int gpu_id)
{
	char buf[256];
	char pstate[8];
	char* card;
	struct cgpu_info *cgpu = NULL;

	for (int g = 0; g < opt_n_threads; g++) {
		if (device_map[g] == gpu_id) {
			cgpu = &thr_info[g].gpu;
			break;
		}
	}

	if (cgpu == NULL)
		return;

	cuda_gpu_info(cgpu);
	cgpu->gpu_plimit = device_plimit[cgpu->gpu_id];

#ifdef USE_WRAPNVML
	cgpu->has_monitoring = true;
	cgpu->gpu_bus = gpu_busid(cgpu);
	cgpu->gpu_temp = gpu_temp(cgpu);
	cgpu->gpu_fan = (uint16_t) gpu_fanpercent(cgpu);
	cgpu->gpu_fan_rpm = (uint16_t) gpu_fanrpm(cgpu);
	cgpu->gpu_pstate = (int16_t) gpu_pstate(cgpu);
	cgpu->gpu_power = gpu_power(cgpu);
	cgpu->gpu_plimit = gpu_plimit(cgpu);
	gpu_info(cgpu);
#ifdef WIN32
	if (opt_debug) nvapi_pstateinfo(cgpu->gpu_id);
#endif
#endif

	memset(pstate, 0, sizeof(pstate));
	if (cgpu->gpu_pstate != -1)
		snprintf(pstate, sizeof(pstate), "P%d", (int) cgpu->gpu_pstate);

	card = device_name[gpu_id];

	snprintf(buf, sizeof(buf), "GPU=%d;BUS=%hd;CARD=%s;SM=%hu;MEM=%u;"
		"TEMP=%.1f;FAN=%hu;RPM=%hu;FREQ=%u;MEMFREQ=%u;GPUF=%u;MEMF=%u;"
		"PST=%s;POWER=%u;PLIM=%u;"
		"VID=%hx;PID=%hx;NVML=%d;NVAPI=%d;SN=%s;BIOS=%s|",
		gpu_id, cgpu->gpu_bus, card, cgpu->gpu_arch, (uint32_t) cgpu->gpu_mem,
		cgpu->gpu_temp, cgpu->gpu_fan, cgpu->gpu_fan_rpm,
		cgpu->gpu_clock/1000U, cgpu->gpu_memclock/1000U, // base clocks
		cgpu->monitor.gpu_clock, cgpu->monitor.gpu_memclock, // current
		pstate, cgpu->gpu_power, cgpu->gpu_plimit,
		cgpu->gpu_vid, cgpu->gpu_pid, cgpu->nvml_id, cgpu->nvapi_id,
		cgpu->gpu_sn, cgpu->gpu_desc);

	strcat(buffer, buf);
}

#ifndef WIN32
static char os_version[64] = "linux ";
#endif

static const char* os_name()
{
#ifdef WIN32
	return "windows";
#else
	FILE *fd = fopen("/proc/version", "r");
	if (!fd)
		return "linux";
	if (!fscanf(fd, "Linux version %48s", &os_version[6])) {
		fclose(fd);
		return "linux";
	}
	fclose(fd);
	os_version[48] = '\0';
	return (const char*) os_version;
#endif
}

/**
 * System and CPU Infos
 */
static void syshwinfos()
{
	char buf[256];

	int cputc = (int) cpu_temp(0);
	uint32_t cpuclk = cpu_clock(0);

	memset(buf, 0, sizeof(buf));
	snprintf(buf, sizeof(buf), "OS=%s;NVDRIVER=%s;CPUS=%d;CPUTEMP=%d;CPUFREQ=%d|",
		os_name(), driver_version, num_cpus, cputc, cpuclk/1000);
	strcat(buffer, buf);
}

/**
 * Returns gpu and system (todo) informations
 */
static char *gethwinfos(char *params)
{
	*buffer = '\0';
	for (int i = 0; i < cuda_num_devices(); i++)
		gpuhwinfos(i);
	syshwinfos();
	return buffer;
}

/*****************************************************************************/

/**
 * Returns the last 50 scans stats
 * optional param thread id (default all)
 */
static char *gethistory(char *params)
{
	struct stats_data data[50];
	int thrid = params ? atoi(params) : -1;
	char *p = buffer;
	int records = stats_get_history(thrid, data, ARRAY_SIZE(data));
	*buffer = '\0';
	for (int i = 0; i < records; i++) {
		time_t ts = data[i].tm_stat;
		p += sprintf(p, "GPU=%d;H=%u;KHS=%.2f;DIFF=%g;"
				"COUNT=%u;FOUND=%u;ID=%u;TS=%u|",
			data[i].gpu_id, data[i].height, data[i].hashrate, data[i].difficulty,
			data[i].hashcount, data[i].hashfound, data[i].uid, (uint32_t)ts);
	}
	return buffer;
}

/**
 * Returns the job scans ranges (debug purpose, only with -D)
 */
static char *getscanlog(char *params)
{
	struct hashlog_data data[50];
	char *p = buffer;
	int records = hashlog_get_history(data, ARRAY_SIZE(data));
	*buffer = '\0';
	for (int i = 0; i < records; i++) {
		time_t ts = data[i].tm_upd;
		p += sprintf(p, "H=%u;P=%u;JOB=%u;ID=%d;DIFF=%g;"
				"N=0x%x;FROM=0x%x;SCANTO=0x%x;"
				"COUNT=0x%x;FOUND=%u;TS=%u|",
			data[i].height, data[i].npool, data[i].njobid, (int)data[i].job_nonce_id, data[i].sharediff,
			data[i].nonce, data[i].scanned_from, data[i].scanned_to,
			(data[i].scanned_to - data[i].scanned_from), data[i].tm_sent ? 1 : 0, (uint32_t)ts);
	}
	return buffer;
}

/**
 * Some debug infos about memory usage
 */
static char *getmeminfo(char *params)
{
	uint64_t smem, hmem, totmem;
	uint32_t srec, hrec;

	stats_getmeminfo(&smem, &srec);
	hashlog_getmeminfo(&hmem, &hrec);
	totmem = smem + hmem;

	*buffer = '\0';
	sprintf(buffer, "STATS=%u;HASHLOG=%u;MEM=%lu|",
		srec, hrec, totmem);

	return buffer;
}

/*****************************************************************************/

/**
 * Set pool by index (pools array in json config)
 * switchpool|1|
 */
static char *remote_switchpool(char *params)
{
	bool ret = false;
	*buffer = '\0';
	if (!params || strlen(params) == 0) {
		// rotate pool test
		ret = pool_switch_next(-1);
	} else {
		int n = atoi(params);
		if (n == cur_pooln)
			ret = true;
		else if (n < num_pools)
			ret = pool_switch(-1, n);
	}
	sprintf(buffer, "%s|", ret ? "ok" : "fail");
	return buffer;
}

/**
 * Change pool url (see --url parameter)
 * seturl|stratum+tcp://<user>:<pass>@mine.xpool.ca:1131|
 */
static char *remote_seturl(char *params)
{
	bool ret;
	*buffer = '\0';
	if (!params || strlen(params) == 0) {
		// rotate pool test
		ret = pool_switch_next(-1);
	} else {
		ret = pool_switch_url(params);
	}
	sprintf(buffer, "%s|", ret ? "ok" : "fail");
	return buffer;
}

/**
 * Ask the miner to quit
 */
static char *remote_quit(char *params)
{
	*buffer = '\0';
	bye = 1;
	sprintf(buffer, "%s", "bye|");
	return buffer;
}

/*****************************************************************************/

static char *gethelp(char *params);
struct CMDS {
	const char *name;
	char *(*func)(char *);
	bool iswritemode;
} cmds[] = {
	{ "summary", getsummary, false },
	{ "threads", getthreads, false },
	{ "pool",    getpoolnfo, false },
	{ "histo",   gethistory, false },
	{ "hwinfo",  gethwinfos, false },
	{ "meminfo", getmeminfo, false },
	{ "scanlog", getscanlog, false },

	/* remote functions */
	{ "seturl",  remote_seturl, true }, /* prefer switchpool, deprecated */
	{ "switchpool", remote_switchpool, true },
	{ "quit", remote_quit, true },

	/* keep it the last */
	{ "help",    gethelp, false },
};
#define CMDMAX ARRAY_SIZE(cmds)

static char *gethelp(char *params)
{
	*buffer = '\0';
	char * p = buffer;
	for (int i = 0; i < CMDMAX-1; i++) {
		bool displayed = !cmds[i].iswritemode || opt_api_allow;
		if (displayed) p += sprintf(p, "%s\n", cmds[i].name);
	}
	sprintf(p, "|");
	return buffer;
}

/*****************************************************************************/

static int send_result(SOCKETTYPE c, char *result)
{
	int n;
	if (!result) {
		n = send(c, "", 1, 0);
	} else {
		// ignore failure - it's closed immediately anyway
		n = send(c, result, (int) strlen(result) + 1, 0);
	}
	return n;
}

/* ---- Base64 Encoding/Decoding Table --- */
static const char table64[]=
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static size_t base64_encode(const uchar *indata, size_t insize, char *outptr, size_t outlen)
{
	uchar ibuf[3];
	uchar obuf[4];
	int i, inputparts, inlen = (int) insize;
	size_t len = 0;
	char *output, *outbuf;

	memset(outptr, 0, outlen);

	outbuf = output = (char*)calloc(1, inlen * 4 / 3 + 4);
	if (outbuf == NULL) {
		return -1;
	}

	while (inlen > 0) {
		for (i = inputparts = 0; i < 3; i++) {
			if (inlen  > 0) {
				inputparts++;
				ibuf[i] = (uchar) *indata;
				indata++; inlen--;
			}
			else
				ibuf[i] = 0;
		}

		obuf[0] = (uchar)  ((ibuf[0] & 0xFC) >> 2);
		obuf[1] = (uchar) (((ibuf[0] & 0x03) << 4) | ((ibuf[1] & 0xF0) >> 4));
		obuf[2] = (uchar) (((ibuf[1] & 0x0F) << 2) | ((ibuf[2] & 0xC0) >> 6));
		obuf[3] = (uchar)   (ibuf[2] & 0x3F);

		switch(inputparts) {
		case 1: /* only one byte read */
			snprintf(output, 5, "%c%c==",
				table64[obuf[0]],
				table64[obuf[1]]);
			break;
		case 2: /* two bytes read */
			snprintf(output, 5, "%c%c%c=",
				table64[obuf[0]],
				table64[obuf[1]],
				table64[obuf[2]]);
			break;
		default:
			snprintf(output, 5, "%c%c%c%c",
				table64[obuf[0]],
				table64[obuf[1]],
				table64[obuf[2]],
				table64[obuf[3]] );
			break;
		}
		if ((len+4) > outlen)
			break;
		output += 4; len += 4;
	}
	len = snprintf(outptr, len, "%s", outbuf);
	// todo: seems to be missing on linux
	if (strlen(outptr) == 27)
		strcat(outptr, "=");
	free(outbuf);

	return len;
}

#include "openssl/sha.h"

/* websocket handshake (tested in Chrome) */
static int websocket_handshake(SOCKETTYPE c, char *result, char *clientkey)
{
	char answer[256];
	char inpkey[128] = { 0 };
	char seckey[64];
	uchar sha1[20];
	SHA_CTX ctx;

	if (opt_protocol)
		applog(LOG_DEBUG, "clientkey: %s", clientkey);

	sprintf(inpkey, "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11", clientkey);

	// SHA-1 test from rfc, returns in base64 "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
	//sprintf(inpkey, "dGhlIHNhbXBsZSBub25jZQ==258EAFA5-E914-47DA-95CA-C5AB0DC85B11");

	SHA1_Init(&ctx);
	SHA1_Update(&ctx, inpkey, strlen(inpkey));
	SHA1_Final(sha1, &ctx);

	base64_encode(sha1, 20, seckey, sizeof(seckey));

	sprintf(answer,
		"HTTP/1.1 101 Switching Protocol\r\n"
		"Upgrade: WebSocket\r\nConnection: Upgrade\r\n"
		"Sec-WebSocket-Accept: %s\r\n"
		"Sec-WebSocket-Protocol: text\r\n"
		"\r\n", seckey);

	// data result as tcp frame

	uchar hd[10] = { 0 };
	hd[0] = 129; // 0x1 text frame (FIN + opcode)
	uint64_t datalen = (uint64_t) strlen(result);
	uint8_t frames = 2;
	if (datalen <= 125) {
		hd[1] = (uchar) (datalen);
	} else if (datalen <= 65535) {
		hd[1] = (uchar) 126;
		hd[2] = (uchar) (datalen >> 8);
		hd[3] = (uchar) (datalen);
		frames = 4;
	} else {
		hd[1] = (uchar) 127;
		hd[2] = (uchar) (datalen >> 56);
		hd[3] = (uchar) (datalen >> 48);
		hd[4] = (uchar) (datalen >> 40);
		hd[5] = (uchar) (datalen >> 32);
		hd[6] = (uchar) (datalen >> 24);
		hd[7] = (uchar) (datalen >> 16);
		hd[8] = (uchar) (datalen >> 8);
		hd[9] = (uchar) (datalen);
		frames = 10;
	}

	size_t handlen = strlen(answer);
	uchar *data = (uchar*) calloc(1, handlen + frames + (size_t) datalen + 1);
	if (data == NULL)
		return -1;
	else {
		uchar *p = data;
		// HTTP header 101
		memcpy(p, answer, handlen);
		p += handlen;
		// WebSocket Frame - Header + Data
		memcpy(p, hd, frames);
		memcpy(p + frames, result, (size_t)datalen);
		send(c, (const char*)data, (int) (strlen(answer) + frames + datalen + 1), 0);
		free(data);
	}
	return 0;
}

/*
 * Interpret --api-groups G:cmd1:cmd2:cmd3,P:cmd4,*,...
 */
static void setup_groups()
{
	const char *api_groups = opt_api_groups ? opt_api_groups : "";
	char *buf, *cmd, *ptr, *next, *colon;
	char commands[512] = { 0 };
	char cmdbuf[128] = { 0 };
	char group;
	bool addstar;
	int i;

	buf = (char *)calloc(1, strlen(api_groups) + 1);
	if (unlikely(!buf))
		proper_exit(1); //, "Failed to malloc ipgroups buf");

	strcpy(buf, api_groups);

	next = buf;
	// for each group defined
	while (next && *next) {
		ptr = next;
		next = strchr(ptr, ',');
		if (next)
			*(next++) = '\0';

		// Validate the group
		if (*(ptr+1) != ':') {
			colon = strchr(ptr, ':');
			if (colon)
				*colon = '\0';
			proper_exit(1); //, "API invalid group name '%s'", ptr);
		}

		group = GROUP(*ptr);
		if (!VALIDGROUP(group))
			proper_exit(1); //, "API invalid group name '%c'", *ptr);

		if (group == PRIVGROUP)
			proper_exit(1); //, "API group name can't be '%c'", PRIVGROUP);

		if (group == NOPRIVGROUP)
			proper_exit(1); //, "API group name can't be '%c'", NOPRIVGROUP);

		if (apigroups[GROUPOFFSET(group)].commands != NULL)
			proper_exit(1); //, "API duplicate group name '%c'", *ptr);

		ptr += 2;

		// Validate the command list (and handle '*')
		cmd = &(commands[0]);
		*(cmd++) = '|';
		*cmd = '\0';
		addstar = false;
		while (ptr && *ptr) {
			colon = strchr(ptr, ':');
			if (colon)
				*(colon++) = '\0';

			if (strcmp(ptr, "*") == 0)
				addstar = true;
			else {
				bool did = false;
				for (i = 0; i < CMDMAX-1; i++) {
					if (strcasecmp(ptr, cmds[i].name) == 0) {
						did = true;
						break;
					}
				}
				if (did) {
					// skip duplicates
					sprintf(cmdbuf, "|%s|", cmds[i].name);
					if (strstr(commands, cmdbuf) == NULL) {
						strcpy(cmd, cmds[i].name);
						cmd += strlen(cmds[i].name);
						*(cmd++) = '|';
						*cmd = '\0';
					}
				} else {
					proper_exit(1); //, "API unknown command '%s' in group '%c'", ptr, group);
				}
			}

			ptr = colon;
		}

		// * = allow all non-iswritemode commands
		if (addstar) {
			for (i = 0; i < CMDMAX-1; i++) {
				if (cmds[i].iswritemode == false) {
					// skip duplicates
					sprintf(cmdbuf, "|%s|", cmds[i].name);
					if (strstr(commands, cmdbuf) == NULL) {
						strcpy(cmd, cmds[i].name);
						cmd += strlen(cmds[i].name);
						*(cmd++) = '|';
						*cmd = '\0';
					}
				}
			}
		}

		ptr = apigroups[GROUPOFFSET(group)].commands = (char *)calloc(1, strlen(commands) + 1);
		if (unlikely(!ptr))
			proper_exit(1); //, "Failed to malloc group commands buf");

		strcpy(ptr, commands);
	}

	// Now define R (NOPRIVGROUP) as all non-iswritemode commands
	cmd = &(commands[0]);
	*(cmd++) = '|';
	*cmd = '\0';
	for (i = 0; i < CMDMAX-1; i++) {
		if (cmds[i].iswritemode == false) {
			strcpy(cmd, cmds[i].name);
			cmd += strlen(cmds[i].name);
			*(cmd++) = '|';
			*cmd = '\0';
		}
	}

	ptr = apigroups[GROUPOFFSET(NOPRIVGROUP)].commands = (char *)calloc(1, strlen(commands) + 1);
	if (unlikely(!ptr))
		proper_exit(1); //, "Failed to malloc noprivgroup commands buf");

	strcpy(ptr, commands);

	// W (PRIVGROUP) is handled as a special case since it simply means all commands

	free(buf);
	return;
}

/*
 * Interpret [W:]IP[/Prefix][,[R|W:]IP2[/Prefix2][,...]] --api-allow option
 *	special case of 0/0 allows /0 (means all IP addresses)
 */
#define ALLIPS "0/0"
/*
 * N.B. IP4 addresses are by Definition 32bit big endian on all platforms
 */
static void setup_ipaccess()
{
	char *buf, *ptr, *comma, *slash, *dot;
	int ipcount, mask, octet, i;
	char group;

	buf = (char*) calloc(1, strlen(opt_api_allow) + 1);
	if (unlikely(!buf))
		proper_exit(1);//, "Failed to malloc ipaccess buf");

	strcpy(buf, opt_api_allow);
	ipcount = 1;
	ptr = buf;
	while (*ptr) if (*(ptr++) == ',')
		ipcount++;

	// possibly more than needed, but never less
	ipaccess = (struct IP4ACCESS *) calloc(ipcount, sizeof(struct IP4ACCESS));
	if (unlikely(!ipaccess))
		proper_exit(1);//, "Failed to calloc ipaccess");

	ips = 0;
	ptr = buf;
	while (ptr && *ptr) {
		while (*ptr == ' ' || *ptr == '\t')
			ptr++;

		if (*ptr == ',') {
			ptr++;
			continue;
		}

		comma = strchr(ptr, ',');
		if (comma)
			*(comma++) = '\0';

		group = NOPRIVGROUP;

		if (isalpha(*ptr) && *(ptr+1) == ':') {
			if (DEFINEDGROUP(*ptr))
				group = GROUP(*ptr);
			ptr += 2;
		}

		ipaccess[ips].group = group;

		if (strcmp(ptr, ALLIPS) == 0 || strcmp(ptr, ALLIP4) == 0)
			ipaccess[ips].ip = ipaccess[ips].mask = 0;
		else
		{
			slash = strchr(ptr, '/');
			if (!slash)
				ipaccess[ips].mask = 0xffffffff;
			else {
				*(slash++) = '\0';
				mask = atoi(slash);
				if (mask < 1 || mask > 32)
					goto popipo; // skip invalid/zero

				ipaccess[ips].mask = 0;
				while (mask-- >= 0) {
					octet = 1 << (mask % 8);
					ipaccess[ips].mask |= (octet << (24 - (8 * (mask >> 3))));
				}
			}

			ipaccess[ips].ip = 0; // missing default to '.0'
			for (i = 0; ptr && (i < 4); i++) {
				dot = strchr(ptr, '.');
				if (dot)
					*(dot++) = '\0';
				octet = atoi(ptr);

				if (octet < 0 || octet > 0xff)
					goto popipo; // skip invalid

				ipaccess[ips].ip |= (octet << (24 - (i * 8)));

				ptr = dot;
			}

			ipaccess[ips].ip &= ipaccess[ips].mask;
		}

		ips++;
popipo:
		ptr = comma;
	}

	free(buf);
}

static bool check_connect(struct sockaddr_in *cli, char **connectaddr, char *group)
{
	bool addrok = false;

	*connectaddr = inet_ntoa(cli->sin_addr);

	*group = NOPRIVGROUP;
	if (opt_api_allow) {
		int client_ip = htonl(cli->sin_addr.s_addr);
		for (int i = 0; i < ips; i++) {
			if ((client_ip & ipaccess[i].mask) == ipaccess[i].ip) {
				addrok = true;
				*group = ipaccess[i].group;
				break;
			}
		}
	}
	else if (strcmp(opt_api_bind, ALLIP4) == 0)
		addrok = true;
	else
		addrok = (strcmp(*connectaddr, localaddr) == 0);

	return addrok;
}

static void mcast()
{
	struct sockaddr_in listen;
	struct ip_mreq grp;
	struct sockaddr_in came_from;
	time_t bindstart;
	char *binderror;
	SOCKETTYPE mcast_sock;
	SOCKETTYPE reply_sock;
	socklen_t came_from_siz;
	char *connectaddr;
	ssize_t rep;
	int bound;
	int count;
	int reply_port;
	bool addrok;
	char group;

	char expect[] = "ccminer-"; // first 8 bytes constant
	char *expect_code;
	size_t expect_code_len;
	char buf[1024];
	char replybuf[1024];

	memset(&grp, 0, sizeof(grp));
	grp.imr_multiaddr.s_addr = inet_addr(opt_api_mcast_addr);
	if (grp.imr_multiaddr.s_addr == INADDR_NONE)
		proper_exit(1); //, "Invalid Multicast Address");
	grp.imr_interface.s_addr = INADDR_ANY;

	mcast_sock = socket(AF_INET, SOCK_DGRAM, 0);

	int optval = 1;
	if (SOCKETFAIL(setsockopt(mcast_sock, SOL_SOCKET, SO_REUSEADDR, (const char *)(&optval), sizeof(optval)))) {
		applog(LOG_ERR, "API mcast setsockopt SO_REUSEADDR failed (%s)%s", strerror(errno), MUNAVAILABLE);
		goto die;
	}

	memset(&listen, 0, sizeof(listen));
	listen.sin_family = AF_INET;
	listen.sin_addr.s_addr = INADDR_ANY;
	listen.sin_port = htons(opt_api_mcast_port);

	// try for more than 1 minute ... in case the old one hasn't completely gone yet
	bound = 0;
	bindstart = time(NULL);
	while (bound == 0) {
		if (SOCKETFAIL(bind(mcast_sock, (struct sockaddr *)(&listen), sizeof(listen)))) {
			binderror = strerror(errno);;
			if ((time(NULL) - bindstart) > 61)
				break;
			else
				sleep(30);
		}
		else
			bound = 1;
	}

	if (bound == 0) {
		applog(LOG_ERR, "API mcast bind to port %d failed (%s)%s", opt_api_port, binderror, MUNAVAILABLE);
		goto die;
	}

	if (SOCKETFAIL(setsockopt(mcast_sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, (const char *)(&grp), sizeof(grp)))) {
		applog(LOG_ERR, "API mcast join failed (%s)%s", strerror(errno), MUNAVAILABLE);
		goto die;
	}

	expect_code_len = sizeof(expect) + strlen(opt_api_mcast_code);
	expect_code = (char *)calloc(1, expect_code_len + 1);
	if (!expect_code)
		proper_exit(1); //, "Failed to malloc mcast expect_code");
	snprintf(expect_code, expect_code_len + 1, "%s%s-", expect, opt_api_mcast_code);

	count = 0;
	while (42) {
		sleep(1);

		count++;
		came_from_siz = sizeof(came_from);
		if (SOCKETFAIL(rep = recvfrom(mcast_sock, buf, sizeof(buf) - 1,
			0, (struct sockaddr *)(&came_from), &came_from_siz))) {
			applog(LOG_DEBUG, "API mcast failed count=%d (%s) (%d)",
				count, strerror(errno), (int)mcast_sock);
			continue;
		}

		addrok = check_connect(&came_from, &connectaddr, &group);
		applog(LOG_DEBUG, "API mcast from %s - %s",
			connectaddr, addrok ? "Accepted" : "Ignored");
		if (!addrok) {
			continue;
		}

		buf[rep] = '\0';
		if (rep > 0 && buf[rep - 1] == '\n')
			buf[--rep] = '\0';

		applog(LOG_DEBUG, "API mcast request rep=%d (%s) from %s:%d",
			(int)rep, buf,
			inet_ntoa(came_from.sin_addr),
			ntohs(came_from.sin_port));

		if ((size_t)rep > expect_code_len && memcmp(buf, expect_code, expect_code_len) == 0) {
			reply_port = atoi(&buf[expect_code_len]);
			if (reply_port < 1 || reply_port > 65535) {
				applog(LOG_DEBUG, "API mcast request ignored - invalid port (%s)",
					&buf[expect_code_len]);
			}
			else {
				applog(LOG_DEBUG, "API mcast request OK port %s=%d",
					&buf[expect_code_len], reply_port);

				came_from.sin_port = htons(reply_port);
				reply_sock = socket(AF_INET, SOCK_DGRAM, 0);

				snprintf(replybuf, sizeof(replybuf),
					"ccm-%s-%d-%s", opt_api_mcast_code, opt_api_port, opt_api_mcast_des);

				rep = sendto(reply_sock, replybuf, (int) strlen(replybuf) + 1,
					0, (struct sockaddr *)(&came_from), (int) sizeof(came_from));
				if (SOCKETFAIL(rep)) {
					applog(LOG_DEBUG, "API mcast send reply failed (%s) (%d)",
						strerror(errno), (int)reply_sock);
				} else {
					applog(LOG_DEBUG, "API mcast send reply (%s) succeeded (%d) (%d)",
						replybuf, (int)rep, (int)reply_sock);
				}

				CLOSESOCKET(reply_sock);
			}
		}
		else
			applog(LOG_DEBUG, "API mcast request was no good");
	}

die:
	CLOSESOCKET(mcast_sock);
}

static void *mcast_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info *)userdata;

	pthread_detach(pthread_self());
	pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

	mcast();

	//PTH(mythr) = 0L;

	return NULL;
}

void mcast_init()
{
	struct thr_info *thr;

	thr = (struct thr_info *)calloc(1, sizeof(*thr));
	if (!thr)
		proper_exit(1); //, "Failed to calloc mcast thr");

	if (unlikely(pthread_create(&thr->pth, NULL, mcast_thread, thr)))
		proper_exit(1); //, "API mcast thread create failed");
}

static void api()
{
	const char *addr = opt_api_bind;
	unsigned short port = (unsigned short) opt_api_port; // 4068
	char buf[MYBUFSIZ];
	int n, bound;
	char *connectaddr;
	char *binderror;
	char group;
	time_t bindstart;
	struct sockaddr_in serv;
	struct sockaddr_in cli;
	socklen_t clisiz;
	bool addrok = false;
	long long counter;
	char *result;
	char *params;
	int i;

	SOCKETTYPE c;
	SOCKETTYPE *apisock;
	if (!opt_api_port && opt_debug) {
		applog(LOG_DEBUG, "API disabled");
		return;
	}

	setup_groups();

	if (opt_api_allow) {
		setup_ipaccess();
		if (ips == 0) {
			applog(LOG_WARNING, "API not running (no valid IPs specified)%s", UNAVAILABLE);
		}
	}

	apisock = (SOCKETTYPE*) calloc(1, sizeof(*apisock));
	*apisock = INVSOCK;

	sleep(1);

	*apisock = socket(AF_INET, SOCK_STREAM, 0);
	if (*apisock == INVSOCK) {
		applog(LOG_ERR, "API initialisation failed (%s)%s", strerror(errno), UNAVAILABLE);
		return;
	}

	memset(&serv, 0, sizeof(serv));
	serv.sin_family = AF_INET;
	serv.sin_addr.s_addr = inet_addr(addr); // TODO: allow bind to ip/interface
	if (serv.sin_addr.s_addr == (in_addr_t)INVINETADDR) {
		applog(LOG_ERR, "API initialisation 2 failed (%s)%s", strerror(errno), UNAVAILABLE);
		// free(apisock); FIXME!!
		return;
	}

	serv.sin_port = htons(port);

#ifndef WIN32
	// On linux with SO_REUSEADDR, bind will get the port if the previous
	// socket is closed (even if it is still in TIME_WAIT) but fail if
	// another program has it open - which is what we want
	int optval = 1;
	// If it doesn't work, we don't really care - just show a debug message
	if (SOCKETFAIL(setsockopt(*apisock, SOL_SOCKET, SO_REUSEADDR, (void *)(&optval), sizeof(optval))))
	        applog(LOG_DEBUG, "API setsockopt SO_REUSEADDR failed (ignored): %s", SOCKERRMSG);
#else
	// On windows a 2nd program can bind to a port>1024 already in use unless
	// SO_EXCLUSIVEADDRUSE is used - however then the bind to a closed port
	// in TIME_WAIT will fail until the timeout - so we leave the options alone
#endif

	// try for 1 minute ... in case the old one hasn't completely gone yet
	bound = 0;
	bindstart = time(NULL);
	while (bound == 0) {
		if (bind(*apisock, (struct sockaddr *)(&serv), sizeof(serv)) < 0) {
			binderror = strerror(errno);
			if ((time(NULL) - bindstart) > 61)
				break;
			else if (opt_api_port == 4068) {
				/* when port is default one, use first available */
				if (opt_debug)
					applog(LOG_DEBUG, "API bind to port %d failed, trying port %u",
						port, (uint32_t) port+1);
				port++;
				serv.sin_port = htons(port);
				sleep(1);
			} else {
				if (!opt_quiet || opt_debug)
					applog(LOG_WARNING, "API bind to port %u failed - trying again in 20sec",
						(uint32_t) port);
				sleep(20);
			}
		}
		else {
			bound = 1;
			if (opt_api_port != port) {
				applog(LOG_WARNING, "API bind to port %d failed - using port %u",
					opt_api_port, (uint32_t)port);
				opt_api_port = port;
			}
		}
	}

	if (bound == 0) {
		applog(LOG_WARNING, "API bind to port %d failed (%s)%s", port, binderror, UNAVAILABLE);
		free(apisock);
		return;
	}

	if (SOCKETFAIL(listen(*apisock, QUEUE))) {
		applog(LOG_ERR, "API initialisation 3 failed (%s)%s", strerror(errno), UNAVAILABLE);
		CLOSESOCKET(*apisock);
		free(apisock);
		return;
	}

	if (opt_api_allow && strcmp(opt_api_bind, "127.0.0.1") == 0)
		applog(LOG_WARNING, "API open locally in full access mode on port %d", opt_api_port);
	else if (opt_api_allow)
		applog(LOG_WARNING, "API open in full access mode to %s on port %d", opt_api_allow, opt_api_port);
	else if (strcmp(opt_api_bind, "127.0.0.1") != 0)
		applog(LOG_INFO, "API open to the network in read-only mode on port %d", opt_api_port);

	if (opt_api_mcast)
		mcast_init();

	buffer = (char *) calloc(1, MYBUFSIZ + 1);

	counter = 0;
	while (bye == 0 && !abort_flag) {
		counter++;

		clisiz = sizeof(cli);
		c = accept(*apisock, (struct sockaddr*) (&cli), &clisiz);
		if (SOCKETFAIL(c)) {
			applog(LOG_ERR, "API failed (%s)%s", strerror(errno), UNAVAILABLE);
			CLOSESOCKET(*apisock);
			free(apisock);
			free(buffer);
			return;
		}

		addrok = check_connect(&cli, &connectaddr, &group);
		if (opt_debug && opt_protocol)
			applog(LOG_DEBUG, "API: connection from %s - %s",
				connectaddr, addrok ? "Accepted" : "Ignored");

		if (addrok) {
			bool fail;
			char *wskey = NULL;
			n = recv(c, &buf[0], SOCK_REC_BUFSZ, 0);

			fail = SOCKETFAIL(n);
			if (fail)
				buf[0] = '\0';
			else if (n > 0 && buf[n-1] == '\n') {
				/* telnet compat \r\n */
				buf[n-1] = '\0'; n--;
				if (n > 0 && buf[n-1] == '\r')
					buf[n-1] = '\0';
			}
			buf[n] = '\0';

			//if (opt_debug && opt_protocol && n > 0)
			//	applog(LOG_DEBUG, "API: recv command: (%d) '%s'+char(%x)", n, buf, buf[n-1]);

			if (!fail) {
				char *msg = NULL;
				/* Websocket requests compat. */
				if ((msg = strstr(buf, "GET /")) && strlen(msg) > 5) {
					char cmd[256] = { 0 };
					sscanf(&msg[5], "%s\n", cmd);
					params = strchr(cmd, '/');
					if (params)
						*(params++) = '|';
					params = strchr(cmd, '/');
					if (params)
						*(params++) = '\0';
					wskey = strstr(msg, "Sec-WebSocket-Key");
					if (wskey) {
						char *eol = strchr(wskey, '\r');
						if (eol) *eol = '\0';
						wskey = strchr(wskey, ':');
						wskey++;
						while ((*wskey) == ' ') wskey++; // ltrim
					}
					n = sprintf(buf, "%s", cmd);
				}

				params = strchr(buf, '|');
				if (params != NULL)
					*(params++) = '\0';

				if (opt_debug && opt_protocol && n > 0)
					applog(LOG_DEBUG, "API: exec command %s(%s)", buf, params ? params : "");

				for (i = 0; i < CMDMAX; i++) {
					if (strcmp(buf, cmds[i].name) == 0 && strlen(buf)) {
						if (params && strlen(params)) {
							// remove possible trailing |
							if (params[strlen(params)-1] == '|')
								params[strlen(params)-1] = '\0';
						}
						result = (cmds[i].func)(params);
						if (wskey) {
							websocket_handshake(c, result, wskey);
							break;
						}
						send_result(c, result);
						break;
					}
				}
			}
			CLOSESOCKET(c);
		}
	}

	CLOSESOCKET(*apisock);
	free(apisock);
	free(buffer);
}

/* external access */
void *api_thread(void *userdata)
{
	struct thr_info *mythr = (struct thr_info*)userdata;

	startup = time(NULL);
	api();
	tq_freeze(mythr->q);

	if (bye) {
		// quit command
		proper_exit(1);
	}

	return NULL;
}

/* to be able to report the default value set in each algo */
void api_set_throughput(int thr_id, uint32_t throughput)
{
	if (thr_id < MAX_GPUS && thr_info) {
		struct cgpu_info *cgpu = &thr_info[thr_id].gpu;
		cgpu->intensity = throughput2intensity(throughput);
		if (cgpu->throughput != throughput) cgpu->throughput = throughput;
	}
	// to display in bench results
	if (opt_benchmark)
		bench_set_throughput(thr_id, throughput);
}
