/**
 * XMR RPC 2.0 Stratum and BBR Scratchpad
 * tpruvot@github - October 2016 - Under GPLv3 Licence
 */

#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h> // mkdir

#include <miner.h>

#ifdef _MSC_VER
#include "mman.h" // mmap
#include <direct.h> // _mkdir
#define chdir(x) _chdir(x)
#define mkdir(x) _mkdir(x)
#define getcwd(d,sz) _getcwd(d,sz)
#define unlink(x) _unlink(x)
#define PATH_MAX MAX_PATH
#else
#include <sys/mman.h> // mmap
#endif

#if defined(__APPLE__) && !defined(MAP_HUGETLB)
#define MAP_ANONYMOUS MAP_ANON
#define MAP_HUGETLB 0
#define MAP_POPULATE 0
#define MADV_HUGEPAGE 0
#endif

#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 0
#endif

#ifndef PRIu64
#define PRIu64 "I64u"
#endif

#include <algos.h>
#include "xmr-rpc.h"
#include "wildkeccak.h"

double target_to_diff_rpc2(uint32_t* target)
{
	// unlike other algos, xmr diff is very low
	if (opt_algo == ALGO_CRYPTONIGHT && target[7]) {
		// simplified to get 1.0 for 1000
		return (double) (UINT32_MAX / target[7]) / 1000;
	}
	else if (opt_algo == ALGO_CRYPTOLIGHT && target[7]) {
		return (double) (UINT32_MAX / target[7]) / 1000;
	}
	else if (opt_algo == ALGO_WILDKECCAK) {
		return target_to_diff(target) * 1000;
	}
	return target_to_diff(target); // util.cpp
}

extern struct stratum_ctx stratum;

bool jobj_binary(const json_t *obj, const char *key, void *buf, size_t buflen);

pthread_mutex_t rpc2_job_lock;
pthread_mutex_t rpc2_work_lock;
pthread_mutex_t rpc2_login_lock;
//pthread_mutex_t rpc2_getscratchpad_lock;

char* opt_scratchpad_url = NULL;
uint64_t* pscratchpad_buff = NULL;

// hide addendums flood on start
static bool opt_quiet_start = true;

static const char * pscratchpad_local_cache = NULL;
static const char cachedir_suffix[] = "boolberry"; /* scratchpad cache saved as ~/.cache/boolberry/scratchpad.bin */
static char scratchpad_file[PATH_MAX];
static time_t prev_save = 0;
static struct scratchpad_hi current_scratchpad_hi;
static struct addendums_array_entry add_arr[WILD_KECCAK_ADDENDUMS_ARRAY_SIZE];

static char *rpc2_job_id = NULL;
static char *rpc2_blob = NULL;
static uint32_t rpc2_target = 0;
static size_t rpc2_bloblen = 0;
static struct work rpc2_work;

static char rpc2_id[64] = { 0 };
static uint64_t last_found_nonce = 0;

static const char* get_json_string_param(const json_t *val, const char* param_name)
{
	json_t *tmp;
	tmp = json_object_get(val, param_name);
	if(!tmp) {
		return NULL;
	}
	return json_string_value(tmp);
}

static size_t hex2bin_len(unsigned char *p, const char *hexstr, size_t len)
{
	char hex_byte[3];
	char *ep;
	size_t count = 0;

	hex_byte[2] = '\0';

	while (*hexstr && len) {
		if (!hexstr[1]) {
			applog(LOG_ERR, "hex2bin str truncated");
			return 0;
		}
		hex_byte[0] = hexstr[0];
		hex_byte[1] = hexstr[1];
		*p = (unsigned char) strtol(hex_byte, &ep, 16);
		if (*ep) {
			applog(LOG_ERR, "hex2bin failed on '%s'", hex_byte);
			return 0;
		}
		count++;
		p++;
		hexstr += 2;
		len--;
	}

	return (/*len == 0 &&*/ *hexstr == 0) ? count : 0;
}

static bool parse_height_info(const json_t *hi_section, struct scratchpad_hi* phi)
{
	unsigned char prevhash[32] = { 0 };
	const char* block_id;
	uint64_t hi_h;
	size_t len;

	if(!phi || !hi_section) {
		applog(LOG_ERR, "parse_height_info: wrong params");
		return false;
	}

	json_t *height = json_object_get(hi_section, "height");
	if(!height) {
		applog(LOG_ERR, "JSON inval hi, no height param");
		goto err_out;
	}

	if(!json_is_integer(height)) {
		applog(LOG_ERR, "JSON inval hi: height is not integer ");
		goto err_out;
	}

	hi_h = (uint64_t)json_integer_value(height);
	if(!hi_h) {
		applog(LOG_ERR, "JSON inval hi: height is 0");
		goto err_out;
	}

	block_id = get_json_string_param(hi_section, "block_id");
	if(!block_id) {
		applog(LOG_ERR, "JSON inval hi: block_id not found ");
		goto err_out;
	}

	len = hex2bin_len(prevhash, block_id, 32);
	if(len != 32) {
		applog(LOG_ERR, "JSON inval hi: block_id wrong len %d", len);
		goto err_out;
	}

	phi->height = hi_h;
	memcpy(phi->prevhash, prevhash, 32);

	return true;
err_out:
	return false;
}

static void reset_scratchpad(void)
{
	current_scratchpad_hi.height = 0;
	scratchpad_size = 0;
	//unlink(scratchpad_file);
}

static bool patch_scratchpad_with_addendum(uint64_t global_add_startpoint, uint64_t* padd_buff, size_t count/*uint64 units*/)
{
	for(size_t i = 0; i < count; i += 4) {
		uint64_t global_offset = (padd_buff[i]%(global_add_startpoint/4))*4;
		for(size_t j = 0; j != 4; j++)
			pscratchpad_buff[global_offset + j] ^= padd_buff[i + j];
	}
	return true;
}

static bool apply_addendum(uint64_t* padd_buff, size_t count/*uint64 units*/)
{
	if(WILD_KECCAK_SCRATCHPAD_BUFFSIZE <= (scratchpad_size + count)*8 ) {
		applog(LOG_ERR, "!!!!!!! WILD_KECCAK_SCRATCHPAD_BUFFSIZE overflowed !!!!!!!! please increase this constant! ");
		return false;
	}

	if(!patch_scratchpad_with_addendum(scratchpad_size, padd_buff, count)) {
		applog(LOG_ERR, "patch_scratchpad_with_addendum is broken, resetting scratchpad");
		reset_scratchpad();
		return false;
	}
	for(int k = 0; k != count; k++)
		pscratchpad_buff[scratchpad_size+k] = padd_buff[k];

	scratchpad_size += count;

	return true;
}

static bool pop_addendum(struct addendums_array_entry* entry)
{
	if(!entry)
		return false;

	if(!entry->add_size || !entry->prev_hi.height) {
		applog(LOG_ERR, "wrong parameters");
		return false;
	}
	patch_scratchpad_with_addendum(scratchpad_size - entry->add_size, &pscratchpad_buff[scratchpad_size - entry->add_size], (size_t) entry->add_size);
	scratchpad_size = scratchpad_size - entry->add_size;
	memcpy(&current_scratchpad_hi, &entry->prev_hi, sizeof(entry->prev_hi));

	memset(entry, 0, sizeof(struct addendums_array_entry));
	return true;
}

// playback scratchpad addendums for whole add_arr
static bool revert_scratchpad()
{
	size_t p = 0;
	size_t i = 0;
	size_t arr_size = ARRAY_SIZE(add_arr);

	for(p=0; p != arr_size; p++) {
		i = arr_size-(p+1);
		if(!add_arr[i].prev_hi.height)
			continue;
		pop_addendum(&add_arr[i]);
	}
	return true;
}

static bool push_addendum_info(struct scratchpad_hi* pprev_hi, uint64_t size /* uint64 units count*/)
{
	size_t i = 0;
	size_t arr_size = ARRAY_SIZE(add_arr);

	// Find last free entry
	for(i=0; i != arr_size; i++) {
		if(!add_arr[i].prev_hi.height)
			break;
	}

	if(i >= arr_size) {
		// Shift array
		memmove(&add_arr[0], &add_arr[1], (arr_size-1)*sizeof(add_arr[0]));
		i = arr_size - 1;
	}
	add_arr[i].prev_hi = *pprev_hi;
	add_arr[i].add_size = size;

	return true;
}

static bool addendum_decode(const json_t *addm)
{
	struct scratchpad_hi hi;
	unsigned char prevhash[32];
	uint64_t* padd_buff;
	uint64_t old_height;

	json_t* hi_section = json_object_get(addm, "hi");
	if (!hi_section) {
		//applog(LOG_ERR, "JSON addms field not found");
		//return false;
		return true;
	}

	if(!parse_height_info(hi_section, &hi)) {
		return false;
	}

	const char* prev_id_str = get_json_string_param(addm, "prev_id");
	if(!prev_id_str) {
		applog(LOG_ERR, "JSON prev_id is not a string");
		return false;
	}
	if(!hex2bin(prevhash, prev_id_str, 32)) {
		applog(LOG_ERR, "JSON prev_id is not valid hex string");
		return false;
	}

	if(current_scratchpad_hi.height != hi.height -1)
	{
		if(current_scratchpad_hi.height > hi.height -1) {
			//skip low scratchpad
			applog(LOG_ERR, "addendum with hi.height=%lld skiped since current_scratchpad_hi.height=%lld", hi.height, current_scratchpad_hi.height);
			return true;
		}

		//TODO: ADD SPLIT HANDLING HERE
		applog(LOG_ERR, "JSON height in addendum-1 (%lld-1) missmatched with current_scratchpad_hi.height(%lld), reverting scratchpad and re-login",
			hi.height, current_scratchpad_hi.height);
		revert_scratchpad();
		//init re-login
		strcpy(rpc2_id, "");
		return false;
	}

	if(memcmp(prevhash, current_scratchpad_hi.prevhash, 32)) {
		//TODO: ADD SPLIT HANDLING HERE
		applog(LOG_ERR, "JSON prev_id in addendum missmatched with current_scratchpad_hi.prevhash");
		return false;
	}

	const char* addm_hexstr = get_json_string_param(addm, "addm");
	if(!addm_hexstr) {
		applog(LOG_ERR, "JSON prev_id in addendum missmatched with current_scratchpad_hi.prevhash");
		return false;
	}
	size_t add_len = strlen(addm_hexstr);
	if(add_len%64) {
		applog(LOG_ERR, "JSON wrong addm hex str len");
		return false;
	}
	padd_buff = (uint64_t*) calloc(1, add_len/2);
	if (!padd_buff) {
		applog(LOG_ERR, "out of memory, wanted %zu", add_len/2);
		return false;
	}

	if(!hex2bin((unsigned char*)padd_buff, addm_hexstr, add_len/2)) {
		applog(LOG_ERR, "JSON wrong addm hex str len");
		goto err_out;
	}

	if(!apply_addendum(padd_buff, add_len/16)) {
		applog(LOG_ERR, "JSON Failed to apply_addendum!");
		goto err_out;
	}
	free(padd_buff);

	push_addendum_info(&current_scratchpad_hi, add_len/16);
	old_height = current_scratchpad_hi.height;
	current_scratchpad_hi = hi;

	if (!opt_quiet && !opt_quiet_start)
		applog(LOG_BLUE, "ADDENDUM APPLIED: Block %lld", (long long) current_scratchpad_hi.height);

	return true;
err_out:
	free(padd_buff);
	return false;
}

static bool addendums_decode(const json_t *job)
{
	json_t* paddms = json_object_get(job, "addms");
	if (!paddms) {
		//applog(LOG_ERR, "JSON addms field not found");
		//return false;
		return true;
	}

	if(!json_is_array(paddms)) {
		applog(LOG_ERR, "JSON addms field is not array");
		return false;
	}

	size_t add_sz = json_array_size(paddms);
	for (size_t i = 0; i < add_sz; i++)
	{
		json_t *addm = json_array_get(paddms, i);
		if (!addm) {
			applog(LOG_ERR, "Internal error: failed to get addm");
			return false;
		}
		if(!addendum_decode(addm))
			return false;
	}

	return true;
}

bool rpc2_job_decode(const json_t *job, struct work *work)
{
	json_t *tmp;
	size_t blobLen;
	const char *job_id;
	const char *hexblob;

	tmp = json_object_get(job, "job_id");
	if (!tmp) {
		applog(LOG_ERR, "JSON inval job id");
		goto err_out;
	}

	if(opt_algo == ALGO_WILDKECCAK && !addendums_decode(job)) {
		applog(LOG_ERR, "JSON failed to process addendums");
		goto err_out;
	}
	// now allow ADDENDUM notices (after the init)
	opt_quiet_start = false;

	job_id = json_string_value(tmp);
	tmp = json_object_get(job, "blob");
	if (!tmp) {
		applog(LOG_ERR, "JSON inval blob");
		goto err_out;
	}
	hexblob = json_string_value(tmp);
	blobLen = strlen(hexblob);
	if (blobLen % 2 != 0 || ((blobLen / 2) < 40 && blobLen != 0) || (blobLen / 2) > 128)
	{
		applog(LOG_ERR, "JSON invalid blob length");
		goto err_out;
	}

	if (blobLen != 0)
	{
		pthread_mutex_lock(&rpc2_job_lock);
		char *blob = (char*) calloc(1, blobLen / 2);
		if (!hex2bin(blob, hexblob, blobLen / 2))
		{
			applog(LOG_ERR, "JSON inval blob");
			pthread_mutex_unlock(&rpc2_job_lock);
			goto err_out;
		}
		if (rpc2_blob) {
			free(rpc2_blob);
		}
		rpc2_bloblen = blobLen / 2;
		rpc2_blob = (char*) malloc(rpc2_bloblen);
		memcpy(rpc2_blob, blob, blobLen / 2);

		free(blob);

		uint32_t target;
		jobj_binary(job, "target", &target, 4);
		if(rpc2_target != target) {
			double difficulty = (((double) UINT32_MAX) / target);
			stratum.job.diff = difficulty;
			rpc2_target = target;
		}

		if (rpc2_job_id) {
			// reset job share counter
			if (strcmp(rpc2_job_id, job_id)) stratum.job.shares_count = 0;
			free(rpc2_job_id);
		}
		rpc2_job_id = strdup(job_id);
		pthread_mutex_unlock(&rpc2_job_lock);
	}

	if(work)
	{
		if (!rpc2_blob) {
			applog(LOG_ERR, "Requested work before work was received");
			goto err_out;
		}
		memcpy(work->data, rpc2_blob, rpc2_bloblen);
		memset(work->target, 0xff, sizeof(work->target));
		work->target[7] = rpc2_target;
		work->targetdiff = target_to_diff_rpc2(work->target);

		snprintf(work->job_id, sizeof(work->job_id), "%s", rpc2_job_id);
	}

	if (opt_algo == ALGO_WILDKECCAK)
		wildkeccak_scratchpad_need_update(pscratchpad_buff);
	return true;

err_out:
	return false;
}

extern struct work _ALIGN(64) g_work;
extern volatile time_t g_work_time;
extern bool submit_old;

bool rpc2_stratum_job(struct stratum_ctx *sctx, json_t *id, json_t *params)
{
	bool ret = false;
	pthread_mutex_lock(&rpc2_work_lock);
	ret = rpc2_job_decode(params, &rpc2_work);
	// update miner threads work
	ret = ret && rpc2_stratum_gen_work(sctx, &g_work);
	restart_threads();
	pthread_mutex_unlock(&rpc2_work_lock);
	return ret;
}

bool rpc2_stratum_gen_work(struct stratum_ctx *sctx, struct work *work)
{
//	pthread_mutex_lock(&rpc2_work_lock);
	memcpy(work, &rpc2_work, sizeof(struct work));
	if (stratum_diff != sctx->job.diff) {
		char sdiff[32] = { 0 };
		stratum_diff = sctx->job.diff;
		if (opt_showdiff && work->targetdiff != stratum_diff)
			snprintf(sdiff, 32, " (%g)", work->targetdiff);
		if (stratum_diff >= 1e6)
			applog(LOG_WARNING, "Stratum difficulty set to %.1f M%s", stratum_diff/1e6, sdiff);
		else
			applog(LOG_WARNING, "Stratum difficulty set to %.0f%s", stratum_diff, sdiff);
	}
	if (work->target[7] != rpc2_target) {
		work->target[7] = rpc2_target;
		work->targetdiff = target_to_diff_rpc2(work->target);
		g_work_time = 0;
		restart_threads();
	}
//	pthread_mutex_unlock(&rpc2_work_lock);
	return (work->data[0] != 0);
}

#define JSON_SUBMIT_BUF_LEN 512
// called by submit_upstream_work()
bool rpc2_stratum_submit(struct pool_infos *pool, struct work *work)
{
	char    _ALIGN(64) s[JSON_SUBMIT_BUF_LEN];
	uint8_t _ALIGN(64) hash[32];
	uint8_t _ALIGN(64) data[88];
	char *noncestr, *hashhex;
	int idnonce = work->submit_nonce_id;

	memcpy(&data[0], work->data, 88);

	if (opt_algo == ALGO_WILDKECCAK) {
		// 64 bits nonce
		memcpy(&data[1], work->nonces, 8);
		// pass if the previous hash is not the current previous hash
		if(!submit_old && memcmp(&work->data[3], &g_work.data[3], 28)) {
			if (opt_debug) applog(LOG_DEBUG, "stale work detected");
			pool->stales_count++;
			return false;
		}
		noncestr = bin2hex((unsigned char*) &data[1], 8);
		// "nonce":"5794ec8000000000" => 0x0000000080ec9457
		memcpy(&last_found_nonce, work->nonces, 8);
		wildkeccak_hash(hash, data, NULL, 0);
		work_set_target_ratio(work, (uint32_t*) hash);
	}

	else if (opt_algo == ALGO_CRYPTOLIGHT) {
		uint32_t nonce = work->nonces[idnonce];
		noncestr = bin2hex((unsigned char*) &nonce, 4);
		last_found_nonce = nonce;
		cryptolight_hash(hash, data, 76);
		work_set_target_ratio(work, (uint32_t*) hash);
	}

	else if (opt_algo == ALGO_CRYPTONIGHT) {
		uint32_t nonce = work->nonces[idnonce];
		noncestr = bin2hex((unsigned char*) &nonce, 4);
		last_found_nonce = nonce;
		int variant = 0;

		if (cryptonight_fork > 1) {
			variant = ((((unsigned char*)work->data)[0] >= cryptonight_fork) ? ((unsigned char*)work->data)[0] - cryptonight_fork + 1 : 0 );
		}

		cryptonight_hash(hash, data, 76, variant);
		work_set_target_ratio(work, (uint32_t*) hash);
	}

	if (hash[31] != 0)
		return false; // prevent bad hashes
	hashhex = bin2hex((unsigned char*)hash, 32);

	snprintf(s, sizeof(s), "{\"method\":\"submit\",\"params\":"
		"{\"id\":\"%s\",\"job_id\":\"%s\",\"nonce\":\"%s\",\"result\":\"%s\"}, \"id\":%u}",
		rpc2_id, work->job_id, noncestr, hashhex, stratum.job.shares_count + 10);

	free(hashhex);
	free(noncestr);

	gettimeofday(&stratum.tv_submit, NULL);

	if(!stratum_send_line(&stratum, s)) {
		applog(LOG_ERR, "%s stratum_send_line failed", __func__);
		return false;
	}

	//stratum.sharediff = target_to_diff_rpc2((uint32_t*)hash);
	stratum.sharediff = work->sharediff[idnonce];

	return true;
}

bool rpc2_login_decode(const json_t *val)
{
	const char *id;
	const char *s;
	json_t *res = json_object_get(val, "result");
	if(!res) {
		applog(LOG_ERR, "JSON invalid result");
		goto err_out;
	}

	json_t *tmp;
	tmp = json_object_get(res, "id");
	if(!tmp) {
		applog(LOG_ERR, "JSON inval id");
		goto err_out;
	}
	id = json_string_value(tmp);
	if(!id) {
		applog(LOG_ERR, "JSON id is not a string");
		goto err_out;
	}

	strncpy(rpc2_id, id, sizeof(rpc2_id)-1);

	if(opt_debug)
		applog(LOG_DEBUG, "Auth id: %s", id);

	tmp = json_object_get(res, "status");
	if(!tmp) {
		applog(LOG_ERR, "JSON inval status");
		goto err_out;
	}
	s = json_string_value(tmp);
	if(!s) {
		applog(LOG_ERR, "JSON status is not a string");
		goto err_out;
	}
	if(strcmp(s, "OK")) {
		applog(LOG_ERR, "JSON returned status \"%s\"", s);
		goto err_out;
	}

	return true;

err_out:
	return false;
}

bool store_scratchpad_to_file(bool do_fsync)
{
	char file_name_buff[PATH_MAX] = { 0 };
	FILE *fp;
	int ret;

	if(opt_algo != ALGO_WILDKECCAK) return true;
	if(!scratchpad_size || !pscratchpad_buff) return true;

	snprintf(file_name_buff, sizeof(file_name_buff), "%s.tmp", pscratchpad_local_cache);
	unlink(file_name_buff);
	fp = fopen(file_name_buff, "wbx");
	if (!fp) {
		applog(LOG_ERR, "failed to create file %s: %s", file_name_buff, strerror(errno));
		return false;
	}

	struct scratchpad_file_header sf = { 0 };
	memcpy(sf.add_arr, add_arr, sizeof(sf.add_arr));
	sf.current_hi = current_scratchpad_hi;
	sf.scratchpad_size = scratchpad_size;

	if ((fwrite(&sf, sizeof(sf), 1, fp) != 1) ||
		(fwrite(pscratchpad_buff, 8, (size_t) scratchpad_size, fp) != scratchpad_size)) {
			applog(LOG_ERR, "failed to write file %s: %s", file_name_buff, strerror(errno));
			fclose(fp);
			unlink(file_name_buff);
			return false;
	}
	fflush(fp);
	/*if (do_fsync) {
		if (fsync(fileno(fp)) == -1) {
			applog(LOG_ERR, "failed to fsync file %s: %s", file_name_buff, strerror(errno));
			fclose(fp);
			unlink(file_name_buff);
			return false;
		}
	}*/
	if (fclose(fp) == EOF) {
		applog(LOG_ERR, "failed to write file %s: %s", file_name_buff, strerror(errno));
		unlink(file_name_buff);
		return false;
	}
	ret = rename(file_name_buff, pscratchpad_local_cache);
	if (ret == -1) {
		applog(LOG_ERR, "failed to rename %s to %s: %s",
			file_name_buff, pscratchpad_local_cache, strerror(errno));
		unlink(file_name_buff);
		return false;
	}
	applog(LOG_DEBUG, "saved scratchpad to %s (%zu+%zu bytes)", pscratchpad_local_cache,
		sizeof(struct scratchpad_file_header), (size_t)scratchpad_size * 8);
	return true;
}

/* TODO: repetitive error+log spam handling */
bool load_scratchpad_from_file(const char *fname)
{
	FILE *fp;
	long flen;

	if(opt_algo != ALGO_WILDKECCAK) return true;

	fp = fopen(fname, "rb");
	if (!fp) {
		if (errno != ENOENT) {
			applog(LOG_ERR, "failed to load %s: %s", fname, strerror(errno));
		}
		return false;
	}

	struct scratchpad_file_header fh = { 0 };
	if ((fread(&fh, sizeof(fh), 1, fp) != 1)) {
		applog(LOG_ERR, "read error from %s: %s", fname, strerror(errno));
		fclose(fp);
		return false;
	}

	if ((fh.scratchpad_size*8 > (WILD_KECCAK_SCRATCHPAD_BUFFSIZE)) ||(fh.scratchpad_size%4)) {
		applog(LOG_ERR, "file %s size invalid (%" PRIu64 "), max=%zu",
			fname, fh.scratchpad_size*8, WILD_KECCAK_SCRATCHPAD_BUFFSIZE);
		fclose(fp);
		return false;
	}

	if (fread(pscratchpad_buff, 8, (size_t) fh.scratchpad_size, fp) != fh.scratchpad_size) {
		applog(LOG_ERR, "read error from %s: %s", fname, strerror(errno));
		fclose(fp);
		return false;
	}

	scratchpad_size = fh.scratchpad_size;
	current_scratchpad_hi = fh.current_hi;
	memcpy(&add_arr[0], &fh.add_arr[0], sizeof(fh.add_arr));
	flen = (long)scratchpad_size*8;

	if (!opt_quiet) {
		applog(LOG_INFO, "Scratchpad size %ld kB at block %" PRIu64, flen/1024, current_scratchpad_hi.height);
	}

	fclose(fp);
	prev_save = time(NULL);

	return true;
}

bool dump_scratchpad_to_file_debug()
{
	char file_name_buff[1024] = { 0 };
	if(opt_algo != ALGO_WILDKECCAK) return true;

	snprintf(file_name_buff, sizeof(file_name_buff), "scratchpad_%" PRIu64 "_%llx.scr",
		current_scratchpad_hi.height, (long long) last_found_nonce);

	/* do not bother rewriting if it exists already */

	FILE *fp = fopen(file_name_buff, "w");
	if(!fp) {
		applog(LOG_WARNING, "failed to open file %s: %s", file_name_buff, strerror(errno));
		return false;
	}
	if (fwrite(pscratchpad_buff, 8, (size_t) scratchpad_size, fp) != scratchpad_size) {
		applog(LOG_ERR, "failed to write file %s: %s", file_name_buff, strerror(errno));
		fclose(fp);
		return false;
	}
	if (fclose(fp) == EOF) {
		applog(LOG_ERR, "failed to write file %s: %s", file_name_buff, strerror(errno));
		return false;
	}

	fclose(fp);
	return true;
}

static bool try_mkdir_chdir(const char *dirn)
{
	if (chdir(dirn) == -1) {
		if (errno == ENOENT) {
#ifdef WIN32
			if (mkdir(dirn) == -1) {
#else
			if (mkdir(dirn, 0700) == -1) {
#endif
				applog(LOG_ERR, "mkdir failed: %s", strerror(errno));
				return false;
			}
			if (chdir(dirn) == -1) {
				applog(LOG_ERR, "chdir failed: %s", strerror(errno));
				return false;
			}
		} else {
			applog(LOG_ERR, "chdir failed: %s", strerror(errno));
			return false;
		}
	}
	return true;
}

static size_t curl_write_data(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
	size_t written = fwrite(ptr, size, nmemb, stream);
	return written;
}

static bool download_inital_scratchpad(const char* path_to, const char* url)
{
	CURL *curl;
	CURLcode res;
	char curl_error_buff[CURL_ERROR_SIZE] = { 0 };
	FILE *fp = fopen(path_to,"wb");
	if (!fp) {
		applog(LOG_ERR, "Failed to create file %s error %d", path_to, errno);
		return false;
	}

	applog(LOG_INFO, "Downloading scratchpad....");

	curl_global_cleanup();
	res = curl_global_init(CURL_GLOBAL_ALL);
	if (res != CURLE_OK) {
		applog(LOG_WARNING, "curl curl_global_init error: %d", (int) res);
	}

	curl = curl_easy_init();
	if (!curl) {
		applog(LOG_INFO, "Failed to curl_easy_init.");
		fclose(fp);
		unlink(path_to);
		return false;
	}

	if (opt_protocol && opt_debug) {
		curl_easy_setopt(curl, CURLOPT_VERBOSE, 1);
	}
	if (opt_proxy) {
		curl_easy_setopt(curl, CURLOPT_PROXY, opt_proxy);
		curl_easy_setopt(curl, CURLOPT_PROXYTYPE, opt_proxy_type);
	}
	curl_easy_setopt(curl, CURLOPT_URL, url);
	curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300);
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
	curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, curl_error_buff);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_data);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
	//curl_easy_setopt(curl, CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2);
	curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0);
	if (opt_cert) {
		curl_easy_setopt(curl, CURLOPT_CAINFO, opt_cert);
	} else {
		curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0);
	}

	res = curl_easy_perform(curl);
	if (res != CURLE_OK) {
		if (res == CURLE_OUT_OF_MEMORY) {
			applog(LOG_ERR, "Failed to download file, not enough memory!");
			applog(LOG_ERR, "curl error: %s", curl_error_buff);
		} else {
			applog(LOG_ERR, "Failed to download file, error: %s", curl_error_buff);
		}
	} else {
		applog(LOG_INFO, "Scratchpad downloaded.");
	}
	/* always cleanup */
	curl_easy_cleanup(curl);

	fflush(fp);
	fclose(fp);

	if (res != CURLE_OK) {
		unlink(path_to);
		return false;
	}
	return true;
}

#ifndef WIN32

void GetScratchpad()
{
	const char *phome_var_name = "HOME";
	size_t sz = WILD_KECCAK_SCRATCHPAD_BUFFSIZE;
	char cachedir[PATH_MAX];

	if(!getenv(phome_var_name)) {
		applog(LOG_ERR, "$%s not set", phome_var_name);
		exit(1);
	}
	else if(!try_mkdir_chdir(getenv(phome_var_name))) {
		exit(1);
	}

	if(!try_mkdir_chdir(".cache")) exit(1);

	if(!try_mkdir_chdir(cachedir_suffix)) exit(1);

	if(getcwd(cachedir, sizeof(cachedir) - 22) == NULL) {
		applog(LOG_ERR, "getcwd failed: %s", strerror(errno));
		exit(1);
	}

	snprintf(scratchpad_file, sizeof(scratchpad_file), "%s/scratchpad.bin", cachedir);
	pscratchpad_local_cache = scratchpad_file;

	if (!opt_quiet)
		applog(LOG_INFO, "Scratchpad file %s", pscratchpad_local_cache);

	pscratchpad_buff = (uint64_t*) mmap(0, sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, 0, 0);
	if(pscratchpad_buff == MAP_FAILED)
	{
		if(opt_debug) applog(LOG_DEBUG, "hugetlb not available");
		pscratchpad_buff = (uint64_t*) malloc(sz);
		if(!pscratchpad_buff) {
			applog(LOG_ERR, "Scratchpad allocation failed");
			exit(1);
		}
	} else {
		if(opt_debug) applog(LOG_DEBUG, "using hugetlb");
	}
	madvise(pscratchpad_buff, sz, MADV_RANDOM | MADV_WILLNEED | MADV_HUGEPAGE);
	mlock(pscratchpad_buff, sz);

	if(!load_scratchpad_from_file(pscratchpad_local_cache))
	{
		if(!opt_scratchpad_url) {
			applog(LOG_ERR, "Scratchpad URL not set. Please specify correct scratchpad url by -k or --scratchpad option");
			exit(1);
		}
		if(!download_inital_scratchpad(pscratchpad_local_cache, opt_scratchpad_url)) {
			applog(LOG_ERR, "Scratchpad not found and not downloaded. Please specify correct scratchpad url by -k or --scratchpad  option");
			exit(1);
		}
		if(!load_scratchpad_from_file(pscratchpad_local_cache)) {
			applog(LOG_ERR, "Failed to load scratchpad data after downloading, probably broken scratchpad link, please restart miner with correct inital scratcpad link(-k or --scratchpad )");
			unlink(pscratchpad_local_cache);
			exit(1);
		}
	}
}

#else /* Windows */

void GetScratchpad()
{
	bool scratchpad_need_update = false;
	size_t sz = WILD_KECCAK_SCRATCHPAD_BUFFSIZE;
	const char* phome_var_name = "LOCALAPPDATA";
	char cachedir[PATH_MAX];

	if(!getenv(phome_var_name)) {
		applog(LOG_ERR, "%s env var is not set", phome_var_name);
		exit(1);
	}
	else if(!try_mkdir_chdir(getenv(phome_var_name))) {
		exit(1);
	}

	if(!try_mkdir_chdir(".cache"))
		exit(1);

	if(!try_mkdir_chdir(cachedir_suffix))
		exit(1);

	if(getcwd(cachedir, sizeof(cachedir) - 22) == NULL) {
		applog(LOG_ERR, "getcwd failed: %s", strerror(errno));
		exit(1);
	}

	snprintf(scratchpad_file, sizeof(scratchpad_file), "%s\\scratchpad.bin", cachedir);
	pscratchpad_local_cache = scratchpad_file;

	if (!opt_quiet)
		applog(LOG_INFO, "Scratchpad file %s", pscratchpad_local_cache);

	if (pscratchpad_buff) {
		reset_scratchpad();
		wildkeccak_scratchpad_need_update(NULL);
		scratchpad_need_update = true;
		free(pscratchpad_buff);
		pscratchpad_buff = NULL;
	}

	pscratchpad_buff = (uint64_t*) malloc(sz);
	if(!pscratchpad_buff) {
		applog(LOG_ERR, "Scratchpad allocation failed");
		exit(1);
	}

	if(!load_scratchpad_from_file(pscratchpad_local_cache))
	{
		if(!opt_scratchpad_url) {
			applog(LOG_ERR, "Scratchpad URL not set. Please specify correct scratchpad url by -k or --scratchpad option");
			exit(1);
		}
		free(pscratchpad_buff);
		pscratchpad_buff = NULL;
		if(!download_inital_scratchpad(pscratchpad_local_cache, opt_scratchpad_url)) {
			applog(LOG_ERR, "Scratchpad not found and not downloaded. Please specify correct scratchpad url by -k or --scratchpad  option");
			exit(1);
		}
		pscratchpad_buff = (uint64_t*) malloc(sz);
		if(!pscratchpad_buff) {
			applog(LOG_ERR, "Scratchpad allocation failed");
			exit(1);
		}
		if(!load_scratchpad_from_file(pscratchpad_local_cache)) {
			applog(LOG_ERR, "Failed to load scratchpad data after downloading, probably broken scratchpad link, please restart miner with correct inital scratcpad link(-k or --scratchpad )");
			unlink(pscratchpad_local_cache);
			exit(1);
		}
	}

	if (scratchpad_need_update)
		wildkeccak_scratchpad_need_update(pscratchpad_buff);
}

#endif /* GetScratchpad() linux */

static bool rpc2_getfullscratchpad_decode(const json_t *val)
{
	const char* status;
	const char* scratch_hex;
	size_t len;
	json_t *hi;
	json_t *res = json_object_get(val, "result");
	if(!res) {
		applog(LOG_ERR, "JSON invalid result in rpc2_getfullscratchpad_decode");
		goto err_out;
	}

	//check status
	status = get_json_string_param(res, "status");
	if (!status ) {
		applog(LOG_ERR, "JSON status is not a string");
		goto err_out;
	}

	if(strcmp(status, "OK")) {
		applog(LOG_ERR, "JSON returned status \"%s\"", status);
		goto err_out;
	}

	//parse scratchpad
	scratch_hex = get_json_string_param(res, "scratchpad_hex");
	if (!scratch_hex) {
		applog(LOG_ERR, "JSON scratch_hex is not a string");
		goto err_out;
	}

	len = hex2bin_len((unsigned char*)pscratchpad_buff, scratch_hex, WILD_KECCAK_SCRATCHPAD_BUFFSIZE);
	if (!len) {
		applog(LOG_ERR, "JSON scratch_hex is not valid hex");
		goto err_out;
	}

	if (len%8 || len%32) {
		applog(LOG_ERR, "JSON scratch_hex is not valid size=%d bytes", len);
		goto err_out;
	}

	//parse hi
	hi = json_object_get(res, "hi");
	if(!hi) {
		applog(LOG_ERR, "JSON inval hi");
		goto err_out;
	}

	if(!parse_height_info(hi, &current_scratchpad_hi))
	{
		applog(LOG_ERR, "JSON inval hi, failed to parse");
		goto err_out;
	}

	applog(LOG_INFO, "Fetched scratchpad size %d bytes", len);
	scratchpad_size = len/8;

	return true;

err_out: return false;
}

static bool rpc2_stratum_getscratchpad(struct stratum_ctx *sctx)
{
	bool ret = false;
	json_t *val = NULL;
	json_error_t err;
	char *s, *sret;
	if(opt_algo != ALGO_WILDKECCAK) return true;

	s = (char*) calloc(1, 1024);
	if (!s)
		goto out;
	sprintf(s, "{\"method\": \"getfullscratchpad\", \"params\": {\"id\": \"%s\", \"agent\": \"" USER_AGENT "\"}, \"id\": 1}", rpc2_id);

	applog(LOG_INFO, "Getting full scratchpad....");
	if (!stratum_send_line(sctx, s))
		goto out;

	//sret = stratum_recv_line_timeout(sctx, 920);
	sret = stratum_recv_line(sctx);
	if (!sret)
		goto out;
	applog(LOG_DEBUG, "Getting full scratchpad received line");

	val = JSON_LOADS(sret, &err);
	free(sret);
	if (!val) {
		applog(LOG_ERR, "JSON decode rpc2_getscratchpad response failed(%d): %s", err.line, err.text);
		goto out;
	}

	applog(LOG_DEBUG, "Getting full scratchpad parsed line");

	ret = rpc2_getfullscratchpad_decode(val);

out:
	free(s);
	if (val)
		json_decref(val);

	return ret;
}

bool rpc2_stratum_authorize(struct stratum_ctx *sctx, const char *user, const char *pass)
{
	bool ret = false;
	json_t *val = NULL, *res_val, *err_val, *job_val = NULL;
	json_error_t err;
	char *sret;
	char *s = (char*) calloc(1, 320 + strlen(user) + strlen(pass));

	if (opt_algo == ALGO_WILDKECCAK) {
		char *prevhash = bin2hex((const unsigned char*)current_scratchpad_hi.prevhash, 32);
		sprintf(s, "{\"method\":\"login\",\"params\":{\"login\":\"%s\",\"pass\":\"%s\","
			   "\"hi\":{\"height\":%" PRIu64 ",\"block_id\":\"%s\"},"
			   "\"agent\":\"" USER_AGENT "\"},\"id\":2}",
			user, pass, current_scratchpad_hi.height, prevhash);
		free(prevhash);
	} else {
		sprintf(s, "{\"method\":\"login\",\"params\":{\"login\":\"%s\",\"pass\":\"%s\","
			   "\"agent\":\"" USER_AGENT "\"},\"id\":2}",
			user, pass);
	}

	if (!stratum_send_line(sctx, s))
		goto out;

	while (1) {
		sret = stratum_recv_line(sctx);
		if (!sret)
			goto out;
		if (!stratum_handle_method(sctx, sret))
			break;
		free(sret);
	}

	val = JSON_LOADS(sret, &err);
	free(sret);
	if (!val) {
		applog(LOG_ERR, "JSON decode failed(%d): %s", err.line, err.text);
		goto out;
	}

	res_val = json_object_get(val, "result");
	err_val = json_object_get(val, "error");

	if (!res_val || json_is_false(res_val) ||
		(err_val && !json_is_null(err_val)))  {
			applog(LOG_ERR, "Stratum authentication failed");
			if (err_val) {
				const char *msg = json_string_value(json_object_get(err_val,"message"));
				if (msg && strlen(msg)) {
					if (strstr(msg, "scratchpad too old") && pscratchpad_local_cache) {
						if (unlink(pscratchpad_local_cache) == 0) {
							applog(LOG_INFO, "Outdated scratchpad, deleted...", pscratchpad_local_cache);
							GetScratchpad();
							goto out;
						}
					}
					applog(LOG_NOTICE, "%s", msg);
				}
			}
			goto out;
	}

	rpc2_login_decode(val);
	job_val = json_object_get(res_val, "job");

	pthread_mutex_lock(&rpc2_work_lock);
	if(job_val) rpc2_job_decode(job_val, &rpc2_work);
	pthread_mutex_unlock(&rpc2_work_lock);

	ret = true;

out:
	free(s);
	if (val)
		json_decref(val);

	return ret;
}

bool rpc2_stratum_request_job(struct stratum_ctx *sctx)
{
	json_t *val = NULL, *res_val, *err_val;
	json_error_t err;
	bool ret = false;
	char *sret;
	char *s = (char*) calloc(1, 10*2048);
	if (!s) {
		applog(LOG_ERR, "Stratum job OOM!");
		return ret;
	}

	if (opt_algo == ALGO_WILDKECCAK) {
		char* prevhash = bin2hex((const unsigned char*)current_scratchpad_hi.prevhash, 32);
		sprintf(s, "{\"method\":\"getjob\",\"params\": {"
			"\"id\":\"%s\", \"hi\": {\"height\": %" PRIu64 ",\"block_id\":\"%s\" }, \"agent\": \"" USER_AGENT "\"},"
			"\"id\":1}",
			rpc2_id, current_scratchpad_hi.height, prevhash);
		free(prevhash);
	} else {
		sprintf(s, "{\"method\":\"getjob\",\"params\":{\"id\":\"%s\"},\"id\":1}", rpc2_id);
	}

	if(!stratum_send_line(sctx, s)) {
		applog(LOG_ERR, "Stratum failed to send getjob line");
		goto out;
	}

	sret = stratum_recv_line(sctx);
	if (!sret) {
		applog(LOG_ERR, "Stratum failed to recv getjob line");
		goto out;
	}

	val = JSON_LOADS(sret, &err);
	free(sret);
	if (!val) {
		applog(LOG_ERR, "JSON getwork decode failed(%d): %s", err.line, err.text);
		goto out;
	}

	res_val = json_object_get(val, "result");
	err_val = json_object_get(val, "error");

	if (!res_val || json_is_false(res_val) ||
		(err_val && !json_is_null(err_val))) {
			applog(LOG_ERR, "Stratum getjob failed");
			goto out;
	}

	pthread_mutex_lock(&rpc2_work_lock);
	rpc2_job_decode(res_val, &rpc2_work);
	pthread_mutex_unlock(&rpc2_work_lock);

	ret = true;
out:
	if (val)
		json_decref(val);

	return ret;
}

int rpc2_stratum_thread_stuff(struct pool_infos* pool)
{
	int opt_fail_pause = 10;

	if(!strcmp(rpc2_id, "")) {
		if (!opt_quiet)
			applog(LOG_DEBUG, "disconnecting...");
		stratum_disconnect(&stratum);
		//not logged in, try to relogin
		if (!opt_quiet)
			applog(LOG_DEBUG, "Re-connect and relogin...");
		if(!stratum_connect(&stratum, stratum.url) || !stratum_authorize(&stratum, pool->user, pool->pass)) {
			stratum_disconnect(&stratum);
			applog(LOG_ERR, "Failed...retry after %d seconds", opt_fail_pause);
			sleep(opt_fail_pause);
		}
	}

	if(!scratchpad_size && opt_algo == ALGO_WILDKECCAK) {
		if(!rpc2_stratum_getscratchpad(&stratum)) {
			stratum_disconnect(&stratum);
			applog(LOG_ERR, "...retry after %d seconds", opt_fail_pause);
			sleep(opt_fail_pause);
		}
		store_scratchpad_to_file(false);
		prev_save = time(NULL);

		if(!rpc2_stratum_request_job(&stratum)) {
			stratum_disconnect(&stratum);
			applog(LOG_ERR, "...retry after %d seconds", opt_fail_pause);
			sleep(opt_fail_pause);
		}
	}

	/* save every 12 hours */
	if ((time(NULL) - prev_save) > 12*3600) {
		store_scratchpad_to_file(false);
		prev_save = time(NULL);
	}

	if (rpc2_work.job_id && (!g_work_time || strcmp(rpc2_work.job_id, g_work.job_id))) {
		pthread_mutex_lock(&rpc2_work_lock);
		rpc2_stratum_gen_work(&stratum, &g_work);
		g_work_time = time(NULL);
		pthread_mutex_unlock(&rpc2_work_lock);

		if (opt_debug) applog(LOG_DEBUG, "Stratum detected new block");
		restart_threads();
	}

	return 0;
}

void rpc2_init()
{
	memset(&current_scratchpad_hi, 0, sizeof(struct scratchpad_hi));
	memset(&rpc2_work, 0, sizeof(struct work));

	pthread_mutex_init(&rpc2_job_lock, NULL);
	pthread_mutex_init(&rpc2_work_lock, NULL);
	pthread_mutex_init(&rpc2_login_lock, NULL);
	//pthread_mutex_init(&rpc2_getscratchpad_lock, NULL);
}
