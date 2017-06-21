/**
 * Unit to read cpu informations
 *
 * TODO: WMI implementation for windows
 *
 * tpruvot 2014
 */

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "miner.h"

#ifndef WIN32

#define HWMON_PATH \
 "/sys/class/hwmon/hwmon1/device/temp1_input"
#define HWMON_ALT \
 "/sys/class/hwmon/hwmon0/temp1_input"

static double linux_cputemp(int core)
{
	double tc = 0.0;
	FILE *fd = fopen(HWMON_PATH, "r");
	uint32_t val = 0;

	if (!fd)
		fd = fopen(HWMON_ALT, "r");

	if (!fd)
		return tc;

	if (fscanf(fd, "%d", &val))
		tc = (double)val / 1000.0;

	fclose(fd);
	return tc;
}

#define CPUFREQ_PATH \
 "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_cur_freq"
static uint32_t linux_cpufreq(int core)
{
	FILE *fd = fopen(CPUFREQ_PATH, "r");
	uint32_t freq = 0;

	if (!fd)
		return freq;

	if (!fscanf(fd, "%d", &freq))
		fclose(fd);
		return freq;

	fclose(fd);
	return freq;
}

#else /* WIN32 */

static float win32_cputemp(int core)
{
	// todo
	return 0.0;
}

#endif /* !WIN32 */


/* exports */


float cpu_temp(int core)
{
#ifdef WIN32
	return win32_cputemp(core);
#else
	return (float) linux_cputemp(core);
#endif
}

uint32_t cpu_clock(int core)
{
#ifdef WIN32
	return 0;
#else
	return linux_cpufreq(core);
#endif
}

int cpu_fanpercent()
{
	return 0;
}

