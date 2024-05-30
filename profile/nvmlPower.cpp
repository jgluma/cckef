#include "nvmlPower.hpp"

/*
These may be encompassed in a class if desired. Trivial CUDA programs written for the purpose of benchmarking might prefer this approach.
*/
bool pollThreadStatus = false;
unsigned int deviceCount = 0;
char deviceNameStr[64];

nvmlReturn_t nvmlResult;
nvmlDevice_t nvmlDeviceID;
nvmlPciInfo_t nvmPCIInfo;
nvmlEnableState_t pmmode;
nvmlComputeMode_t computeMode;

pthread_t powerPollThread;

/*
Poll the GPU using nvml APIs.
*/
void *powerPollingFunc(void *ptr)
{

	unsigned int powerLevel = 0;
	FILE *fp = fopen("Power_data.txt", "w+");
	struct timespec now;
	double curtime, starttime;
	clock_gettime(CLOCK_MONOTONIC, &now);
	starttime = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;

	while (pollThreadStatus)
	{
		pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);

		// Get the power management mode of the GPU.
		nvmlResult = nvmlDeviceGetPowerManagementMode(nvmlDeviceID, &pmmode);

		// The following function may be utilized to handle errors as needed.
		getNVMLError(nvmlResult);

		// Check if power management mode is enabled.
		//	if (pmmode == NVML_FEATURE_ENABLED)
		//	{
		// Get the power usage in milliWatts.
		nvmlResult = nvmlDeviceGetPowerUsage(nvmlDeviceID, &powerLevel);
		// }
		// else {
		// 	fprintf(fp, "power not enabled\n");
		// }

		// The output file stores power in Watts.
		clock_gettime(CLOCK_MONOTONIC, &now);
		curtime = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
		fprintf(fp, "%lu\t%lu\t%e\t%.3lf\n", now.tv_sec, now.tv_nsec, (curtime - starttime) * 1000, (powerLevel) / 1000.0);
		pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
	}

	fclose(fp);
	pthread_exit(0);
}

/*
Start power measurement by spawning a pthread that polls the GPU.
Function needs to be modified as per usage to handle errors as seen fit.
*/
void nvmlAPIRun()
{
	int i;

	// Initialize nvml.
	nvmlResult = nvmlInit();
	if (NVML_SUCCESS != nvmlResult)
	{
		printf("NVML Init fail: %s\n", nvmlErrorString(nvmlResult));
		exit(0);
	}

	// Count the number of GPUs available.
	nvmlResult = nvmlDeviceGetCount(&deviceCount);
	if (NVML_SUCCESS != nvmlResult)
	{
		printf("Failed to query device count: %s\n", nvmlErrorString(nvmlResult));
		exit(0);
	}

	for (i = 0; i < deviceCount; i++)
	{
		// Get the device ID.
		nvmlResult = nvmlDeviceGetHandleByIndex(i, &nvmlDeviceID);
		if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get handle for device %d: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}

		// Get the name of the device.
		nvmlResult = nvmlDeviceGetName(nvmlDeviceID, deviceNameStr, sizeof(deviceNameStr) / sizeof(deviceNameStr[0]));
		if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get name of device %d: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}

		// Get PCI information of the device.
		nvmlResult = nvmlDeviceGetPciInfo(nvmlDeviceID, &nvmPCIInfo);
		if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get PCI info of device %d: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}

		// Get the compute mode of the device which indicates CUDA capabilities.
		nvmlResult = nvmlDeviceGetComputeMode(nvmlDeviceID, &computeMode);
		if (NVML_ERROR_NOT_SUPPORTED == nvmlResult)
		{
			printf("This is not a CUDA-capable device.\n");
		}
		else if (NVML_SUCCESS != nvmlResult)
		{
			printf("Failed to get compute mode for device %i: %s\n", i, nvmlErrorString(nvmlResult));
			exit(0);
		}
	}

	// This statement assumes that the first indexed GPU will be used.
	// If there are multiple GPUs that can be used by the system, this needs to be done with care.
	// Test thoroughly and ensure the correct device ID is being used.
	nvmlResult = nvmlDeviceGetHandleByIndex(0, &nvmlDeviceID);

	pollThreadStatus = true;

	const char *message = "Test";
	int iret = pthread_create(&powerPollThread, NULL, powerPollingFunc, (void *)message);
	if (iret)
	{
		fprintf(stderr, "Error - pthread_create() return code: %d\n", iret);
		exit(0);
	}
}

/*
End power measurement. This ends the polling thread.
*/
void nvmlAPIEnd()
{
	pollThreadStatus = false;
	pthread_join(powerPollThread, NULL);

	nvmlResult = nvmlShutdown();
	if (NVML_SUCCESS != nvmlResult)
	{
		printf("Failed to shut down NVML: %s\n", nvmlErrorString(nvmlResult));
		exit(0);
	}
}

/*
Return a number with a specific meaning. This number needs to be interpreted and handled appropriately.
*/
int getNVMLError(nvmlReturn_t resultToCheck)
{
	if (resultToCheck == NVML_ERROR_UNINITIALIZED)
		return 1;
	if (resultToCheck == NVML_ERROR_INVALID_ARGUMENT)
		return 2;
	if (resultToCheck == NVML_ERROR_NOT_SUPPORTED)
		return 3;
	if (resultToCheck == NVML_ERROR_NO_PERMISSION)
		return 4;
	if (resultToCheck == NVML_ERROR_ALREADY_INITIALIZED)
		return 5;
	if (resultToCheck == NVML_ERROR_NOT_FOUND)
		return 6;
	if (resultToCheck == NVML_ERROR_INSUFFICIENT_SIZE)
		return 7;
	if (resultToCheck == NVML_ERROR_INSUFFICIENT_POWER)
		return 8;
	if (resultToCheck == NVML_ERROR_DRIVER_NOT_LOADED)
		return 9;
	if (resultToCheck == NVML_ERROR_TIMEOUT)
		return 10;
	if (resultToCheck == NVML_ERROR_IRQ_ISSUE)
		return 11;
	if (resultToCheck == NVML_ERROR_LIBRARY_NOT_FOUND)
		return 12;
	if (resultToCheck == NVML_ERROR_FUNCTION_NOT_FOUND)
		return 13;
	if (resultToCheck == NVML_ERROR_CORRUPTED_INFOROM)
		return 14;
	if (resultToCheck == NVML_ERROR_GPU_IS_LOST)
		return 15;
	if (resultToCheck == NVML_ERROR_UNKNOWN)
		return 16;

	return 0;
}

// K20power11

#define DEVICE 0 /* may need to be changed */

#define NEAR_IDLE_DELTA 500 /* mW */
#define IDLE_DELTA 250		/* mW */
// #define SAMPLE_DELAY 14000  /* usec */
// #define RAMP_DELAY 4000000  /* usec */
// #define TIME_OUT 30000000  /* usec */
#define STABLE_COUNT 5 /* sec */

#define SAMPLE_DELAY 14000000	 /* nsec */
#define RAMP_DELAY 4000000000	 /* nsec */
#define TIME_OUT 30000000000	 /* nsec */
#define time2seconds 0.000000001 /* nsec -> sec */

#define power2watts 0.001 /* mW -> W */
// #define time2seconds 0.000001  /* usec -> sec */
#define capacitance 840000.0 /* usec */
// #define ACTIVE_IDLE 55  /* W */
#define ACTIVE_IDLE 45		  /* W */
#define SAMPLES (1024 * 1024) /* 4.3 hours */

static int samples = 0;
static int p_sample[SAMPLES];		/* power */
static long long t_sample[SAMPLES]; /* time */
static double truepower[SAMPLES];	/* true power */
static double max_power;			/* power cap in W */
static FILE *fPowerData;

static nvmlDevice_t initAndTest()
{
	nvmlReturn_t result;
	nvmlDevice_t device;
	int power;

	result = nvmlInit();
	if (NVML_SUCCESS != result)
	{
		printf("failed to initialize NVML: %s\n", nvmlErrorString(result));
		exit(-1);
	}

	result = nvmlDeviceGetHandleByIndex(DEVICE, &device);
	if (NVML_SUCCESS != result)
	{
		printf("failed to get handle for device: %s\n", nvmlErrorString(result));
		exit(-1);
	}

	result = nvmlDeviceGetPowerUsage(device, (unsigned int *)&power);
	if (NVML_SUCCESS != result)
	{
		printf("failed to read power: %s\n", nvmlErrorString(result));
		exit(-1);
	}

	result = nvmlDeviceGetPowerManagementLimit(device, (unsigned int *)&power);
	if (NVML_SUCCESS != result)
	{
		printf("failed to read power limit: %s\n", nvmlErrorString(result));
		exit(-1);
	}
	max_power = power * power2watts;

	printf("Max power %f W\n", max_power);
	return device;
}

static inline long long getTime() /* usec */
{
	//   struct timeval time;
	//   gettimeofday(&time, NULL);
	//   return time.tv_sec * 1000000 + time.tv_usec;
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return now.tv_sec * 1000000000 + now.tv_nsec;
}

static void getSample(nvmlDevice_t device, int *power, long long *time) /* mW usec */
{
	nvmlReturn_t result;
	int samplepower;
	static long long sampletime = LONG_LONG_MIN;

	sampletime += SAMPLE_DELAY;
	do
	{
	} while (getTime() < sampletime);
	result = nvmlDeviceGetPowerUsage(device, (unsigned int *)&samplepower);
	sampletime = getTime();

	if (NVML_SUCCESS != result)
	{
		printf("failed to read power: %s\n", nvmlErrorString(result));
		exit(-1);
	}

	p_sample[samples] = samplepower;
	t_sample[samples] = sampletime;
	samples++;
	if (samples >= SAMPLES)
	{
		printf("out of memory for storing samples\n");
		exit(-1);
	}

	if (samples >= 3)
	{
		int s = samples - 2;
		double tp = (p_sample[s] + capacitance * (p_sample[s + 1] - p_sample[s - 1]) / (t_sample[s + 1] - t_sample[s - 1])) * power2watts;
		if (tp < 0.0)
			tp = 0.0;
		if (tp > max_power)
			tp = max_power;
		truepower[s] = tp;
	}

	*power = samplepower;
	*time = sampletime;
}

nvmlDevice_t device;
int power, nearidlepower, active_samples;
long long cur_time, timeout;
double activetime, activeenergy, mindt;

void nvmlPowerInit()
{
	int i, count;

	int prevpower, diff;
	long long endtime;
	char filename[1100];

	sprintf(filename, "power.txt");
	fPowerData = fopen(filename, "wt");

	device = initAndTest();

	getSample(device, &power, &cur_time);
	timeout = cur_time + TIME_OUT;
	count = 0;
	do
	{
		prevpower = power;
		sleep(1);
		getSample(device, &power, &cur_time);
		count++;
		diff = power - prevpower;
		if (diff < 0)
			diff = -diff;
		if (diff >= IDLE_DELTA)
			count = 0;
		//   printf("%lld (%lld): %f W\n", cur_time, timeout - cur_time, power * power2watts);
	} while ((count < STABLE_COUNT) && (cur_time < timeout));

	if (cur_time >= timeout)
	{
		printf("timed out waiting for idle power to stabilize\n");
		exit(-1);
	}

	samples = 0;
	getSample(device, &power, &cur_time);
	endtime = cur_time + RAMP_DELAY;
	do
	{
		getSample(device, &power, &cur_time);
		//   printf("%lld (%lld): %f W\n", cur_time, endtime - cur_time, power * power2watts);
	} while (cur_time < endtime);
	nearidlepower = power + NEAR_IDLE_DELTA;
	printf("Idle power %f: %f W\n", (cur_time - t_sample[1]) * time2seconds, nearidlepower * power2watts);

	pollThreadStatus = true;

	const char *message = "Test";
	int iret = pthread_create(&powerPollThread, NULL, nvmlPowerStart, (void *)message);
	if (iret)
	{
		fprintf(stderr, "Error - pthread_create() return code: %d\n", iret);
		exit(0);
	}
}

void *nvmlPowerStart(void *ptr)
{
	while (pollThreadStatus)
	{
		pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);

		//   getSample(device, &power, &cur_time);
		//   timeout = cur_time + TIME_OUT;
		//   do {
		getSample(device, &power, &cur_time);
		//     if (power > nearidlepower) {
		//       timeout = cur_time + TIME_OUT;
		//     }
		//   } while (cur_time < timeout);

		//   getSample(device, &power, &cur_time);
		//   getSample(device, &power, &cur_time);
		pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
	}

	pthread_exit(0);
}

void nvmlPowerEnd()
{
	pollThreadStatus = false;
	pthread_join(powerPollThread, NULL);

	// Wait for idle power
	getSample(device, &power, &cur_time);
	timeout = cur_time + 50000000;
	do
	{
		getSample(device, &power, &cur_time);
		// if (power > nearidlepower)
		// {
		// 	timeout = cur_time + 50000000;
		// }
	} while (cur_time < timeout);

	samples--;
	active_samples = 0;
	activetime = 0.0;
	activeenergy = 0.0;
	mindt = TIME_OUT;
	int cnt = 0;
	for (int i = 3; i < samples; i++)
	{
		if (truepower[i] > ACTIVE_IDLE)
		{
			active_samples++;
			double dt = (t_sample[i] - t_sample[i - 1]) * time2seconds;
			if (mindt > dt)
				mindt = dt;
			activetime += dt;
			activeenergy += dt * truepower[i];
		}
		else
			cnt++;
	}
	printf("Discarded %d samples out of %d\n", cnt, samples);

	fprintf(fPowerData, "%.4f\t#active time [s]\n", activetime);
	fprintf(fPowerData, "%.4f\t#active energy [J]\n", activeenergy);

	fprintf(fPowerData, "\ntime [s]\tpower [W]\ttrue power [W]\n");
	for (int i = 1; i < samples; i++)
	{
		fprintf(fPowerData, "%.6f\t%.3f\t%.3f\n", (t_sample[i] - t_sample[1]) * time2seconds, p_sample[i] * power2watts, truepower[i]);
	}
	fclose(fPowerData);

	nvmlShutdown();
}

long long getRefTime() { return t_sample[1]; }
void setRefTime(long long ref) {
	t_sample[0] = ref;
	p_sample[0] = ACTIVE_IDLE;
	truepower[0] = ACTIVE_IDLE;
	t_sample[1] = ref;
	p_sample[1] = ACTIVE_IDLE;
	truepower[1] = ACTIVE_IDLE;
	samples=2;
}