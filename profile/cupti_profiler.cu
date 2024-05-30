/**
 * @file cupti_profiler.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief
 * @version 0.1
 * @date 2021-03-18
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#include "cupti_profiler.h"

// Static variables
CUcontext m_context = 0;
CUdevice m_device = 0;
static uint64_t m_kernelDuration;
static bool PRINT_VALUES = false;
static bool MAX_VERBOSITY = false;
static FILE *m_fp = stderr;
static char m_metricName[255];
static char m_eventName[255];
#define SAMPLE_PERIOD_MS 50
#define SAMPLE_PERIOD_US 50000					// With a low sampling period many samples are lost
#define NUM_SAMPLES 100000						// 100000*50000*1e-06 = 5000s -> more than 80 minutes
static unsigned long long samples[NUM_SAMPLES]; // S
static long long t_samples[NUM_SAMPLES];
static unsigned current_sample;
static unsigned long long totalEventVal;
static pthread_t m_pThread;
static key_t shmkey;								   /*      shared memory key       */
static int shmid;									   /*      shared memory id        */
static sem_t *sem; /*      synch semaphore         */ /*shared */
static int *p; /*      shared variable         */	   /*shared */

struct samples_st {
	long long sample;
	long long time;
};

// Static Routines

static inline long long
getTime() /* usec */
{
	//   struct timeval time;
	//   gettimeofday(&time, NULL);
	//   return time.tv_sec * 1000000 + time.tv_usec;
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return now.tv_sec * 1000000000 + now.tv_nsec;
}

static int compare(const void *a, const void *b)
{
	return (*(int *)a - *(int *)b);
}

static int compare_st(const void *a, const void *b)
{
	return ((*(struct samples_st *)a).sample - (*(struct samples_st *)b).sample);
}

static void CUPTIAPI
getMetricValueCallback(void *userdata, CUpti_CallbackDomain domain,
					   CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
	MetricData_t *metricData = (MetricData_t *)userdata;
	unsigned int i, j, k;

	// This callback is enabled only for launch so we shouldn't see anything else.
	if ((cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) &&
		(cbid != CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
	{
		printf("%s:%d: unexpected cbid %d\n", __FILE__, __LINE__, cbid);
		exit(-1);
	}

	// on entry, enable all the event groups being collected this pass,
	// for metrics we collect for all instances of the event

	if (cbInfo->callbackSite == CUPTI_API_ENTER)
	{
		cudaDeviceSynchronize();
		CUPTI_CALL(cuptiSetEventCollectionMode(cbInfo->context, CUPTI_EVENT_COLLECTION_MODE_KERNEL));

		for (i = 0; i < metricData->eventGroups->numEventGroups; i++)
		{
			uint32_t all = 1;
			CUPTI_CALL(cuptiEventGroupSetAttribute(metricData->eventGroups->eventGroups[i],
												   CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
												   sizeof(all), &all));
			CUPTI_CALL(cuptiEventGroupEnable(metricData->eventGroups->eventGroups[i]));
		}
	}

	// on exit, read and record event values
	if (cbInfo->callbackSite == CUPTI_API_EXIT)
	{
		cudaDeviceSynchronize();
		// for each group, read the event values from the group and record in metricData
		for (i = 0; i < metricData->eventGroups->numEventGroups; i++)
		{
			CUpti_EventGroup group = metricData->eventGroups->eventGroups[i];
			CUpti_EventDomainID groupDomain;
			uint32_t numEvents, numInstances, numTotalInstances;
			CUpti_EventID *eventIds;
			size_t groupDomainSize = sizeof(groupDomain);
			size_t numEventsSize = sizeof(numEvents);
			size_t numInstancesSize = sizeof(numInstances);
			size_t numTotalInstancesSize = sizeof(numTotalInstances);
			uint64_t *values, normalized, sum;
			size_t valuesSize, eventIdsSize;

			CUPTI_CALL(cuptiEventGroupGetAttribute(group,
												   CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
												   &groupDomainSize, &groupDomain));
			CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(metricData->device, groupDomain,
														  CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
														  &numTotalInstancesSize, &numTotalInstances));
			CUPTI_CALL(cuptiEventGroupGetAttribute(group,
												   CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
												   &numInstancesSize, &numInstances));
			CUPTI_CALL(cuptiEventGroupGetAttribute(group,
												   CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
												   &numEventsSize, &numEvents));
			eventIdsSize = numEvents * sizeof(CUpti_EventID);
			eventIds = (CUpti_EventID *)malloc(eventIdsSize);
			CUPTI_CALL(cuptiEventGroupGetAttribute(group,
												   CUPTI_EVENT_GROUP_ATTR_EVENTS,
												   &eventIdsSize, eventIds));
			valuesSize = sizeof(uint64_t) * numInstances;
			values = (uint64_t *)malloc(valuesSize);

			for (j = 0; j < numEvents; j++)
			{
				CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
													eventIds[j], &valuesSize, values));
				if (metricData->eventIdx >= metricData->numEvents)
				{
					fprintf(stderr, "error: too many events collected, metric expects only %d\n", (int)metricData->numEvents);
					exit(-1);
				}

				// sum collect event values from all instances
				sum = 0;
				for (k = 0; k < numInstances; k++)
					sum += values[k];

				// normalize the event value to represent the total number of
				// domain instances on the device
				normalized = (sum * numTotalInstances) / numInstances;

				metricData->eventIdArray[metricData->eventIdx] = eventIds[j];
				metricData->eventValueArray[metricData->eventIdx] = normalized;
				metricData->eventNumInstances[metricData->eventIdx] = numInstances;
				metricData->eventInstancesArray[metricData->eventIdx] = (uint64_t *)malloc(numInstances * sizeof(uint64_t));
				for (k = 0; k < numInstances; k++)
					metricData->eventInstancesArray[metricData->eventIdx][k] = values[k];
				metricData->eventIdx++;

				// print collected value
				if (PRINT_VALUES)
				{
					char eventName[128];
					size_t eventNameSize = sizeof(eventName) - 1;
					CUPTI_CALL(cuptiEventGetAttribute(eventIds[j], CUPTI_EVENT_ATTR_NAME,
													  &eventNameSize, eventName));

					eventName[127] = '\0';
					fprintf(m_fp, "%s, %s, %llu, %llu, ", m_metricName, eventName, (unsigned long long)eventIds[j], (unsigned long long)sum);
					fprintf(m_fp, "%u, %u, %llu", numTotalInstances, numInstances, (unsigned long long)normalized);
					for (k = 0; k < numInstances; k++)
					{
						fprintf(m_fp, "%llu, ", (unsigned long long)values[k]);
					}
					fprintf(m_fp, "\n");
				}
			}
			free(values);
		}
		for (i = 0; i < metricData->eventGroups->numEventGroups; i++)
			CUPTI_CALL(cuptiEventGroupDisable(metricData->eventGroups->eventGroups[i]));
	}
}

static void CUPTIAPI
bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
	uint8_t *rawBuffer;

	*size = 16 * 1024;
	rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

	*buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
	*maxNumRecords = 0;

	if (*buffer == NULL)
	{
		printf("Error: out of memory\n");
		exit(-1);
	}
}

static void CUPTIAPI
bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
	CUpti_Activity *record = NULL;
	CUpti_ActivityKernel4 *kernel;

	//since we launched only 1 kernel, we should have only 1 kernel record
	CUPTI_CALL(cuptiActivityGetNextRecord(buffer, validSize, &record));

	kernel = (CUpti_ActivityKernel4 *)record;
	if (kernel->kind != CUPTI_ACTIVITY_KIND_KERNEL)
	{
		fprintf(stderr, "Error: expected kernel activity record, got %d\n", (int)kernel->kind);
		exit(-1);
	}

	m_kernelDuration = kernel->end - kernel->start;
	free(buffer);
}

// CuptiProfiler routines
void CuptiProfiler::init_cupti_sampler(const int device_num)
{
	int deviceCount;
	char deviceName[32];

	// Init CUDA and create context
	DRIVER_API_CALL(cuInit(0));
	DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0)
	{
		printf("There is no device supporting CUDA.\n");
		exit(-1);
	}
	if (device_num >= deviceCount)
	{
		printf("Device %d does not exist. Device count is %d\n", device_num, deviceCount);
		exit(-2);
	}
	DRIVER_API_CALL(cuDeviceGet(&m_device, device_num));
	DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, m_device));
	printf("CUDA Device Name: %s\n", deviceName);
	DRIVER_API_CALL(cuCtxGetCurrent(&m_context));
	if (m_context == 0)
	{
		printf("There is no CUDA context, creating one\n");
		DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
	}

	current_sample = 0;
}

void CuptiProfiler::set_cupti_sampler(const char *eventName)
{
	strcpy(m_eventName, eventName);
}

static void *sampling_func(void *arg)
{
	CUptiResult cuptiErr;
	CUpti_EventGroup eventGroup;
	CUpti_EventID eventId;
	size_t bytesRead, valueSize;
	uint32_t numInstances = 0, j = 0;
	uint64_t *eventValues = NULL, eventVal = 0;
	uint32_t profile_all = 1;

	cuptiErr = cuptiSetEventCollectionMode(m_context,
										   CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiSetEventCollectionMode");

	cuptiErr = cuptiEventGroupCreate(m_context, &eventGroup, 0);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupCreate");

	cuptiErr = cuptiEventGetIdFromName(m_device, m_eventName, &eventId);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGetIdFromName");

	cuptiErr = cuptiEventGroupAddEvent(eventGroup, eventId);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupAddEvent");

	cuptiErr = cuptiEventGroupSetAttribute(eventGroup,
										   CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
										   sizeof(profile_all), &profile_all);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupSetAttribute");

	cuptiErr = cuptiEventGroupEnable(eventGroup);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupEnable");

	valueSize = sizeof(numInstances);
	cuptiErr = cuptiEventGroupGetAttribute(eventGroup,
										   CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
										   &valueSize, &numInstances);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupGetAttribute");

	bytesRead = sizeof(uint64_t) * numInstances;
	eventValues = (uint64_t *)malloc(bytesRead);
	if (eventValues == NULL)
	{
		printf("%s:%d: Failed to allocate memory.\n", __FILE__, __LINE__);
		exit(-1);
	}
	*p = 1;
	while (*p < 2)
	{
		cuptiErr = cuptiEventGroupReadEvent(eventGroup,
											CUPTI_EVENT_READ_FLAG_NONE,
											eventId, &bytesRead, eventValues);
		CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupReadEvent");
		if (bytesRead != (sizeof(uint64_t) * numInstances))
		{
			printf("Failed to read value for \"%s\"\n", m_eventName);
			exit(-1);
		}

		eventVal = 0;
		for (j = 0; j < numInstances; j++)
		{
			eventVal += eventValues[j];
		}
		// printf("%s: %llu\n", m_eventName, (unsigned long long)eventVal);
		if (eventVal > 0 && current_sample < NUM_SAMPLES)
		{
			samples[current_sample] = eventVal;
			t_samples[current_sample] = getTime();
			if (m_fp /*current_sample == 15*/)
			{
				fprintf(m_fp, "%s\t%llu\t%llu\t", m_eventName, t_samples[current_sample], samples[current_sample]);
				for (j = 0; j < numInstances; j++)
				{
					fprintf(m_fp, "%llu\t", eventValues[j]);
				}
				fprintf(m_fp, "\n");
			}
			current_sample++;
			totalEventVal += eventVal;
		}
		usleep(SAMPLE_PERIOD_US);
	}

	cuptiErr = cuptiEventGroupDisable(eventGroup);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDisable");

	cuptiErr = cuptiEventGroupDestroy(eventGroup);
	CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDestroy");

	free(eventValues);
	return NULL;
}

void CuptiProfiler::start_cupti_sampler(const char *eventName)
{

	/* initialize a shared variable in shared memory */
	shmkey = ftok("/dev/null", 5); /* valid directory name and a number */
	shmid = shmget(shmkey, sizeof(int), 0644 | IPC_CREAT);
	if (shmid < 0)
	{ /* shared memory error check */
		perror("shmget\n");
		exit(1);
	}
	p = (int *)shmat(shmid, NULL, 0); /* attach p to shared memory */
	*p = 0;

	/* initialize semaphores for shared processes */
	// sem = sem_open("pSem", O_CREAT | O_EXCL, 0644, 1); // Binary semaphore
													   /* name of semaphore is "pSem", semaphore is reached using this name */

	totalEventVal = 0;
	for (int i = 0; i < current_sample; i++)
		samples[i] = 0;
	current_sample = 0;
	set_cupti_sampler(eventName);
	int status = pthread_create(&m_pThread, NULL, sampling_func, NULL);
	if (status != 0)
	{
		// sem_unlink("pSem");
		// sem_close(sem);
		perror("pthread_create");
		exit(-1);
	}
}

void CuptiProfiler::end_cupti_sampler(long long reftime)
{
	// sleep(1);
	// Barrier
	*p = 2;

	pthread_join(m_pThread, NULL);

	unsigned long long max = 0;
	struct samples_st *tmp_samples;
	tmp_samples = (struct samples_st *)malloc(current_sample * sizeof(struct samples_st));
	tmp_samples[0].sample = samples[0];
	tmp_samples[0].time = SAMPLE_PERIOD_US * 1000;
	for (int i = 1; i < current_sample; i++)
	{
		if (samples[i] > max)
			max = samples[i];
		tmp_samples[i].sample = samples[i];
		tmp_samples[i].time = t_samples[i] - t_samples[i - 1];
	}
	qsort(tmp_samples, current_sample, sizeof(struct samples_st), compare_st);
	// printf("%f : %llu - max %llu\n", tmp_t[current_sample / 2] * 0.000000001, tmp_s[current_sample / 2], max);
	// printf("Total : %llu\n", totalEventVal);
	printf("%llu %f ", tmp_samples[current_sample / 2].sample, 32*tmp_samples[current_sample / 2].sample / (tmp_samples[current_sample / 2].time * 0.000000001) / 1e09);
	fflush(stdout);
	/* shared memory detach */
	shmdt(p);
	shmctl(shmid, IPC_RMID, 0);

	/* cleanup semaphores */
	sem_unlink("pSem");
	sem_close(sem);
}

std::vector<std::string>
CuptiProfiler::init_cupti_profiler(const int device_num)
{
	int deviceCount;
	char deviceName[32];

	// Make sure activity is enabled before any CUDA API
	CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));

	// Init CUDA and create context
	DRIVER_API_CALL(cuInit(0));
	DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
	if (deviceCount == 0)
	{
		printf("There is no device supporting CUDA.\n");
		exit(-1);
	}
	if (device_num >= deviceCount)
	{
		printf("Device %d does not exist. Device count is %d\n", device_num, deviceCount);
		exit(-2);
	}
	DRIVER_API_CALL(cuDeviceGet(&m_device, device_num));
	DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, m_device));
	// printf("CUDA Device Name: %s\n", deviceName);
	//	DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
	DRIVER_API_CALL(cuCtxGetCurrent(&m_context));
	if (m_context == 0)
	{
		printf("There is no CUDA context, creating one\n");
		DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
	}
	m_kernelDuration = 0;

	return available_metrics_cupti_profiler(m_device, false);
}

void CuptiProfiler::start_kernelduration_cupti_profiler()
{
	CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
}

uint64_t
CuptiProfiler::end_kernelduration_cupti_profiler()
{
	cudaDeviceSynchronize();
	CUPTI_CALL(cuptiActivityFlushAll(0));
	return m_kernelDuration;
}

CUpti_EventGroupSets *
CuptiProfiler::start_cupti_profiler(const char *metricName)
{
	CUpti_EventGroupSets *passData;

	sprintf(m_metricName, "%s", metricName);

	// setup launch callback for event collection
	CUPTI_CALL(cuptiSubscribe(&m_subscriber, (CUpti_CallbackFunc)getMetricValueCallback, &m_metricData));
	CUPTI_CALL(cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
								   CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020));
	CUPTI_CALL(cuptiEnableCallback(1, m_subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
								   CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

	// allocate space to hold all the events needed for the metric
	CUPTI_CALL(cuptiMetricGetIdFromName(m_device, metricName, &m_metricId));
	CUPTI_CALL(cuptiMetricGetNumEvents(m_metricId, &m_metricData.numEvents));

	m_metricData.device = m_device;
	m_eventId = (CUpti_EventID *)malloc(m_metricData.numEvents * sizeof(CUpti_EventID));
	m_metricData.eventIdArray = m_eventId;
	m_numEvents = (uint64_t *)malloc(m_metricData.numEvents * sizeof(uint64_t));
	m_metricData.eventValueArray = m_numEvents;
	m_numInstances = (uint64_t *)malloc(m_metricData.numEvents * sizeof(uint64_t));
	m_metricData.eventNumInstances = m_numInstances;
	m_numInstancesArray = (uint64_t **)malloc(m_metricData.numEvents * sizeof(uint64_t *));
	m_metricData.eventInstancesArray = m_numInstancesArray;
	m_metricData.eventIdx = 0;

	// get the number of passes required to collect all the events
	// needed for the metric and the event groups for each pass
	CUPTI_CALL(cuptiMetricCreateEventGroupSets(m_context, sizeof(m_metricId), &m_metricId, &passData));

	return passData;
}

void CuptiProfiler::advance_cupti_profiler(CUpti_EventGroupSets *passData, int pass)
{
	m_metricData.eventGroups = passData->sets + pass;
}

void CuptiProfiler::stop_cupti_profiler(bool getvalue)
{
	CUpti_MetricValue metricValue;
	//	printf("Kernel duration %llu\n", (unsigned long long) m_kernelDuration);

	// use all the collected events to calculate the metric value
	if (getvalue)
	{
		CUPTI_CALL(cuptiMetricGetValue(m_device, m_metricId,
									   m_metricData.numEvents * sizeof(CUpti_EventID),
									   m_metricData.eventIdArray,
									   m_metricData.numEvents * sizeof(uint64_t),
									   m_metricData.eventValueArray,
									   m_kernelDuration, &metricValue));

		// print metric value, we format based on the value kind
		{
			CUpti_MetricValueKind valueKind;
			size_t valueKindSize = sizeof(valueKind);
			CUPTI_CALL(cuptiMetricGetAttribute(m_metricId, CUPTI_METRIC_ATTR_VALUE_KIND,
											   &valueKindSize, &valueKind));
			switch (valueKind)
			{
			case CUPTI_METRIC_VALUE_KIND_DOUBLE:
				// printf("Metric %f\n", metricValue.metricValueDouble);
				fprintf(m_fp, "%s, %f\n", m_metricName, metricValue.metricValueDouble);
				break;
			case CUPTI_METRIC_VALUE_KIND_UINT64:
				// printf("Metric %llu\n", (unsigned long long)metricValue.metricValueUint64);
				fprintf(m_fp, "%s, %llu\n", m_metricName, (unsigned long long)metricValue.metricValueUint64);
				break;
			case CUPTI_METRIC_VALUE_KIND_INT64:
				// printf("Metric %lld\n", (long long)metricValue.metricValueInt64);
				fprintf(m_fp, "%s, %lld\n", m_metricName, (long long)metricValue.metricValueInt64);
				break;
			case CUPTI_METRIC_VALUE_KIND_PERCENT:
				// printf("Metric %f%%\n", metricValue.metricValuePercent);
				fprintf(m_fp, "%s, %f\n", m_metricName, metricValue.metricValuePercent);
				break;
			case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
				fprintf(m_fp, "%s, %llu\n", m_metricName, (unsigned long long)metricValue.metricValueThroughput);
				break;
			case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
				// printf("Metric utilization level %u\n", (unsigned int)metricValue.metricValueUtilizationLevel);
				fprintf(m_fp, "%s, %u\n", m_metricName, (unsigned int)metricValue.metricValueUtilizationLevel);
				break;
			default:
				fprintf(stderr, "error: unknown value kind\n");
				exit(-1);
			}
		}
	}
}

void CuptiProfiler::unsubscribe_cupti_profiler()
{
	// Unsubscribe and free data
	CUPTI_CALL(cuptiUnsubscribe(m_subscriber));
	//CUPTI_CALL(cuptiEventGroupSetsDestroy(m_metric_pass_data));
	free(m_eventId);
	free(m_numEvents);
}

void CuptiProfiler::free_cupti_profiler()
{
	// Unsubscribe and free data
	CUPTI_CALL(cuptiUnsubscribe(m_subscriber));
	//CUPTI_CALL(cuptiEventGroupSetsDestroy(m_metric_pass_data));
	free(m_eventId);
	free(m_numEvents);
	free(m_numInstances);
	for (int i = 0; i < m_metricData.numEvents; i++)
		free(m_numInstancesArray[i]);
	free(m_numInstancesArray);
}

std::vector<std::string>
CuptiProfiler::available_metrics_cupti_profiler(CUdevice device,
												bool print_names)
{
	std::vector<std::string> metric_names;
	uint32_t numMetric;
	size_t size;
	char metricName[__CUPTI_PROFILER_NAME_SHORT];
	CUpti_MetricValueKind metricKind;
	CUpti_MetricID *metricIdArray;

	CUPTI_CALL(cuptiDeviceGetNumMetrics(device, &numMetric));

	size = sizeof(CUpti_MetricID) * numMetric;
	metricIdArray = (CUpti_MetricID *)malloc(size);
	if (NULL == metricIdArray)
	{
		printf("Memory could not be allocated for metric array");
		exit(-1);
	}

	CUPTI_CALL(cuptiDeviceEnumMetrics(device, &size, metricIdArray));

	if (print_names)
		printf("%d available metrics:\n", numMetric);

	for (int i = 0; i < numMetric; i++)
	{
		size = __CUPTI_PROFILER_NAME_SHORT;
		CUPTI_CALL(cuptiMetricGetAttribute(metricIdArray[i], CUPTI_METRIC_ATTR_NAME,
										   &size, (void *)&metricName));
		size = sizeof(CUpti_MetricValueKind);
		CUPTI_CALL(cuptiMetricGetAttribute(metricIdArray[i], CUPTI_METRIC_ATTR_VALUE_KIND,
										   &size, (void *)&metricKind));

		if ((metricKind == CUPTI_METRIC_VALUE_KIND_THROUGHPUT) || (metricKind == CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL))
		{
			if (print_names && MAX_VERBOSITY)
				printf("Metric %s cannot be profiled as metric requires GPU time duration for kernel run.\n", metricName);
		}
		else
		{
			metric_names.push_back(metricName);
			if (print_names)
				if (i > 0)
					printf(", %s", metricName);
				else
					printf("%s", metricName);
		}
	}

	if (print_names)
		printf("\n %lu metrics will be profiled\n", metric_names.size());

	free(metricIdArray);
	return (metric_names);
}

FILE *
CuptiProfiler::open_metric_file(const char *name)
{
	m_fp = fopen(name, "a");
	return m_fp;
}

void CuptiProfiler::close_metric_file()
{
	fclose(m_fp);
}

bool CuptiProfiler::checkConsistency(unsigned long long expected)
{
	int numMax = 0;

	if (m_metricData.numEvents != 2)
		return false;

	if (m_metricData.eventNumInstances[0] != m_metricData.eventNumInstances[1])
		return false;

	for (int k = 0; k < m_metricData.eventNumInstances[0]; k++)
	{
		unsigned long long sum = m_metricData.eventInstancesArray[0][k] + m_metricData.eventInstancesArray[1][k];
		if (sum > expected)
			numMax++;
	}
	if (numMax > 1)
	{
		printf("%d values greater than %llu found\n", numMax, expected);
		for (int k = 0; k < m_metricData.eventNumInstances[0]; k++)
		{
			printf("%d - %d\n", m_metricData.eventInstancesArray[0][k], m_metricData.eventInstancesArray[1][k]);
		}
		return false;
	}
	return true;
}

int CuptiProfiler::getMaxIdxEvent(unsigned long long expected)
{

	FILE *fp = fopen("assigns.txt", "a");
	for (int k = 0; k < m_metricData.eventNumInstances[0]; k++)
		fprintf(fp, "%d\t", m_metricData.eventInstancesArray[0][k] + m_metricData.eventInstancesArray[1][k]);
	fprintf(fp, "\n");
	fclose(fp);

	for (int k = 0; k < m_metricData.eventNumInstances[0]; k++)
	{
		unsigned long long sum = m_metricData.eventInstancesArray[0][k] + m_metricData.eventInstancesArray[1][k];
		if (sum > expected)
			return k;
	}
	return -1;
}

void CuptiProfiler::print_event_instances()
{

	char eventName[128];
	size_t eventNameSize = sizeof(eventName) - 1;
	// printf("There are %d events\n", m_metricData.numEvents);
	for (int j = 0; j < m_metricData.numEvents; j++)
	{
		CUPTI_CALL(cuptiEventGetAttribute(m_metricData.eventIdArray[j], CUPTI_EVENT_ATTR_NAME, &eventNameSize, eventName));
		eventName[127] = '\0';
		printf("%s, id %llu, val %llu, inst %llu: ",
			   eventName, (unsigned long long)m_metricData.eventIdArray[j],
			   (unsigned long long)m_metricData.eventValueArray[j],
			   (unsigned long long)m_metricData.eventNumInstances[j]);
		for (int k = 0; k < m_metricData.eventNumInstances[j]; k++)
			printf("%llu, ", (unsigned long long)m_metricData.eventInstancesArray[j][k]);
		printf("\n");
	}
}