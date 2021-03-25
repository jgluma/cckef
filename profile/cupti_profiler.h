/**
 * @file cupti_profiler.h
 * @author José María González Linares (jgl@uma.es)
 * @brief 
 * @version 0.1
 * @date 2021-03-23
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#pragma once

#include <stdio.h>

#include <vector>
#include <map>
#include <string>

#include <cuda.h>
#include <cupti.h>

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                       \
      exit(-1);                                                         \
    }                                                                   \
  } while (0)

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

#ifndef __CUPTI_PROFILER_NAME_SHORT
	#define __CUPTI_PROFILER_NAME_SHORT 128
#endif

// User data for event collection callback
typedef struct MetricData_st {
  // the device where metric is being collected
  CUdevice device;
  // the set of event groups to collect for a pass
  CUpti_EventGroupSet *eventGroups;
  // the current number of events collected in eventIdArray and
  // eventValueArray
  uint32_t eventIdx;
  // the number of entries in eventIdArray and eventValueArray
  uint32_t numEvents;
  // array of event ids
  CUpti_EventID *eventIdArray;
  // array of event values
  uint64_t *eventValueArray;
  // array of instances values
  uint64_t *eventNumInstances;
  uint64_t **eventInstancesArray;  
} MetricData_t;

class CuptiProfiler {

public:
std::vector<std::string> init_cupti_profiler(	const int device_num );

//void CUPTIAPI getMetricValueCallback(void *userdata, CUpti_CallbackDomain domain,
//                       CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo);
//void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
//void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);

void start_kernelduration_cupti_profiler();
uint64_t end_kernelduration_cupti_profiler();
CUpti_EventGroupSets *start_cupti_profiler(	const char *metricName );
void advance_cupti_profiler( CUpti_EventGroupSets *passData, int pass );
void stop_cupti_profiler( bool getvalue );
void free_cupti_profiler();

std::vector<std::string> available_metrics_cupti_profiler(	CUdevice device, bool print_names);

FILE *open_metric_file( const char *name );
void close_metric_file();
bool checkConsistency( unsigned long long expected );
int getMaxIdxEvent( unsigned long long expected );

void print_event_instances();

private:

CUcontext m_context = 0;
CUdevice m_device = 0;
CUpti_SubscriberHandle m_subscriber;
MetricData_t m_metricData;
CUpti_MetricID m_metricId;

CUpti_EventID *m_eventId;
uint64_t *m_numEvents;
uint64_t *m_numInstances;
uint64_t **m_numInstancesArray;
};
