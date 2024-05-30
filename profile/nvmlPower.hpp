/*
Header file including necessary nvml headers.
*/

#ifndef INCLNVML
#define INCLNVML

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <nvml.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <time.h>
#include <unistd.h>

void nvmlAPIRun();
void nvmlAPIEnd();
void *powerPollingFunc(void *ptr);
int getNVMLError(nvmlReturn_t resultToCheck);

void nvmlPowerInit();
void *nvmlPowerStart(void *ptr);
void nvmlPowerEnd();

long long getRefTime();
void setRefTime(long long ref);

#endif