/**
 * @file cuda_tasks.h
 * @details This file includes routines to implement CUDA tasks.
 * @author José María González Linares.
 * @date 15/12/2021
 */

#ifndef _CUDA_TASKS_H_
#define _CUDA_TASKS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "cuda_task_class.h"
#include "../profile/memory_profiler.h"
#include "memBench/memBench.h"
#include "../vectorAdd/vectorAdd.h"
#include "../dummy/dummy.h"
#include "../matrixMul/matrixMul.h"
#include "../BlackScholes/BlackScholes.h"
#include "../PathFinder/PathFinder.h"

using namespace std;

CUDAtask *createCUDATask(CUDAtaskNames t, int deviceID);
void setupCUDATask(CUDAtask *task, int n, int b);
int getTaskSize(CUDAtask *task);
void sendTaskData(CUDAtask *task, float *d_ptr, MemoryProfiler *memprof, int c);

#endif
