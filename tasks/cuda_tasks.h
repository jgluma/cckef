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
#include "../vectorAdd/vectorAdd.h"
#include "../dummy/dummy.h"


using namespace std;

CUDAtask *createCUDATask(CUDAtaskNames t);

#endif
