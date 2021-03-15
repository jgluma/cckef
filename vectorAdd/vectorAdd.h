/**
 * @file vectorAdd.h
 * @author José María González Linares (jgl@uma.es)
 * @brief 
 * @version 0.1
 * @date 2021-02-15
 * 
 */
#ifndef _VECTORADD_H_
#define _VECTORADD_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "tasks/cuda_tasks.h"

using namespace std;

/**
 * @brief Vector addition task
 * 
 */
class vectorAddTask : public CUDAtask
{
public:

vectorAddTask();
vectorAddTask(unsigned long n, bool p);
~vectorAddTask();
CUDAtaskNames getName();

void allocHostMemory(void);
void freeHostMemory(void);
void allocDeviceMemory(void);
void freeDeviceMemory(void);
void dataGeneration(void);
void memHostToDeviceAsync(cudaStream_t stream);
void memHostToDevice(void);
void memDeviceToHostAsync(cudaStream_t stream);
void memDeviceToHost(void);
void launchKernelAsync(cudaStream_t stream);
void launchKernel(void);
bool checkResults(void);

void setNumElements(unsigned long n);
unsigned long getNumElements() {return numElements;};

private:

// Host and device arrays
float *h_A;
float *h_B;
float *h_C;

float *d_A;
float *d_B;
float *d_C;

unsigned long numElements;
size_t size;


};

#endif