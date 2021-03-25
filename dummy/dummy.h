/**
 * @file dummy.h
 * @author José María González Linares (jgl@uma.es)
 * @brief 
 * @version 0.1
 * @date 2021-02-15
 * 
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "tasks/cuda_tasks.h"

using namespace std;

/**
 * @brief Dummy task
 * 
 */
class dummyTask : public CUDAtask
{
public:

dummyTask();
dummyTask(unsigned long nI, bool cb, unsigned long nE, bool mb);
~dummyTask();
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
unsigned long getNumElements() {return m_numElements;}

void setNumIter(unsigned long n) { m_numIter = n; }
unsigned long getNumIter() {return m_numIter;}

void setCB(bool b) { m_launchCB = b; }
void setMB(bool b) { m_launchMB = b; }

private:

// Host and device arrays
float *h_A;
float *h_B;

float *d_A;
float *d_B;

unsigned long m_numElements;
unsigned long m_numIter;

bool m_launchCB;
bool m_launchMB;
};
