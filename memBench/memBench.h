/**
 * @file memBench.h
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
#include "profile/cupti_profiler.h"

using namespace std;

/**
 * @brief Memory Benchmarking task
 * 
 */
class memBenchTask : public CUDAtask
{
public:

memBenchTask();
memBenchTask(unsigned long n, bool p);
~memBenchTask();
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
bool checkResults(void) {return true;}

void setPersistentBlocks(){};
void setNumElements(unsigned long n);
unsigned long getNumElements() {return numElements;}
void setOffset(unsigned long o) {offset=o;}

void assignDeviceMemory(float *ptr) { d_A = ptr;
    m_allocated = false;
}

void setAllModules(int reverse);

private:

// Host and device arrays
float *h_A;
float *d_A;

unsigned long offset;
unsigned long numElements;
size_t size;
bool m_allocated;
};

class memBench
{
    public:
    memBench();
    ~memBench();

    void init( int deviceID );
    void init( int deviceID, size_t ms );
    void init( int deviceID, size_t ms, size_t mr );
    void init(int d, float *ptr, unsigned long n);

    size_t getMaxMR() { return maxMR; }
    void setMaxMR(size_t m) { maxMR = m; }

    size_t getMemorySize() { return memorySize; }
    void setMemorySize(size_t m) { memorySize = m; }
    
    unsigned long long getNumAssignments() { return numAssignments; }
    void setNumAssignments(unsigned long long m) { numAssignments = m; }
    
    int getNumChips() { return numChips; }
    int *returnNumChipAssignments() { return numChipAssignments; }
    int *returnChipMR() { return chipMR; }

    void getChipAssignments();
    void getMemoryRanges();

    void writeAssignments();
    void readAssignments();

    int *getAssignmentsIdx(size_t bytes, int *modules, int *num_indices);

private:
    memBenchTask *t; // Task to profile memory    
    int deviceId; // CUDA device
    int numChips; // Number of memory chips
    
    vector<string> event_names{};
    vector<string> metric_names{"l2_read_transactions"};
    CuptiProfiler profiler;

    size_t memorySize; // Total global memory size in bytes
    size_t maxMR; // Maximum continuous range in a chip
    unsigned long long numAssignments; // Number of assignments to be computed
    int *chipAssignments; // Memory chips assignments
    int *numChipAssignments; // Number of memory ranges in each chip
    int offsetMR; // Max number of memory ranges in each chip
    int *chipMR; // Memory ranges in each chip
    FILE *fpA, *fpM;
};