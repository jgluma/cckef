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
#include "../memBench/memBench.h"

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
    unsigned long getNumElements() { return m_numElements; }
    void setSizeA() { m_sizeA = m_numElements * sizeof(float); }
    void setSizeB() { m_sizeB = m_numElements * sizeof(float); }
    unsigned long getSizeA() { return m_sizeA; }
    unsigned long getSizeB() { return m_sizeB; }

    void setNumIterations(unsigned long n) { m_numIterations = n; }
    unsigned long getNumIterations() { return m_numIterations; }

    void setCB(bool b) { m_launchCB = b; }
    void setMB(bool b) { m_launchMB = b; }
    void setMBR(bool b) { m_launchMBR = b; }

    void setComputeAssigns(bool b) { m_computeAssigns = b; }
    bool getComputeAssigns() { return m_computeAssigns; }

    void setPersistentBlocks();

    void setPartitions(int pA, int pB)
    {
        partitionA = pA;
        partitionB = pB;
    }

    void allocDeviceMemory(float *ptr, int *assignsA, int numAssignsA, int *assignsB, int numAssignsB);
    void memHostToDeviceAssigns(void);
    void memDeviceToHostAssigns(void);

    void setModules(int *mA, int *mB)
    {
        modulesA = mA;
        modulesB = mB;
    }
    void setAllModules(int reverse);
    int getNumCombinations() { return m_numCombinations; }
    int *getModulesA() { return modulesA; }
    int *getModulesA(int idx) { return allModulesA[idx]; }
    int *getModulesB() { return modulesB; }
    int *getModulesB(int idx) { return allModulesB[idx]; }

private:
    // Host and device arrays
    float *h_A;
    float *h_B;

    float *d_A;
    float *d_B;

    int numChipAssignmentsA;
    int *chipAssignmentsA;
    int *modulesA;
    int **allModulesA;
    int numChipAssignmentsB;
    int *chipAssignmentsB;
    int *modulesB;
    int **allModulesB;
    int m_numCombinations;

    int partitionA, partitionB;

    unsigned long m_numElements;
    unsigned long m_numIterations;
    unsigned long m_sizeA, m_sizeB;

    bool m_launchCB;
    bool m_launchMB;
    bool m_launchMBR;
    bool m_computeAssigns;
};
