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

    void setPersistentBlocks();

    void setNumElements(unsigned long n);
    unsigned long getNumElements() { return numElements; }
    size_t getSizeN() { return size; }
    void setNumIterations(unsigned long n) { m_numIterations = n; }
    unsigned long getNumIterations() { return m_numIterations; }
    
    void allocDeviceMemory(float *ptr,
                           int *assignsA, int numAssignsA,
                           int *assignsB, int numAssignsB,
                           int *assignsC, int numAssignsC);
    void memHostToDeviceAssigns(void);
    void memDeviceToHostAssigns(void);

    void setModules(int *mA, int *mB, int *mC)
    {
        modulesA = mA;
        modulesB = mB;
        modulesC = mC;
    }
    void setAllModules(int reverse);
    int getNumCombinations() { return m_numCombinations; }
    int *getModulesA() { return modulesA; }
    int *getModulesA(int idx) { return allModulesA[idx]; }
    int *getModulesB() { return modulesB; }
    int *getModulesB(int idx) { return allModulesB[idx]; }
    int *getModulesC() { return modulesC; }
    int *getModulesC(int idx) { return allModulesC[idx]; }

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
    int m_numIterations;

    int numChipAssignmentsA;
    int *chipAssignmentsA;
    int *modulesA;
    int **allModulesA;

    int numChipAssignmentsB;
    int *chipAssignmentsB;
    int *modulesB;
    int **allModulesB;

    int numChipAssignmentsC;
    int *chipAssignmentsC;
    int *modulesC;
    int **allModulesC;

    int m_numCombinations;
};

#endif