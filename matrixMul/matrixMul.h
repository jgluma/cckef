/**
 * @file matrixMul.h
 * @author José María González Linares (jgl@uma.es)
 * @brief 
 * @version 0.1
 * @date 2021-05-12
 * 
 */
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "tasks/cuda_tasks.h"

using namespace std;

/**
 * @brief matrixMul task
 * 
 */
class matrixMulTask : public CUDAtask
{
public:
    matrixMulTask();
    matrixMulTask(dim3 &dA, dim3 &dB);
    ~matrixMulTask();
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

    void setMatrixDims(dim3 &dimsA, dim3 &dimsB);
    dim3 getADims() { return m_dimsA; }
    dim3 getBDims() { return m_dimsB; }
    void setSizeA() { m_size_A = m_dimsA.x * m_dimsA.y * sizeof(float); }
    void setSizeB() { m_size_B = m_dimsB.x * m_dimsB.y * sizeof(float); }
    void setSizeC() { m_size_C = m_dimsC.x * m_dimsC.y * sizeof(float); }
    unsigned int getSizeA() { return m_size_A; }
    unsigned int getSizeB() { return m_size_B; }
    unsigned int getSizeC() { return m_size_C; }

    void allocDeviceMemory(float *ptr, int *assignsA, int numAssignsA, int *assignsB, int numAssignsB, int *assignsC, int numAssignsC);
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

    int m_block_size;
    dim3 m_dimsA, m_dimsB, m_dimsC;
    unsigned int m_size_A, m_size_B, m_size_C;

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
