/**
 * @file BlackScholes.h
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
#include <math.h>
#include "tasks/cuda_tasks.h"
#include "memBench/memBench.h"

using namespace std;

/**
 * @brief BlackScholes task
 * 
 */
class BlackScholesTask : public CUDAtask
{
public:
    BlackScholesTask();
    BlackScholesTask(int optN, int numIter);
    ~BlackScholesTask();
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

    int getOptN() { return m_optN; }
    int getNumIter() { return m_numIter; }
    void setOptN(int optN);
    int getSizeN() { return m_optN_SZ; }
    void setNumIter(int numIter) { m_numIter = numIter; }
    void setOptions(int optN, int numIter)
    {
        setOptN(optN);
        setNumIter(numIter);
    }

    void setComputeAssigns(bool b) { m_computeAssigns = b; }
    bool getComputeAssigns() { return m_computeAssigns; }
    void setPersistentBlocks();

    void allocDeviceMemory(float *ptr,
                           int *assignsA, int numAssignsA,
                           int *assignsB, int numAssignsB,
                           int *assignsC, int numAssignsC,
                           int *assignsD, int numAssignsD,
                           int *assignsE, int numAssignsE);
    void memHostToDeviceAssigns(void);
    void memDeviceToHostAssigns(void);

    void setModules(int *mA, int *mB, int *mC, int *mD, int *mE)
    {
        modulesA = mA;
        modulesB = mB;
        modulesC = mC;
        modulesD = mD;
        modulesE = mE;
    }
    void setAllModules(int reverse);
    int getNumCombinations() { return m_numCombinations; }
    int *getModulesA() { return modulesA; }
    int *getModulesA(int idx) { return allModulesA[idx]; }    
    int *getModulesB() { return modulesB; }
    int *getModulesB(int idx) { return allModulesB[idx]; }    
    int *getModulesC() { return modulesC; }
    int *getModulesC(int idx) { return allModulesC[idx]; }    
    int *getModulesD() { return modulesD; }
    int *getModulesD(int idx) { return allModulesD[idx]; }    
    int *getModulesE() { return modulesE; }
    int *getModulesE(int idx) { return allModulesE[idx]; }    

private:
    // Host and device arrays
    float *h_CallResultCPU;
    float *h_PutResultCPU;

    float *h_CallResultGPU;
    float *h_PutResultGPU;
    float *h_StockPrice;
    float *h_OptionStrike;
    float *h_OptionYears;

    float *d_CallResultGPU;
    float *d_PutResultGPU;
    float *d_StockPrice;
    float *d_OptionStrike;
    float *d_OptionYears;

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

    int numChipAssignmentsD;
    int *chipAssignmentsD;
    int *modulesD;
    int **allModulesD;

    int numChipAssignmentsE;
    int *chipAssignmentsE;
    int *modulesE;
    int **allModulesE;

    int m_numCombinations;

    int m_optN;
    int m_numIter;
    int m_optN_SZ;

    bool m_computeAssigns;
};
