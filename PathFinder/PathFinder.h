/**
 * @file PathFinder.h
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
 * @brief PathFinder task
 * 
 */
class PathFinderTask : public CUDAtask
{
public:
    PathFinderTask();
    PathFinderTask(int rows, int cols, int height);
    ~PathFinderTask();
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

    int getRows() { return m_rows; }
    int getCols() { return m_cols; }
    int getHeight() { return m_pyramid_height; }
    int getDataSize() { return m_datasize; }
    int getResultSize() { return m_resultsize; }
    void setParameters(int rows, int cols, int height);

    void setComputeAssigns(bool b) { m_computeAssigns = b; }
    bool getComputeAssigns() { return m_computeAssigns; }
    void setPersistentBlocks();

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
    int *h_data;
    int *h_result;

    int *d_wall;
    int *d_result[2];

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

    int m_rows;
    int m_cols;
    int m_datasize;
    int m_resultsize;
    int m_pyramid_height;
    int m_borderCols;
    int m_smallBlockCol;
    int m_blockCols;
    int m_final_ret;

    bool m_computeAssigns;

    void setRows(int rows) { m_rows = rows; }
    void setCols(int cols) { m_cols = cols; }
    void setHeight(int height) { m_pyramid_height = height; }
};
