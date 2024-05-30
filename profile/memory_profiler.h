/**
 * @file memory_profiler.h
 * @author José María González Linares (jgl@uma.es)
 * @brief 
 * @version 0.1
 * @date 2021-06-04
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#pragma once

#include <stdio.h>
#include <unistd.h>

#include <vector>
#include <map>
#include <string>

#include <cuda.h>
#include <cupti.h>

#include <sys/shm.h>   /* shmat(), IPC_RMID        */
#include <errno.h>     /* errno, ECHILD            */
#include <semaphore.h> /* sem_open(), sem_destroy(), sem_wait().. */
#include <fcntl.h>     /* O_CREAT, O_EXEC          */

#define DRIVER_API_CALL(apiFuncCall)                                             \
    do                                                                           \
    {                                                                            \
        CUresult _status = apiFuncCall;                                          \
        if (_status != CUDA_SUCCESS)                                             \
        {                                                                        \
            fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", \
                    __FILE__, __LINE__, #apiFuncCall, _status);                  \
            exit(-1);                                                            \
        }                                                                        \
    } while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                               \
    do                                                                              \
    {                                                                               \
        cudaError_t _status = apiFuncCall;                                          \
        if (_status != cudaSuccess)                                                 \
        {                                                                           \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
                    __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status)); \
            exit(-1);                                                               \
        }                                                                           \
    } while (0)

#define CUPTI_CALL(call)                                                         \
    do                                                                           \
    {                                                                            \
        CUptiResult _status = call;                                              \
        if (_status != CUPTI_SUCCESS)                                            \
        {                                                                        \
            const char *errstr;                                                  \
            cuptiGetResultString(_status, &errstr);                              \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                    __FILE__, __LINE__, #call, errstr);                          \
            exit(-1);                                                            \
        }                                                                        \
    } while (0)

#define CHECK_CUPTI_ERROR(err, cuptifunc)                       \
    if (err != CUPTI_SUCCESS)                                   \
    {                                                           \
        const char *errstr;                                     \
        cuptiGetResultString(err, &errstr);                     \
        printf("%s:%d:Error %s for CUPTI API function '%s'.\n", \
               __FILE__, __LINE__, errstr, cuptifunc);          \
        return 0;                                               \
    }

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) \
    (((uintptr_t)(buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) : (buffer))

#ifndef __CUPTI_PROFILER_NAME_SHORT
#define __CUPTI_PROFILER_NAME_SHORT 128
#endif

class MemoryProfiler
{
public:
    void initMemoryProfiler(const int device_num);
    void initMemoryProfiler(const int device_num, float *ptr, const size_t ptr_size);
    void setPtr(float *ptr, const size_t ptr_size);
    void profileMemory();

    void writeAssignments();
    void writeAssignments(char *filenameA, char *filenameM);
    void readAssignments();
    void readAssignments(char *filenameA, char *filenameM);

    int *getAssignmentsIdx(size_t bytes, int *modules, int *num_indices);

    unsigned long long getNumAssignments() { return m_numAssignments; }
    int *getChipAssignments() { return m_chipAssignments; }

	virtual void setThreadsPerBlock(dim3 &t) { m_threadsPerBlock = t; }
    void setBlocksPerGrid(dim3 &b) { m_blocksPerGrid = b; }
	dim3 getThreadsPerBlock() { return m_threadsPerBlock; }
	dim3 getBlocksPerGrid() { return m_blocksPerGrid; }

private:
    float *m_ptr;
    size_t m_ptr_size;
    int m_flush_size;

    dim3 m_threadsPerBlock, m_blocksPerGrid;

    int m_numChips;
    unsigned long long m_numAssignments; // Number of assignments to be computed
    int *m_chipAssignments; // Memory chips assignments
    int *m_numAssignmentsPerChip; // Number of memory ranges in each chip
    int m_offsetMR; // Max number of memory ranges in each chip
    int *m_chipMR; // Memory ranges in each chip

    FILE *m_fpA, *m_fpM;
    char m_filenameA[255], m_filenameM[255];
};
