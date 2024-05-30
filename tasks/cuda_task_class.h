/**
 * @file cuda_task_class.h
 * @details This file describes a class to implement CUDA tasks.
 * @author José María González Linares.
 * @date 15/12/2021
 */

#ifndef _CUDA_TASK_CLASS_H_
#define _CUDA_TASK_CLASS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

typedef enum
{
	Dummy = 0,
	matrixMul,
	VA,
	BS,
	PF,
	MemBench
} CUDAtaskNames;
typedef enum
{
	NoProfile = 0,
	TimerProf,
	EventsProf,
	CUPTIProf,
	CUPTISample
} ProfileMode;
typedef enum
{
	SYNC = 0,
	ASYNC
} CKEmode;
typedef enum
{
	None = 0,
	Shared,
	NonShared
} MemoryRangeMode;
typedef enum
{
	All = 0,
	Half,
	ThreeQuarters
} AssignmentMode;

typedef enum
{
	Original = 0,
	Persistent,
	MemoryRanges,
	PersistentMemoryRanges
} ExecutionMode;

/**
 * @class CUDAtask
 * @brief A generic CUDA task
 * @details This class defines virtual methods for a generic CUDA task
 * @author José María González Linares.
 * 
 */
class CUDAtask
{

public:
	virtual CUDAtaskNames getName() = 0;
	/**
	 * @brief Alloc host memory
	 * @details Virtual function to alloc host pinned memory
	 * @author José María González Linares.
	 */
	virtual void allocHostMemory() = 0;
	/**
	 * @brief free host memory
	 * @details Virtual function to free host pinned memory
	 * @author José María González Linares.
	 */
	virtual void freeHostMemory() = 0;
	/**
	 * @brief Alloc device memory
	 * @details Virtual function to alloc device memory
	 * @author José María González Linares.
	 */
	virtual void allocDeviceMemory(void) = 0;
	/**
	 * @brief Free device memory
	 * @details Virtual function to free device memory
	 * @author José María González Linares.
	 */
	virtual void freeDeviceMemory(void) = 0;
	/**
	 * @brief Data generation
	 * @details Virtual function to generate input data.
	 * @author José María González Linares.
	 */
	virtual void dataGeneration(void) = 0;
	/**
	 * @brief Asynchronous HTD memory transfer.
	 * @details Virtual function to asynchronously perfom HTD memory tranfers.
	 * @author José María González Linares.
 	 * 
 	 * @param stream CUDA stream to launch the memory transfer.
	 */
	virtual void memHostToDeviceAsync(cudaStream_t stream) = 0;
	/**
	 * @brief Synchronous HTD memory transfer.
	 * @details Virtual function to synchronously perform HTD memory transfers.
	 * @author José María González Linares.
	 */
	virtual void memHostToDevice(void) = 0;
	/**
	 * @brief Asynchronous DTH memory transfer.
	 * @details Virtual function to asynchronously perform DTH memory tranfers.
	 * @author José María González Linares.
 	 * 
 	 * @param stream CUDA stream to launch the memory transfer.
	 */
	virtual void memDeviceToHostAsync(cudaStream_t stream) = 0;
	/**
	 * @brief Synchronous DTH memory transfer.
	 * @details Virtual function to synchronously perform DTH memory tranfers.
	 * @author José María González Linares.
	 */
	virtual void memDeviceToHost(void) = 0;
	/**
	 * @brief Asynchronous kernel launching.
	 * @details Virtual function to launch asynchronously a CUDA kernel.
	 * @author José María González Linares.
 	 * 
 	 * @param stream CUDA stream to launch the kernel.
	 */
	virtual void launchKernelAsync(cudaStream_t stream) = 0;
	/**
	 * @brief Synchronous kernel launching.
	 * @details Virtual function to launch synchronously a CUDA kernel.
	 * @author José María González Linares.
	 */
	virtual void launchKernel(void) = 0;
	/**
	 * @brief Check results.
	 * @details Virtual function to check results correctness.
	 * @author José María González Linares.
	 */
	virtual bool checkResults(void) = 0;

	virtual void setExecMode(ExecutionMode m)
	{
		exec_mode = m;
	}

	virtual ExecutionMode getExecMode() { return exec_mode; }

	virtual void setCKEMode(CKEmode m)
	{
		cke_mode = m;
		if (cke_mode == ASYNC)
			cudaStreamCreate(&t_stream);
	}

	virtual CKEmode getCKEMode() { return cke_mode; }

	virtual void setMRMode(MemoryRangeMode m)
	{
		mr_mode = m;
	}

	virtual MemoryRangeMode getMRMode() { return mr_mode; }

	virtual void htdTransfer()
	{
		if (cke_mode == SYNC)
			memHostToDevice();
		else
			memHostToDeviceAsync(t_stream);
	}

	virtual void dthTransfer()
	{
		if (cke_mode == SYNC)
			memDeviceToHost();
		else
			memDeviceToHostAsync(t_stream);
	}

	virtual void kernelExec()
	{
		if (cke_mode == SYNC)
			launchKernel();
		else
			launchKernelAsync(t_stream);
	}

	virtual void setTaskName(CUDAtaskNames t) { name = t; }
	virtual void setPinned(bool p) { pinned = p; }
	virtual void setDeviceID(int d) { deviceID = d; }
	virtual void setThreadsPerBlock(dim3 &t) { m_threadsPerBlock = t; }
	virtual void setBlocksPerGrid(dim3 &b) { m_blocksPerGrid = b; }
	virtual void setPersistentBlocks(void) = 0;

	virtual dim3 getThreadsPerBlock() { return m_threadsPerBlock; }
	virtual dim3 getBlocksPerGrid() { return m_blocksPerGrid; }

	// Provide a default implementation for next methods
	virtual void setProfileMode(ProfileMode p)
	{
		if (p == EventsProf)
		{
			cudaEventCreate(&htdStart);
			cudaEventCreate(&htdEnd);
			cudaEventCreate(&dthStart);
			cudaEventCreate(&dthEnd);
			cudaEventCreate(&kernelStart);
			cudaEventCreate(&kernelEnd);
		}
		profile = p;
	}

	virtual ProfileMode getProfileMode() { return profile; }

	virtual float getHtDElapsedTime()
	{
		if (profile == EventsProf)
		{
			cudaEventSynchronize(htdEnd);
			cudaEventElapsedTime(&htdElapsedTime, htdStart, htdEnd);
		}
		return (htdElapsedTime);
	}

	virtual float getDtHElapsedTime()
	{
		if (profile == EventsProf)
		{
			cudaEventSynchronize(dthEnd);
			cudaEventElapsedTime(&dthElapsedTime, dthStart, dthEnd);
		}
		return (dthElapsedTime);
	}

	virtual float getKernelElapsedTime()
	{
		if (profile == EventsProf)
		{
			cudaEventSynchronize(kernelEnd);
			cudaEventElapsedTime(&kernelElapsedTime, kernelStart, kernelEnd);
		}
		return (kernelElapsedTime);
	}

	virtual void setAssignments(int numChips, int *numChipAssignments, int *chipMR);

	virtual void transferHeaderHtD(void *d_A, void *h_A, size_t count, int *chunks, int idx);
	virtual void transferChunksHtD(void *d_A, void *h_A, size_t count, int *chunks, int idx);
	virtual void transferChunksDtH(void *d_A, void *h_A, size_t count, int *chunks, int idx);

	virtual void setNumLaunchs(int numLaunchs) { m_numLaunchs = numLaunchs; }
	virtual int getNumLaunchs() { return m_numLaunchs; }

	virtual void setAssignmentMode(AssignmentMode m) { assig_mode = m; }

	virtual void setDeviceMemory(float *ptr) { d_ptr = (void *)ptr; }

	virtual void setAllModules(int reverse) { m_numCombinations = 0; }
	virtual int getNumCombinations() { return m_numCombinations; }
	virtual int *selectModules(int init, int step, int n);
	virtual void sendData(void *d_ptr, void *h_ptr, int *ptEntries, int numEntries);
	virtual void sendPageTables(void *d_ptr, void *h_ptr, int numEntries);
	virtual void receiveData(void *d_ptr, void *h_ptr, int *ptEntries, int numEntries);

protected:
	// Task generic variables
	CUDAtaskNames name;
	// CUDA configuration
	int deviceID;
	CKEmode cke_mode;		 // CKE mode: SYNC or ASYNC
	MemoryRangeMode mr_mode; // Memory range mode: none, shared or unshared
	cudaStream_t t_stream;
	bool pinned = false; // True to request pinned memory
	cudaError_t err;	 // Error code to check return values for CUDA calls
	dim3 m_threadsPerBlock;
	dim3 m_blocksPerGrid;

	ExecutionMode exec_mode;

	// Memory partition variables
	void *d_ptr;									// Device memory pointer to allocate memory for all arrays
	int chunkSize, chunkInts;						// Size in bytes of a chunk, and number of ints in a chunk
    int m_numCombinations;

	int *chipAssignments0, *chipAssignments1;		// Memory chips assignments for partition 0 (1)
	int maxChipsAssignments0, maxChipsAssignments1; // Max avalaible chips assignments for 0 (1)
	AssignmentMode assig_mode;						// Chips assignments mode

	// CUDA profiling
	ProfileMode profile = NoProfile;
	cudaEvent_t htdStart, htdEnd, dthStart, dthEnd, kernelStart, kernelEnd;
	float htdElapsedTime, dthElapsedTime, kernelElapsedTime;
	int m_numLaunchs;
};

#endif