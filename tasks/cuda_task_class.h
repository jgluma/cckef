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

typedef enum {VA=0, BS} CUDAtaskNames;
typedef enum {NoProfile=0, TimerProf, EventsProf, CUPTIProf} ProfileMode;
typedef enum {SYNC=0, ASYNC} CKEmode;
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

	virtual void setCKEMode( CKEmode m ) {
		mode = m;
		if ( mode == ASYNC )
			cudaStreamCreate( &t_stream );
	}

	virtual void htdTransfer() {
		if ( mode == SYNC )
			memHostToDevice();
		else
			memHostToDeviceAsync( t_stream );
	}

	virtual void dthTransfer() {
		if ( mode == SYNC )
			memDeviceToHost();
		else
			memDeviceToHostAsync( t_stream );
	}

	virtual void kernelExec() {
		if ( mode == SYNC )
			launchKernel();
		else
			launchKernelAsync( t_stream );
	}

	virtual void setTaskName( CUDAtaskNames t) { name = t;}
	virtual void setPinned(bool p) { pinned = p;}
	virtual void setThreadsPerBlock(int t) { threadsPerBlock = t; }
	virtual void setBlocksPerGrid(int b) { blocksPerGrid = b; }

	virtual void setProfileMode(ProfileMode p) {
		if ( p == EventsProf )
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

	virtual float getHtDElapsedTime() {
		if ( profile == EventsProf )
		{
			cudaEventSynchronize(htdEnd);
			cudaEventElapsedTime(&htdElapsedTime, htdStart, htdEnd);
		}
		return(htdElapsedTime);
	}

	virtual float getDtHElapsedTime() {
		if ( profile == EventsProf )
		{
			cudaEventSynchronize(dthEnd);
			cudaEventElapsedTime(&dthElapsedTime, dthStart, dthEnd);
		}
		return(dthElapsedTime);
	}

	virtual float getKernelElapsedTime() {
		if ( profile == EventsProf )
		{
			cudaEventSynchronize(kernelEnd);
			cudaEventElapsedTime(&kernelElapsedTime, kernelStart, kernelEnd);
		}
		return(kernelElapsedTime);
	}


protected:
	// Task generic variables
	CUDAtaskNames name;
	// CUDA configuration
	CKEmode mode; // CKE mode: SYNC or ASYNC
	cudaStream_t t_stream;
	bool pinned = false; // True to request pinned memory
	cudaError_t err; // Error code to check return values for CUDA calls
	int threadsPerBlock;
	int blocksPerGrid;

	// CUDA profiling
	ProfileMode profile = NoProfile;
	cudaEvent_t htdStart, htdEnd, dthStart, dthEnd, kernelStart, kernelEnd;
	float htdElapsedTime, dthElapsedTime, kernelElapsedTime;

};

#endif