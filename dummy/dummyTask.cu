/**
 * @file dummyTask.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief
 * @version 0.1
 * @date 2021-02-15
 *
 */
#include "dummy.h"
#include "dummyKernel.cuh"

/**
 * @brief Construct a new dummy task with default values
 *
 */
dummyTask::dummyTask() {

  setTaskName(Dummy);

  h_A = NULL;
  h_B = NULL;
  d_A = NULL;
  d_B = NULL;

  setMRMode(None);
  setNumElements(50000);
  setNumIter(50000);
  setCB(true);
  setMB(true);
  setPinned(false);
  err = cudaSuccess;
}

/**
 * @brief Construct a new dummy task
 *
 * @param n
 */
dummyTask::dummyTask(unsigned long nI, bool cb, unsigned long nE, bool mb) {

  setTaskName(Dummy);

  h_A = NULL;
  h_B = NULL;
  d_A = NULL;
  d_B = NULL;

  setMRMode(None);
  setNumElements(nE);
  setNumIter(nI);
  setCB(cb);
  setMB(mb);
  setPinned(false);
  err = cudaSuccess;
}

CUDAtaskNames dummyTask::getName() { return name; }

void dummyTask::setNumElements(unsigned long n) {
  m_numElements = n;
  setThreadsPerBlock(256);
  setBlocksPerGrid((m_numElements+threadsPerBlock-1)/threadsPerBlock);
}

/**
 * @brief Destroy the vectorAdd task object
 *
 */
dummyTask::~dummyTask() {
  // Free host memory
  if (pinned) {
    if (h_A != NULL)
      cudaFreeHost(h_A);
    if (h_B != NULL)
      cudaFreeHost(h_B);
  } else {
    free(h_A);
    free(h_B);
  }

  // Free device memory
  if (d_A != NULL)
    cudaFree(d_A);
  if (d_B != NULL)
    cudaFree(d_B);
}

/**
 * @brief Allocates pinned or non pinned host memory for vectors
 *
 */
void dummyTask::allocHostMemory(void) {
  if (pinned) {
    err = cudaMallocHost((void **)&h_A, m_numElements * sizeof(float));
    if (err != cudaSuccess) {
      fprintf(stderr,
              "Failed to allocate pinned host vector A (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **)&h_B, m_numElements * sizeof(float));
    if (err != cudaSuccess) {
      fprintf(stderr,
              "Failed to allocate pinned host vector B (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  } else {
    h_A = (float *)malloc(m_numElements * sizeof(float));
    h_B = (float *)malloc(m_numElements * sizeof(float));

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL) {
      fprintf(stderr, "Failed to allocate host vectors!\n");
      exit(EXIT_FAILURE);
    }
  }
}

/**
 * @brief Free vectors in host memory
 *
 */
void dummyTask::freeHostMemory(void) {
  if (pinned) {
    if (h_A != NULL)
      cudaFreeHost(h_A);
    if (h_B != NULL)
      cudaFreeHost(h_B);
  } else {
    free(h_A);
    free(h_B);
  }
}

/**
 * @brief Allocates device memory for vectors
 *
 */
void dummyTask::allocDeviceMemory(void) {
  // Allocate the device input vector A
  err = cudaMalloc((void **)&d_A, m_numElements * sizeof(float));

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  err = cudaMalloc((void **)&d_B, m_numElements * sizeof(float));

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Free vectors in device memory
 *
 */
void dummyTask::freeDeviceMemory(void) {
  if (d_A != NULL)
    cudaFree(d_A);
  if (d_B != NULL)
    cudaFree(d_B);
}

/**
 * @brief Generate random data for input vectors
 *
 */
void dummyTask::dataGeneration(void) {
  for (int i = 0; i < m_numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
  }
}

/**
 * @brief Asynchronous HTD memory transfer using stream
 *
 * @param stream
 */
void dummyTask::memHostToDeviceAsync(cudaStream_t stream) {

  if (profile == EventsProf)
    cudaEventRecord(htdStart, stream);

  err = cudaMemcpyAsync(d_A, h_A, m_numElements * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  if (profile == EventsProf)
    cudaEventRecord(htdEnd, stream);
}

/**
 * @brief Synchronous HTD memory transfer
 *
 */
void dummyTask::memHostToDevice(void) {

  if (profile == EventsProf)
    cudaEventRecord(htdStart, 0);

  err = cudaMemcpy(d_A, h_A, m_numElements * sizeof(float),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  if (profile == EventsProf)
    cudaEventRecord(htdEnd);
}

/**
 * @brief Asynchronous DTH memory transfer using stream
 *
 * @param stream
 */
void dummyTask::memDeviceToHostAsync(cudaStream_t stream) {

  if (profile == EventsProf)
    cudaEventRecord(htdStart, stream);

  err = cudaMemcpyAsync(h_B, d_B, m_numElements * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  if (profile == EventsProf)
    cudaEventRecord(htdEnd, stream);
}

/**
 * @brief Synchronous HTD memory transfer
 *
 */
void dummyTask::memDeviceToHost(void) {
  if (profile == EventsProf)
    cudaEventRecord(dthStart);

  err = cudaMemcpy(h_B, d_B, m_numElements * sizeof(float),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  if (profile == EventsProf)
    cudaEventRecord(dthEnd);
}

/**
 * @brief Launch vectorAdd kernel asynchronously using stream
 *
 * @param stream
 */
void dummyTask::launchKernelAsync(cudaStream_t stream) {

  if (profile == EventsProf)
    cudaEventRecord(kernelStart, stream);

  if (m_launchCB)
    cbDummy<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, m_numIter);

  if (m_launchMB)
    mbDummy<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, m_numElements);

  if (profile == EventsProf)
    cudaEventRecord(kernelEnd, stream);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch dummy kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Launch vectorAdd kernel synchronously
 *
 */
void dummyTask::launchKernel(void) {

  if (profile == EventsProf)
    cudaEventRecord(kernelStart);

  if (m_launchCB)
    cbDummy<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, m_numIter);

  if (m_launchMB)
    mbDummy<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, m_numElements);

  if (profile == EventsProf)
    cudaEventRecord(kernelEnd);

  err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch dummy kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Verify the dumbness of the dummy kernel
 *
 * @return true Test passed
 * @return false Test failed
 */
bool dummyTask::checkResults(void) { return true; }