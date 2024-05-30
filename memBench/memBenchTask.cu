/**
 * @file memBenchTask.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief
 * @version 0.1
 * @date 2021-02-15
 *
 */
#include "memBench.h"
#include "memBenchKernel.cuh"

/**
 * @brief Construct a new memBench task with default values
 *
 */
memBenchTask::memBenchTask()
{

  setTaskName(MemBench);

  h_A = NULL;
  d_A = NULL;
  m_allocated = false;

  setMRMode(None);
  setNumElements(50000);
  setPinned(false);
  setCKEMode(SYNC);
  err = cudaSuccess;
}

/**
 * @brief Construct a new memBench task with a vector of n elements
 *
 * @param n
 */
memBenchTask::memBenchTask(unsigned long n, bool p)
{

  setTaskName(MemBench);

  h_A = NULL;
  d_A = NULL;
  m_allocated = false;

  setMRMode(None);
  setNumElements(n);
  setPinned(p);
  setCKEMode(SYNC);
  err = cudaSuccess;
}

CUDAtaskNames memBenchTask::getName() { return name; }

void memBenchTask::setNumElements(unsigned long n)
{
  numElements = n;
  size = numElements * sizeof(float);

  dim3 t(256, 1, 1);
  setThreadsPerBlock(t);
  dim3 b(1, 1, 1);
  setBlocksPerGrid(b);
  setOffset(0);
}


void memBenchTask::setAllModules(int reverse)
{
  m_numCombinations = 0;
}

/**
 * @brief Destroy the memBench task object
 *
 */
memBenchTask::~memBenchTask()
{

  // Free host memory
  if (pinned)
  {
    if (h_A != NULL)
      cudaFreeHost(h_A);
  }
  else
  {
    if (h_A != NULL)
      free(h_A);
  }

  // Free device memory
  if (m_allocated == true)
  {
    if (d_A != NULL)
      cudaFree(d_A);
  }
}

/**
 * @brief Allocates pinned or non pinned host memory for vectors
 *
 */
void memBenchTask::allocHostMemory(void)
{
  if (pinned)
  {
    err = cudaMallocHost((void **)&h_A, size);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host vector A (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    h_A = (float *)malloc(size);
    // Verify that allocation succeeded
    if (h_A == NULL)
    {
      fprintf(stderr, "Failed to allocate host vectors!\n");
      exit(EXIT_FAILURE);
    }
  }
}

/**
 * @brief Free vectors in host memory
 *
 */
void memBenchTask::freeHostMemory(void)
{
  if (pinned)
  {
    if (h_A != NULL)
      cudaFreeHost(h_A);
  }
  else
  {
    free(h_A);
  }
}

/**
 * @brief Allocates device memory for vectors
 *
 */
void memBenchTask::allocDeviceMemory(void)
{
  // Allocate the device input vector A
  err = cudaMalloc((void **)&d_A, size);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  m_allocated = true;
}

/**
 * @brief Free vectors in device memory
 *
 */
void memBenchTask::freeDeviceMemory(void)
{
  if (d_A != NULL)
    cudaFree(d_A);
}

/**
 * @brief Generate random data for input vector
 *
 */
void memBenchTask::dataGeneration(void)
{
  for (int i = 0; i < numElements; ++i)
  {
    h_A[i] = rand() / (float)RAND_MAX;
  }
}

/**
 * @brief Asynchronous HTD memory transfer using stream
 *
 * @param stream
 */
void memBenchTask::memHostToDeviceAsync(cudaStream_t stream)
{

  if (profile == EventsProf)
    cudaEventRecord(htdStart, stream);

  err = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess)
  {
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
void memBenchTask::memHostToDevice(void)
{

  if (profile == EventsProf)
    cudaEventRecord(htdStart, 0);

  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
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
void memBenchTask::memDeviceToHostAsync(cudaStream_t stream)
{

  if (profile == EventsProf)
    cudaEventRecord(htdStart, stream);

  err = cudaMemcpyAsync(h_A, d_A, size, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess)
  {
    fprintf(stderr,
            "Failed to copy vector A from device to host (error code %s)!\n",
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
void memBenchTask::memDeviceToHost(void)
{
  if (profile == EventsProf)
    cudaEventRecord(dthStart);

  err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    fprintf(stderr,
            "Failed to copy vector A from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  if (profile == EventsProf)
    cudaEventRecord(dthEnd);
}

/**
 * @brief Launch memBench kernel asynchronously using stream
 *
 * @param stream
 */
void memBenchTask::launchKernelAsync(cudaStream_t stream)
{

  if (profile == EventsProf)
    cudaEventRecord(kernelStart, stream);

  memBench<<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_A, offset);

  if (profile == EventsProf)
    cudaEventRecord(kernelEnd, stream);

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch memBench kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Launch memBench kernel synchronously
 *
 */
void memBenchTask::launchKernel(void)
{

  if (profile == EventsProf)
    cudaEventRecord(kernelStart);

  memBench<<<m_blocksPerGrid, m_threadsPerBlock>>>(d_A, offset);

  if (profile == EventsProf)
    cudaEventRecord(kernelEnd);

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch memBench kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
