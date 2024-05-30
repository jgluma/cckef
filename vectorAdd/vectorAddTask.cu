/**
 * @file vectorAddTask.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief
 * @version 0.1
 * @date 2021-02-15
 *
 */
#include "vectorAdd.h"
#include "vectorAddKernel.cuh"

/**
 * @brief Construct a new vectorAdd task with default values
 *
 */
vectorAddTask::vectorAddTask()
{

  setTaskName(VA);

  h_A = NULL;
  h_B = NULL;
  h_C = NULL;
  d_A = NULL;
  d_B = NULL;
  d_C = NULL;

  setExecMode(PersistentMemoryRanges);
  setMRMode(None);
  setNumElements(2097152);
  setNumIterations(100);
  setPinned(false);
  chunkSize = 1024; // TODO: read this value from parameters
  chunkInts = chunkSize / sizeof(int);
  m_numLaunchs = 0;
  err = cudaSuccess;
}

/**
 * @brief Construct a new vectorAdd task with vectors of n elements
 *
 * @param n
 */
vectorAddTask::vectorAddTask(unsigned long n, bool p)
{

  setTaskName(VA);

  h_A = NULL;
  h_B = NULL;
  h_C = NULL;
  d_A = NULL;
  d_B = NULL;
  d_C = NULL;

  setExecMode(PersistentMemoryRanges);
  setNumElements(n);
  setNumIterations(100);
  setPinned(p);
  chunkSize = 1024; // TODO: read this value from parameters
  chunkInts = chunkSize / sizeof(int);
  m_numLaunchs = 0;
  err = cudaSuccess;
}

CUDAtaskNames vectorAddTask::getName() { return name; }

void vectorAddTask::setNumElements(unsigned long n)
{
  numElements = n;
  size = numElements * sizeof(float);

  dim3 t(256, 1, 1);
  setThreadsPerBlock(t);
  dim3 b((numElements + t.x - 1) / t.x, 1, 1);
  setBlocksPerGrid(b);

  // printf("VA task: %d elements, with size %lu bytes. Using %d blocks\n", numElements, size, b.x);

}

void vectorAddTask::setPersistentBlocks()
{
  int maxBlocksPerMulti;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceID);
  dim3 b(1,1,1);
  switch (exec_mode)
  {
  case Original:
    printf("Original kernel cannot use persistent blocks\n");
    break;
  case Persistent:
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMulti, vectorAddP, m_threadsPerBlock.x, 0);
    b.x = maxBlocksPerMulti * deviceProp.multiProcessorCount;
    setBlocksPerGrid(b);
    break;
  case MemoryRanges:
    printf("Original kernel cannot use persistent blocks\n");
    break;
  case PersistentMemoryRanges:
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMulti, vectorAddPMR, m_threadsPerBlock.x, 0);
    b.x = maxBlocksPerMulti * deviceProp.multiProcessorCount;
    setBlocksPerGrid(b);
    break;
  }
}

void vectorAddTask::setAllModules(int reverse)
{
  m_numCombinations = 14;
  allModulesA = (int **)malloc(m_numCombinations * sizeof(int *));
  allModulesB = (int **)malloc(m_numCombinations * sizeof(int *));
  allModulesC = (int **)malloc(m_numCombinations * sizeof(int *));
  // First, combinations where all share the modules
  int j = 0;
  for (int i = 24; i >= 4; i -= 4)
  {
    if (reverse == 0)
    {
      allModulesA[j] = selectModules(0, 1, i);
      allModulesB[j] = selectModules(0, 1, i);
      allModulesC[j] = selectModules(0, 1, i);
    }
    else
    {
      allModulesA[j] = selectModules(24 - i, 1, i);
      allModulesB[j] = selectModules(24 - i, 1, i);
      allModulesC[j] = selectModules(24 - i, 1, i);
    }
    j++;
  }
  // Then, combinations where input and output vectors use different modules
  for (int i = 12; i >= 4; i -= 2)
  {
    if (reverse == 0)
    {
      allModulesA[j] = selectModules(0, 2, i);
      allModulesB[j] = selectModules(0, 2, i);
      allModulesC[j] = selectModules(1, 2, i);
    }
    else
    {
      allModulesA[j] = selectModules(24 - 2 * i, 2, i);
      allModulesB[j] = selectModules(24 - 2 * i, 2, i);
      allModulesC[j] = selectModules(24 - 2 * i + 1, 2, i);
    }
    j++;
  }
  // Finally, combinations where all vectors use different modules
  for (int i = 8; i >= 4; i -= 2)
  {
    if (reverse == 0)
    {
      allModulesA[j] = selectModules(0, 3, i);
      allModulesB[j] = selectModules(1, 3, i);
      allModulesC[j] = selectModules(2, 3, i);
    }
    else
    {
      allModulesA[j] = selectModules(24 - 3 * i, 3, i);
      allModulesB[j] = selectModules(24 - 3 * i + 1, 3, i);
      allModulesC[j] = selectModules(24 - 3 * i + 2, 3, i);
    }
    j++;
  }
}

/**
 * @brief Destroy the vectorAdd task object
 *
 */
vectorAddTask::~vectorAddTask()
{
  // Free host memory
  if (pinned)
  {
    if (h_A != NULL)
      cudaFreeHost(h_A);
    if (h_B != NULL)
      cudaFreeHost(h_B);
    if (h_C != NULL)
      cudaFreeHost(h_C);
  }
  else
  {
    free(h_A);
    free(h_B);
    free(h_C);
  }

  // Free device memory
  if (d_A != NULL)
    cudaFree(d_A);
  if (d_B != NULL)
    cudaFree(d_B);
  if (d_C != NULL)
    cudaFree(d_C);
}

/**
 * @brief Allocates pinned or non pinned host memory for vectors
 *
 */
void vectorAddTask::allocHostMemory(void)
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

    err = cudaMallocHost((void **)&h_B, size);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host vector B (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **)&h_C, size);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host vector C (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
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
void vectorAddTask::freeHostMemory(void)
{
  if (pinned)
  {
    if (h_A != NULL)
      cudaFreeHost(h_A);
    if (h_B != NULL)
      cudaFreeHost(h_B);
    if (h_C != NULL)
      cudaFreeHost(h_C);
  }
  else
  {
    free(h_A);
    free(h_B);
    free(h_C);
  }
}

/**
 * @brief Allocates device memory for vectors
 *
 */
void vectorAddTask::allocDeviceMemory(void)
{
  // Allocate the device input vector A
  err = cudaMalloc((void **)&d_A, size);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  err = cudaMalloc((void **)&d_B, size);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  err = cudaMalloc((void **)&d_C, size);

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Allocates device memory for vectors
 *
 */
void vectorAddTask::allocDeviceMemory(float *ptr,
                                      int *assignsA, int numAssignsA,
                                      int *assignsB, int numAssignsB,
                                      int *assignsC, int numAssignsC)
{

  d_ptr = ptr;

  numChipAssignmentsA = numAssignsA;
  chipAssignmentsA = (int *)malloc(numChipAssignmentsA * sizeof(int));
  d_A = (float *)((int *)d_ptr + assignsA[0] * chunkInts);
  for (int i = 0; i < numChipAssignmentsA; i++)
  {
    chipAssignmentsA[i] = assignsA[i] - assignsA[0];
  }

  numChipAssignmentsB = numAssignsB;
  chipAssignmentsB = (int *)malloc(numChipAssignmentsB * sizeof(int));
  d_B = (float *)((int *)d_ptr + assignsB[0] * chunkInts);
  for (int i = 0; i < numChipAssignmentsB; i++)
  {
    chipAssignmentsB[i] = assignsB[i] - assignsB[0];
  }

  numChipAssignmentsC = numAssignsC;
  chipAssignmentsC = (int *)malloc(numChipAssignmentsC * sizeof(int));
  d_C = (float *)((int *)d_ptr + assignsC[0] * chunkInts);
  for (int i = 0; i < numChipAssignmentsB; i++)
  {
    chipAssignmentsC[i] = assignsC[i] - assignsC[0];
  }

  // printf("VectorAdd Task %p\n", ptr);
  // printf("A: %p, %p, ... %p\n", d_A + chipAssignmentsA[257] * chunkInts, d_A + chipAssignmentsA[258] * chunkInts, d_A + chipAssignmentsA[numChipAssignmentsA - 1] * chunkInts);
  // printf("B: %p, %p, ... %p\n", d_B + chipAssignmentsB[257] * chunkInts, d_B + chipAssignmentsB[258] * chunkInts, d_B + chipAssignmentsB[numChipAssignmentsB - 1] * chunkInts);
  // printf("C: %p, %p, ... %p\n", d_C + chipAssignmentsC[257] * chunkInts, d_C + chipAssignmentsC[258] * chunkInts, d_C + chipAssignmentsC[numChipAssignmentsC - 1] * chunkInts);
  // for (int i = 257; i < numChipAssignmentsC; i++)
  //   printf("%p\n", d_C + chipAssignmentsC[i] * chunkInts);
}

/**
 * @brief Free vectors in device memory
 *
 */
void vectorAddTask::freeDeviceMemory(void)
{
  if (d_A != NULL)
    cudaFree(d_A);
  if (d_B != NULL)
    cudaFree(d_B);
  if (d_C != NULL)
    cudaFree(d_C);
}

/**
 * @brief Generate random data for input vectors
 *
 */
void vectorAddTask::dataGeneration(void)
{
  for (int i = 0; i < numElements; ++i)
  {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }
}

/**
 * @brief Asynchronous HTD memory transfer using stream
 *
 * @param stream
 */
void vectorAddTask::memHostToDeviceAsync(cudaStream_t stream)
{

  if (profile == EventsProf)
    cudaEventRecord(htdStart, stream);

  if (mr_mode == None)
  {
    err = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy vector A from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    err = cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy vector B from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
    memHostToDeviceAssigns();

  if (profile == EventsProf)
    cudaEventRecord(htdEnd, stream);
}

/**
 * @brief Synchronous HTD memory transfer
 *
 */
void vectorAddTask::memHostToDevice(void)
{

  if (profile == EventsProf)
    cudaEventRecord(htdStart, 0);

  if (mr_mode == None)
  {
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy vector A from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy vector B from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
    memHostToDeviceAssigns();

  cudaDeviceSynchronize();

  if (profile == EventsProf)
    cudaEventRecord(htdEnd);
}

void vectorAddTask::memHostToDeviceAssigns(void)
{

  if (profile == EventsProf)
    cudaEventRecord(htdStart, 0);

  sendPageTables(d_A, chipAssignmentsA, numChipAssignmentsA);
  sendData(d_A, h_A, chipAssignmentsA, numChipAssignmentsA);

  sendPageTables(d_B, chipAssignmentsB, numChipAssignmentsB);
  sendData(d_B, h_B, chipAssignmentsB, numChipAssignmentsB);

  sendPageTables(d_C, chipAssignmentsC, numChipAssignmentsC);

  cudaDeviceSynchronize();

  if (profile == EventsProf)
    cudaEventRecord(htdEnd);
}

/**
 * @brief Asynchronous DTH memory transfer using stream
 *
 * @param stream
 */
void vectorAddTask::memDeviceToHostAsync(cudaStream_t stream)
{

  if (profile == EventsProf)
    cudaEventRecord(htdStart, stream);

  if (mr_mode == None)
  {
    err = cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy vector C from device to host (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
    memDeviceToHostAssigns();

  if (profile == EventsProf)
    cudaEventRecord(htdEnd, stream);
}

/**
 * @brief Synchronous HTD memory transfer
 *
 */
void vectorAddTask::memDeviceToHost(void)
{
  if (profile == EventsProf)
    cudaEventRecord(dthStart);

  if (mr_mode == None)
  {
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy vector C from device to host (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
    memDeviceToHostAssigns();

  if (profile == EventsProf)
    cudaEventRecord(dthEnd);
}

void vectorAddTask::memDeviceToHostAssigns(void)
{
  if (profile == EventsProf)
    cudaEventRecord(dthStart);

  receiveData(d_C, h_C, chipAssignmentsC, numChipAssignmentsC);

  if (profile == EventsProf)
    cudaEventRecord(dthEnd);
}

/**
 * @brief Launch vectorAdd kernel asynchronously using stream
 *
 * @param stream
 */
void vectorAddTask::launchKernelAsync(cudaStream_t stream)
{

  if (m_numLaunchs > 0)
    if (cudaEventQuery(kernelEnd) != cudaSuccess)
      return;

  cudaEventRecord(kernelStart, stream);

  switch (exec_mode)
  {
  case Original:
    m_blocksPerGrid.y = m_numIterations;
    vectorAdd<<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, numElements, 1);
    break;
  case Persistent:
    vectorAddP<<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, numElements, m_numIterations);
    break;
  case MemoryRanges:
    m_blocksPerGrid.y = m_numIterations;
    vectorAddMR<<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, numElements, 1);
    break;
  case PersistentMemoryRanges:
    vectorAddPMR<<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, numElements, m_numIterations);
    break;
  }

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  m_numLaunchs++;

  cudaEventRecord(kernelEnd, stream);
}

/**
 * @brief Launch vectorAdd kernel synchronously
 *
 */
void vectorAddTask::launchKernel(void)
{

  cudaEventRecord(kernelStart);

  switch (exec_mode)
  {
  case Original:
    vectorAdd<<<m_blocksPerGrid, m_threadsPerBlock>>>(d_A, d_B, d_C, numElements, m_numIterations);
    break;
  case Persistent:
    vectorAddP<<<m_blocksPerGrid, m_threadsPerBlock>>>(d_A, d_B, d_C, numElements, m_numIterations);
    break;
  case MemoryRanges:
    vectorAddMR<<<m_blocksPerGrid, m_threadsPerBlock>>>(d_A, d_B, d_C, numElements, m_numIterations);
    break;
  case PersistentMemoryRanges:
    vectorAddPMR<<<m_blocksPerGrid, m_threadsPerBlock>>>(d_A, d_B, d_C, numElements, m_numIterations);
    break;
  }

  m_numLaunchs++;
  cudaDeviceSynchronize();

  cudaEventRecord(kernelEnd);

  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Verify that the result vector is correct
 *
 * @return true Test passed
 * @return false Test failed
 */
bool vectorAddTask::checkResults(void)
{
  for (int i = 0; i < numElements; ++i)
  {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
    {
      fprintf(stderr, "Result verification failed at element %d! CPU %f GPU %f\n", i, h_A[i] + h_B[i], h_C[i]);
      return false;
    }
  }
  return true;
}