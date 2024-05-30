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
dummyTask::dummyTask()
{

  setTaskName(Dummy);

  h_A = NULL;
  h_B = NULL;
  d_A = NULL;
  d_B = NULL;

  setExecMode(PersistentMemoryRanges);
  setMRMode(None);
  setNumElements(4 * 1024 * 1024);
  setNumIterations(1);
  setCB(false);
  setMB(true);
  setMBR(true);

  chunkSize = 1024; // TODO: read this value from parameters
  chunkInts = chunkSize / sizeof(int);
  setComputeAssigns(true);
  setPinned(false);
  setPartitions(0, 1);

  m_numLaunchs = 0;
  err = cudaSuccess;
}

/**
 * @brief Construct a new dummy task
 *
 * @param n
 */
dummyTask::dummyTask(unsigned long nI, bool cb, unsigned long nE, bool mb)
{

  setTaskName(Dummy);

  h_A = NULL;
  h_B = NULL;
  d_A = NULL;
  d_B = NULL;

  setExecMode(PersistentMemoryRanges);
  setMRMode(None);
  setNumElements(nE);
  setNumIterations(nI);
  setCB(cb);
  setMB(mb);
  setMBR(false);
  chunkSize = 1024; // TODO: read this value from parameters
  chunkInts = chunkSize / sizeof(int);
  setComputeAssigns(true);
  setPinned(false);
  setPartitions(0, 1);
  m_numLaunchs = 0;
  err = cudaSuccess;
}

CUDAtaskNames dummyTask::getName() { return name; }

void dummyTask::setNumElements(unsigned long n)
{
  m_numElements = n;
  setSizeA();
  setSizeB();
  dim3 t(256, 1, 1);
  setThreadsPerBlock(t);
  dim3 b((m_numElements + m_threadsPerBlock.x - 1) / m_threadsPerBlock.x, 1, 1);
  setBlocksPerGrid(b);
}

void dummyTask::setPersistentBlocks()
{
  int maxBlocksPerMulti;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceID);
  dim3 b;
  switch (exec_mode)
  {
  case Original:
    printf("Original kernel cannot use persistent blocks\n");
    break;
  case Persistent:
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMulti, mbDummyP, m_threadsPerBlock.x, 0);
    b.x = maxBlocksPerMulti * deviceProp.multiProcessorCount;
    setBlocksPerGrid(b);
    break;
  case MemoryRanges:
    printf("Original kernel cannot use persistent blocks\n");
    break;
  case PersistentMemoryRanges:
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMulti, mbDummyPMR, m_threadsPerBlock.x, 0);
    b.x = maxBlocksPerMulti * deviceProp.multiProcessorCount;
    setBlocksPerGrid(b);
    break;
  }
}

void dummyTask::setAllModules(int reverse)
{
  m_numCombinations = 12;
  allModulesA = (int **)malloc(m_numCombinations * sizeof(int *));
  allModulesB = (int **)malloc(m_numCombinations * sizeof(int *));
  // First, combinations where A and B share the modules
  int j = 0;
  for (int i = 24; i >= 4; i -= 4)
  {
    if (reverse == 0)
    {
      allModulesA[j] = selectModules(0, 1, i);
      allModulesB[j] = selectModules(0, 1, i);
    }
    else
    {
      allModulesA[j] = selectModules(24 - i, 1, i);
      allModulesB[j] = selectModules(24 - i, 1, i);
    }
    j++;
  }
  // Then, combinations where A and B use different modules
  for (int i = 12; i >= 2; i -= 2)
  {
    if (reverse == 0)
    {
      allModulesA[j] = selectModules(0, 2, i);
      allModulesB[j] = selectModules(1, 2, i);
    }
    else
    {
      allModulesA[j] = selectModules(24 - 2 * i, 2, i);
      allModulesB[j] = selectModules(24 - 2 * i + 1, 2, i);
    }
    j++;
  }
}

/**
 * @brief Destroy the vectorAdd task object
 *
 */
dummyTask::~dummyTask()
{
  // Free host memory
  if (pinned)
  {
    if (h_A != NULL)
      cudaFreeHost(h_A);
    if (h_B != NULL)
      cudaFreeHost(h_B);
  }
  else
  {
    free(h_A);
    free(h_B);
  }

  // Free device memory
  if (d_A != NULL)
    cudaFree(d_A);
  if (mr_mode == None && d_B != NULL)
    cudaFree(d_B);
}

/**
 * @brief Allocates pinned or non pinned host memory for vectors
 *
 */
void dummyTask::allocHostMemory(void)
{
  if (pinned)
  {
    err = cudaMallocHost((void **)&h_A, m_sizeA);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host vector A (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **)&h_B, m_sizeB);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host vector B (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    h_A = (float *)malloc(m_sizeA);
    h_B = (float *)malloc(m_sizeB);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL)
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
void dummyTask::freeHostMemory(void)
{
  if (pinned)
  {
    if (h_A != NULL)
      cudaFreeHost(h_A);
    if (h_B != NULL)
      cudaFreeHost(h_B);
  }
  else
  {
    free(h_A);
    free(h_B);
  }
}

/**
 * @brief Allocates device memory for vectors
 *
 */
void dummyTask::allocDeviceMemory(void)
{
  // Allocate the device input vector A
  if (mr_mode == None)
  {
    err = cudaMalloc((void **)&d_A, m_sizeA);
  }
  else if (mr_mode == Shared)
  {
    // Allocate enough memory for all the vectors and page tables
    // The page tables are stored in the first 257 chunks (assuming a chunk size of 256 ints)

    numChipAssignmentsA = m_sizeA / chunkSize + 257;
    numChipAssignmentsB = numChipAssignmentsA;
    if (getComputeAssigns() == true)
    {
      int size = 1.25 * (numChipAssignmentsA + numChipAssignmentsB);
      err = cudaMalloc((void **)&d_ptr, size * chunkSize);
      printf("Profile %d chunks (%d elements, %d assigns)\n", size, m_numElements, numChipAssignmentsA);
      memBench *mb = new memBench();
      mb->init(deviceID, (float *)d_ptr, size);
      mb->getChipAssignments();
      mb->getMemoryRanges();
      mb->writeAssignments();

      setAssignments(mb->getNumChips(), mb->returnNumChipAssignments(), mb->returnChipMR());
      if (partitionA == 0)
        d_A = (float *)((int *)d_ptr + chipAssignments0[0] * chunkInts);
      else
        d_A = (float *)((int *)d_ptr + chipAssignments1[0] * chunkInts);
    }
    else
    {
      if (numChipAssignmentsA > maxChipsAssignments0 || numChipAssignmentsB > maxChipsAssignments1)
      {
        fprintf(stderr, "Failed to allocate device vector A, not enough chunks!\n");
        exit(EXIT_FAILURE);
      }
      int newSize = chipAssignments0[numChipAssignmentsA] > chipAssignments1[numChipAssignmentsB] ? chipAssignments0[numChipAssignmentsA] + 2 : chipAssignments1[numChipAssignmentsB] + 2;
      err = cudaMalloc((void **)&d_ptr, newSize * chunkSize);
      if (partitionA == 0)
        d_A = (float *)((int *)d_ptr + chipAssignments0[0] * chunkInts);
      else
        d_A = (float *)((int *)d_ptr + chipAssignments1[0] * chunkInts);
    }
  }
  else if (mr_mode == NonShared)
  {
    numChipAssignmentsA = m_sizeA / chunkSize + 257;
    numChipAssignmentsB = numChipAssignmentsA;
    if (getComputeAssigns() == true)
    { // Allocate enough space for A and B, and their page tables
      int size = 2.1 * (numChipAssignmentsA /*+ numChipAssignmentsB*/);
      err = cudaMalloc((void **)&d_ptr, size * chunkSize);
      printf("Profile %d chunks\n", size);
      memBench *mb = new memBench();
      mb->init(deviceID, (float *)d_ptr, size);
      mb->getChipAssignments();
      mb->getMemoryRanges();
      mb->writeAssignments();
      setAssignments(mb->getNumChips(), mb->returnNumChipAssignments(), mb->returnChipMR());
    }
    if (partitionA == 0)
    {
      d_A = (float *)((int *)d_ptr + chipAssignments0[0] * chunkInts);
    }
    else
    {
      d_A = (float *)((int *)d_ptr + chipAssignments1[0] * chunkInts);
    }
  }

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  if (mr_mode == None)
  {
    err = cudaMalloc((void **)&d_B, m_sizeB);
    if (err != cudaSuccess)
    {
      fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else if (mr_mode == Shared)
  {
    if (partitionB == 0)
      d_B = (float *)((int *)d_ptr + chipAssignments0[0] * chunkInts);
    else
      d_B = (float *)((int *)d_ptr + chipAssignments1[0] * chunkInts);
  }
  else if (mr_mode == NonShared)
  {
    if (partitionB == 0)
    {
      if (partitionA == 0)
        d_B = (float *)((int *)d_ptr + chipAssignments0[numChipAssignmentsA] * chunkInts);
      else
        d_B = (float *)((int *)d_ptr + chipAssignments0[0] * chunkInts);
    }
    else if (partitionA == 0)
      d_B = (float *)((int *)d_ptr + chipAssignments1[0] * chunkInts);
    else
      d_B = (float *)((int *)d_ptr + chipAssignments1[numChipAssignmentsA] * chunkInts);
  }
}

/**
 * @brief Allocates device memory for vectors
 *
 */
void dummyTask::allocDeviceMemory(float *ptr, int *assignsA, int numAssignsA, int *assignsB, int numAssignsB)
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
  for (int i = 0; i < numChipAssignmentsA; i++)
  {
    chipAssignmentsB[i] = assignsB[i] - assignsB[0];
  }

  // printf("Dummy Task %p\n", ptr);
  // printf("A: %p (%d), %p (%d), ... %p (%d) - %d pages\n", d_A + chipAssignmentsA[0] * chunkInts, chipAssignmentsA[0], d_A + chipAssignmentsA[1] * chunkInts, chipAssignmentsA[1], d_A + chipAssignmentsA[numChipAssignmentsA - 1] * chunkInts, chipAssignmentsA[numChipAssignmentsA - 1], numChipAssignmentsA);
  // printf("B: %p (%d), %p (%d), ... %p (%d) - %d pages\n", d_B + chipAssignmentsB[0] * chunkInts, chipAssignmentsB[0], d_B + chipAssignmentsB[1] * chunkInts, chipAssignmentsB[1], d_B + chipAssignmentsB[numChipAssignmentsB - 1] * chunkInts, chipAssignmentsB[numChipAssignmentsB - 1], numChipAssignmentsB);
}

/**
 * @brief Free vectors in device memory
 *
 */
void dummyTask::freeDeviceMemory(void)
{
  if (mr_mode != NULL)
  {
    if (d_ptr != NULL)
      cudaFree(d_ptr);
  }
  else
  {
    if (d_A != NULL)
      cudaFree(d_A);
    if (d_B != NULL)
      cudaFree(d_B);
  }
}

/**
 * @brief Generate random data for input vectors
 *
 */
void dummyTask::dataGeneration(void)
{
  for (int i = 0; i < m_numElements; ++i)
  {
    h_A[i] = rand() / (float)RAND_MAX;
  }
}

/**
 * @brief Asynchronous HTD memory transfer using stream
 *
 * @param stream
 */
void dummyTask::memHostToDeviceAsync(cudaStream_t stream)
{

  if (profile == EventsProf)
    cudaEventRecord(htdStart, stream);

  if (mr_mode == None)
  {
    err = cudaMemcpyAsync(d_A, h_A, m_sizeA, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy vector A from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else if (mr_mode == Shared)
  {
    if (partitionA == 0)
    {
      transferChunksHtD(d_A, h_A, m_sizeA, chipAssignments0, 0);
      if (partitionB == 0)
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments0, numChipAssignmentsA);
      else
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments1, 0);
    }
    else
    {
      transferChunksHtD(d_A, h_A, m_sizeA, chipAssignments1, 0);
      if (partitionB == 0)
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments0, 0);
      else
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments1, numChipAssignmentsA);
    }
  }
  else if (mr_mode == NonShared)
  {
    if (partitionA == 0)
    {
      transferChunksHtD(d_A, h_A, m_sizeA, chipAssignments0, 0);
      if (partitionB == 0)
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments0, numChipAssignmentsA);
      else
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments1, 0);
    }
    else
    {
      transferChunksHtD(d_A, h_A, m_sizeA, chipAssignments1, 0);
      if (partitionB == 0)
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments0, 0);
      else
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments1, numChipAssignmentsA);
    }
  }

  if (profile == EventsProf)
    cudaEventRecord(htdEnd, stream);
}

/**
 * @brief Synchronous HTD memory transfer
 *
 */
void dummyTask::memHostToDevice(void)
{

  if (profile == EventsProf)
    cudaEventRecord(htdStart, 0);

  if (mr_mode == None)
  {
    err = cudaMemcpy(d_A, h_A, m_sizeA, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy vector A from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else if (mr_mode == Shared)
  {
    if (partitionA == 0)
    {
      transferChunksHtD(d_A, h_A, m_sizeA, chipAssignments0, 0);
      if (partitionB == 0)
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments0, numChipAssignmentsA);
      else
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments1, 0);
    }
    else
    {
      transferChunksHtD(d_A, h_A, m_sizeA, chipAssignments1, 0);
      if (partitionB == 0)
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments0, 0);
      else
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments1, numChipAssignmentsA);
    }
  }
  else if (mr_mode == NonShared)
  {
    if (partitionA == 0)
    {
      transferChunksHtD(d_A, h_A, m_sizeA, chipAssignments0, 0);
      if (partitionB == 0)
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments0, numChipAssignmentsA);
      else
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments1, 0);
    }
    else
    {
      transferChunksHtD(d_A, h_A, m_sizeA, chipAssignments1, 0);
      if (partitionB == 0)
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments0, 0);
      else
        transferHeaderHtD(d_B, h_B, m_sizeB, chipAssignments1, numChipAssignmentsA);
    }
  }

  cudaDeviceSynchronize();

  if (profile == EventsProf)
    cudaEventRecord(htdEnd);
}

void dummyTask::memHostToDeviceAssigns(void)
{

  if (profile == EventsProf)
  {
    err = cudaEventRecord(htdStart, 0);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to record event (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  sendPageTables(d_A, chipAssignmentsA, numChipAssignmentsA);
  sendData(d_A, h_A, chipAssignmentsA, numChipAssignmentsA);
  sendPageTables(d_B, chipAssignmentsB, numChipAssignmentsB);

  cudaDeviceSynchronize();

  if (profile == EventsProf)
    cudaEventRecord(htdEnd);
}

/**
 * @brief Asynchronous DTH memory transfer using stream
 *
 * @param stream
 */
void dummyTask::memDeviceToHostAsync(cudaStream_t stream)
{

  if (profile == EventsProf)
    cudaEventRecord(dthStart, stream);

  if (mr_mode == None)
  {
    err = cudaMemcpyAsync(h_B, d_B, m_sizeB, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy vector B from device to host (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else if (mr_mode == Shared)
  {
    if (partitionB == 0)
    {
      if (partitionA == 0)
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments0, numChipAssignmentsA);
      else
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments0, 0);
    }
    else
    {
      if (partitionA == 0)
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments1, 0);
      else
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments1, numChipAssignmentsA);
    }
  }
  else if (mr_mode == NonShared)
  {
    if (partitionB == 0)
    {
      if (partitionA == 0)
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments0, numChipAssignmentsA);
      else
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments0, 0);
    }
    else
    {
      if (partitionA == 0)
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments1, 0);
      else
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments1, numChipAssignmentsA);
    }
  }

  if (profile == EventsProf)
    cudaEventRecord(dthEnd, stream);
}

/**
 * @brief Synchronous HTD memory transfer
 *
 */
void dummyTask::memDeviceToHost(void)
{
  if (profile == EventsProf)
    cudaEventRecord(dthStart);

  if (mr_mode == None)
  {
    err = cudaMemcpy(h_B, d_B, m_sizeB, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy vector B from device to host (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else if (mr_mode == Shared)
  {
    if (partitionB == 0)
    {
      if (partitionA == 0)
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments0, numChipAssignmentsA);
      else
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments0, 0);
    }
    else
    {
      if (partitionA == 0)
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments1, 0);
      else
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments1, numChipAssignmentsA);
    }
  }
  else if (mr_mode == NonShared)
  {
    if (partitionB == 0)
    {
      if (partitionA == 0)
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments0, numChipAssignmentsA);
      else
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments0, 0);
    }
    else
    {
      if (partitionA == 0)
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments1, 0);
      else
        transferChunksDtH(d_B, h_B, m_sizeB, chipAssignments1, numChipAssignmentsA);
    }
  }

  if (profile == EventsProf)
    cudaEventRecord(dthEnd);
}

void dummyTask::memDeviceToHostAssigns(void)
{
  if (profile == EventsProf)
    cudaEventRecord(dthStart);

  receiveData(d_B, h_B, chipAssignmentsB, numChipAssignmentsB);

  if (profile == EventsProf)
    cudaEventRecord(dthEnd);
}

/**
 * @brief Launch vectorAdd kernel asynchronously using stream
 *
 * @param stream
 */
void dummyTask::launchKernelAsync(cudaStream_t stream)
{

  if (m_numLaunchs > 0)
    if (cudaEventQuery(kernelEnd) != cudaSuccess)
      return;

  cudaEventRecord(kernelStart, stream);

  switch (exec_mode)
  {
  case Original:
    m_blocksPerGrid.y = m_numIterations;
    mbDummy<<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_A, d_B, m_numElements, 1);
    break;
  case Persistent:
    mbDummyP<<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_A, d_B, m_numElements, m_numIterations);
    break;
  case MemoryRanges:
    m_blocksPerGrid.y = m_numIterations;
    mbDummyMR<<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_A, d_B, m_numElements, 1);
    break;
  case PersistentMemoryRanges:
    mbDummyPMR<<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_A, d_B, m_numElements, m_numIterations);
    break;
  }

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch dummy kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  m_numLaunchs++;

  cudaEventRecord(kernelEnd, stream);
}

/**
 * @brief Launch kernel synchronously
 *
 */
void dummyTask::launchKernel(void)
{

  if (profile == EventsProf)
    cudaEventRecord(kernelStart);

  switch (exec_mode)
  {
  case Original:
    m_blocksPerGrid.y = m_numIterations;
    mbDummy<<<m_blocksPerGrid, m_threadsPerBlock>>>(d_A, d_B, m_numElements, 1);
    break;
  case Persistent:
    mbDummyP<<<m_blocksPerGrid, m_threadsPerBlock>>>(d_A, d_B, m_numElements, m_numIterations);
    break;
  case MemoryRanges:
    m_blocksPerGrid.y = m_numIterations;
    mbDummyMR<<<m_blocksPerGrid, m_threadsPerBlock>>>(d_A, d_B, m_numElements, 1);
    break;
  case PersistentMemoryRanges:
    mbDummyPMR<<<m_blocksPerGrid, m_threadsPerBlock>>>(d_A, d_B, m_numElements, m_numIterations);
    break;
  }

  m_numLaunchs++;
  cudaDeviceSynchronize();

  cudaEventRecord(kernelEnd);

  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
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
bool dummyTask::checkResults(void)
{
  int rep = 0;
  for (int i = 0; i < m_numElements; ++i)
  {
    if (fabs(h_A[i] + 1 - h_B[i]) > 1e-5)
    {
      fprintf(stderr, "Result verification failed at element %d! CPU %f GPU %f\n", i, h_A[i] + 1, h_B[i]);
      rep++;
      if (rep > 0)
        return false;
    }
  }
  return true;
}