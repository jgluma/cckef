/**
 * @file BlackScholesTask.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief 
 * @version 0.1
 * @date 2021-05-25
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "BlackScholes.h"
#include "BlackScholes_kernel.cuh"

/**
 * @brief Construct a new BlackScholes task with default values
 *
 */
BlackScholesTask::BlackScholesTask()
{

  setTaskName(BS);

  h_CallResultCPU = NULL;
  h_PutResultCPU = NULL;

  h_CallResultGPU = NULL;
  h_PutResultGPU = NULL;
  h_StockPrice = NULL;
  h_OptionStrike = NULL;
  h_OptionYears = NULL;

  d_CallResultGPU = NULL;
  d_PutResultGPU = NULL;
  d_StockPrice = NULL;
  d_OptionStrike = NULL;
  d_OptionYears = NULL;

  setExecMode(PersistentMemoryRanges);
  setMRMode(None);
  setOptions(2000000, 128);
  chunkSize = 1024; // TODO: read this value from parameters
  chunkInts = chunkSize / sizeof(int);
  setPinned(false);
  m_numLaunchs = 0;
  err = cudaSuccess;
}

/**
 * @brief Construct a new BlackScholes task
 *
 * @param n
 */
BlackScholesTask::BlackScholesTask(int optN, int numIter)
{

  setTaskName(BS);

  h_CallResultCPU = NULL;
  h_PutResultCPU = NULL;

  h_CallResultGPU = NULL;
  h_PutResultGPU = NULL;
  h_StockPrice = NULL;
  h_OptionStrike = NULL;
  h_OptionYears = NULL;

  d_CallResultGPU = NULL;
  d_PutResultGPU = NULL;
  d_StockPrice = NULL;
  d_OptionStrike = NULL;
  d_OptionYears = NULL;

  setExecMode(PersistentMemoryRanges);
  setMRMode(None);
  setOptions(optN, numIter);
  chunkSize = 1024; // TODO: read this value from parameters
  chunkInts = chunkSize / sizeof(int);
  setPinned(false);
  m_numLaunchs = 0;
  err = cudaSuccess;
}

CUDAtaskNames BlackScholesTask::getName() { return name; }

void BlackScholesTask::setOptN(int optN)
{
  m_optN = optN;
  m_optN_SZ = 256 * ((m_optN + 255) / 256) * sizeof(float); // Size should be a multiple of chunkSize
  dim3 t(256, 1, 1);
  setThreadsPerBlock(t);
  dim3 b((m_optN + m_threadsPerBlock.x - 1) / m_threadsPerBlock.x, 1, 1);
  setBlocksPerGrid(b);
}

void BlackScholesTask::setPersistentBlocks()
{
  int maxBlocksPerMulti;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, deviceID);
  dim3 b(1, 1, 1);
  switch (exec_mode)
  {
  case Original:
    printf("Original kernel cannot use persistent blocks\n");
    break;
  case Persistent:
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMulti, BlackScholesGPUP, m_threadsPerBlock.x, 0);
    b.x = maxBlocksPerMulti * deviceProp.multiProcessorCount;
    setBlocksPerGrid(b);
    break;
  case MemoryRanges:
    printf("Original kernel cannot use persistent blocks\n");
    break;
  case PersistentMemoryRanges:
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMulti, BlackScholesGPUPMR, m_threadsPerBlock.x, 0);
    b.x = maxBlocksPerMulti * deviceProp.multiProcessorCount;
    setBlocksPerGrid(b);
    break;
  }
}

void BlackScholesTask::setAllModules(int reverse)
{
  m_numCombinations = 12;
  allModulesA = (int **)malloc(m_numCombinations * sizeof(int *));
  allModulesB = (int **)malloc(m_numCombinations * sizeof(int *));
  allModulesC = (int **)malloc(m_numCombinations * sizeof(int *));
  allModulesD = (int **)malloc(m_numCombinations * sizeof(int *));
  allModulesE = (int **)malloc(m_numCombinations * sizeof(int *));
  // First, combinations where all share the modules
  int j = 0;
  for (int i = 24; i >= 4; i -= 4)
  {
    if (reverse == 0)
    {
      allModulesA[j] = selectModules(0, 1, i);
      allModulesB[j] = selectModules(0, 1, i);
      allModulesC[j] = selectModules(0, 1, i);
      allModulesD[j] = selectModules(0, 1, i);
      allModulesE[j] = selectModules(0, 1, i);
    }
    else
    {
      allModulesA[j] = selectModules(24 - i, 1, i);
      allModulesB[j] = selectModules(24 - i, 1, i);
      allModulesC[j] = selectModules(24 - i, 1, i);
      allModulesD[j] = selectModules(24 - i, 1, i);
      allModulesE[j] = selectModules(24 - i, 1, i);
    }
    j++;
  }
  // Then, combinations where input and output vectors use different modules
  for (int i = 12; i >= 2; i -= 2)
  {
    if (reverse == 0)
    {
      allModulesA[j] = selectModules(0, 2, i);
      allModulesB[j] = selectModules(0, 2, i);
      allModulesC[j] = selectModules(1, 2, i);
      allModulesD[j] = selectModules(1, 2, i);
      allModulesE[j] = selectModules(1, 2, i);
    }
    else
    {
      allModulesA[j] = selectModules(24 - 2 * i, 2, i);
      allModulesB[j] = selectModules(24 - 2 * i, 2, i);
      allModulesC[j] = selectModules(24 - 2 * i + 1, 2, i);
      allModulesD[j] = selectModules(24 - 2 * i + 1, 2, i);
      allModulesE[j] = selectModules(24 - 2 * i + 1, 2, i);
    }
    j++;
  }
}

/**
 * @brief Destroy the BlackScholes task object
 *
 */
BlackScholesTask::~BlackScholesTask()
{
  // Free host memory
  freeHostMemory();

  // Free device memory
  if (mr_mode == None)
  {
    if (d_CallResultGPU != NULL)
      cudaFree(d_CallResultGPU);
    if (d_PutResultGPU != NULL)
      cudaFree(d_PutResultGPU);
    if (d_StockPrice != NULL)
      cudaFree(d_StockPrice);
    if (d_OptionStrike != NULL)
      cudaFree(d_OptionStrike);
    if (d_OptionYears != NULL)
      cudaFree(d_OptionYears);
  }
}

/**
 * @brief Allocates pinned or non pinned host memory for vectors
 *
 */
void BlackScholesTask::allocHostMemory(void)
{
  if (pinned)
  {
    err = cudaMallocHost((void **)&h_CallResultGPU, m_optN_SZ);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host matrix Call (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **)&h_PutResultGPU, m_optN_SZ);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host matrix Put (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **)&h_StockPrice, m_optN_SZ);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host matrix Price (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **)&h_OptionStrike, m_optN_SZ);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host matrix Strike (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **)&h_OptionYears, m_optN_SZ);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host matrix Years (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    h_CallResultGPU = (float *)malloc(m_optN_SZ);
    h_PutResultGPU = (float *)malloc(m_optN_SZ);
    h_StockPrice = (float *)malloc(m_optN_SZ);
    h_OptionStrike = (float *)malloc(m_optN_SZ);
    h_OptionYears = (float *)malloc(m_optN_SZ);

    // Verify that allocations succeeded
    if (h_CallResultGPU == NULL || h_PutResultGPU == NULL || h_StockPrice == NULL || h_OptionStrike == NULL || h_OptionYears == NULL)
    {
      fprintf(stderr, "Failed to allocate host vectors!\n");
      exit(EXIT_FAILURE);
    }
  }
  h_CallResultCPU = (float *)malloc(m_optN_SZ);
  h_PutResultCPU = (float *)malloc(m_optN_SZ);
}

/**
 * @brief Free vectors in host memory
 *
 */
void BlackScholesTask::freeHostMemory(void)
{
  // Free host memory
  free(h_CallResultCPU);
  free(h_PutResultCPU);

  if (pinned)
  {
    if (h_CallResultGPU != NULL)
      cudaFreeHost(h_CallResultGPU);
    if (h_PutResultGPU != NULL)
      cudaFreeHost(h_PutResultGPU);
    if (h_StockPrice != NULL)
      cudaFreeHost(h_StockPrice);
    if (h_OptionStrike != NULL)
      cudaFreeHost(h_OptionStrike);
    if (h_OptionYears != NULL)
      cudaFreeHost(h_OptionYears);
  }
  else
  {
    free(h_CallResultGPU);
    free(h_PutResultGPU);
    free(h_StockPrice);
    free(h_OptionStrike);
    free(h_OptionYears);
  }
}

/**
 * @brief Allocates device memory for vectors
 *
 */
void BlackScholesTask::allocDeviceMemory(void)
{
  err = cudaMalloc((void **)&d_CallResultGPU, m_optN_SZ);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector Call (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **)&d_PutResultGPU, m_optN_SZ);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector Put (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **)&d_StockPrice, m_optN_SZ);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector Price (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **)&d_OptionStrike, m_optN_SZ);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector Strike (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **)&d_OptionYears, m_optN_SZ);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector Years (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Allocates device memory for vectors
 *
 */
void BlackScholesTask::allocDeviceMemory(float *ptr,
                                         int *assignsA, int numAssignsA,
                                         int *assignsB, int numAssignsB,
                                         int *assignsC, int numAssignsC,
                                         int *assignsD, int numAssignsD,
                                         int *assignsE, int numAssignsE)
{

  d_ptr = ptr;

  numChipAssignmentsA = numAssignsA;
  chipAssignmentsA = (int *)malloc(numChipAssignmentsA * sizeof(int));
  d_CallResultGPU = (float *)((int *)d_ptr + assignsA[0] * chunkInts);
  for (int i = 0; i < numChipAssignmentsA; i++)
  {
    chipAssignmentsA[i] = assignsA[i] - assignsA[0];
  }

  numChipAssignmentsB = numAssignsB;
  chipAssignmentsB = (int *)malloc(numChipAssignmentsB * sizeof(int));
  d_PutResultGPU = (float *)((int *)d_ptr + assignsB[0] * chunkInts);
  for (int i = 0; i < numChipAssignmentsB; i++)
  {
    chipAssignmentsB[i] = assignsB[i] - assignsB[0];
  }

  numChipAssignmentsC = numAssignsC;
  chipAssignmentsC = (int *)malloc(numChipAssignmentsC * sizeof(int));
  d_StockPrice = (float *)((int *)d_ptr + assignsC[0] * chunkInts);
  for (int i = 0; i < numChipAssignmentsB; i++)
  {
    chipAssignmentsC[i] = assignsC[i] - assignsC[0];
  }

  numChipAssignmentsD = numAssignsD;
  chipAssignmentsD = (int *)malloc(numChipAssignmentsD * sizeof(int));
  d_OptionStrike = (float *)((int *)d_ptr + assignsD[0] * chunkInts);
  for (int i = 0; i < numChipAssignmentsD; i++)
  {
    chipAssignmentsD[i] = assignsD[i] - assignsD[0];
  }

  numChipAssignmentsE = numAssignsE;
  chipAssignmentsE = (int *)malloc(numChipAssignmentsE * sizeof(int));
  d_OptionYears = (float *)((int *)d_ptr + assignsE[0] * chunkInts);
  for (int i = 0; i < numChipAssignmentsE; i++)
  {
    chipAssignmentsE[i] = assignsE[i] - assignsE[0];
  }

  // printf("BlackScholes Task %p\n", ptr);
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
void BlackScholesTask::freeDeviceMemory(void)
{
  if (mr_mode != NULL)
  {
    if (d_ptr != NULL)
      cudaFree(d_ptr);
  }
  else
  {
    if (d_CallResultGPU != NULL)
      cudaFree(d_CallResultGPU);
    if (d_PutResultGPU != NULL)
      cudaFree(d_PutResultGPU);
    if (d_StockPrice != NULL)
      cudaFree(d_StockPrice);
    if (d_OptionStrike != NULL)
      cudaFree(d_OptionStrike);
    if (d_OptionYears != NULL)
      cudaFree(d_OptionYears);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
static float RandFloat(float low, float high)
{
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}

/**
 * @brief Generate random data for input vectors
 *
 */
void BlackScholesTask::dataGeneration(void)
{
  srand(5347);

  //Generate options set
  for (int i = 0; i < m_optN; i++)
  {
    h_CallResultCPU[i] = 0.0f;
    h_PutResultCPU[i] = -1.0f;
    h_StockPrice[i] = RandFloat(5.0f, 30.0f);
    h_OptionStrike[i] = RandFloat(1.0f, 100.0f);
    h_OptionYears[i] = RandFloat(0.25f, 10.0f);
  }
}

/**
 * @brief Asynchronous HTD memory transfer using stream
 *
 * @param stream
 */
void BlackScholesTask::memHostToDeviceAsync(cudaStream_t stream)
{

  cudaEventRecord(htdStart, stream);

  if (mr_mode == None)
  {
    err = cudaMemcpyAsync(d_StockPrice, h_StockPrice, m_optN_SZ, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix Price from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    err = cudaMemcpyAsync(d_OptionStrike, h_OptionStrike, m_optN_SZ, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix Strike from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    err = cudaMemcpyAsync(d_OptionYears, h_OptionYears, m_optN_SZ, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix Years from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
    memHostToDeviceAssigns();

  cudaEventRecord(htdEnd, stream);
}

/**
 * @brief Synchronous HTD memory transfer
 *
 */
void BlackScholesTask::memHostToDevice(void)
{

  cudaEventRecord(htdStart, 0);

  if (mr_mode == None)
  {
    err = cudaMemcpy(d_StockPrice, h_StockPrice, m_optN_SZ, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix Price from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_OptionStrike, h_OptionStrike, m_optN_SZ, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix Strike from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_OptionYears, h_OptionYears, m_optN_SZ, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix Years from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
    memHostToDeviceAssigns();

  cudaDeviceSynchronize();

  cudaEventRecord(htdEnd, 0);
}

void BlackScholesTask::memHostToDeviceAssigns(void)
{

  if (profile == EventsProf)
    cudaEventRecord(htdStart, 0);

  sendPageTables(d_CallResultGPU, chipAssignmentsA, numChipAssignmentsA);

  sendPageTables(d_PutResultGPU, chipAssignmentsB, numChipAssignmentsB);

  sendPageTables(d_StockPrice, chipAssignmentsC, numChipAssignmentsC);
  sendData(d_StockPrice, h_StockPrice, chipAssignmentsC, numChipAssignmentsC);

  sendPageTables(d_OptionStrike, chipAssignmentsD, numChipAssignmentsD);
  sendData(d_OptionStrike, h_OptionStrike, chipAssignmentsD, numChipAssignmentsD);

  sendPageTables(d_OptionYears, chipAssignmentsE, numChipAssignmentsE);
  sendData(d_OptionYears, h_OptionYears, chipAssignmentsE, numChipAssignmentsE);

  cudaDeviceSynchronize();

  if (profile == EventsProf)
    cudaEventRecord(htdEnd);
}

/**
 * @brief Asynchronous DTH memory transfer using stream
 *
 * @param stream
 */
void BlackScholesTask::memDeviceToHostAsync(cudaStream_t stream)
{

  cudaEventRecord(dthStart, stream);

  if (mr_mode == None)
  {
    err = cudaMemcpyAsync(h_CallResultGPU, d_CallResultGPU, m_optN_SZ, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix Call from device to host (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    err = cudaMemcpyAsync(h_PutResultGPU, d_PutResultGPU, m_optN_SZ, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix Put from device to host (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
    memDeviceToHostAssigns();

  cudaEventRecord(dthEnd, stream);
}

/**
 * @brief Synchronous HTD memory transfer
 *
 */
void BlackScholesTask::memDeviceToHost(void)
{

  cudaEventRecord(dthStart);

  if (mr_mode == None)
  {
    err = cudaMemcpy(h_CallResultGPU, d_CallResultGPU, m_optN_SZ, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix Call from device to host (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(h_PutResultGPU, d_PutResultGPU, m_optN_SZ, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix Put from device to host (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
    memDeviceToHostAssigns();

  cudaEventRecord(dthEnd);
}

void BlackScholesTask::memDeviceToHostAssigns(void)
{
  if (profile == EventsProf)
    cudaEventRecord(dthStart);

  receiveData(d_CallResultGPU, h_CallResultGPU, chipAssignmentsA, numChipAssignmentsA);
  receiveData(d_PutResultGPU, h_PutResultGPU, chipAssignmentsB, numChipAssignmentsB);

  if (profile == EventsProf)
    cudaEventRecord(dthEnd);
}

/**
 * @brief Launch vectorAdd kernel asynchronously using stream
 *
 * @param stream
 */
void BlackScholesTask::launchKernelAsync(cudaStream_t stream)
{

  if (m_numLaunchs > 0)
    if (cudaEventQuery(kernelEnd) != cudaSuccess)
      return;

  cudaEventRecord(kernelStart, stream);

  switch (exec_mode)
  {
  case Original:
    m_blocksPerGrid.y = m_numIter;
    BlackScholesGPU<<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_CallResultGPU, d_PutResultGPU, d_StockPrice, d_OptionStrike, d_OptionYears, m_optN, 1);
    break;
  case Persistent:
    BlackScholesGPUP<<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_CallResultGPU, d_PutResultGPU, d_StockPrice, d_OptionStrike, d_OptionYears, m_optN, m_numIter);
    break;
  case MemoryRanges:
    m_blocksPerGrid.y = m_numIter;
    BlackScholesGPUMR<<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_CallResultGPU, d_PutResultGPU, d_StockPrice, d_OptionStrike, d_OptionYears, m_optN, 1);
    break;
  case PersistentMemoryRanges:
    BlackScholesGPUPMR<<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_CallResultGPU, d_PutResultGPU, d_StockPrice, d_OptionStrike, d_OptionYears, m_optN, m_numIter);
    break;
  }

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch BlackScholes kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  m_numLaunchs++;

  // if (profile == EventsProf)
  cudaEventRecord(kernelEnd, stream);
}

/**
 * @brief Launch kernel synchronously
 *
 */
void BlackScholesTask::launchKernel(void)
{

  cudaEventRecord(kernelStart);

  switch (exec_mode)
  {
  case Original:
    m_blocksPerGrid.y = m_numIter;
    BlackScholesGPU<<<m_blocksPerGrid, m_threadsPerBlock>>>(d_CallResultGPU, d_PutResultGPU, d_StockPrice, d_OptionStrike, d_OptionYears, m_optN, 1);
    break;
  case Persistent:
    BlackScholesGPUP<<<m_blocksPerGrid, m_threadsPerBlock>>>(d_CallResultGPU, d_PutResultGPU, d_StockPrice, d_OptionStrike, d_OptionYears, m_optN, m_numIter);
    break;
  case MemoryRanges:
    m_blocksPerGrid.y = m_numIter;
    BlackScholesGPUMR<<<m_blocksPerGrid, m_threadsPerBlock>>>(d_CallResultGPU, d_PutResultGPU, d_StockPrice, d_OptionStrike, d_OptionYears, m_optN, 1);
    break;
  case PersistentMemoryRanges:
    BlackScholesGPUPMR<<<m_blocksPerGrid, m_threadsPerBlock>>>(d_CallResultGPU, d_PutResultGPU, d_StockPrice, d_OptionStrike, d_OptionYears, m_optN, m_numIter);
    break;
  }

  m_numLaunchs++;

  cudaDeviceSynchronize();

  cudaEventRecord(kernelEnd);

  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch BlackScholes kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
static double CND(double d)
{
  const double A1 = 0.31938153;
  const double A2 = -0.356563782;
  const double A3 = 1.781477937;
  const double A4 = -1.821255978;
  const double A5 = 1.330274429;
  const double RSQRT2PI = 0.39894228040143267793994605993438;

  double
      K = 1.0 / (1.0 + 0.2316419 * fabs(d));

  double
      cnd = RSQRT2PI * exp(-0.5 * d * d) *
            (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0)
    cnd = 1.0 - cnd;

  return cnd;
}

///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
static void BlackScholesBodyCPU(
    float &callResult,
    float &putResult,
    float Sf, //Stock price
    float Xf, //Option strike
    float Tf, //Option years
    float Rf, //Riskless rate
    float Vf  //Volatility rate
)
{
  double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

  double sqrtT = sqrt(T);
  double d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
  double d2 = d1 - V * sqrtT;
  double CNDD1 = CND(d1);
  double CNDD2 = CND(d2);

  //Calculate Call and Put simultaneously
  double expRT = exp(-R * T);
  callResult = (float)(S * CNDD1 - X * expRT * CNDD2);
  putResult = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options
////////////////////////////////////////////////////////////////////////////////
void BlackScholesCPU(
    float *h_CallResult,
    float *h_PutResult,
    float *h_StockPrice,
    float *h_OptionStrike,
    float *h_OptionYears,
    int optN)
{
  for (int opt = 0; opt < optN; opt++)
    BlackScholesBodyCPU(
        h_CallResult[opt],
        h_PutResult[opt],
        h_StockPrice[opt],
        h_OptionStrike[opt],
        h_OptionYears[opt],
        Riskfree,
        Volatility);
}

/**
 * @brief Verify the correcteness of the BlackScholes kernel
 *
 * @return true Test passed
 * @return false Test failed
 */
bool BlackScholesTask::checkResults(void)
{
  bool correct = true;

  BlackScholesCPU(
      h_CallResultCPU,
      h_PutResultCPU,
      h_StockPrice,
      h_OptionStrike,
      h_OptionYears,
      m_optN);

  //Calculate max absolute difference and L1 distance
  //between CPU and GPU results
  double eps = 1.e-6; // machine zero
  double sum_delta = 0, sum_ref = 0, max_delta = 0;

  for (int i = 0; i < m_optN; i++)
  {
    double ref = h_CallResultCPU[i];
    double delta = fabs(h_CallResultCPU[i] - h_CallResultGPU[i]);

    if (delta > max_delta)
    {
      max_delta = delta;
    }

    sum_delta += delta;
    sum_ref += fabs(ref);
  }

  double L1norm = sum_delta / sum_ref;

  if (L1norm > eps)
  {
    correct = false;
    printf("L1 norm: %E\n", L1norm);
    printf("Max absolute error: %E\n\n", max_delta);
  }
  return (correct);
}
