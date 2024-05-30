/**
 * @file matrixMulTask.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief
 * @version 0.1
 * @date 2021-05-12
 *
 */
#include "matrixMul.h"
#include "matrixMulKernel.cuh"

/**
 * @brief Construct a new matrixMul task with default values
 *
 */
matrixMulTask::matrixMulTask()
{

  setTaskName(matrixMul);

  h_A = NULL;
  h_B = NULL;
  h_C = NULL;
  d_A = NULL;
  d_B = NULL;
  d_C = NULL;

  setExecMode(PersistentMemoryRanges);
  setMRMode(None);

  m_block_size = 32;
  dim3 dimsA(2048, 2048, 1);
  dim3 dimsB(2048, 2048, 1);
  setMatrixDims(dimsA, dimsB);
  setPinned(false);

  chunkSize = 1024; // TODO: read this value from parameters
  chunkInts = chunkSize / sizeof(int);
  m_numLaunchs = 0;
  err = cudaSuccess;
}

/**
 * @brief Construct a new matrixMul task
 *
 * @param n
 */
matrixMulTask::matrixMulTask(dim3 &dA, dim3 &dB)
{

  setTaskName(matrixMul);

  h_A = NULL;
  h_B = NULL;
  h_C = NULL;
  d_A = NULL;
  d_B = NULL;
  d_C = NULL;

  setExecMode(PersistentMemoryRanges);
  setMRMode(None);

  m_block_size = 32;
  setMatrixDims(dA, dB);
  setPinned(false);
  chunkSize = 1024; // TODO: read this value from parameters
  chunkInts = chunkSize / sizeof(int);
  m_numLaunchs = 0;
  err = cudaSuccess;
}

CUDAtaskNames matrixMulTask::getName() { return name; }

void matrixMulTask::setMatrixDims(dim3 &dimsA, dim3 &dimsB)
{
  m_dimsA = dimsA;
  setSizeA();
  m_dimsB = dimsB;
  setSizeB();
  m_dimsC.x = dimsB.x;
  m_dimsC.y = dimsA.y;
  setSizeC();
  m_block_size = 16;
  dim3 t(m_block_size, m_block_size, 1);
  setThreadsPerBlock(t);
  dim3 b((dimsB.x / t.x), (dimsA.y / t.y), 1);
  setBlocksPerGrid(b);
  // printf("Dims %d x %d - %d x %d - threads %d x %d - blocks %d\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y, t.x, t.y, b.x);
}

void matrixMulTask::setPersistentBlocks()
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
    if (m_block_size == 16)
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMulti, matrixMulCUDAP<16>, m_threadsPerBlock.x * m_threadsPerBlock.y, 0);
    else
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMulti, matrixMulCUDAP<32>, m_threadsPerBlock.x * m_threadsPerBlock.y, 0);
    b.x = maxBlocksPerMulti * deviceProp.multiProcessorCount;
    setBlocksPerGrid(b);
    break;
  case MemoryRanges:
    printf("Original kernel cannot use persistent blocks\n");
    break;
  case PersistentMemoryRanges:
    if (m_block_size == 16)
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMulti, matrixMulCUDAPMR<16>, m_threadsPerBlock.x * m_threadsPerBlock.y, 0);
    else
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMulti, matrixMulCUDAPMR<32>, m_threadsPerBlock.x * m_threadsPerBlock.y, 0);
    b.x = maxBlocksPerMulti * deviceProp.multiProcessorCount;
    setBlocksPerGrid(b);
    break;
  }
}

void matrixMulTask::setAllModules(int reverse)
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
matrixMulTask::~matrixMulTask()
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
void matrixMulTask::allocHostMemory(void)
{

  if (pinned)
  {
    err = cudaMallocHost((void **)&h_A, m_size_A);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host matrix A (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **)&h_B, m_size_B);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host matrix B (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **)&h_C, m_size_C);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host matrix C (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    h_A = (float *)malloc(m_size_A);
    h_B = (float *)malloc(m_size_B);
    h_C = (float *)malloc(m_size_C);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
      fprintf(stderr, "Failed to allocate host matrices!\n");
      exit(EXIT_FAILURE);
    }
  }
}

/**
 * @brief Free vectors in host memory
 *
 */
void matrixMulTask::freeHostMemory(void)
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
void matrixMulTask::allocDeviceMemory(void)
{

  err = cudaMalloc((void **)&d_A, m_size_A);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device matrix A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMalloc((void **)&d_B, m_size_B);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device matrix B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMalloc((void **)&d_C, m_size_C);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device matrix C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Allocates device memory for vectors
 *
 */
void matrixMulTask::allocDeviceMemory(float *ptr, int *assignsA, int numAssignsA, int *assignsB, int numAssignsB, int *assignsC, int numAssignsC)
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

  // printf("MatrixMul Task %p\n", ptr);
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
void matrixMulTask::freeDeviceMemory(void)
{
  if (d_A != NULL)
    cudaFree(d_A);
  if (d_B != NULL)
    cudaFree(d_B);
  if (d_C != NULL)
    cudaFree(d_C);
}

/**
 * @brief Generate data for input matrices
 *
 */
void matrixMulTask::dataGeneration(void)
{
  for (int i = 0; i < m_dimsA.x * m_dimsA.y; ++i)
    h_A[i] = 1.0f;

  for (int i = 0; i < m_dimsB.x * m_dimsB.y; ++i)
    h_B[i] = 0.01f;
}

/**
 * @brief Asynchronous HTD memory transfer using stream
 *
 * @param stream
 */
void matrixMulTask::memHostToDeviceAsync(cudaStream_t stream)
{
  if (profile == EventsProf)
    cudaEventRecord(htdStart, stream);

  if (mr_mode == None)
  {
    err = cudaMemcpyAsync(d_A, h_A, m_size_A, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix A from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    err = cudaMemcpyAsync(d_B, h_B, m_size_B, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix B from host to device (error code %s)!\n",
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
void matrixMulTask::memHostToDevice(void)
{

  if (profile == EventsProf)
    cudaEventRecord(htdStart);

  if (mr_mode == None)
  {
    err = cudaMemcpy(d_A, h_A, m_size_A, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix A from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_B, h_B, m_size_B, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix B from host to device (error code %s)!\n",
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

void matrixMulTask::memHostToDeviceAssigns(void)
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
void matrixMulTask::memDeviceToHostAsync(cudaStream_t stream)
{

  cudaEventRecord(dthStart, stream);

  if (mr_mode == None)
  {
    err = cudaMemcpyAsync(h_C, d_C, m_size_C, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix C from device to host (error code %s)!\n",
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
void matrixMulTask::memDeviceToHost(void)
{

  cudaEventRecord(dthStart);

  if (mr_mode == None)
  {
    err = cudaMemcpy(h_C, d_C, m_size_C, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix C from device to host (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
    memDeviceToHostAssigns();

  if (profile == EventsProf)
    cudaEventRecord(dthEnd);
}

void matrixMulTask::memDeviceToHostAssigns(void)
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
void matrixMulTask::launchKernelAsync(cudaStream_t stream)
{

  if (m_numLaunchs > 0)
    if (cudaEventQuery(kernelEnd) != cudaSuccess)
      return;

  cudaEventRecord(kernelStart, stream);

  switch (exec_mode)
  {
  case Original:
    if (m_block_size == 16)
      matrixMulCUDA<16><<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    else
      matrixMulCUDA<32><<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    break;
  case Persistent:
    if (m_block_size == 16)
      matrixMulCUDAP<16><<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    else
      matrixMulCUDAP<32><<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    break;
  case MemoryRanges:
    if (m_block_size == 16)
      matrixMulCUDAMR<16><<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    else
      matrixMulCUDAMR<32><<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    break;
  case PersistentMemoryRanges:
    if (m_block_size == 16)
      matrixMulCUDAPMR<16><<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    else
      matrixMulCUDAPMR<32><<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    break;
  }

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch matrixMul kernel (error code %s)!\n",
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
void matrixMulTask::launchKernel(void)
{

  cudaEventRecord(kernelStart);

  switch (exec_mode)
  {
  case Original:
    if (m_block_size == 16)
      matrixMulCUDA<16><<<m_blocksPerGrid, m_threadsPerBlock>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    else
      matrixMulCUDA<32><<<m_blocksPerGrid, m_threadsPerBlock>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    break;
  case Persistent:
    if (m_block_size == 16)
      matrixMulCUDAP<16><<<m_blocksPerGrid, m_threadsPerBlock>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    else
      matrixMulCUDAP<32><<<m_blocksPerGrid, m_threadsPerBlock>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    break;
  case MemoryRanges:
    if (m_block_size == 16)
      matrixMulCUDAMR<16><<<m_blocksPerGrid, m_threadsPerBlock>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    else
      matrixMulCUDAMR<32><<<m_blocksPerGrid, m_threadsPerBlock>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    break;
  case PersistentMemoryRanges:
    if (m_block_size == 16)
      matrixMulCUDAPMR<16><<<m_blocksPerGrid, m_threadsPerBlock>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    else
      matrixMulCUDAPMR<32><<<m_blocksPerGrid, m_threadsPerBlock>>>(d_C, d_A, d_B, m_dimsA, m_dimsB);
    break;
  }

  m_numLaunchs++;
  cudaDeviceSynchronize();

  cudaEventRecord(kernelEnd);

  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch matrixMul kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Verify the correcteness of the matrixMul kernel
 *
 * @return true Test passed
 * @return false Test failed
 */
bool matrixMulTask::checkResults(void)
{
  bool correct = true;
  const float valB = 0.01f;

  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-6; // machine zero

  for (int i = 0; i < (int)(m_dimsC.x * m_dimsC.y); i++)
  {
    double abs_err = fabs(h_C[i] - (m_dimsA.x * valB));
    double dot_length = m_dimsA.x;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;

    if (rel_err > eps)
    {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], m_dimsA.x * valB, eps);
      correct = false;
      return correct;
    }
  }
  return correct;
}