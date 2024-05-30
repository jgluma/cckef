/**
 * @file PathFinderTask.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief 
 * @version 0.1
 * @date 2021-05-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "PathFinder.h"
#include "PathFinder_kernel.cuh"

/**
 * @brief Construct a new PathFinder task with default values
 *
 */
PathFinderTask::PathFinderTask()
{

  setTaskName(PF);

  h_data = NULL;
  h_result = NULL;

  d_result[0] = NULL;
  d_result[1] = NULL;
  d_wall = NULL;

  setExecMode(PersistentMemoryRanges);
  setMRMode(None);

  // setParameters(500, 30000, 126);
  setParameters(512, 16384, 126);

  chunkSize = 1024; // TODO: read this value from parameters
  chunkInts = chunkSize / sizeof(int);

  setComputeAssigns(true);
  setPinned(false);

  m_numLaunchs = 0;
  err = cudaSuccess;
}

/**
 * @brief Construct a new PathFinder task
 *
 * @param n
 */
PathFinderTask::PathFinderTask(int rows, int cols, int height)
{

  setTaskName(PF);

  h_data = NULL;
  h_result = NULL;

  d_result[0] = NULL;
  d_result[1] = NULL;
  d_wall = NULL;

  setExecMode(PersistentMemoryRanges);
  setMRMode(None);

  setParameters(rows, cols, height);

  chunkSize = 1024; // TODO: read this value from parameters
  chunkInts = chunkSize / sizeof(int);

  setComputeAssigns(true);
  setPinned(false);

  m_numLaunchs = 0;
  err = cudaSuccess;
}

CUDAtaskNames PathFinderTask::getName() { return name; }

void PathFinderTask::setParameters(int rows, int cols, int height)
{
  setRows(rows);
  setCols(cols);
  setHeight(height);
  m_datasize = rows * cols * sizeof(int);
  m_resultsize = cols * sizeof(int);

  dim3 t(256, 1, 1);
  setThreadsPerBlock(t);

  m_borderCols = m_pyramid_height * HALO;
  m_smallBlockCol = m_threadsPerBlock.x - m_pyramid_height * HALO * 2;
  m_blockCols = m_cols / m_smallBlockCol + ((m_cols % m_smallBlockCol == 0) ? 0 : 1);
  dim3 b(m_blockCols, 1, 1);
  setBlocksPerGrid(b);
}

void PathFinderTask::setPersistentBlocks()
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
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMulti, pathFinderGPUP<256>, m_threadsPerBlock.x, 0);
    b.x = maxBlocksPerMulti * deviceProp.multiProcessorCount;
    setBlocksPerGrid(b);
    break;
  case MemoryRanges:
    printf("Original kernel cannot use persistent blocks\n");
    break;
  case PersistentMemoryRanges:
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerMulti, pathFinderGPUPMR<256>, m_threadsPerBlock.x, 0);
    b.x = maxBlocksPerMulti * deviceProp.multiProcessorCount;
    setBlocksPerGrid(b);
    break;
  }
}

void PathFinderTask::setAllModules(int reverse)
{
  m_numCombinations = 14;
  allModulesA = (int **)malloc(m_numCombinations * sizeof(int *));
  allModulesB = (int **)malloc(m_numCombinations * sizeof(int *));
  allModulesC = (int **)malloc(m_numCombinations * sizeof(int *));
  // First, combinations where A and B share the modules
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
  for (int i = 12; i >= 2; i -= 2)
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
 * @brief Destroy the PathFinder task object
 *
 */
PathFinderTask::~PathFinderTask()
{
  // Free host memory
  freeHostMemory();

  // Free device memory
  if (mr_mode == None)
  {
    if (d_result != NULL)
      cudaFree(d_result);
    if (d_wall != NULL)
      cudaFree(d_wall);
  }
}

/**
 * @brief Allocates pinned or non pinned host memory for vectors
 *
 */
void PathFinderTask::allocHostMemory(void)
{
  if (pinned)
  {
    err = cudaMallocHost((void **)&h_data, m_datasize);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host matrix data (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **)&h_result, m_resultsize);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to allocate pinned host matrix result (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    h_data = (int *)malloc(m_datasize);
    h_result = (int *)malloc(m_resultsize);

    // Verify that allocations succeeded
    if (h_data == NULL || h_result == NULL)
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
void PathFinderTask::freeHostMemory(void)
{
  // Free host memory

  if (pinned)
  {
    if (h_data != NULL)
      cudaFreeHost(h_data);
    if (h_result != NULL)
      cudaFreeHost(h_result);
  }
  else
  {
    free(h_data);
    free(h_result);
  }
}

/**
 * @brief Allocates device memory for vectors
 *
 */
void PathFinderTask::allocDeviceMemory(void)
{
  // if (mr_mode == None)
  // {
  err = cudaMalloc((void **)&d_wall, m_datasize - m_resultsize);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector wall (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **)&d_result[0], m_resultsize);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector result 0 (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **)&d_result[1], m_resultsize);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to allocate device vector result 1 (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  // }
}

/**
 * @brief Allocates device memory for vectors
 *
 */
void PathFinderTask::allocDeviceMemory(float *ptr,
                                       int *assignsA, int numAssignsA,
                                       int *assignsB, int numAssignsB,
                                       int *assignsC, int numAssignsC)
{
  d_ptr = ptr;

  numChipAssignmentsA = numAssignsA;
  chipAssignmentsA = (int *)malloc(numChipAssignmentsA * sizeof(int));
  d_wall = (int *)((int *)d_ptr + assignsA[0] * chunkInts);
  for (int i = 0; i < numChipAssignmentsA; i++)
  {
    chipAssignmentsA[i] = assignsA[i] - assignsA[0];
  }

  numChipAssignmentsB = numAssignsB;
  chipAssignmentsB = (int *)malloc(numChipAssignmentsB * sizeof(int));
  d_result[0] = (int *)((int *)d_ptr + assignsB[0] * chunkInts);
  for (int i = 0; i < numChipAssignmentsB; i++)
  {
    chipAssignmentsB[i] = assignsB[i] - assignsB[0];
  }

  numChipAssignmentsC = numAssignsC;
  chipAssignmentsC = (int *)malloc(numChipAssignmentsC * sizeof(int));
  d_result[1] = (int *)((int *)d_ptr + assignsC[0] * chunkInts);
  for (int i = 0; i < numChipAssignmentsB; i++)
  {
    chipAssignmentsC[i] = assignsC[i] - assignsC[0];
  }
}

/**
 * @brief Free vectors in device memory
 *
 */
void PathFinderTask::freeDeviceMemory(void)
{
  if (mr_mode != NULL)
  {
    if (d_ptr != NULL)
      cudaFree(d_ptr);
  }
  else
  {
    if (d_wall != NULL)
      cudaFree(d_wall);
    if (d_result[0] != NULL)
      cudaFree(d_result[0]);
    if (d_result[1] != NULL)
      cudaFree(d_result[1]);
  }
}

/**
 * @brief Generate random data for input vectors
 *
 */
void PathFinderTask::dataGeneration(void)
{
  srand(5347);
  //Generate options set
  for (int i = 0; i < m_rows; i++)
    for (int j = 0; j < m_cols; j++)
      h_data[i * m_cols + j] = rand() % 10;
}

/**
 * @brief Asynchronous HTD memory transfer using stream
 *
 * @param stream
 */
void PathFinderTask::memHostToDeviceAsync(cudaStream_t stream)
{

  cudaEventRecord(htdStart, stream);

  // if (mr_mode == None)
  // {
  err = cudaMemcpyAsync(d_result[0], h_data, m_resultsize, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess)
  {
    fprintf(stderr,
            "Failed to copy matrix result 0 from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpyAsync(d_wall, h_data + m_cols, m_datasize - m_resultsize, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess)
  {
    fprintf(stderr,
            "Failed to copy matrix data from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  // }
  // else
  //   memHostToDeviceAssigns();

  cudaEventRecord(htdEnd, stream);
}

/**
 * @brief Synchronous HTD memory transfer
 *
 */
void PathFinderTask::memHostToDevice(void)
{

  cudaEventRecord(htdStart, 0);

  if (mr_mode == None)
  {
    err = cudaMemcpy(d_result[0], h_data, m_resultsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix result from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_wall, h_data + m_cols, m_datasize - m_resultsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix data from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
    memHostToDeviceAssigns();

  cudaDeviceSynchronize();

  cudaEventRecord(htdEnd, 0);
}

void PathFinderTask::memHostToDeviceAssigns(void)
{

  if (profile == EventsProf)
    cudaEventRecord(htdStart, 0);

  sendPageTables(d_wall, chipAssignmentsA, numChipAssignmentsA);
  sendData(d_wall, h_data + m_cols, chipAssignmentsA, numChipAssignmentsA);

  sendPageTables(d_result[0], chipAssignmentsB, numChipAssignmentsB);
  sendData(d_result[0], h_data, chipAssignmentsB, numChipAssignmentsB);

  sendPageTables(d_result[1], chipAssignmentsC, numChipAssignmentsC);

  cudaDeviceSynchronize();

  if (profile == EventsProf)
    cudaEventRecord(htdEnd);
}

/**
 * @brief Asynchronous DTH memory transfer using stream
 *
 * @param stream
 */
void PathFinderTask::memDeviceToHostAsync(cudaStream_t stream)
{

  cudaEventRecord(dthStart, stream);

  if (mr_mode == None)
  {
    err = cudaMemcpyAsync(h_result, d_result[m_final_ret], m_resultsize, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix result from device to host (error code %s)!\n",
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
void PathFinderTask::memDeviceToHost(void)
{

  cudaEventRecord(dthStart);

  if (mr_mode == None)
  {
    err = cudaMemcpy(h_result, d_result[m_final_ret], m_resultsize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
      fprintf(stderr,
              "Failed to copy matrix Call from device to host (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else
    memDeviceToHostAssigns();

  cudaEventRecord(dthEnd);
}

void PathFinderTask::memDeviceToHostAssigns(void)
{
  if (profile == EventsProf)
    cudaEventRecord(dthStart);

  if (m_final_ret == 0)
    receiveData(d_result[m_final_ret], h_result, chipAssignmentsB, numChipAssignmentsB);
  else
    receiveData(d_result[m_final_ret], h_result, chipAssignmentsC, numChipAssignmentsC);

  if (profile == EventsProf)
    cudaEventRecord(dthEnd);
}

/**
 * @brief Launch vectorAdd kernel asynchronously using stream
 *
 * @param stream
 */
void PathFinderTask::launchKernelAsync(cudaStream_t stream)
{

  if (m_numLaunchs > 0)
    if (cudaEventQuery(kernelEnd) != cudaSuccess)
      return;

  cudaEventRecord(kernelStart, stream);

  int src = 1, dst = 0;
  for (int t = 0; t < m_rows - 1; t += m_pyramid_height)
  {
    int temp = src;
    src = dst;
    dst = temp;
    switch (exec_mode)
    {
    case Original:
      pathFinderGPU<256><<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(
          MIN(m_pyramid_height, m_rows - t - 1),
          d_wall, d_result[src], d_result[dst],
          m_cols, m_rows, t, m_borderCols);
      break;
    case Persistent:
      pathFinderGPUP<256><<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(
          MIN(m_pyramid_height, m_rows - t - 1),
          d_wall, d_result[src], d_result[dst],
          m_cols, m_rows, t, m_borderCols, m_blockCols);
      break;
    case MemoryRanges:
      pathFinderGPUMR<256><<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(
          MIN(m_pyramid_height, m_rows - t - 1),
          d_wall, d_result[src], d_result[dst],
          m_cols, m_rows, t, m_borderCols);
      break;
    case PersistentMemoryRanges:
      pathFinderGPUPMR<256><<<m_blocksPerGrid, m_threadsPerBlock, 0, stream>>>(
          MIN(m_pyramid_height, m_rows - t - 1),
          d_wall, d_result[src], d_result[dst],
          m_cols, m_rows, t, m_borderCols, m_blockCols);
      break;
    }
  }
  m_final_ret = dst;

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch PathFinder kernel (error code %s)!\n",
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
void PathFinderTask::launchKernel(void)
{

  cudaEventRecord(kernelStart);

  int src = 1, dst = 0;
  for (int t = 0; t < m_rows - 1; t += m_pyramid_height)
  {
    int temp = src;
    src = dst;
    dst = temp;
    switch (exec_mode)
    {
    case Original:
      pathFinderGPU<256><<<m_blocksPerGrid, m_threadsPerBlock>>>(
          MIN(m_pyramid_height, m_rows - t - 1),
          d_wall, d_result[src], d_result[dst],
          m_cols, m_rows, t, m_borderCols);
      break;
    case Persistent:
      pathFinderGPUP<256><<<m_blocksPerGrid, m_threadsPerBlock>>>(
          MIN(m_pyramid_height, m_rows - t - 1),
          d_wall, d_result[src], d_result[dst],
          m_cols, m_rows, t, m_borderCols, m_blockCols);
      break;
    case MemoryRanges:
      pathFinderGPUMR<256><<<m_blocksPerGrid, m_threadsPerBlock>>>(
          MIN(m_pyramid_height, m_rows - t - 1),
          d_wall, d_result[src], d_result[dst],
          m_cols, m_rows, t, m_borderCols);
      break;
    case PersistentMemoryRanges:
      pathFinderGPUPMR<256><<<m_blocksPerGrid, m_threadsPerBlock>>>(
          MIN(m_pyramid_height, m_rows - t - 1),
          d_wall, d_result[src], d_result[dst],
          m_cols, m_rows, t, m_borderCols, m_blockCols);
      break;
    }
  }
  m_final_ret = dst;

  m_numLaunchs++;

  cudaDeviceSynchronize();

  cudaEventRecord(kernelEnd);

  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed to launch PathFinder kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Verify the correcteness of the PathFinder kernel
 *
 * @return true Test passed
 * @return false Test failed
 */
bool PathFinderTask::checkResults(void)
{
  bool correct = true;

  return (correct);
}
