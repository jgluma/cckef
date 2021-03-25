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
vectorAddTask::vectorAddTask() {

  setTaskName(VA);

  h_A = NULL;
  h_B = NULL;
  h_C = NULL;
  d_A = NULL;
  d_B = NULL;
  d_C = NULL;

  setMRMode(None);
  setNumElements(50000);
  setPinned(false);
  err = cudaSuccess;
}

/**
 * @brief Construct a new vectorAdd task with vectors of n elements
 *
 * @param n
 */
vectorAddTask::vectorAddTask(unsigned long n, bool p) {

  setTaskName(VA);

  h_A = NULL;
  h_B = NULL;
  h_C = NULL;
  d_A = NULL;
  d_B = NULL;
  d_C = NULL;

  setNumElements(n);
  setPinned(p);
  err = cudaSuccess;
}

CUDAtaskNames vectorAddTask::getName() { return name; }

void vectorAddTask::setNumElements(unsigned long n) {
  numElements = n;
  size = numElements * sizeof(float);

  printf("VA task: %d elements, with size %lu bytes\n", numElements, size);
  // Memory range parameters
  if ( mr_mode != None ) {
    FullRowElements = FullRowSize / sizeof(float);
    ChunkSize = ChipRowSize; 
    numChunks = floor( size / ChunkSize );
    LastChunkSize = size - numChunks * ChunkSize;
    size = numChunks * FullRowSize;
    if ( LastChunkSize > 0 )
      size += FullRowSize;
    ChunkElements = ChunkSize / sizeof(float);
    printf("%d - %d - %d - %d - %d - %d\n", numElements, size, numChunks, ChunkSize, LastChunkSize, FullRowElements);  
  }

  setThreadsPerBlock(256);
  setBlocksPerGrid((numElements+threadsPerBlock-1)/threadsPerBlock);
}

/**
 * @brief Destroy the vectorAdd task object
 *
 */
vectorAddTask::~vectorAddTask() {
  // Free host memory
  if (pinned) {
    if (h_A != NULL)
      cudaFreeHost(h_A);
    if (h_B != NULL)
      cudaFreeHost(h_B);
    if (h_C != NULL)
      cudaFreeHost(h_C);
  } else {
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
void vectorAddTask::allocHostMemory(void) {
  if (pinned) {
    err = cudaMallocHost((void **)&h_A, size);
    if (err != cudaSuccess) {
      fprintf(stderr,
              "Failed to allocate pinned host vector A (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **)&h_B, size);
    if (err != cudaSuccess) {
      fprintf(stderr,
              "Failed to allocate pinned host vector B (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMallocHost((void **)&h_C, size);
    if (err != cudaSuccess) {
      fprintf(stderr,
              "Failed to allocate pinned host vector C (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  } else {
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
      fprintf(stderr, "Failed to allocate host vectors!\n");
      exit(EXIT_FAILURE);
    }
  }
}

/**
 * @brief Free vectors in host memory
 *
 */
void vectorAddTask::freeHostMemory(void) {
  if (pinned) {
    if (h_A != NULL)
      cudaFreeHost(h_A);
    if (h_B != NULL)
      cudaFreeHost(h_B);
    if (h_C != NULL)
      cudaFreeHost(h_C);
  } else {
    free(h_A);
    free(h_B);
    free(h_C);
  }
}

/**
 * @brief Allocates device memory for vectors
 *
 */
void vectorAddTask::allocDeviceMemory(void) {
  // Allocate the device input vector A
  err = cudaMalloc((void **)&d_A, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device input vector B
  err = cudaMalloc((void **)&d_B, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Allocate the device output vector C
  err = cudaMalloc((void **)&d_C, size);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Free vectors in device memory
 *
 */
void vectorAddTask::freeDeviceMemory(void) {
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
void vectorAddTask::dataGeneration(void) {
  for (int i = 0; i < numElements; ++i) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }
}

/**
 * @brief Asynchronous HTD memory transfer using stream
 *
 * @param stream
 */
void vectorAddTask::memHostToDeviceAsync(cudaStream_t stream) {

  if ( profile == EventsProf )
    cudaEventRecord( htdStart, stream);

  if ( mr_mode == None ) {
    err = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      fprintf(stderr,
              "Failed to copy vector A from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    err = cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      fprintf(stderr,
              "Failed to copy vector B from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else { // Memory range mode
    int ChunkOff = 0, RowOff = 0;
    for ( int i = 0; i < numChunks; i++ ) {
      err = cudaMemcpyAsync(d_A + RowOff, h_A + ChunkOff, ChunkSize, cudaMemcpyHostToDevice, stream);
      if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy vector A from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
      err = cudaMemcpyAsync(d_B + RowOff, h_B + ChunkOff, ChunkSize, cudaMemcpyHostToDevice, stream);
      if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy vector B from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
      ChunkOff += ChunkElements;
      RowOff += FullRowElements;
      if ( i == 1 )
        printf("%d %d %p %p %p %p\n", ChunkOff, RowOff, d_A, d_A + RowOff, h_A, h_A + ChunkOff);
    }
  }

  if ( profile == EventsProf )
    cudaEventRecord( htdEnd, stream);

}

/**
 * @brief Synchronous HTD memory transfer
 *
 */
void vectorAddTask::memHostToDevice(void) {

  if ( profile == EventsProf )
    cudaEventRecord( htdStart, 0);

  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector A from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector B from host to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  if ( profile == EventsProf )
    cudaEventRecord( htdEnd);

}

/**
 * @brief Asynchronous DTH memory transfer using stream
 *
 * @param stream
 */
void vectorAddTask::memDeviceToHostAsync(cudaStream_t stream) {

  if ( profile == EventsProf )
    cudaEventRecord( htdStart, stream); 

  if ( mr_mode == None ) {
    printf("Copy %d bytes\n", size);
    err = cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
      fprintf(stderr,
              "Failed to copy vector C from device to host (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }
  else { // Memory range mode
    int ChunkOff = 0, RowOff = 0;
    for ( int i = 0; i < numChunks; i++ ) {
      err = cudaMemcpyAsync(h_C + ChunkOff, d_C + RowOff, ChunkSize, cudaMemcpyDeviceToHost, stream);
      if (err != cudaSuccess) {
        fprintf(stderr,
                "Failed to copy vector A from device to host (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
      }
      ChunkOff += ChunkElements;
      RowOff += FullRowElements;
    }
  }

  if ( profile == EventsProf )
    cudaEventRecord( htdEnd, stream);

}

/**
 * @brief Synchronous HTD memory transfer
 *
 */
void vectorAddTask::memDeviceToHost(void) {
  if ( profile == EventsProf )
    cudaEventRecord( dthStart);

  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  if ( profile == EventsProf )
    cudaEventRecord( dthEnd);

}

/**
 * @brief Launch vectorAdd kernel asynchronously using stream
 *
 * @param stream
 */
void vectorAddTask::launchKernelAsync(cudaStream_t stream) {

  if ( profile == EventsProf )
    cudaEventRecord( kernelStart, stream);

  if ( mr_mode == None )
  {
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, kk /*numElements*/);
  }
  else
  {
    printf("MR launched\n");
    vectorAddMR<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, numElements, ChunkElements, FullRowElements);
  }

  if ( profile == EventsProf )
    cudaEventRecord( kernelEnd, stream);
  
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

}

/**
 * @brief Launch vectorAdd kernel synchronously
 *
 */
void vectorAddTask::launchKernel(void) {

  if ( profile == EventsProf )
    cudaEventRecord( kernelStart);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

  if ( profile == EventsProf )
    cudaEventRecord( kernelEnd);

    err = cudaGetLastError();

  if (err != cudaSuccess) {
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
bool vectorAddTask::checkResults(void) {
  for (int i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d! CPU %f GPU %f\n", i, h_A[i] + h_B[i], h_C[i]);
      return false;
    }
  }
  return true;
}