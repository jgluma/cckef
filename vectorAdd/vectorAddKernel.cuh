/**
 * @file vectorAddKernel.cuh
 * @author José María González Linares (jgl@uma.es)
 * @brief Kernel for vector addition from CUDA samples
 * @version 0.1
 * @date 2021-02-15
 * 
 */

/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    i = threadIdx.x;
    C[i] = A[i+numElements*256];

//    if ( i < numElements)
    {
//        C[i] = A[i] + B[i];
    }
}

/* Memory ranges version */

__global__ void
vectorAddMR(const float *A, const float *B, float *C, int numElements, int ChunkElements, int FullRowElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < numElements)
    {
        int ChunkIdx = i/ChunkElements;
        int ChunkOff = i%ChunkElements;
        i = FullRowElements*ChunkIdx + ChunkOff;
        C[i] = A[i] + B[i];
    }
}
