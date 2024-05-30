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
vectorAdd(const float *A, const float *B, float *C, int numElements, int numIterations)
{
    int iter = 0;
    while (iter < numIterations)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if (i < numElements)
            C[i] = A[i] + B[i];

        iter++;
    }
}

/* Persistent blocks version */

__global__ void
vectorAddP(const float *A, const float *B, float *C, int numElements, int numIterations)
{
    int nTasks = (numElements + blockDim.x - 1) / blockDim.x;
    int iter = 0;
    while (iter < numIterations)
    {
        int task = blockIdx.x;
        while (task < nTasks)
        {
            int i = blockDim.x * task + threadIdx.x;

            if (i < numElements)
                C[i] = A[i] + B[i];

            task += gridDim.x;
        }
        iter++;
    }
}

/*  Memory ranges version */

__global__ void
vectorAddMR(const float *A, const float *B, float *C, int numElements, int numIterations)
{
    int iter = 0;
    while (iter < numIterations)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        __shared__ int offsetA, offsetB, offsetC;
        if (threadIdx.x == 0)
        {
            int i2d = i / (256 * 256);
            int i2r = (i % (256 * 256)) / 256;

            int first_level = (*((int *)A + i2d)) * 256;
            offsetA = (*((int *)A + first_level + i2r)) * 256;
        }
        else if (threadIdx.x == 1)
        {
            int i2d = i / (256 * 256);
            int i2r = (i % (256 * 256)) / 256;
            int first_level = (*((int *)B + i2d)) * 256;
            offsetB = (*((int *)B + first_level + i2r)) * 256;
        }
        else if (threadIdx.x == 2)
        {
            int i2d = i / (256 * 256);
            int i2r = (i % (256 * 256)) / 256;
            int first_level = (*((int *)C + i2d)) * 256;
            offsetC = (*((int *)C + first_level + i2r)) * 256;
        }
        __syncthreads();

        if (i < numElements)
            C[offsetC + threadIdx.x] = A[offsetA + threadIdx.x] + B[offsetB + threadIdx.x];

        iter++;
    }
}

/* Persistent blocks + Memory ranges version */

__global__ void
vectorAddPMR(const float *A, const float *B, float *C, int numElements, int numIterations)
{
    int nTasks = (numElements + blockDim.x - 1) / blockDim.x;
    int iter = 0;
    while (iter < numIterations)
    {
        int task = blockIdx.x;
        while (task < nTasks)
        {
            int i = blockDim.x * task + threadIdx.x;
            // Compute chunk offset for A and B
            __shared__ int offsetA, offsetB, offsetC;
            if (threadIdx.x == 0)
            {
                int i2d = i / (256 * 256);
                int i2r = (i % (256 * 256)) / 256;

                int first_level = (*((int *)A + i2d)) * 256;
                offsetA = (*((int *)A + first_level + i2r)) * 256;
            }
            else if (threadIdx.x == 1)
            {
                int i2d = i / (256 * 256);
                int i2r = (i % (256 * 256)) / 256;
                int first_level = (*((int *)B + i2d)) * 256;
                offsetB = (*((int *)B + first_level + i2r)) * 256;
            }
            else if (threadIdx.x == 2)
            {
                int i2d = i / (256 * 256);
                int i2r = (i % (256 * 256)) / 256;
                int first_level = (*((int *)C + i2d)) * 256;
                offsetC = (*((int *)C + first_level + i2r)) * 256;
            }
            __syncthreads();

            if (i < numElements)
            {
                C[offsetC + threadIdx.x] = A[offsetA + threadIdx.x] + B[offsetB + threadIdx.x];
                // if ( threadIdx.x == 0 )
                //     printf("Element %d of %d, Tasks %d of %d : %f + %f = %f\t%d - %d - %d\n", i, numElements, task, nTasks, A[offsetA + threadIdx.x], B[offsetB + threadIdx.x], C[offsetC + threadIdx.x], offsetA, offsetB, offsetC);
            }

            task += gridDim.x;
        }
        iter++;
    }
}
