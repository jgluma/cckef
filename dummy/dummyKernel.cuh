/**
 * @file vdummyKernel.cuh
 * @author José María González Linares (jgl@uma.es)
 * @brief Dummy kernels
 * @version 0.1
 * @date 2021-02-15
 * 
 */

/**
 * Memory bound dummy kernel
 *
 */

__global__ void
mbDummy(float *A, float *B, unsigned long numElements, int numIterations)
{
    int iter = 0;
    while (iter < numIterations)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if (i < numElements)
            B[i] = A[i] + 1;
        iter++;
    }
}

/* Persistent blocks version */

__global__ void
mbDummyP(float *A, float *B, unsigned long numElements, int numIterations)
{
    int iter = 0;
    int nTasks = (numElements + blockDim.x - 1) / blockDim.x;
    while (iter < numIterations)
    {
        int task = blockIdx.x;
        while (task < nTasks)
        {
            int i = blockDim.x * task + threadIdx.x;

            if (i < numElements)
            {
                B[i] = A[i] + 1;
            }
            task += gridDim.x;
        }
        iter++;
    }
}

/*  Memory ranges version */

__global__ void
mbDummyMR(float *A, float *B, unsigned long numElements, int numIterations)
{
    int iter = 0;
    while (iter < numIterations)
    {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        // Compute chunk offset for A and B
        __shared__ int offsetA, offsetB;
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

        __syncthreads();

        if (i < numElements)
            B[offsetB + threadIdx.x] = A[offsetA + threadIdx.x] + 1;

        iter++;
    }
}

/* Persistent blocks + Memory ranges version */

__global__ void
mbDummyPMR(float *A, float *B, unsigned long numElements, int numIterations)
{
    int iter = 0;
    int nTasks = (numElements + blockDim.x - 1) / blockDim.x;
    while (iter < numIterations)
    {
        int task = blockIdx.x;
        while (task < nTasks)
        {
            int i = blockDim.x * task + threadIdx.x;
            // Compute chunk offset for A and B
            __shared__ int offsetA, offsetB;
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

            __syncthreads();

            if (i < numElements)
                B[offsetB + threadIdx.x] = A[offsetA + threadIdx.x] + 1;

            task += gridDim.x;
        }
        iter++;
    }
}

/**
 * Compute bound dummy kernel
 *
 */

__global__ void
cbDummy(const float *A, float *B, unsigned long numIterations)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float data, tmp = 0;

    if (i == 0)
        data = A[i];

    __syncthreads();

    for (int j = 0; j < numIterations; j++)
    {
        for (int k = 0; k < numIterations; k++)
            tmp += sqrtf(data * k);
        tmp *= j;
    }

    __syncthreads();

    if (i == 0)
        B[i] = data;
}