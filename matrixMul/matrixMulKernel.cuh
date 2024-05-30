/**
 * @file matrixMulKernel.cuh
 * @author José María González Linares (jgl@uma.es)
 * @brief Dummy kernels
 * @version 0.1
 * @date 2021-05-12
 * 
 */

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE>
__global__ void
matrixMulCUDA(float *C, float *A, float *B, dim3 dimA, dim3 dimB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Width of A and B

    int wA = dimA.x;
    int wB = dimB.x;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

/* Persistent blocks version */

template <int BLOCK_SIZE>
__global__ void
matrixMulCUDAP(float *C, float *A, float *B, dim3 dimA, dim3 dimB)
{
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Width of A and B
    int wA = dimA.x;
    int wB = dimB.x;

    int task = blockIdx.x;
    while (1)
    {
        int bx = task % (wB / blockDim.x);
        int by = task / (wB / blockDim.x);
        if (by >= dimA.y / blockDim.y)
            return;

        // Index of the first sub-matrix of A processed by the block
        int aBegin = wA * BLOCK_SIZE * by;

        // Index of the last sub-matrix of A processed by the block
        int aEnd = aBegin + wA - 1;

        // Step size used to iterate through the sub-matrices of A
        int aStep = BLOCK_SIZE;

        // Index of the first sub-matrix of B processed by the block
        int bBegin = BLOCK_SIZE * bx;

        // Step size used to iterate through the sub-matrices of B
        int bStep = BLOCK_SIZE * wB;

        // Csub is used to store the element of the block sub-matrix
        // that is computed by the thread
        float Csub = 0;

        // Loop over all the sub-matrices of A and B
        // required to compute the block sub-matrix
        for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep)
        {

            // Declaration of the shared memory array As used to
            // store the sub-matrix of A
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

            // Declaration of the shared memory array Bs used to
            // store the sub-matrix of B
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

            // Load the matrices from device memory
            // to shared memory; each thread loads
            // one element of each matrix
            As[ty][tx] = A[a + wA * ty + tx];
            Bs[ty][tx] = B[b + wB * ty + tx];

            // Synchronize to make sure the matrices are loaded
            __syncthreads();

            // Multiply the two matrices together;
            // each thread computes one element
            // of the block sub-matrix
#pragma unroll

            for (int k = 0; k < BLOCK_SIZE; ++k)
            {
                Csub += As[ty][k] * Bs[k][tx];
            }

            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads();
        }

        // Write the block sub-matrix to device memory;
        // each thread writes one element
        int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
        C[c + wB * ty + tx] = Csub;

        task += gridDim.x;
    }
}

/* Memory ranges version */

template <int BLOCK_SIZE>
__global__ void
matrixMulCUDAMR(float *C, float *A, float *B, dim3 dimA, dim3 dimB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Width of A and B

    int wA = dimA.x;
    int wB = dimB.x;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix

        // Compute chunk offset for A and B
        __shared__ int offsetA[BLOCK_SIZE], offsetB[BLOCK_SIZE];
        if (tx == 0)
        {
            int i = a + wA * ty;
            int i2d = i / (256 * 256);
            int i2r = (i % (256 * 256)) / 256;
            int first_level = (*((int *)A + i2d)) * 256;
            offsetA[ty] = (*((int *)A + first_level + i2r)) * 256;
        }
        else if (tx == 1)
        {
            int i = b + wB * ty;
            int i2d = i / (256 * 256);
            int i2r = (i % (256 * 256)) / 256;
            int first_level = (*((int *)B + i2d)) * 256;
            offsetB[ty] = (*((int *)B + first_level + i2r)) * 256;
        }

        __syncthreads();

        As[ty][tx] = A[offsetA[ty] + tx];
        Bs[ty][tx] = B[offsetB[ty] + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    __shared__ int offsetC[BLOCK_SIZE];
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx + wB * ty + tx;
    if (tx == 0)
    {
        int i2d = c / (256 * 256);
        int i2r = (c % (256 * 256)) / 256;
        int first_level = (*((int *)C + i2d)) * 256;
        offsetC[ty] = (*((int *)C + first_level + i2r)) * 256;
        // printf("t %d, %d - b %d, %d - i %d, i2d %d, i2r %d, c1 %d, c2 %d, c3 %d : Write at %d a value of %f\n", tx, ty, bx, by, i, i2d, i2r, wB * BLOCK_SIZE * by, BLOCK_SIZE * bx, wB * ty, offsetC, Csub);
    }
    __syncthreads();
    C[offsetC[ty] + c % 256] = Csub;
    // C[c + wB * ty + tx] = Csub;
}

/* Persistent blocks + Memory ranges version */

template <int BLOCK_SIZE>
__global__ void
matrixMulCUDAPMR(float *C, float *A, float *B, dim3 dimA, dim3 dimB)
{
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int wA = dimA.x;
    int wB = dimB.x;

    int task = blockIdx.x;
    while (1)
    {

        int bx = task % (wB / blockDim.x);
        int by = task / (wB / blockDim.x);
        if (by >= dimA.y / blockDim.y)
            return;
        // Index of the first sub-matrix of A processed by the block
        int aBegin = wA * BLOCK_SIZE * by;

        // Index of the last sub-matrix of A processed by the block
        int aEnd = aBegin + wA - 1;

        // Step size used to iterate through the sub-matrices of A
        int aStep = BLOCK_SIZE;

        // Index of the first sub-matrix of B processed by the block
        int bBegin = BLOCK_SIZE * bx;

        // Step size used to iterate through the sub-matrices of B
        int bStep = BLOCK_SIZE * wB;

        // Csub is used to store the element of the block sub-matrix
        // that is computed by the thread
        float Csub = 0;

        // Loop over all the sub-matrices of A and B
        // required to compute the block sub-matrix
        for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep)
        {

            // Declaration of the shared memory array As used to
            // store the sub-matrix of A
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

            // Declaration of the shared memory array Bs used to
            // store the sub-matrix of B
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

            // Load the matrices from device memory
            // to shared memory; each thread loads
            // one element of each matrix

            // Compute chunk offset for A and B
            __shared__ int offsetA[BLOCK_SIZE], offsetB[BLOCK_SIZE];
            if (tx == 0)
            {
                int i = a + wA * ty;
                int i2d = i / (256 * 256);
                int i2r = (i % (256 * 256)) / 256;
                int first_level = (*((int *)A + i2d)) * 256;
                offsetA[ty] = (*((int *)A + first_level + i2r)) * 256;
            }
            else if (tx == 1)
            {
                int i = b + wB * ty;
                int i2d = i / (256 * 256);
                int i2r = (i % (256 * 256)) / 256;
                int first_level = (*((int *)B + i2d)) * 256;
                offsetB[ty] = (*((int *)B + first_level + i2r)) * 256;
            }

            __syncthreads();

            As[ty][tx] = A[offsetA[ty] + tx];
            Bs[ty][tx] = B[offsetB[ty] + tx];

            // Synchronize to make sure the matrices are loaded
            __syncthreads();

            // Multiply the two matrices together;
            // each thread computes one element
            // of the block sub-matrix
#pragma unroll

            for (int k = 0; k < BLOCK_SIZE; ++k)
            {
                Csub += As[ty][k] * Bs[k][tx];
            }

            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            __syncthreads();
        }

        // Write the block sub-matrix to device memory;
        // each thread writes one element
        __shared__ int offsetC[BLOCK_SIZE];
        int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx + wB * ty + tx;
        if (tx == 0)
        {
            int i2d = c / (256 * 256);
            int i2r = (c % (256 * 256)) / 256;
            int first_level = (*((int *)C + i2d)) * 256;
            offsetC[ty] = (*((int *)C + first_level + i2r)) * 256;
            // printf("t %d, %d - b %d, %d - i %d, i2d %d, i2r %d, c1 %d, c2 %d, c3 %d : Write at %d a value of %f\n", tx, ty, bx, by, i, i2d, i2r, wB * BLOCK_SIZE * by, BLOCK_SIZE * bx, wB * ty, offsetC, Csub);
        }
        __syncthreads();
        C[offsetC[ty] + c % 256] = Csub;
        // C[c + wB * ty + tx] = Csub;
        task += gridDim.x;
    }
}