/**
 * @file vdummyKernel.cuh
 * @author José María González Linares (jgl@uma.es)
 * @brief Dummy kernels
 * @version 0.1
 * @date 2021-02-15
 * 
 */

/**
 * Compute bound dummy kernel
 *
 */

__global__ void
cbDummy(const float *A, const float *B, unsigned long numIter)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float data, tmp = 0;

    if ( i == 0 )
        data = A[i];

    __syncthreads();

    for ( int j = 0; j < numIter; j++ )
        tmp += sqrtf(data*i);

    __syncthreads();

    if ( i == 0 )
        B[i] = data;

}

/**
 * Memory bound dummy kernel
 *
 */

 __global__ void
 mbDummy(const float *A, const float *B, unsigned long numElements)
 {
     int i = blockDim.x * blockIdx.x + threadIdx.x;

     if ( i < numElements )
        B[i] = A[1] + 1;
 }
