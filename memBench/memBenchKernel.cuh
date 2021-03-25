/**
 * @file memBenchKernel.cuh
 * @author José María González Linares (jgl@uma.es)
 * @brief Kernel for memory benchmarking
 * @version 0.1
 * @date 2021-02-15
 * 
 */

/**
 *  memBench: A very simple kernel that access a range of addresses
 */

__global__ void
memBench(float *A, int offset)
{
    unsigned long long i = threadIdx.x;
    A[i+offset*blockDim.x] = 2*A[i+offset*blockDim.x];
}