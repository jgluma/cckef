/*
LICENSE TERMS

Copyright (c)2008-2014 University of Virginia
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted without royalty fees or other restrictions, provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of the University of Virginia, the Dept. of Computer Science, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF VIRGINIA OR THE SOFTWARE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * @brief PathFinder uses dynamic programming to find a path on a 2-D grid
 *        from the bottom row to the top row with the smallest accumulated
 *        weights, where each step of the path moves straight ahead or
 *        diagonally ahead. It iterates row by row, each node picks a
 *        neighboring node in the previous row that has the smallest 
 *        accumulated weight, and adds its own weight to the sum. This
 *        kernel uses the technique of ghost zone optimization.
 * 
 */

#define HALO 1
#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x < (min)) ? min : ((x > (max)) ? max : x)
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

template <int BLOCK_SIZE>
__global__ void pathFinderGPU(
    int iteration,
    int *gpuWall,
    int *gpuSrc,
    int *gpuResults,
    int cols,
    int rows,
    int startStep,
    int border)
{

    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // each block finally computes result for a small block
    // after N iterations.
    // it is the non-overlapping small blocks that cover
    // all the input data

    // calculate the small block size
    int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

    // calculate the boundary for the block according to
    // the boundary of its small block
    int blkX = small_block_cols * bx - border;
    int blkXmax = blkX + BLOCK_SIZE - 1;

    // calculate the global thread coordination
    int xidx = blkX + tx;

    // effective range within this block that falls within
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1) : BLOCK_SIZE - 1;

    int W = tx - 1;
    int E = tx + 1;

    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

    if (IN_RANGE(xidx, 0, cols - 1))
    {
        prev[tx] = gpuSrc[xidx];
    }
    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    bool computed;
    for (int i = 0; i < iteration; i++)
    {
        computed = false;
        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) &&
            isValid)
        {
            computed = true;
            int left = prev[W];
            int up = prev[tx];
            int right = prev[E];
            int shortest = MIN(left, up);
            shortest = MIN(shortest, right);
            int index = cols * (startStep + i) + xidx;
            result[tx] = shortest + gpuWall[index];
        }
        __syncthreads();
        if (i == iteration - 1)
            break;
        if (computed) //Assign the computation range
            prev[tx] = result[tx];
        __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    }

    // update the global memory
    // after the last iteration, only threads coordinated within the
    // small block perform the calculation and switch on ``computed''
    if (computed)
    {
        gpuResults[xidx] = result[tx];
    }
}

/* Persistent blocks version */

template <int BLOCK_SIZE>
__global__ void pathFinderGPUP(
    int iteration,
    int *gpuWall,
    int *gpuSrc,
    int *gpuResults,
    int cols,
    int rows,
    int startStep,
    int border,
    int nblocks)
{

    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    while (bx < nblocks)
    {

        // each block finally computes result for a small block
        // after N iterations.
        // it is the non-overlapping small blocks that cover
        // all the input data

        // calculate the small block size
        int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

        // calculate the boundary for the block according to
        // the boundary of its small block
        int blkX = small_block_cols * bx - border;
        int blkXmax = blkX + BLOCK_SIZE - 1;

        // calculate the global thread coordination
        int xidx = blkX + tx;

        // effective range within this block that falls within
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1) : BLOCK_SIZE - 1;

        int W = tx - 1;
        int E = tx + 1;

        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

        if (IN_RANGE(xidx, 0, cols - 1))
        {
            prev[tx] = gpuSrc[xidx];
        }
        __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (int i = 0; i < iteration; i++)
        {
            computed = false;
            if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) &&
                isValid)
            {
                computed = true;
                int left = prev[W];
                int up = prev[tx];
                int right = prev[E];
                int shortest = MIN(left, up);
                shortest = MIN(shortest, right);
                int index = cols * (startStep + i) + xidx;
                result[tx] = shortest + gpuWall[index];
            }
            __syncthreads();
            if (i == iteration - 1)
                break;
            if (computed) //Assign the computation range
                prev[tx] = result[tx];
            __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        }

        // update the global memory
        // after the last iteration, only threads coordinated within the
        // small block perform the calculation and switch on ``computed''
        if (computed)
        {
            gpuResults[xidx] = result[tx];
        }
        bx += gridDim.x;
    }
}

/*  Memory ranges version */

template <int BLOCK_SIZE>
__global__ void pathFinderGPUMR(
    int iteration,
    int *gpuWall,
    int *gpuSrc,
    int *gpuResults,
    int cols,
    int rows,
    int startStep,
    int border)
{

    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    // each block finally computes result for a small block
    // after N iterations.
    // it is the non-overlapping small blocks that cover
    // all the input data

    // calculate the small block size
    int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

    // calculate the boundary for the block according to
    // the boundary of its small block
    int blkX = small_block_cols * bx - border;
    int blkXmax = blkX + BLOCK_SIZE - 1;

    // calculate the global thread coordination
    int xidx = blkX + tx;

    // effective range within this block that falls within
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1) : BLOCK_SIZE - 1;

    int W = tx - 1;
    int E = tx + 1;

    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

    if (IN_RANGE(xidx, 0, cols - 1))
    {
        int i2d = xidx / (256 * 256);
        int i2r = (xidx % (256 * 256)) / 256;
        int first_level = (*((int *)gpuSrc + i2d)) * 256;
        int offset = (*((int *)gpuSrc + first_level + i2r)) * 256;

        prev[tx] = gpuSrc[offset];
    }
    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    bool computed;
    for (int i = 0; i < iteration; i++)
    {
        computed = false;
        if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) &&
            isValid)
        {
            computed = true;
            int left = prev[W];
            int up = prev[tx];
            int right = prev[E];
            int shortest = MIN(left, up);
            shortest = MIN(shortest, right);
            int index = cols * (startStep + i) + xidx;

            int i2d = index / (256 * 256);
            int i2r = (index % (256 * 256)) / 256;
            int first_level = (*((int *)gpuWall + i2d)) * 256;
            int offset = (*((int *)gpuWall + first_level + i2r)) * 256;

            result[tx] = shortest + gpuWall[offset];
        }
        __syncthreads();
        if (i == iteration - 1)
            break;
        if (computed) //Assign the computation range
            prev[tx] = result[tx];
        __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    }

    // update the global memory
    // after the last iteration, only threads coordinated within the
    // small block perform the calculation and switch on ``computed''
    if (computed)
    {
        int i2d = xidx / (256 * 256);
        int i2r = (xidx % (256 * 256)) / 256;
        int first_level = (*((int *)gpuResults + i2d)) * 256;
        int offset = (*((int *)gpuResults + first_level + i2r)) * 256;

        gpuResults[offset] = result[tx];
    }
}

/* Persistent blocks + Memory ranges version */

template <int BLOCK_SIZE>
__global__ void pathFinderGPUPMR(
    int iteration,
    int *gpuWall,
    int *gpuSrc,
    int *gpuResults,
    int cols,
    int rows,
    int startStep,
    int border,
    int nblocks)
{

    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result[BLOCK_SIZE];

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    while (bx < nblocks)
    {

        // each block finally computes result for a small block
        // after N iterations.
        // it is the non-overlapping small blocks that cover
        // all the input data

        // calculate the small block size
        int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

        // calculate the boundary for the block according to
        // the boundary of its small block
        int blkX = small_block_cols * bx - border;
        int blkXmax = blkX + BLOCK_SIZE - 1;

        // calculate the global thread coordination
        int xidx = blkX + tx;

        // effective range within this block that falls within
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1) : BLOCK_SIZE - 1;

        int W = tx - 1;
        int E = tx + 1;

        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

        if (IN_RANGE(xidx, 0, cols - 1))
        {
            int i2d = xidx / (256 * 256);
            int i2r = (xidx % (256 * 256)) / 256;
            int first_level = (*((int *)gpuSrc + i2d)) * 256;
            int offset = (*((int *)gpuSrc + first_level + i2r)) * 256;

            prev[tx] = gpuSrc[offset];
        }
        __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (int i = 0; i < iteration; i++)
        {
            computed = false;
            if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) &&
                isValid)
            {
                computed = true;
                int left = prev[W];
                int up = prev[tx];
                int right = prev[E];
                int shortest = MIN(left, up);
                shortest = MIN(shortest, right);
                int index = cols * (startStep + i) + xidx;

                int i2d = index / (256 * 256);
                int i2r = (index % (256 * 256)) / 256;
                int first_level = (*((int *)gpuWall + i2d)) * 256;
                int offset = (*((int *)gpuWall + first_level + i2r)) * 256;

                result[tx] = shortest + gpuWall[offset];
            }
            __syncthreads();
            if (i == iteration - 1)
                break;
            if (computed) //Assign the computation range
                prev[tx] = result[tx];
            __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        }

        // update the global memory
        // after the last iteration, only threads coordinated within the
        // small block perform the calculation and switch on ``computed''
        if (computed)
        {
            int i2d = xidx / (256 * 256);
            int i2r = (xidx % (256 * 256)) / 256;
            int first_level = (*((int *)gpuResults + i2d)) * 256;
            int offset = (*((int *)gpuResults + first_level + i2r)) * 256;

            gpuResults[offset] = result[tx];
        }
        bx += gridDim.x;
    }
}