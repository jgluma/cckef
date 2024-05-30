/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Avoid unnecessary parameters
#define Riskfree 0.02f
#define Volatility 0.30f

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d)
{
    const float A1 = 0.31938153f;
    const float A2 = -0.356563782f;
    const float A3 = 1.781477937f;
    const float A4 = -1.821255978f;
    const float A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
        K = __fdividef(1.0f, (1.0f + 0.2316419f * fabsf(d)));

    float
        cnd = RSQRT2PI * __expf(-0.5f * d * d) *
              (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}

///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    sqrtT = __fdividef(1.0F, rsqrtf(T));
    d1 = __fdividef(__logf(S / X) + (R + 0.5f * V * V) * T, V * sqrtT);
    d2 = d1 - V * sqrtT;

    CNDD1 = cndGPU(d1);
    CNDD2 = cndGPU(d2);

    //Calculate Call and Put simultaneously
    expRT = __expf(-R * T);
    CallResult = S * CNDD1 - X * expRT * CNDD2;
    PutResult = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}

////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
//To keep things simple we use floats instead of float2
//We also remove __restrict keyword to force reloading data
////////////////////////////////////////////////////////////////////////////////
__launch_bounds__(256) // Better 256 threads to compensate using floats
    __global__ void BlackScholesGPU(
        float *d_CallResult,
        float *d_PutResult,
        float *d_StockPrice,
        float *d_OptionStrike,
        float *d_OptionYears,
        int optN,
        int nIter)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

    for (int i = 0; i < nIter; i++)
    {
        const int opt = blockDim.x * blockIdx.x + threadIdx.x;
        if (opt < optN)
        {
            float callResult, putResult;
            BlackScholesBodyGPU(
                callResult,
                putResult,
                d_StockPrice[opt],
                d_OptionStrike[opt],
                d_OptionYears[opt],
                Riskfree,
                Volatility);
            d_CallResult[opt] = callResult;
            d_PutResult[opt] = putResult;
        }
    }
}

/* Persistent blocks version */

__launch_bounds__(256) // Better 256 threads to compensate using floats
    __global__ void BlackScholesGPUP(
        float *d_CallResult,
        float *d_PutResult,
        float *d_StockPrice,
        float *d_OptionStrike,
        float *d_OptionYears,
        int optN,
        int nIter)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

    for (int i = 0; i < nIter; i++)
    {
        int task = blockIdx.x;
        int opt = blockDim.x * task + threadIdx.x;
        while (opt < optN)
        {
            float callResult, putResult;
            BlackScholesBodyGPU(
                callResult,
                putResult,
                d_StockPrice[opt],
                d_OptionStrike[opt],
                d_OptionYears[opt],
                Riskfree,
                Volatility);
            d_CallResult[opt] = callResult;
            d_PutResult[opt] = putResult;
            task += gridDim.x;
            opt = blockDim.x * task + threadIdx.x;
        }
    }
}

/*  Memory ranges version */

__launch_bounds__(256)
    __global__ void BlackScholesGPUMR(
        float *d_CallResult,
        float *d_PutResult,
        float *d_StockPrice,
        float *d_OptionStrike,
        float *d_OptionYears,
        int optN,
        int nIter)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

    for (int i = 0; i < nIter; i++)
    {
        const int opt = blockDim.x * blockIdx.x + threadIdx.x;
        if (opt < optN)
        {
            float callResult, putResult;
            __shared__ int oCall, oPut, oPrice, oStrike, oYears;
            if (threadIdx.x == 0)
            {
                int i2d = opt / (256 * 256);
                int i2r = (opt % (256 * 256)) / 256;
                int first_level;

                first_level = (*((int *)d_CallResult + i2d)) * 256;
                oCall = (*((int *)d_CallResult + first_level + i2r)) * 256;

                first_level = (*((int *)d_PutResult + i2d)) * 256;
                oPut = (*((int *)d_PutResult + first_level + i2r)) * 256;

                first_level = (*((int *)d_StockPrice + i2d)) * 256;
                oPrice = (*((int *)d_StockPrice + first_level + i2r)) * 256;

                first_level = (*((int *)d_OptionStrike + i2d)) * 256;
                oStrike = (*((int *)d_OptionStrike + first_level + i2r)) * 256;

                first_level = (*((int *)d_OptionYears + i2d)) * 256;
                oYears = (*((int *)d_OptionYears + first_level + i2r)) * 256;
                //                printf("iter %d - opt %d i2d %d - i2r %03d - off %d   %d   %d   %d   %d\n", i, opt, i2d, i2r, offsetCall, offsetPut, offsetPrice, offsetStrike, offsetYears);
            }
            __syncthreads();

            BlackScholesBodyGPU(
                callResult,
                putResult,
                d_StockPrice[oPrice + threadIdx.x],
                d_OptionStrike[oStrike + threadIdx.x],
                d_OptionYears[oYears + threadIdx.x],
                Riskfree,
                Volatility);
            d_CallResult[oCall + threadIdx.x] = callResult;
            d_PutResult[oPut + threadIdx.x] = putResult;

        }
    }
}

/* Persistent blocks + Memory ranges version */

__launch_bounds__(256)
    __global__ void BlackScholesGPUPMR(
        float *d_CallResult,
        float *d_PutResult,
        float *d_StockPrice,
        float *d_OptionStrike,
        float *d_OptionYears,
        int optN,
        int nIter)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

    for (int i = 0; i < nIter; i++)
    {
        int task = blockIdx.x;
        int opt = blockDim.x * task + threadIdx.x;
        while (opt < optN)
        {
            float callResult, putResult;
            __shared__ int oCall, oPut, oPrice, oStrike, oYears;
            if (threadIdx.x == 0)
            {
                int i2d = opt / (256 * 256);
                int i2r = (opt % (256 * 256)) / 256;
                int first_level;

                first_level = (*((int *)d_CallResult + i2d)) * 256;
                oCall = (*((int *)d_CallResult + first_level + i2r)) * 256;

                first_level = (*((int *)d_PutResult + i2d)) * 256;
                oPut = (*((int *)d_PutResult + first_level + i2r)) * 256;

                first_level = (*((int *)d_StockPrice + i2d)) * 256;
                oPrice = (*((int *)d_StockPrice + first_level + i2r)) * 256;

                first_level = (*((int *)d_OptionStrike + i2d)) * 256;
                oStrike = (*((int *)d_OptionStrike + first_level + i2r)) * 256;

                first_level = (*((int *)d_OptionYears + i2d)) * 256;
                oYears = (*((int *)d_OptionYears + first_level + i2r)) * 256;
                //                printf("iter %d - opt %d i2d %d - i2r %03d - off %d   %d   %d   %d   %d\n", i, opt, i2d, i2r, offsetCall, offsetPut, offsetPrice, offsetStrike, offsetYears);
            }
            __syncthreads();

            BlackScholesBodyGPU(
                callResult,
                putResult,
                d_StockPrice[oPrice + threadIdx.x],
                d_OptionStrike[oStrike + threadIdx.x],
                d_OptionYears[oYears + threadIdx.x],
                Riskfree,
                Volatility);
            d_CallResult[oCall + threadIdx.x] = callResult;
            d_PutResult[oPut + threadIdx.x] = putResult;
            task += gridDim.x;
            opt = blockDim.x * task + threadIdx.x;
        }
    }
}
