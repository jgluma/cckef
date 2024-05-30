/**
 * @file memBench.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief 
 * @version 0.1
 * @date 2021-03-22
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "memBench.h"

memBench::memBench()
{
    chipAssignments = 0;
    numChipAssignments = 0;
    chipMR = 0;
    t = 0;
}

memBench::~memBench()
{
    if (t)
        delete (t);
    if (chipAssignments)
        free(chipAssignments);
    if (numChipAssignments)
        free(numChipAssignments);
    if (chipMR)
        free(chipMR);
    cuptiFinalize();
}

void memBench::init(int d)
{
    deviceId = d;
    cudaSetDevice(deviceId);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    setMemorySize(deviceProp.totalGlobalMem);
    numChips = 24; // TODO: obtain the value from the number of instances of the event

    // Init profiler

    profiler.init_cupti_profiler(deviceId);

    // Chip assignments for half the available memory
    //setNumAssignments( (memorySize/2)/1024 );
    setNumAssignments(1024 * 4 * 32 * 8);
    printf("Profiling %llu chunks\n", numAssignments);
    chipAssignments = (int *)malloc(numAssignments * sizeof(int));
    for (int i = 0; i < numAssignments; i++)
        chipAssignments[i] = -1;
    numChipAssignments = (int *)calloc(numChips, sizeof(int));

    // Memory benchmarking task
    unsigned long bigMR = 2 * 256 * getNumAssignments();
    t = new memBenchTask(bigMR, false);

    dim3 th(256, 1, 1);
    t->setThreadsPerBlock(th);
    dim3 b(deviceProp.multiProcessorCount, 1, 1);
    t->setBlocksPerGrid(b);

    t->allocDeviceMemory();
}

void memBench::init(int d, float *ptr, unsigned long n)
{
    deviceId = d;
    cudaSetDevice(deviceId);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    numChips = 24; // TODO: obtain the value from the number of instances of the event

    // Init profiler

    profiler.init_cupti_profiler(deviceId);

    // Chip assignments for n floats
    //setNumAssignments( (memorySize/2)/1024 );
    setNumAssignments(n);
    printf("Profiling %llu chunks\n", numAssignments);
    chipAssignments = (int *)malloc(numAssignments * sizeof(int));
    for (int i = 0; i < numAssignments; i++)
        chipAssignments[i] = -1;
    numChipAssignments = (int *)calloc(numChips, sizeof(int));

    // Memory benchmarking task
    unsigned long bigMR = 2 * 256 * getNumAssignments();
    t = new memBenchTask(bigMR, false);

    dim3 th(256, 1, 1);
    t->setThreadsPerBlock(th);
    dim3 b(deviceProp.multiProcessorCount, 1, 1);
    t->setBlocksPerGrid(b);
    t->assignDeviceMemory(ptr);
}

void memBench::getChipAssignments()
{
    int offset = 0;
    int numRepeats = 0;
    float p = 0;

    t->kernelExec();

    // There are 4 memory acceses per warp
    dim3 th = t->getThreadsPerBlock();
    dim3 b = t->getBlocksPerGrid();
    unsigned long long expected = 0.8 * (4 * th.x / 32) * b.x;

    struct timespec now;
    double time0, time1, time2, time3, time4 = 0;
    int ntimes = 0, prev = 0;

    clock_gettime(CLOCK_MONOTONIC, &now);
    time0 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;

    while (numRepeats < 5)
    {
        cudaDeviceSynchronize();

        t->setOffset(offset);
        CUpti_EventGroupSets *passData = profiler.start_cupti_profiler(metric_names[0].c_str());
        int num_passes = passData->numSets;
        profiler.advance_cupti_profiler(passData, 0);
        if (num_passes > 1)
        {
            profiler.stop_cupti_profiler(false);
            printf("Ignoring metric %s because it needs %d passes\n", metric_names[0].c_str(), num_passes);
        }
        else
        {
            t->kernelExec();
            profiler.stop_cupti_profiler(false);
        }

        clock_gettime(CLOCK_MONOTONIC, &now);
        time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;

        //profiler.print_event_instances();
        if (profiler.checkConsistency(expected))
        {
            chipAssignments[offset] = profiler.getMaxIdxEvent(expected);
            offset += 1;
        }
        else
        {
            printf("Checking again\n");
            numRepeats++;
        }
        if (offset >= numAssignments)
            numRepeats = 10;

        if (offset > p * numAssignments)
        {
            printf("Completed %llu out of %llu (%f percent)\n", offset, numAssignments, p * 100);
            p += 0.25;
        }
        clock_gettime(CLOCK_MONOTONIC, &now);
        time3 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
        time4 += time3 - time2;

        profiler.free_cupti_profiler();

        clock_gettime(CLOCK_MONOTONIC, &now);
        time1 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
        if ((time1 - time0) > 180)
        {
            cuptiFinalize();
            time0 = time1;
            ntimes++;
            fprintf(stderr, "Resting 1 second (%d times), completed %d (%d), time profiling %f\r", ntimes, offset, offset - prev, 180 - time4);
            sleep(1);
            profiler.init_cupti_profiler(deviceId);
            time4 = 0;
            prev = offset;
        }
    }
}

void memBench::getMemoryRanges()
{

    offsetMR = 0;

    for (int i = 0; i < numAssignments; i++)
        numChipAssignments[chipAssignments[i]]++;

    for (int i = 0; i < numChips; i++)
    {
        if (offsetMR < numChipAssignments[i])
            offsetMR = numChipAssignments[i];
    }

    chipMR = (int *)calloc(offsetMR * numChips, sizeof(int));

    int *c = (int *)calloc(numChips, sizeof(int));

    for (int i = 0; i < numAssignments; i++)
    {
        int a = chipAssignments[i];
        int idx = a * offsetMR + c[a];
        chipMR[idx] = i;
        c[a]++;
    }

    //    free(c);
}

void memBench::writeAssignments()
{
    fpA = fopen("chip_assignments.txt", "w");
    fpM = fopen("memory_ranges.txt", "w");

    int a = fprintf(fpA, "%llu\n", numAssignments);

    fflush(fpA);
    for (int i = 0; i < numAssignments; i++)
        fprintf(fpA, "%d\n", chipAssignments[i]);

    fprintf(fpM, "%d\n", numChips);
    for (int i = 0; i < numChips; i++)
        fprintf(fpM, "%d\t", numChipAssignments[i]);
    fprintf(fpM, "\n");

    for (int i = 0; i < numChips; i++)
        for (int j = 0; j < numChipAssignments[i]; j++)
            fprintf(fpM, "%d\n", chipMR[i * offsetMR + j]);

    fclose(fpA);
    fclose(fpM);
}

void memBench::readAssignments()
{
    fpA = fopen("chip_assignments.txt", "r");
    fpM = fopen("memory_ranges.txt", "r");

    fscanf(fpA, "%llu\n", &numAssignments);
    chipAssignments = (int *)malloc(numAssignments * sizeof(int));
    for (int i = 0; i < numAssignments; i++)
        fscanf(fpA, "%d\n", &(chipAssignments[i]));

    fscanf(fpM, "%d\n", &numChips);
    numChipAssignments = (int *)calloc(numChips, sizeof(int));
    long long total = 0;
    for (int i = 0; i < numChips; i++)
    {
        fscanf(fpM, "%d\t", &(numChipAssignments[i]));
        if (total < numChipAssignments[i])
            total = numChipAssignments[i];
    }

    offsetMR = total;

    chipMR = (int *)calloc(offsetMR * numChips, sizeof(int));

    for (int i = 0; i < numChips; i++)
        for (int j = 0; j < numChipAssignments[i]; j++)
            fscanf(fpM, "%d\n", &(chipMR[i * offsetMR + j]));

    fclose(fpA);
    fclose(fpM);
}

static int mycmpfunc(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

/**
 * @brief Assignments are like page tables for virtual addresses. The mapping
 *        has two levels and the page size is 1024 bytes (256 entries). First
 *        level has just one page with pointers to the 256 tables of second
 *        level. This function returns the indices of 
 * 
 * @param numChips 
 * @param numChipAssignments 
 * @param chipMR 
 */
int *memBench::getAssignmentsIdx(size_t bytes, int *modules, int *num_indices)
{
    int chunksize = 1024; // This should be a parameter
    int numAssignments = (bytes + chunksize - 1) / chunksize + 257;
    int i = 0, j = 0, left = 0;
    int *indices = NULL, *first = NULL;

    indices = (int *)malloc(numAssignments * sizeof(int));
    first = (int *)calloc(numChips, sizeof(int));

    // Check if there are enough assignments
    for (int i = 0; i < numChips; i++)
        if (modules[i] > 0)
            for (int j = 0; j < numChipAssignments[i]; j++)
                if (chipMR[i * offsetMR + j] >= 0)
                    left++;
    if (left < numAssignments)
    {
        fprintf(stderr, "Not enough assignments left (requested %d, available %d)\n", numAssignments, left);
        return (indices);
    }

    // Look for first valid assignment in each module
    for (int i = 0; i < numChips; i++)
    {
        j = 0;
        while (chipMR[i * offsetMR + j] < 0)
            j++;
        first[i] = j;
    }

    // Get the assignments
    while (1)
    {
        for (int k = 0; k < numChips; k++)
        {
            if (modules[k] > 0)
            {
                if (chipMR[k * offsetMR + first[k]] >= 0)
                {
                    indices[i] = chipMR[k * offsetMR + first[k]];
                    chipMR[k * offsetMR + first[k]] = -1;
                    // printf("%d - %d;", k, first[k]);
                    i++;
                    if (i >= numAssignments)
                    {
                        num_indices[0] = i;
                        qsort(indices, i, sizeof(int), mycmpfunc);
                        return (indices);
                    }
                    first[k]++;
                }
                else
                    fprintf(stderr, "There are not assignments left!\n");
            }
        }
    }
}