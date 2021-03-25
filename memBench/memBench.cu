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
    chipMR = 0;
}

memBench::~memBench()
{
    if (chipAssignments)
        free(chipAssignments);
    if (chipMR)
    {
        for (int i = 0; i < numChips; i++)
            free(chipMR[i]);
        free(chipMR);
    }
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
    setNumAssignments( (memorySize/2)/1024 );
    chipAssignments = (int *)malloc(numAssignments * sizeof(int));
    for (int i = 0; i < numAssignments; i++)
        chipAssignments[i] = -1;
    chipMR = (int **)calloc(numChips, sizeof(int *));

    printf("Profiling %llu chunks\n", numAssignments);
    // Memory benchmarking task
    unsigned long bigMR = 256 * getNumAssignments();
    t = new memBenchTask(bigMR, false);

    t->setThreadsPerBlock(256);
    t->setBlocksPerGrid(deviceProp.multiProcessorCount);
 //   t->allocHostMemory();
 //   t->dataGeneration();
    t->allocDeviceMemory();
 //   t->htdTransfer();
    t->kernelExec();
}

void memBench::getChipAssignments()
{
    int offset = 0;
    int numRepeats = 0;
    // There are 4 memory acceses per warp
    unsigned long long expected = 0.8 * (4 * t->getThreadsPerBlock() / 32) * t->getBlocksPerGrid();
    printf("Threads %d Blocks %d Expected %llu\n", t->getThreadsPerBlock(), t->getBlocksPerGrid(), expected);
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
        if (offset > numAssignments)
            numRepeats = 10;
        profiler.free_cupti_profiler();
    }
}

void memBench::getMemoryRanges()
{
    numChipAssignments = (int *)calloc(numChips, sizeof(int));
    for (int i = 0; i < numAssignments; i++)
        numChipAssignments[chipAssignments[i]]++;

    for (int i = 0; i < numChips; i++)
        chipMR[i] = (int *)malloc(numChipAssignments[i] * sizeof(int));

    int *c = (int *)calloc(numChips, sizeof(int));

    for (int i = 0; i < numAssignments; i++)
    {
        int a = chipAssignments[i];
        chipMR[a][c[a]] = i;
        c[a]++;
    }

    free(c);
}

void memBench::writeAssignments()
{
    fpA = fopen("chip_assignments.txt", "w");
    fpM = fopen("memory_ranges.txt", "w");

    fprintf(fpA, "%llu\n", numAssignments);
    for (int i = 0; i < numAssignments; i++)
        fprintf(fpA, "%d\n", chipAssignments[i]);

    fprintf(fpM, "%d\n", numChips);
    for (int i = 0; i < numChips; i++)
        fprintf(fpM, "%d\t", numChipAssignments[i]);
    fprintf(fpM, "\n");

    for (int i = 0; i < numChips; i++)
        for (int j = 0; j < numChipAssignments[i]; j++)
            fprintf(fpM, "%d\n", chipMR[i][j]);

    fclose(fpA);
    fclose(fpM);
}

void memBench::readAssignments()
{
}