/**
 * @file soloTest.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief
 * @version 0.1
 * @date 2021-05-03
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <stdio.h>  /* printf()                 */
#include <stdlib.h> /* exit(), malloc(), free() */
#include <unistd.h>
#include <errno.h> /* errno, ECHILD            */
#include <time.h>

#include <cuda_profiler_api.h>
#include <helper_functions.h> // helper functions for string parsing
#include <helper_cuda.h>

#include "tasks/cuda_tasks.h"
#include "profile/nvmlPower.hpp"
#include "profile/cupti_profiler.h"
#include "memBench/memBench.h"
#include "dummy/dummy.h"
#include "matrixMul/matrixMul.h"
#include "BlackScholes/BlackScholes.h"
#include "PathFinder/PathFinder.h"

#include <argp.h>

const char *argp_program_version =
    "soloTest 0.1";

/* Program documentation. */
static char doc[] =
    "soloTest -- a program to test the solo execution of a kernel";

/* A description of the arguments we accept. */
// static char args_doc[] = "ARG1 ARG2"; // We don´t need mandatory arguments

/* The options we understand. */
static struct argp_option options[] = {
    {"verbose", 'v', 0, 0, "Produce verbose output"},
    {"quiet", 'q', 0, 0, "Don't produce any output"},
    {"output", 'o', "FILE", 0, "Output to FILE instead of standard output"},
    {"device", 'd', "ID", 0, "CUDA device id"},
    {"task", 't', "TID", 0, "Task id"},
    {"execution", 'x', "EX", 0, "Execution mode. EX: 0 (Original), 1 (Persistent), 2 (MemoryRanges), 3 (PersistentMemoryRanges"},
    {"memory", 'm', "MR", 0, "Memory ranges mode. MR: 0 (None), 1 (Shared), 2 (Non Shared)"},
    {"profile", 'p', "PM", 0, "Profiling mode. PM: 0 (None), 1 (CPU Timers), 2 (GPU Events), 3 (CUPTI)"},
    {"nelements", 'n', "N", 0, "Number of 1024 elements"},
    {"blocks", 'b', "B", 0, "Number of blocks per grid"},
    {"power", 'e', 0, 0, "Measure power drawn"},
    {0}};

/* Used by main to communicate with parse_opt. */
struct arguments
{
    //  char *args[2];                /* arg1 & arg2 */
    int quiet, verbose;
    char *output_file;
    int deviceID;
    CUDAtaskNames taskID;
    ExecutionMode exMode;
    MemoryRangeMode mrMode;
    ProfileMode profMode;
    int nElements;
    int blocksPerGrid;
    bool powerMeasure;
};

/* Parse a single option. */
static error_t
parse_opt(int key, char *arg, struct argp_state *state)
{
    /* Get the input argument from argp_parse, which we
     know is a pointer to our arguments structure. */
    struct arguments *arguments = (struct arguments *)state->input;

    switch (key)
    {
    case 'q':
        arguments->quiet = 1;
        break;
    case 'v':
        arguments->verbose = 1;
        break;
    case 'o':
        arguments->output_file = arg;
        break;
    case 'd':
        arguments->deviceID = arg ? atoi(arg) : 0;
        break;
    case 't':
        arguments->taskID = arg ? (CUDAtaskNames)atoi(arg) : (CUDAtaskNames)0;
        break;
    case 'x':
        arguments->exMode = (ExecutionMode)atoi(arg);
        break;
    case 'm':
        arguments->mrMode = (MemoryRangeMode)atoi(arg);
        break;
    case 'p':
        arguments->profMode = (ProfileMode)atoi(arg);
        break;
    case 'n':
        arguments->nElements = atoi(arg);
        break;
    case 'b':
        arguments->blocksPerGrid = atoi(arg);
        break;
    case 'e':
        arguments->powerMeasure = 1;
        break;
        // case ARGP_KEY_ARG:
        //   if (state->arg_num >= 2)
        //     /* Too many arguments. */
        //     argp_usage (state);

        //   arguments->args[state->arg_num] = arg;

        //   break;
        // case ARGP_KEY_END:
        //   if (state->arg_num < 2)
        //     /* Not enough arguments. */
        //     argp_usage (state);
        //   break;

    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

void initArg(struct arguments *arg)
{
    arg->deviceID = 0;
    arg->taskID = Dummy;
    arg->blocksPerGrid = 0;
    arg->exMode = Original;
    arg->mrMode = None;
    arg->nElements = 1024;
    arg->output_file = 0;
    arg->powerMeasure = 0;
    arg->profMode = NoProfile;
    arg->quiet = 0;
    arg->verbose = 0;
}

void printArg(struct arguments arg, char **argv)
{
    printf("%s\n", argv[0]);
    printf("\tProduce verbose output: %d\n", arg.verbose);
    printf("\tDon't produce any output: %d\n", arg.quiet);
    printf("\tOutput file: %s\n", arg.output_file);
    printf("\tCUDA device id: %d\n", arg.deviceID);
    printf("\tTask id: %d\n", arg.taskID);
    printf("\tExecution mode: %d\n", arg.exMode);
    printf("\tMemory ranges mode: %d\n", arg.mrMode);
    printf("\tProfiling mode: %d\n", arg.profMode);
    printf("\tNumber of 1024 elements: %d\n", arg.nElements);
    printf("\tNumber of blocks per grid: %d\n", arg.blocksPerGrid);
    printf("\tMeasure power drawn: %d\n", arg.powerMeasure);
}

/* Our argp parser. */
static struct argp argp = {options, parse_opt, "", doc};

//#include "profile/cupti_profiler.h"

int main(int argc, char **argv)
{
    struct arguments arguments;
    arguments.deviceID = 0;
    arguments.taskID = Dummy;
    arguments.blocksPerGrid = 0;
    arguments.exMode = Original;
    arguments.mrMode = None;
    arguments.nElements = 1024;
    arguments.output_file = 0;
    arguments.powerMeasure = 0;
    arguments.profMode = NoProfile;
    arguments.quiet = 0;
    arguments.verbose = 0;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    printArg(arguments, argv);

    // Select device
    cudaError_t err;
    int deviceId = arguments.deviceID;
    // Init memory profiler at the beginning to set MapHostMemory
    float *d_ptr = 0;
    MemoryProfiler memprof;
    memprof.initMemoryProfiler(deviceId);

    cudaSetDevice(deviceId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    printf("Working on Device %s. Memory size %llu \n", deviceProp.name, deviceProp.totalGlobalMem);

    // Profilers
    vector<string> event_names{};
    //    vector<string> metric_names{"dram_read_transactions", "dram_write_transactions", "ipc", "dram_read_throughput", "dram_write_throughput", "dram_utilization", "stall_memory_dependency", "stall_memory_throttle"};
    // vector<string> metric_names{"inst_executed", "stall_sync", "warp_execution_efficiency", "sm_efficiency", "achieved_occupancy", "eligible_warps_per_cycle"};
    vector<string> metric_names{"ipc", "dram_read_throughput", "dram_write_throughput"};
    CuptiProfiler profiler;

    // Timers
    double ProfilingTimeThreshold = 1.0; // Kernels are launched many times during this interval
    struct timespec now;
    double time0, time1, time2, elapsed_time = 0.0, exec_time = 0.0;

    // Create task
    ProfileMode prof = NoProfile; //CUPTIProf; //TimerProf;//EventsProf;
    int mprof = arguments.profMode;
    switch (mprof)
    {
    case 0:
        prof = NoProfile;
        break;
    case 1:
        prof = TimerProf;
        break;
    case 2:
        prof = EventsProf;
        break;
    case 3:
        prof = CUPTIProf;
        break;
    case 4:
        prof = CUPTISample;
        break;
    default:
        break;
    }

    CUDAtaskNames task_name = arguments.taskID;
    CKEmode cke_mode = ASYNC;
    MemoryRangeMode mr_mode = Shared; //None; //Shared;
    int m_mr_mode = arguments.mrMode;
    switch (m_mr_mode)
    {
    case 0:
        mr_mode = None;
        break;
    case 1:
        mr_mode = Shared;
        break;
    case 2:
        mr_mode = NonShared;
        break;
    default:
        break;
    }

    ExecutionMode ex_mode = Original;
    int m_ex_mode = arguments.exMode;
    switch (m_ex_mode)
    {
    case 0:
        ex_mode = Original;
        arguments.blocksPerGrid = -1;
        break;
    case 1:
        ex_mode = Persistent;
        break;
    case 2:
        ex_mode = MemoryRanges;
        arguments.blocksPerGrid = -1;
        break;
    case 3:
        ex_mode = PersistentMemoryRanges;
        break;
    default:
        break;
    }

    CUDAtask *task = createCUDATask(task_name, deviceId);
    task->setPinned(true);
    task->setProfileMode(EventsProf);
    task->setProfileMode(prof);
    task->setCKEMode(cke_mode);
    task->setMRMode(mr_mode);

    setupCUDATask(task, arguments.nElements, arguments.blocksPerGrid);

    task->setExecMode(ex_mode);

    task->allocHostMemory();
    task->dataGeneration();

    clock_gettime(CLOCK_MONOTONIC, &now);
    printf("%lu\t%lu\tAllocating device memory\n", now.tv_sec, now.tv_nsec);

    if (mr_mode == None)
        task->allocDeviceMemory();
    else
    {
        int size = getTaskSize(task);
        size *= 7;
        printf("allocating %d bytes\n", size);

        err = cudaMalloc((void **)&d_ptr, size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        memprof.setPtr(d_ptr, size);
        memprof.profileMemory();
        memprof.writeAssignments();

        // cuptiFinalize();
        cudaMemset(d_ptr, 0, size);
    }

    cudaProfilerStart();

    clock_gettime(CLOCK_MONOTONIC, &now);
    long long reftime = now.tv_sec * 1000000000 + now.tv_nsec;
    if (arguments.powerMeasure)
    {
        printf("Wait 30 seconds\n");
        sleep(30);
        nvmlPowerInit();
        setRefTime(reftime); // For power measurement
    }

    printf("%lu\t%lu\t%f s\tTransfering to device\n", now.tv_sec, now.tv_nsec, (1000000000 * now.tv_sec + now.tv_nsec - reftime) * 0.000000001);

    if (mr_mode == None)
        task->htdTransfer();

    cudaDeviceSynchronize();

    if (prof == CUPTIProf)
        profiler.init_cupti_profiler(deviceId);
    else if (prof == CUPTISample)
        profiler.init_cupti_sampler(deviceId);

    clock_gettime(CLOCK_MONOTONIC, &now);
    printf("%lu\t%lu\t%f s\tExecuting kernel\n", now.tv_sec, now.tv_nsec, (1000000000 * now.tv_sec + now.tv_nsec - reftime) * 0.000000001);

    // Solo original profiling
    unsigned long int numLaunchs;
    int comb;
    task->setAllModules(0);
    comb = task->getNumCombinations();

    task->setProfileMode(EventsProf);
    if (ex_mode == Persistent || ex_mode == PersistentMemoryRanges)
        task->setPersistentBlocks();
    dim3 b = task->getBlocksPerGrid();
    int maxBlocksPerMulti = b.x / deviceProp.multiProcessorCount;

    for (int nb = 1; nb <= maxBlocksPerMulti; nb++)
    {
        if (ex_mode == Original || ex_mode == MemoryRanges)
            nb = maxBlocksPerMulti;
        else
        {
            b.x = deviceProp.multiProcessorCount * nb;
            task->setBlocksPerGrid(b);
        }
        // printf("Blocks %d (%d)", b.x, nb);

        for (int c = 0; c < comb; c++)
        {
            if (ex_mode == Original || ex_mode == Persistent)
                c = comb;

            if (mr_mode != None)
                sendTaskData(task, d_ptr, &memprof, c);
            else
                c = comb;

            if (prof == TimerProf)
            {
                clock_gettime(CLOCK_MONOTONIC, &now);
                time0 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                while (elapsed_time < ProfilingTimeThreshold)
                {
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    time1 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                    task->kernelExec();
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                    exec_time += time2 - time1;
                    elapsed_time = time2 - time0;
                    numLaunchs++;
                }
            }
            else if (prof == CUPTIProf)
            {
                for (int i = 0; i < metric_names.size(); i++)
                {
                    task->setNumLaunchs(0);
                    profiler.start_kernelduration_cupti_profiler();
                    task->kernelExec();
                    int dur = profiler.end_kernelduration_cupti_profiler();
                    printf("\tKernel duration %f s\t", (float)dur / 1e09);
                    fflush(stdout);
                    CUpti_EventGroupSets *passData = profiler.start_cupti_profiler(metric_names[i].c_str());
                    int num_passes = passData->numSets;
                    profiler.advance_cupti_profiler(passData, 0);
                    if (num_passes > 1)
                    {
                        task->kernelExec();
                        for (int j = 1; j < num_passes; j++)
                        {
                            profiler.advance_cupti_profiler(passData, j);
                            task->kernelExec();
                        }
                        profiler.stop_cupti_profiler(true);
                    }
                    else
                    {
                        task->kernelExec();
                        profiler.stop_cupti_profiler(true);
                    }
                    profiler.print_event_instances();
                    profiler.free_cupti_profiler();
                }
            }
            else if (prof == CUPTISample)
            {
                ProfilingTimeThreshold = 1.0;
                vector<string> event_name{"active_cycles", "inst_executed", "l2_subp0_read_sector_misses", "l2_subp1_read_sector_misses", "l2_subp0_write_sector_misses", "l2_subp1_write_sector_misses"};
                profiler.open_metric_file("events.txt");
                for (int i = 0; i < event_name.size(); i++)
                {
                    profiler.start_cupti_sampler(event_name[i].c_str());
                    task->setNumLaunchs(0);
                    sleep(1);
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    time0 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                    elapsed_time = 0;
                    while (elapsed_time < ProfilingTimeThreshold)
                    {
                        clock_gettime(CLOCK_MONOTONIC, &now);
                        time1 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                        task->kernelExec();
                        clock_gettime(CLOCK_MONOTONIC, &now);
                        time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                        elapsed_time = time2 - time0;
                    }
                    cudaDeviceSynchronize();
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                    elapsed_time = time2 - time0;
                    profiler.end_cupti_sampler(reftime);
                    printf("Task launchs: %d, elapsed time %f\n", task->getNumLaunchs(), elapsed_time);
                }
                profiler.close_metric_file();
            }
            else if (prof == NoProfile)
                task->kernelExec();
            else if (prof == EventsProf)
            {
                ProfilingTimeThreshold = 5.0;
                task->setNumLaunchs(0);
                clock_gettime(CLOCK_MONOTONIC, &now);
                time0 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                elapsed_time = 0;
                exec_time = 0;
                while (elapsed_time < ProfilingTimeThreshold)
                {
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    time1 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                    task->kernelExec();
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                    exec_time += time2 - time1;
                    elapsed_time = time2 - time0;
                }
                cudaDeviceSynchronize();
                float tk = task->getKernelElapsedTime();
                dim3 bb = task->getBlocksPerGrid();
                printf("%d %d %d\n", bb.x, c, task->getNumLaunchs());
                // printf("\nNumLaunchs %d, Time %f, Elapsed %f, TimePerKernel %f ms, %f ms\n", task->getNumLaunchs(), exec_time, elapsed_time, 1000 * exec_time / task->getNumLaunchs(), tk);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &now);
    printf("%lu\t%lu\t%f s\tTransfering from device\n", now.tv_sec, now.tv_nsec, (1000000000 * now.tv_sec + now.tv_nsec - reftime) * 0.000000001);

    cudaProfilerStop();
    cudaDeviceSynchronize();

    task->dthTransfer();

    cudaDeviceSynchronize();
    if (task->checkResults())
        printf("Test passed\n");

    if (arguments.powerMeasure)
        nvmlPowerEnd();

    switch (prof)
    {
    case NoProfile:
        break;
    case TimerProf:
        printf("Child %lu Kernel %lu NumLaunchs %lu Time %f TimePerKernel %f us\n", (ulong)getpid(), (ulong)task->getName(), (ulong)numLaunchs, exec_time, 1000000 * exec_time / numLaunchs);
        break;
    case EventsProf:
        float th = task->getHtDElapsedTime();
        float td = task->getDtHElapsedTime();
        float tk = task->getKernelElapsedTime();
        printf("HtD %f K %f DtH %f\n", th, tk, td);
        printf("NumLaunchs %d, Time %f, TimePerKernel %f us\n", task->getNumLaunchs(), exec_time, 1000000 * exec_time / task->getNumLaunchs());
        break;
    }
    exit(0);
}