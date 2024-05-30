/**
 * @file concTest.cu
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
#include "profile/memory_profiler.h"
#include "profile/nvmlPower.hpp"
#include "profile/cupti_profiler.h"
#include "memBench/memBench.h"
#include "dummy/dummy.h"
#include "matrixMul/matrixMul.h"
#include "BlackScholes/BlackScholes.h"
#include "PathFinder/PathFinder.h"

#include <argp.h>

const char *argp_program_version =
    "concTest 0.1";

/* Program documentation. */
static char doc[] =
    "concTest -- a program to test concurrent execution of two kernels";

/* A description of the arguments we accept. */
// static char args_doc[] = "ARG1 ARG2"; // We don´t need mandatory arguments

/* The options we understand. */
static struct argp_option options[] = {
    {"verbose", 'v', 0, 0, "Produce verbose output"},
    {"quiet", 'q', 0, 0, "Don't produce any output"},
    {"output", 'o', "FILE", 0, "Output to FILE instead of standard output"},
    {"device", 'd', "ID", 0, "CUDA device id"},
    {"task1", 't', "TID1", 0, "Task 1 id"},
    {"task2", 'a', "TID2", 0, "Task 2 id"},
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
    CUDAtaskNames taskID1;
    CUDAtaskNames taskID2;
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
        arguments->taskID1 = arg ? (CUDAtaskNames)atoi(arg) : (CUDAtaskNames)0;
        break;
    case 'a':
        arguments->taskID2 = arg ? (CUDAtaskNames)atoi(arg) : (CUDAtaskNames)0;
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
    arg->taskID1 = Dummy;
    arg->taskID2 = Dummy;
    arg->blocksPerGrid = 0;
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
    printf("\tTask 1 id: %d\n", arg.taskID1);
    printf("\tTask 2 id: %d\n", arg.taskID2);
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
    arguments.taskID1 = Dummy;
    arguments.taskID2 = Dummy;
    arguments.blocksPerGrid = 0;
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
    printf("Working on Device %s\n", deviceProp.name);

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

    CUDAtaskNames task_name1 = arguments.taskID1;
    CUDAtaskNames task_name2 = arguments.taskID2;
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
    // Task 1
    CUDAtask *task1 = createCUDATask(task_name1, deviceId);
    task1->setPinned(true);
    task1->setProfileMode(EventsProf);
    task1->setProfileMode(prof);
    task1->setCKEMode(cke_mode);
    task1->setMRMode(mr_mode);

    setupCUDATask(task1, arguments.nElements, arguments.blocksPerGrid);

    CUDAtask *task2 = createCUDATask(task_name2, deviceId);
    task2->setPinned(true);
    task2->setProfileMode(EventsProf);
    task2->setProfileMode(prof);
    task2->setCKEMode(cke_mode);
    task2->setMRMode(mr_mode);

    setupCUDATask(task2, arguments.nElements, arguments.blocksPerGrid);

    if (mr_mode == None)
        task1->setExecMode(Persistent);
    else
        task1->setExecMode(PersistentMemoryRanges);
    task1->allocHostMemory();
    task1->dataGeneration();

    if (mr_mode == None)
        task2->setExecMode(Persistent);
    else
        task2->setExecMode(PersistentMemoryRanges);
    task2->allocHostMemory();
    task2->dataGeneration();

    clock_gettime(CLOCK_MONOTONIC, &now);
    printf("%lu\t%lu\tAllocating device memory\n", now.tv_sec, now.tv_nsec);

    if (mr_mode == None)
    {
        task1->allocDeviceMemory();
        task2->allocDeviceMemory();
    }
    else
    {
        // Create memory partitions

        int size = getTaskSize(task1);
        size += getTaskSize(task2);
        size *= 7;

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
    {
        task1->htdTransfer();
        task2->htdTransfer();
    }

    cudaDeviceSynchronize();

    if (prof == CUPTIProf)
        profiler.init_cupti_profiler(deviceId);
    else if (prof == CUPTISample)
        profiler.init_cupti_sampler(deviceId);

    // if (mr_mode != None)
    // {
    //     dummyTask *vt = dynamic_cast<dummyTask *>(task1);
    //     vt->memHostToDeviceAssigns();
    //     matrixMulTask *vtm = dynamic_cast<matrixMulTask *>(task2);
    //     vtm->memHostToDeviceAssigns();
    // }
    // else
    // {
    //     task1->htdTransfer();
    //     printf("1\n");
    //     task2->htdTransfer();
    //     printf("2\n");
    // }

    clock_gettime(CLOCK_MONOTONIC, &now);
    printf("%lu\t%lu\t%f s\tExecuting kernel\n", now.tv_sec, now.tv_nsec, (1000000000 * now.tv_sec + now.tv_nsec - reftime) * 0.000000001);

    unsigned long int numLaunchs;
    int nb1, nb2;
    int comb1, comb2;

    task1->setAllModules(0);
    comb1 = task1->getNumCombinations();
    task2->setAllModules(1);
    comb2 = task2->getNumCombinations();

    // for (int nb = 10; nb <= 50; nb += 10)
    for (int nb = 1; nb <= 7; nb++)
    {
        task1->setProfileMode(EventsProf);
        task1->setPersistentBlocks();
        dim3 b1 = task1->getBlocksPerGrid();
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, arguments.deviceID);
        int maxBlocksPerMulti = b1.x / deviceProp.multiProcessorCount;
        int n = (nb * maxBlocksPerMulti) / 100;
        // if (n < 1)
        //     n = 1;
        n = nb;
        dim3 b(n * deviceProp.multiProcessorCount, 1, 1);
        task1->setBlocksPerGrid(b);
        nb1 = b.x;
        // printf("Blocks task 1 %d\t", b.x);

        task2->setProfileMode(EventsProf);
        task2->setPersistentBlocks();
        dim3 b2 = task2->getBlocksPerGrid();
        maxBlocksPerMulti = b2.x / deviceProp.multiProcessorCount;
        n = ((100 - nb) * maxBlocksPerMulti) / 100;
        // if (n < 1)
        //     n = 1;
        n = 8 - nb;
        b.x = n * deviceProp.multiProcessorCount;
        task2->setBlocksPerGrid(b);
        // printf("task2 %d\n", b.x);
        nb2 = b.x;

        for (int c1 = 0; c1 < comb1; c1++)
            for (int c2 = 0; c2 < comb2; c2++)
            {
                if (mr_mode != None)
                {
                    memprof.readAssignments();
                    int nA, nB, nC, nD, nE;
                    int *assA = 0, *assB = 0, *assC = 0, *assD = 0, *assE = 0;

                    if (task_name1 == Dummy)
                    {
                        dummyTask *vt = dynamic_cast<dummyTask *>(task1);
                        int *tmp;
                        tmp = vt->getModulesA(c1);
                        assA = memprof.getAssignmentsIdx(vt->getSizeA(), tmp, &nA);
                        tmp = vt->getModulesB(c1);
                        assB = memprof.getAssignmentsIdx(vt->getSizeA(), tmp, &nB);
                        vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB);
                        vt->memHostToDeviceAssigns();
                    }
                    else if (task_name1 == matrixMul)
                    {
                        matrixMulTask *vt = dynamic_cast<matrixMulTask *>(task1);
                        int *tmp;
                        tmp = vt->getModulesA(c1);
                        assA = memprof.getAssignmentsIdx(vt->getSizeA(), tmp, &nA);

                        tmp = vt->getModulesB(c1);
                        assB = memprof.getAssignmentsIdx(vt->getSizeB(), tmp, &nB);

                        tmp = vt->getModulesC(c1);
                        assC = memprof.getAssignmentsIdx(vt->getSizeC(), tmp, &nC);
                        vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC);
                        vt->memHostToDeviceAssigns();
                    }
                    else if (task_name1 == VA)
                    {
                        vectorAddTask *vt = dynamic_cast<vectorAddTask *>(task1);
                        int *tmp;
                        tmp = vt->getModulesA(c1);
                        assA = memprof.getAssignmentsIdx(vt->getSizeN(), tmp, &nA);

                        tmp = vt->getModulesB(c1);
                        assB = memprof.getAssignmentsIdx(vt->getSizeN(), tmp, &nB);

                        tmp = vt->getModulesC(c1);
                        assC = memprof.getAssignmentsIdx(vt->getSizeN(), tmp, &nC);
                        vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC);
                        vt->memHostToDeviceAssigns();
                    }
                    else if (task_name1 == BS)
                    {
                        BlackScholesTask *vb = dynamic_cast<BlackScholesTask *>(task1);
                        int *tmp;
                        tmp = vb->getModulesA(c1);
                        assA = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nA);

                        tmp = vb->getModulesB(c1);
                        assB = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nB);

                        tmp = vb->getModulesC(c1);
                        assC = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nC);

                        tmp = vb->getModulesD(c1);
                        assD = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nD);

                        tmp = vb->getModulesE(c1);
                        assE = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nE);

                        vb->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC, assD, nD, assE, nE);
                        vb->memHostToDeviceAssigns();
                    }
                    else if (task_name1 == PF)
                    {
                        PathFinderTask *vt = dynamic_cast<PathFinderTask *>(task1);
                        int *tmp;
                        tmp = vt->getModulesA(c1);
                        assA = memprof.getAssignmentsIdx(vt->getDataSize() - vt->getResultSize(), tmp, &nA);
                        tmp = vt->getModulesB(c1);
                        assB = memprof.getAssignmentsIdx(vt->getResultSize(), tmp, &nB);
                        tmp = vt->getModulesC(c1);
                        assC = memprof.getAssignmentsIdx(vt->getResultSize(), tmp, &nC);
                        vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC);
                        vt->memHostToDeviceAssigns();
                    }

                    if (task_name2 == Dummy)
                    {
                        dummyTask *vt = dynamic_cast<dummyTask *>(task2);
                        int *tmp;
                        tmp = vt->getModulesA(c2);
                        assA = memprof.getAssignmentsIdx(vt->getSizeA(), tmp, &nA);
                        tmp = vt->getModulesB(c2);
                        assB = memprof.getAssignmentsIdx(vt->getSizeA(), tmp, &nB);
                        vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB);
                        vt->memHostToDeviceAssigns();
                    }
                    else if (task_name2 == matrixMul)
                    {
                        matrixMulTask *vt = dynamic_cast<matrixMulTask *>(task2);
                        int *tmp;
                        tmp = vt->getModulesA(c2);
                        assA = memprof.getAssignmentsIdx(vt->getSizeA(), tmp, &nA);

                        tmp = vt->getModulesB(c2);
                        assB = memprof.getAssignmentsIdx(vt->getSizeB(), tmp, &nB);

                        tmp = vt->getModulesC(c2);
                        assC = memprof.getAssignmentsIdx(vt->getSizeC(), tmp, &nC);
                        vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC);
                        vt->memHostToDeviceAssigns();
                    }
                    else if (task_name2 == VA)
                    {
                        vectorAddTask *vt = dynamic_cast<vectorAddTask *>(task2);
                        int *tmp;
                        tmp = vt->getModulesA(c2);
                        assA = memprof.getAssignmentsIdx(vt->getSizeN(), tmp, &nA);

                        tmp = vt->getModulesB(c2);
                        assB = memprof.getAssignmentsIdx(vt->getSizeN(), tmp, &nB);

                        tmp = vt->getModulesC(c2);
                        assC = memprof.getAssignmentsIdx(vt->getSizeN(), tmp, &nC);
                        vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC);
                        vt->memHostToDeviceAssigns();
                    }
                    else if (task_name2 == BS)
                    {
                        BlackScholesTask *vb = dynamic_cast<BlackScholesTask *>(task2);
                        int *tmp;
                        tmp = vb->getModulesA(c2);
                        assA = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nA);

                        tmp = vb->getModulesB(c2);
                        assB = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nB);

                        tmp = vb->getModulesC(c2);
                        assC = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nC);

                        tmp = vb->getModulesD(c2);
                        assD = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nD);

                        tmp = vb->getModulesE(c2);
                        assE = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nE);

                        vb->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC, assD, nD, assE, nE);
                        vb->memHostToDeviceAssigns();
                    }
                    else if (task_name2 == PF)
                    {
                        PathFinderTask *vt = dynamic_cast<PathFinderTask *>(task2);
                        int *tmp;
                        tmp = vt->getModulesA(c2);
                        assA = memprof.getAssignmentsIdx(vt->getDataSize() - vt->getResultSize(), tmp, &nA);
                        tmp = vt->getModulesB(c2);
                        assB = memprof.getAssignmentsIdx(vt->getResultSize(), tmp, &nB);
                        tmp = vt->getModulesC(c2);
                        assC = memprof.getAssignmentsIdx(vt->getResultSize(), tmp, &nC);
                        vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC);
                        vt->memHostToDeviceAssigns();
                    }
                }
                else
                {
                    c1 = comb1;
                    c2 = comb2;
                }

                numLaunchs = 0;
                if (prof == TimerProf)
                {
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    time0 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                    while (elapsed_time < ProfilingTimeThreshold)
                    {
                        clock_gettime(CLOCK_MONOTONIC, &now);
                        time1 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                        task1->kernelExec();
                        task2->kernelExec();
                        clock_gettime(CLOCK_MONOTONIC, &now);
                        time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                        exec_time += time2 - time1;
                        elapsed_time = time2 - time0;
                        numLaunchs++;
                    }
                }
                else if (prof == CUPTIProf)
                {
                    fprintf(stderr, "Continuous mode profiling not supported for concurrent execution, ");
                    fprintf(stderr, "switching to sampling mode\n");
                    prof = CUPTISample;
                }
                else if (prof == CUPTISample)
                {
                    ProfilingTimeThreshold = 15.0;
                    vector<string> event_name{"active_cycles", "inst_executed", "l2_subp0_read_sector_misses", "l2_subp1_read_sector_misses", "l2_subp0_write_sector_misses", "l2_subp1_write_sector_misses"};
                    profiler.open_metric_file("events.txt");
                    for (int i = 0; i < event_name.size(); i++)
                    {
                        profiler.start_cupti_sampler(event_name[i].c_str());
                        task1->setNumLaunchs(0);
                        task2->setNumLaunchs(0);
                        sleep(1);
                        clock_gettime(CLOCK_MONOTONIC, &now);
                        time0 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                        elapsed_time = 0;
                        while (elapsed_time < ProfilingTimeThreshold)
                        {
                            clock_gettime(CLOCK_MONOTONIC, &now);
                            time1 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                            task1->kernelExec();
                            task2->kernelExec();
                            clock_gettime(CLOCK_MONOTONIC, &now);
                            time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                            elapsed_time = time2 - time0;
                        }
                        cudaDeviceSynchronize();
                        clock_gettime(CLOCK_MONOTONIC, &now);
                        time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                        elapsed_time = time2 - time0;
                        profiler.end_cupti_sampler(reftime);
                        printf("Task launchs: 1 %d, 2 %d, elapsed time %f\n", task1->getNumLaunchs(), task2->getNumLaunchs(), elapsed_time);
                    }
                    profiler.close_metric_file();
                }
                else if (prof == NoProfile)
                {
                    cudaDeviceSynchronize();
                    task1->kernelExec();
                    task2->kernelExec();
                }
                else if (prof == EventsProf)
                {
                    ProfilingTimeThreshold = 5.0;
                    task1->setNumLaunchs(0);
                    task2->setNumLaunchs(0);
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    time0 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                    elapsed_time = 0;
                    exec_time = 0;
                    while (elapsed_time < ProfilingTimeThreshold)
                    {
                        clock_gettime(CLOCK_MONOTONIC, &now);
                        time1 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                        task1->kernelExec();
                        task2->kernelExec();
                        clock_gettime(CLOCK_MONOTONIC, &now);
                        time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                        exec_time += time2 - time1;
                        elapsed_time = time2 - time0;
                    }
                    cudaDeviceSynchronize();
                    int key = 100 * c1 + c2;
                    printf("%d %d %d %d %d %d %d\n", nb1, nb2, key, c1, c2, task1->getNumLaunchs(), task2->getNumLaunchs());

                    // printf("Task 1 NumLaunchs %d, Time %f, Elapsed %f, TimePerKernel %f ms, %f ms\n", task1->getNumLaunchs(), exec_time, elapsed_time, 1000 * exec_time / task1->getNumLaunchs(), task1->getKernelElapsedTime());
                    // printf("Task 2 NumLaunchs %d, Time %f, Elapsed %f, TimePerKernel %f ms, %f ms\n", task2->getNumLaunchs(), exec_time, elapsed_time, 1000 * exec_time / task2->getNumLaunchs(), task2->getKernelElapsedTime());
                }
            }
    }
    clock_gettime(CLOCK_MONOTONIC, &now);
    printf("\n%lu\t%lu\t%f s\tTransfering from device\n", now.tv_sec, now.tv_nsec, (1000000000 * now.tv_sec + now.tv_nsec - reftime) * 0.000000001);

    cudaProfilerStop();

    if (mr_mode == None)
    {
        task1->dthTransfer();
        task2->dthTransfer();
    }
    else
    {
        if (task_name1 == Dummy)
        {
            dummyTask *vt = dynamic_cast<dummyTask *>(task1);
            vt->memDeviceToHostAssigns();
        }
        else if (task_name1 == matrixMul)
        {
            matrixMulTask *vtm = dynamic_cast<matrixMulTask *>(task1);
            vtm->memDeviceToHostAssigns();
        }
        else if (task_name1 == BS)
        {
            BlackScholesTask *vb = dynamic_cast<BlackScholesTask *>(task1);
            vb->memDeviceToHostAssigns();
        }
        else if (task_name1 == PF)
        {
            PathFinderTask *vb = dynamic_cast<PathFinderTask *>(task1);
            vb->memDeviceToHostAssigns();
        }

        if (task_name2 == Dummy)
        {
            dummyTask *vt = dynamic_cast<dummyTask *>(task2);
            vt->memDeviceToHostAssigns();
        }
        else if (task_name2 == matrixMul)
        {
            matrixMulTask *vtm = dynamic_cast<matrixMulTask *>(task2);
            vtm->memDeviceToHostAssigns();
        }
        else if (task_name2 == BS)
        {
            BlackScholesTask *vb = dynamic_cast<BlackScholesTask *>(task2);
            vb->memDeviceToHostAssigns();
        }
        else if (task_name2 == PF)
        {
            PathFinderTask *vb = dynamic_cast<PathFinderTask *>(task2);
            vb->memDeviceToHostAssigns();
        }
    }

    // if (mr_mode != None)
    // {
    //     dummyTask *vt = dynamic_cast<dummyTask *>(task1);
    //     vt->memDeviceToHostAssigns();
    //     matrixMulTask *vtm = dynamic_cast<matrixMulTask *>(task2);
    //     vtm->memDeviceToHostAssigns();
    // }
    // else
    // {
    //     task1->dthTransfer();
    //     task2->dthTransfer();
    // }

    if (arguments.powerMeasure)
        nvmlPowerEnd();

    cudaDeviceSynchronize();
    if (task1->checkResults())
        printf("Test passed for task 1\n");
    if (task2->checkResults())
        printf("Test passed for task 2\n");

    switch (prof)
    {
    case NoProfile:
        break;
    case TimerProf:
        printf("Child %lu Kernel %lu NumLaunchs %lu Time %f TimePerKernel %f us\n", (ulong)getpid(), (ulong)task1->getName(), (ulong)numLaunchs, exec_time, 1000000 * exec_time / numLaunchs);
        break;
    case EventsProf:
        float th = task1->getHtDElapsedTime();
        float td = task1->getDtHElapsedTime();
        float tk = task1->getKernelElapsedTime();
        printf("Task 1: HtD %f K %f DtH %f\n", th, tk, td);
        th = task2->getHtDElapsedTime();
        td = task2->getDtHElapsedTime();
        tk = task2->getKernelElapsedTime();
        printf("Task 2: HtD %f K %f DtH %f\n", th, tk, td);
        printf("NumLaunchs 1 %d, 2 %d, Time %f, TimePerKernel 1 %f, 2 %f us\n", task1->getNumLaunchs(), task2->getNumLaunchs(), exec_time, 1000000 * exec_time / task1->getNumLaunchs(), 1000000 * exec_time / task2->getNumLaunchs());
        break;
    }
    exit(0);
}