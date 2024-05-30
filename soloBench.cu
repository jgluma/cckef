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

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    printf("Working on Device %s. Memory size %llu \n", deviceProp.name, deviceProp.totalGlobalMem);

    // Profilers
    vector<string> event_names{};
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
    CUDAtask *t = createCUDATask(task_name, deviceId);

    t->setPinned(true);
    t->setProfileMode(EventsProf);
    t->setProfileMode(prof);
    t->setCKEMode(cke_mode);
    t->setMRMode(mr_mode);

    if (task_name == Dummy)
    {
        // Dummy task parameters
        dummyTask *vt = dynamic_cast<dummyTask *>(t);
        vt->setCB(false);
        vt->setMB(true);
        if (arguments.nElements != 1024)
            vt->setNumElements(1024 * arguments.nElements);
        if (arguments.blocksPerGrid == 0)
            vt->setPersistentBlocks();
        else
        {
            dim3 b(arguments.blocksPerGrid, 1, 1);
            vt->setBlocksPerGrid(b);
        }
        vt->setNumIterations(100); // CB 6000  MB 12000
    }
    else if (task_name == matrixMul)
    {
        matrixMulTask *vt = dynamic_cast<matrixMulTask *>(t);
        // vt->setComputeAssigns(true);
        dim3 dA, dB;
        if (arguments.nElements != 1024)
        {
            dA.x = arguments.nElements;
            dA.y = arguments.nElements;
            dB.x = arguments.nElements;
            dB.y = arguments.nElements;
            vt->setMatrixDims(dA, dB);
        }
        if (arguments.blocksPerGrid == 0)
            vt->setPersistentBlocks();
        else
        {
            dim3 b(arguments.blocksPerGrid, 1, 1);
            vt->setBlocksPerGrid(b);
        }
        // vt->setPartitions(0, 1, 0);
    }
    else if (task_name == VA)
    {
        vectorAddTask *vt = dynamic_cast<vectorAddTask *>(t);
        if (arguments.nElements != 1024)
            vt->setNumElements(arguments.nElements * 1024);
        if (arguments.blocksPerGrid == 0)
            vt->setPersistentBlocks();
        else
        {
            dim3 b(arguments.blocksPerGrid, 1, 1);
            vt->setBlocksPerGrid(b);
        }
        vt->setNumIterations(100); // CB 6000  MB 12000
    }
    else if (task_name == BS)
    {
        BlackScholesTask *vb = dynamic_cast<BlackScholesTask *>(t);
        if (arguments.nElements != 1024)
            vb->setOptions(arguments.nElements, 256);
        if (arguments.blocksPerGrid == 0)
            vb->setPersistentBlocks();
        else
        {
            dim3 b(arguments.blocksPerGrid, 1, 1);
            vb->setBlocksPerGrid(b);
        }
    }
    else if (task_name == PF)
    {
        PathFinderTask *vt = dynamic_cast<PathFinderTask *>(t);
        if (arguments.nElements != 1024)
            vt->setParameters(500, arguments.nElements, 126);
        if (arguments.blocksPerGrid == 0)
            vt->setPersistentBlocks();
        else
        {
            dim3 b(arguments.blocksPerGrid, 1, 1);
            vt->setBlocksPerGrid(b);
        }
    }

    t->allocHostMemory();
    t->dataGeneration();

    clock_gettime(CLOCK_MONOTONIC, &now);
    printf("%lu\t%lu\tAllocating device memory\n", now.tv_sec, now.tv_nsec);

    if (mr_mode == None)
        t->allocDeviceMemory();
    else
    {
        // Create memory partitions

        int size = 0;
        if (task_name == Dummy)
        {
            dummyTask *vt = dynamic_cast<dummyTask *>(t);
            size += vt->getSizeA() + vt->getSizeB();
        }
        else if (task_name == matrixMul)
        {
            matrixMulTask *vtm = dynamic_cast<matrixMulTask *>(t);
            size += vtm->getSizeA() + vtm->getSizeB();
        }
        else if (task_name == VA)
        {
            vectorAddTask *vt = dynamic_cast<vectorAddTask *>(t);
            size += 3 * vt->getSizeN();
        }
        else if (task_name == BS)
        {
            BlackScholesTask *vb = dynamic_cast<BlackScholesTask *>(t);
            size += 6 * vb->getSizeN();
        }
        else if (task_name == PF)
        {
            fprintf(stderr, "Memory ranges for PathFinder is not supported\n");
            exit(-1);
        }

        size += 2 * 257 * 1024;
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

        cuptiFinalize();
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
        t->htdTransfer();

    cudaDeviceSynchronize();

    if (prof == CUPTIProf)
        profiler.init_cupti_profiler(deviceId);
    else if (prof == CUPTISample)
        profiler.init_cupti_sampler(deviceId);

    clock_gettime(CLOCK_MONOTONIC, &now);
    printf("%lu\t%lu\t%f s\tExecuting kernel\n", now.tv_sec, now.tv_nsec, (1000000000 * now.tv_sec + now.tv_nsec - reftime) * 0.000000001);

    // Solo original profiling

    unsigned long int numLaunchs = 0;
    int comb = 0;
    for (int nb = 80; nb <= 640; nb += 80)
    {
        t->setProfileMode(EventsProf);
        dim3 b(nb, 1, 1);
        t->setBlocksPerGrid(b);
        // printf("Blocks %d", nb);

        if (task_name == Dummy)
        {
            dummyTask *vt = dynamic_cast<dummyTask *>(t);
            vt->setAllModules(0);
            comb = vt->getNumCombinations();
        }
        else if (task_name == VA)
        {
            vectorAddTask *vt = dynamic_cast<vectorAddTask *>(t);
            vt->setAllModules(0);
            comb = vt->getNumCombinations();
        }
        else if (task_name == BS)
        {
            BlackScholesTask *vb = dynamic_cast<BlackScholesTask *>(t);
            vb->setAllModules(0);
            comb = vb->getNumCombinations();
        }

        for (int nm = 0; nm < comb; nm++)
        {
            if (mr_mode != None)
            {
                // printf("\tMemory modules %d\n", nm);
                memprof.readAssignments();
                int nA, nB, nC, nD, nE;
                int *assA = 0, *assB = 0, *assC = 0, *assD = 0, *assE = 0;
                if (task_name == Dummy)
                {
                    dummyTask *vt = dynamic_cast<dummyTask *>(t);
                    int *tmp;
                    tmp = vt->getModulesA(nm);
                    // printf("A: ");
                    // for (int k = 0; k < 24; k++)
                    //     printf("%d ", tmp[k]);
                    assA = memprof.getAssignmentsIdx(vt->getSizeA(), tmp, &nA);
                    tmp = vt->getModulesB(nm);
                    // printf("\nB: ");
                    // for (int k = 0; k < 24; k++)
                    //     printf("%d ", tmp[k]);
                    // printf("\n");
                    assB = memprof.getAssignmentsIdx(vt->getSizeA(), tmp, &nB);
                    vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB);
                    vt->memHostToDeviceAssigns();
                }
                else if (task_name == matrixMul)
                {
                    matrixMulTask *vtm = dynamic_cast<matrixMulTask *>(t);
                    if (nm == 24)
                        vtm->setModules(t->selectModules(0, 1, nm), t->selectModules(0, 1, nm), t->selectModules(0, 1, nm));
                    // else if (nm == 12)
                    //     vtm->setModules(t->selectModules(0, 2, nm), t->selectModules(1, 2, nm), t->selectModules(0, 1, 24));
                    else
                        vtm->setModules(t->selectModules(0, 2, nm), t->selectModules(1, 2, nm), t->selectModules(0, 1, 24));
                    assA = memprof.getAssignmentsIdx(vtm->getSizeA(), vtm->getModulesA(), &nA);
                    assB = memprof.getAssignmentsIdx(vtm->getSizeB(), vtm->getModulesB(), &nB);
                    nC = 0;
                    // mb->getAssignmentsIdx(vtm->getSizeC(), vtm->getModulesC(), &nC);
                    vtm->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC);
                    vtm->memHostToDeviceAssigns();
                }
                else if (task_name == VA)
                {
                    vectorAddTask *vt = dynamic_cast<vectorAddTask *>(t);
                    int *tmp;
                    tmp = vt->getModulesA(nm);
                    assA = memprof.getAssignmentsIdx(vt->getSizeN(), tmp, &nA);

                    tmp = vt->getModulesB(nm);
                    assB = memprof.getAssignmentsIdx(vt->getSizeN(), tmp, &nB);

                    tmp = vt->getModulesC(nm);
                    assC = memprof.getAssignmentsIdx(vt->getSizeN(), tmp, &nC);
                    vt->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC);
                    vt->memHostToDeviceAssigns();
                }
                else if (task_name == BS)
                {
                    BlackScholesTask *vb = dynamic_cast<BlackScholesTask *>(t);
                    int *tmp;
                    tmp = vb->getModulesA(nm);
                    assA = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nA);

                    tmp = vb->getModulesB(nm);
                    assB = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nB);

                    tmp = vb->getModulesC(nm);
                    assC = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nC);

                    tmp = vb->getModulesD(nm);
                    assD = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nD);

                    tmp = vb->getModulesE(nm);
                    assE = memprof.getAssignmentsIdx(vb->getSizeN(), tmp, &nE);

                    vb->allocDeviceMemory(d_ptr, assA, nA, assB, nB, assC, nC, assD, nD, assE, nE);
                    vb->memHostToDeviceAssigns();
                }
            }
            else
                nm = 24;

            if (prof == TimerProf)
            {
                clock_gettime(CLOCK_MONOTONIC, &now);
                time0 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                while (elapsed_time < ProfilingTimeThreshold)
                {
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    time1 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                    t->kernelExec();
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
                    t->setNumLaunchs(0);
                    profiler.start_kernelduration_cupti_profiler();
                    t->kernelExec();
                    int dur = profiler.end_kernelduration_cupti_profiler();
                    printf("\tKernel duration %f s\t", (float)dur / 1e09);
                    fflush(stdout);
                    CUpti_EventGroupSets *passData = profiler.start_cupti_profiler(metric_names[i].c_str());
                    int num_passes = passData->numSets;
                    profiler.advance_cupti_profiler(passData, 0);
                    if (num_passes > 1)
                    {
                        t->kernelExec();
                        for (int j = 1; j < num_passes; j++)
                        {
                            profiler.advance_cupti_profiler(passData, j);
                            t->kernelExec();
                        }
                        profiler.stop_cupti_profiler(true);
                    }
                    else
                    {
                        t->kernelExec();
                        profiler.stop_cupti_profiler(true);
                    }
                    // profiler.print_event_instances();
                    profiler.free_cupti_profiler();
                }
            }
            else if (prof == CUPTISample)
            {
                ProfilingTimeThreshold = 5.0;
                vector<string> event_name{"active_cycles", "inst_executed", "l2_subp0_read_sector_misses", "l2_subp1_read_sector_misses", "l2_subp0_write_sector_misses", "l2_subp1_write_sector_misses"};
                profiler.open_metric_file("events.txt");
                for (int i = 0; i < event_name.size(); i++)
                {
                    printf("%d %d ", nb, nm);
                    profiler.start_cupti_sampler(event_name[i].c_str());
                    t->setNumLaunchs(0);
                    sleep(1);
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    time0 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                    elapsed_time = 0;
                    while (elapsed_time <= ProfilingTimeThreshold)
                    {
                        clock_gettime(CLOCK_MONOTONIC, &now);
                        time1 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                        t->kernelExec();
                        clock_gettime(CLOCK_MONOTONIC, &now);
                        time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                        elapsed_time = time2 - time0;
                    }
                    cudaDeviceSynchronize();
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                    elapsed_time = time2 - time0;
                    profiler.end_cupti_sampler(reftime);
                    // printf("Task launchs: %d, elapsed time %f\n", t->getNumLaunchs(), elapsed_time);
                    printf("%d %f\n", t->getNumLaunchs(), elapsed_time);
                }
                profiler.close_metric_file();
            }
            else if (prof == NoProfile)
                t->kernelExec();
            else if (prof == EventsProf)
            {
                ProfilingTimeThreshold = 5.0;
                t->setNumLaunchs(0);
                clock_gettime(CLOCK_MONOTONIC, &now);
                time0 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                elapsed_time = 0;
                exec_time = 0;
                while (elapsed_time < ProfilingTimeThreshold)
                {
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    time1 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                    t->kernelExec();
                    clock_gettime(CLOCK_MONOTONIC, &now);
                    time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                    exec_time += time2 - time1;
                    elapsed_time = time2 - time0;
                }
                cudaDeviceSynchronize();
                float tk = t->getKernelElapsedTime();
                printf("NumLaunchs %d, Time %f, Elapsed %f, TimePerKernel %f ms, %f ms\n", t->getNumLaunchs(), exec_time, elapsed_time, 1000 * exec_time / t->getNumLaunchs(), tk);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &now);
    printf("%lu\t%lu\t%f s\tTransfering from device\n", now.tv_sec, now.tv_nsec, (1000000000 * now.tv_sec + now.tv_nsec - reftime) * 0.000000001);

    cudaProfilerStop();
    cudaDeviceSynchronize();

    if (mr_mode == None)
        t->dthTransfer();
    else
    {
        if (task_name == Dummy)
        {
            dummyTask *vt = dynamic_cast<dummyTask *>(t);
            vt->memDeviceToHostAssigns();
        }
        else if (task_name == matrixMul)
        {
            matrixMulTask *vtm = dynamic_cast<matrixMulTask *>(t);
            vtm->memDeviceToHostAssigns();
        }
        else if (task_name == VA)
        {
            vectorAddTask *vt = dynamic_cast<vectorAddTask *>(t);
            vt->memDeviceToHostAssigns();
        }
        else if (task_name == BS)
        {
            BlackScholesTask *vb = dynamic_cast<BlackScholesTask *>(t);
            vb->memDeviceToHostAssigns();
        }
    }

    cudaDeviceSynchronize();
    if (t->checkResults())
        printf("Test passed\n");

    if (arguments.powerMeasure)
        nvmlPowerEnd();

    switch (prof)
    {
    case NoProfile:
        break;
    case TimerProf:
        printf("Child %lu Kernel %lu NumLaunchs %lu Time %f TimePerKernel %f us\n", (ulong)getpid(), (ulong)t->getName(), (ulong)numLaunchs, exec_time, 1000000 * exec_time / numLaunchs);
        break;
    case EventsProf:
        float th = t->getHtDElapsedTime();
        float td = t->getDtHElapsedTime();
        float tk = t->getKernelElapsedTime();
        printf("HtD %f K %f DtH %f\n", th, tk, td);
        printf("NumLaunchs %d, Time %f, TimePerKernel %f us\n", t->getNumLaunchs(), exec_time, 1000000 * exec_time / t->getNumLaunchs());
        break;
    }
    exit(0);
}