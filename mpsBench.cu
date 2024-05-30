/**
 * @file mpsBench.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief 
 * @version 0.1
 * @date 2021-06-10
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include <stdio.h>  /* printf()                 */
#include <stdlib.h> /* exit(), malloc(), free() */
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <sys/wait.h>
#include <sys/types.h> /* key_t, sem_t, pid_t      */
#include <errno.h>     /* errno, ECHILD            */
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

sem_t *semMutex, *semBarrier; /* sync barrier semaphores */
sem_t *semTurnstile1, *semTurnstile2;
struct sharedStr
{
    int shSync, shNum, shTurn; /*      shared variables         */
    int shTid1, shTid2;
    int shNumComb1, shNumComb2;
    int shPid1, shPid2;
    int shNumLaunchs1, shNumLaunchs2;
    int numProcs;
};

struct sharedStr *shStr;

void barrierAll()
{
    sem_wait(semMutex);
    shStr->shNum = shStr->shNum + 1;
    // printf("Soy %d bar with %d\n", getpid(), shStr->shNum);
    if (shStr->shNum == shStr->numProcs)
    {
        shStr->shNum = 0;
        sem_post(semBarrier);
    }
    sem_post(semMutex);

    sem_wait(semBarrier);
    sem_post(semBarrier);
}

void turnstileIn()
{
    sem_wait(semMutex);
    shStr->shTurn = shStr->shTurn + 1;
    // printf("Soy %d in with %d\n", getpid(), shStr->shTurn);
    if (shStr->shTurn == shStr->numProcs)
    {
        for (int i = 0; i < shStr->numProcs; i++)
            sem_post(semTurnstile1);
    }
    sem_post(semMutex);

    sem_wait(semTurnstile1);
}

void turnstileOut()
{
    sem_wait(semMutex);
    shStr->shTurn = shStr->shTurn - 1;
    // printf("Soy %d out with %d\n", getpid(), shStr->shTurn);
    if (shStr->shTurn == 0)
    {
        for (int i = 0; i < shStr->numProcs; i++)
            sem_post(semTurnstile2);
    }
    sem_post(semMutex);

    sem_wait(semTurnstile2);
}

const char *argp_program_version =
    "mpsBench 0.1";

/* Program documentation. */
static char doc[] =
    "mpsBench -- a program to benchmark the execution of a kernel using MPS";

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
} arguments;

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
    arg->blocksPerGrid = -1;
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

int change_thread_percentage(int percentage)
{
    FILE *server_list = NULL;
    char server_string[256], command_string[256];
    int server_pid, status;
    server_list = popen("echo get_server_list | nvidia-cuda-mps-control", "r");
    if (!server_list)
    {
        perror("Error reading MPS server list");
        exit(-1);
    }
    fgets(server_string, 1000, server_list);
    while (!feof(server_list))
    {
        server_pid = atoi(server_string);
        fgets(server_string, 1000, server_list);
    }
    sprintf(command_string, "echo set_active_thread_percentage %d %d | nvidia-cuda-mps-control > /dev/null", server_pid, percentage);
    status = system(command_string);
    return (status);
}

void *launchTask(void *arg)
{

    // Select device
    cudaError_t err;
    int deviceId = arguments.deviceID;

    // Init memory profiler at the beginning to set MapHostMemory
    float *d_ptr = 0;
    MemoryProfiler memprof;
    memprof.initMemoryProfiler(deviceId);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    // printf("Working on Device %s. Memory size %llu \n", deviceProp.name, deviceProp.totalGlobalMem);

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

    if (prof == CUPTISample)
        if (getpid() == shStr->shPid2)
            prof = EventsProf;

    CUDAtaskNames task_name;
    if (getpid() == shStr->shPid1)
        task_name = arguments.taskID1;
    else
        task_name = arguments.taskID2;
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

    setupCUDATask(t, arguments.nElements, arguments.blocksPerGrid);

    if (mr_mode == None)
        t->setExecMode(Original);
    else
        t->setExecMode(MemoryRanges);

    t->allocHostMemory();
    t->dataGeneration();

    clock_gettime(CLOCK_MONOTONIC, &now);
    printf("%lu\t%lu\tAllocating device memory\n", now.tv_sec, now.tv_nsec);

    if (mr_mode == None)
        t->allocDeviceMemory();
    else
    {
        int size = getTaskSize(t);
        size *= 7;

        err = cudaMalloc((void **)&d_ptr, size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        memprof.setPtr(d_ptr, size);

        // int tmp;
        // sem_getvalue(semMutex, &tmp);
        sem_wait(semMutex); // Only one process profiling with CUPTI
        memprof.profileMemory();
        memprof.writeAssignments();
        sem_post(semMutex);

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
    dim3 b = t->getBlocksPerGrid();

    if (getpid() == shStr->shPid1)
        t->setAllModules(0);
    else
        t->setAllModules(1);
    comb = t->getNumCombinations();

    if (mr_mode != None)
    {
        if (getpid() == shStr->shPid1)
            shStr->shNumComb1 = comb;
        else
            shStr->shNumComb2 = comb;
    }
    else
    {
        if (getpid() == shStr->shPid1)
            shStr->shNumComb1 = 1;
        else
            shStr->shNumComb2 = 1;
    }
    // printf("Soy %d (%d, %d) -> %d, %d\n", getpid(), shStr->shPid1, shStr->shPid2, shStr->shNumComb1, shStr->shNumComb2);
    // printf("%d ready\n", getpid());
    // barrierAll();
    // printf("%d go\n", getpid());
    turnstileIn();
    turnstileOut();

    int nm = -1;
    while (1)
    {
        // printf("%d ready to launch tasks\n", getpid());

        turnstileIn();
        if (getpid() == shStr->shPid1)
            nm = shStr->shNumComb1;
        else
            nm = shStr->shNumComb2;
        // printf("%d wants to use comb %d of %d (%d procs)\n", getpid(), nm, comb, shStr->shNum);
        if (nm == -1)
            break;
        if (mr_mode != None)
            sendTaskData(t, d_ptr, &memprof, nm);
        else
            nm = 24;
        // printf("%d sent data\n", getpid());

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
            vector<string> event_name{/*"active_cycles", "inst_executed",*/ "l2_subp0_read_sector_misses" /*, "l2_subp1_read_sector_misses", "l2_subp0_write_sector_misses", "l2_subp1_write_sector_misses"*/};
            char filename[256];
            sprintf(filename, "events%d.txt", getpid());
            profiler.open_metric_file(filename);
            for (int i = 0; i < event_name.size(); i++)
            {
                profiler.start_cupti_sampler(event_name[i].c_str());
                t->setNumLaunchs(0);
                // sleep(1);
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
                printf("%d %d %d %d %f\n", getpid(), deviceProp.multiProcessorCount, nm, t->getNumLaunchs(), elapsed_time);
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
            clock_gettime(CLOCK_MONOTONIC, &now);
            time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
            elapsed_time = time2 - time0;
            // float tk = t->getKernelElapsedTime();
            int key_comb = shStr->shNumComb1 * 100 + shStr->shNumComb2;
            // printf("%d %d %d %d %d %d %f %f %f\n", key_comb, getpid(), perc, deviceProp.multiProcessorCount, nm, t->getNumLaunchs(), elapsed_time, 1000 * exec_time / t->getNumLaunchs(), tk);
            if (shStr->shPid1 == getpid())
                shStr->shNumLaunchs1 = t->getNumLaunchs();
            else
                shStr->shNumLaunchs2 = t->getNumLaunchs();
            // printf("NumLaunchs %d, Time %f, Elapsed %f, TimePerKernel %f ms\n", t->getNumLaunchs(), exec_time, elapsed_time, 1000 * exec_time / t->getNumLaunchs());
        }
        turnstileOut();
    }

    clock_gettime(CLOCK_MONOTONIC, &now);
    printf("%lu\t%lu\t%f s\tTransfering from device\n", now.tv_sec, now.tv_nsec, (1000000000 * now.tv_sec + now.tv_nsec - reftime) * 0.000000001);

    cudaProfilerStop();
    cudaDeviceSynchronize();

    t->dthTransfer();

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

    // printf("%d bye %d, %d\n", getpid(), shStr->shNumComb1, shStr->shNumComb2);
    return NULL;
}

#define OBJ_PERMS (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)

int main(int argc, char **argv)
{
    arguments.deviceID = 0;
    arguments.taskID1 = Dummy;
    arguments.taskID2 = Dummy;
    arguments.blocksPerGrid = -1;
    arguments.mrMode = None;
    arguments.nElements = 1024;
    arguments.output_file = 0;
    arguments.powerMeasure = 0;
    arguments.profMode = NoProfile;
    arguments.quiet = 0;
    arguments.verbose = 0;
    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    printArg(arguments, argv);

    /* initialize a shared variable in shared memory */
    key_t shmkey; /*      shared memory key       */
    int shmid;    /*      shared memory id        */
    pid_t pid;    /*      fork pid                */
    //int *shSync; /*      shared variable         */ /*shared */
    unsigned int value = 1; /*      semaphore value         */

    shmkey = ftok("/dev/null", 5); /* valid directory name and a number */
    shmid = shmget(shmkey, sizeof(struct sharedStr), 0644 | IPC_CREAT);
    if (shmid < 0)
    { /* shared memory error check */
        perror("shmget\n");
        /* shared memory detach */
        if (shmctl(shmid, IPC_RMID, 0) < 0)
            perror("shmctl\n");
        exit(1);
    }

    shStr = (struct sharedStr *)shmat(shmid, NULL, 0); /* attach p to shared memory */
    if (shStr == (void *)-1)
    { /* shared memory error check */
        perror("shmat\n");
        /* shared memory detach */
        if (shmctl(shmid, IPC_RMID, 0) < 0)
            perror("shmctl\n");
        exit(1);
    }
    shStr->numProcs = 3; // Master + 2 slaves
    shStr->shNum = 0;
    shStr->shTurn = 0;
    shStr->shNumComb1 = 0;
    shStr->shNumComb1 = 0;
    shStr->shPid1 = 0;
    shStr->shPid2 = 0;
    shStr->shSync = 0;

    /* initialize semaphores for sync barrier */
    semMutex = sem_open("pSemMutex", O_CREAT | O_EXCL, 0644, value);
    semBarrier = sem_open("pSemBarrier", O_CREAT | O_EXCL, 0644, 0);
    semTurnstile1 = sem_open("pSemTurnstile1", O_CREAT | O_EXCL, 0644, 0);
    semTurnstile2 = sem_open("pSemTurnstile2", O_CREAT | O_EXCL, 0644, 0);
    if (semMutex == SEM_FAILED || semBarrier == SEM_FAILED || semTurnstile1 == SEM_FAILED || semTurnstile2 == SEM_FAILED)
    {
        printf("Error opening semaphores. Releasing and trying again\n");
        // cleanup semaphores
        sem_unlink("pSemMutex");
        sem_close(semMutex);
        sem_unlink("pSemBarrier");
        sem_close(semBarrier);
        sem_unlink("pSemTurnstile1");
        sem_close(semTurnstile1);
        sem_unlink("pSemTurnstile2");
        sem_close(semTurnstile2);
        // Try again
        semMutex = sem_open("pSemMutex", O_CREAT | O_EXCL, 0644, value);
        semBarrier = sem_open("pSemBarrier", O_CREAT | O_EXCL, 0644, 0);
        semTurnstile1 = sem_open("pSemTurnstile1", O_CREAT | O_EXCL, 0644, 0);
        semTurnstile2 = sem_open("pSemTurnstile2", O_CREAT | O_EXCL, 0644, 0);
        if (semMutex == SEM_FAILED || semBarrier == SEM_FAILED || semTurnstile1 == SEM_FAILED || semTurnstile2 == SEM_FAILED)
        {
            printf("Error again, exiting\n");
            // cleanup semaphores
            sem_unlink("pSemMutex");
            sem_close(semMutex);
            sem_unlink("pSemBarrier");
            sem_close(semBarrier);
            sem_unlink("pSemTurnstile1");
            sem_close(semTurnstile1);
            sem_unlink("pSemTurnstile2");
            sem_close(semTurnstile2);
            /* shared memory detach */
            shmdt(shStr);
            shmctl(shmid, IPC_RMID, 0);
            exit(-1);
        }
    }

    int perc = 120 / (shStr->numProcs - 1);
    change_thread_percentage(perc);
    for (int i = 0; i < shStr->numProcs - 1; i++)
    {
        pid = fork();
        if (pid < 0)
        {
            /* check for error      */
            /* shared memory detach */
            shmdt(shStr);
            shmctl(shmid, IPC_RMID, 0);
            /* cleanup semaphores */
            sem_unlink("pSemMutex");
            sem_close(semMutex);
            sem_unlink("pSemBarrier");
            sem_close(semBarrier);
            sem_unlink("pSemTurnstile1");
            sem_close(semTurnstile1);
            sem_unlink("pSemTurnstile2");
            sem_close(semTurnstile2);
            /* unlink prevents the semaphore existing forever */
            /* if a crash occurs during the execution         */
            printf("Fork error.\n");
        }
        else if (pid == 0)
            break;
        if (i == 0)
            shStr->shPid1 = pid;
        else
            shStr->shPid2 = pid;
    }

    if (pid != 0)
    {
        // CuptiProfiler profiler;
        // vector<string> event_name{"l2_subp0_read_sector_misses", "l2_subp1_read_sector_misses", "l2_subp0_write_sector_misses", "l2_subp1_write_sector_misses"};
        // profiler.open_metric_file("events.txt");
        printf("Host ready\n");
        turnstileIn();
        int nm1 = shStr->shNumComb1;
        int nm2 = shStr->shNumComb2;
        // profiler.init_cupti_sampler(arguments.deviceID);
        turnstileOut();
        for (int i = 0; i < nm1; i++)
            for (int j = 0; j < nm2; j++)
            // for (int i = 6; i < 7; i++)
            //     for (int j = 5; j < 6; j++)
            {
                sem_wait(semMutex);
                shStr->shNumComb1 = i;
                shStr->shNumComb2 = j;
                sem_post(semMutex);
                // printf("Comb %d %d\n", i, j);
                turnstileIn();
                // profiler.start_cupti_sampler(event_name[0].c_str());
                turnstileOut();
                int key = 100 * shStr->shNumComb1 + shStr->shNumComb2;
                printf("%d %d %d %d %d\n", key, shStr->shNumComb1, shStr->shNumComb2, shStr->shNumLaunchs1, shStr->shNumLaunchs2);
                // profiler.end_cupti_sampler(0);
            }
        // profiler.close_metric_file();
        sem_wait(semMutex);
        shStr->shNumComb1 = -1;
        shStr->shNumComb2 = -1;
        sem_post(semMutex);
        turnstileIn();
        int retval;
        // printf("Host waiting\n");
        while (pid = waitpid(-1, &retval, 0))
        {
            if (errno == ECHILD)
                break;
        }

        printf("\nParent: All children have exited.\n");
        change_thread_percentage(100);

        /* shared memory detach */
        shmdt(shStr);
        shmctl(shmid, IPC_RMID, 0);

        /* cleanup semaphores */
        sem_unlink("pSemMutex");
        sem_close(semMutex);
        sem_unlink("pSemBarrier");
        sem_close(semBarrier);
        sem_unlink("pSemTurnstile1");
        sem_close(semTurnstile1);
        sem_unlink("pSemTurnstile2");
        sem_close(semTurnstile2);
        /* unlink prevents the semaphore existing forever */
        /* if a crash occurs during the execution         */
        exit(0);
    }
    else
    {
        launchTask(NULL);

        exit(0);
    }
}