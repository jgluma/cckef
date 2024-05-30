/**
 * @file ckeTest.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief
 * @version 0.1
 * @date 2021-03-12
 *
 */
#include <stdio.h>  /* printf()                 */
#include <stdlib.h> /* exit(), malloc(), free() */
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h> /* key_t, sem_t, pid_t      */
#include <sys/shm.h>   /* shmat(), IPC_RMID        */
#include <errno.h>     /* errno, ECHILD            */
#include <semaphore.h> /* sem_open(), sem_destroy(), sem_wait().. */
#include <fcntl.h>     /* O_CREAT, O_EXEC          */
#include <pthread.h>
#include <time.h>

#include <cuda_profiler_api.h>
#include <helper_functions.h> // helper functions for string parsing
#include <helper_cuda.h>

#include "tasks/cuda_tasks.h"
#include "profile/nvmlPower.hpp"
#include "profile/cupti_profiler.h"
#include "memBench/memBench.h"
#include "dummy/dummy.h"

//#include "profile/cupti_profiler.h"

int main(int argc, char **argv)
{
    int it;                                        /*      loop variables          */
    key_t shmkey;                                  /*      shared memory key       */
    int shmid;                                     /*      shared memory id        */
    sem_t *sem; /*      synch semaphore         */ /*shared */
    pid_t pid;                                     /*      fork pid                */
    int *p; /*      shared variable         */     /*shared */

    int num_kernels = 1;
    if (argc > 2)
        num_kernels = atoi(argv[2]);

    /* initialize a shared variable in shared memory */
    shmkey = ftok("/dev/null", 5); /* valid directory name and a number */
    printf("shmkey for p = %d\n", shmkey);
    shmid = shmget(shmkey, sizeof(int), 0644 | IPC_CREAT);
    if (shmid < 0)
    { /* shared memory error check */
        perror("shmget\n");
        exit(1);
    }

    p = (int *)shmat(shmid, NULL, 0); /* attach p to shared memory */
    *p = 0;

    /* initialize semaphores for shared processes */
    sem = sem_open("pSem", O_CREAT | O_EXCL, 0644, 1); // Binary semaphore
    /* name of semaphore is "pSem", semaphore is reached using this name */

    /* fork child processes */
    for (it = 0; it < num_kernels; it++)
    {
        pid = fork();
        if (pid < 0)
        {
            /* check for error      */
            sem_unlink("pSem");
            sem_close(sem);
            /* unlink prevents the semaphore existing forever */
            /* if a crash occurs during the execution         */
            fprintf(stderr, "Fork error.\n");
        }
        else if (pid == 0)
            break; /* child processes */
    }

    /* PARENT PROCESS */
    if (pid != 0)
    {
        /* wait for all children to exit */
        while (pid = waitpid(-1, NULL, 0))
        {
            if (errno == ECHILD)
                break;
        }

        /* shared memory detach */
        shmdt(p);
        shmctl(shmid, IPC_RMID, 0);

        /* cleanup semaphores */
        sem_unlink("pSem");
        sem_close(sem);
        /* unlink prevents the semaphore existing forever */
        /* if a crash occurs during the execution         */
        exit(0);
    }
    /* CHILD PROCESS */
    else
    {
        // Select device
        cudaError_t err;
        int deviceId = 0;
        if (argc > 1)
            deviceId = atoi(argv[1]);
        cudaSetDevice(deviceId);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceId);
        printf("Working on Device %s\n", deviceProp.name);

        // Memory ranges
        // memBench *mb = new memBench();
        // mb->readAssignments();
        // mb->init(deviceId);
        // mb->getChipAssignments();
        // mb->getMemoryRanges();
        // printf("Write\n");
        // mb->writeAssignments();
        // exit(0);

        // Profilers
        vector<string> event_names{};
        vector<string> metric_names{"dram_read_transactions", "l2_read_transactions", "dram_write_transactions", "l2_write_transactions"};
        CuptiProfiler profiler;

        // Timers
        double ProfilingTimeThreshold = 1.0; // Kernels are launched many times during this interval
        struct timespec now;
        double time0, time1, time2, elapsed_time = 0.0, exec_time = 0.0;

        // Create task
        ProfileMode prof = NoProfile; //CUPTIProf; //TimerProf;//EventsProf;
        if (argc > 3)
        {
            int mprof = atoi(argv[3]);
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
            default:
                break;
            }
        }

        CUDAtaskNames task_name = Dummy;
        CKEmode cke_mode = SYNC;
        MemoryRangeMode mr_mode = Shared; //None; //Shared;
        if (argc > 4)
        {
            int m_mr_mode = atoi(argv[4]);
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
        }
        CUDAtask *t = createCUDATask(task_name, deviceId);
        t->setPinned(false);
        t->setProfileMode(prof);
        t->setCKEMode(cke_mode);
        t->setMRMode(mr_mode);

        // t->setAssignments(mb->getNumChips(), mb->returnNumChipAssignments(), mb->returnChipMR());
        // delete (mb);
        //      vectorAddTask *vt = dynamic_cast<vectorAddTask*>(t);
        //      vt->setNumElements(16*16*1024);

        dummyTask *vt = dynamic_cast<dummyTask *>(t);
        vt->setCB(false);
        vt->setMB(true);
        vt->setComputeAssigns(true);
        if (argc > 5)
            vt->setNumElements(1024 * atoi(argv[5]));
        else
            vt->setNumElements(1024 * 1024 * 32);

        if (argc > 6)
        {
            dim3 b(atoi(argv[6]), 1, 1);
            vt->setBlocksPerGrid(b);
        }
        else
            vt->setPersistentBlocks();

        vt->setNumIterations(125000); // CB 6000  MB 125000
        t->allocHostMemory();
        t->dataGeneration();

        clock_gettime(CLOCK_MONOTONIC, &now);
        printf("%lu\t%lu\tAllocating device memory\n", now.tv_sec, now.tv_nsec);

        t->allocDeviceMemory();

if ( mr_mode != None)
{
    printf("Wait 30 seconds\n");
    sleep(30);
}
    // printf("Wait 5 seconds\n");
    // sleep(4);
    // nvmlAPIRun();
    // sleep(1);
    nvmlPowerInit();
    //nvmlPowerStart();

    clock_gettime(CLOCK_MONOTONIC, &now);
    long long reftime = getRefTime();

    printf("%lu\t%lu\t%f s\tTransfering to device\n", now.tv_sec, now.tv_nsec, (1000000000 * now.tv_sec + now.tv_nsec - reftime) * 0.000000001);
    t->htdTransfer();

    if (prof == CUPTIProf)
        profiler.init_cupti_profiler(deviceId);

    // Barrier
    *p += 1;
    while (*p < num_kernels)
        ; // Spin lock

    clock_gettime(CLOCK_MONOTONIC, &now);
    printf("%lu\t%lu\t%f s\tExecuting kernel\n", now.tv_sec, now.tv_nsec, (1000000000 * now.tv_sec + now.tv_nsec - reftime) * 0.000000001);

    // Solo original profiling

    unsigned long int numLaunchs = 0;
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
                printf("Profiling %s\n", metric_names[i].c_str());
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
                    profiler.stop_cupti_profiler(false);
//                    printf("Ignoring metric %s because it needs %d passes\n", metric_names[i].c_str(), num_passes);
                }
                else
                {
                    t->kernelExec();
                    profiler.stop_cupti_profiler(false);
                }
                profiler.print_event_instances();
                profiler.free_cupti_profiler();
            }
        }
        else if (prof == NoProfile)
            t->kernelExec();

        clock_gettime(CLOCK_MONOTONIC, &now);
        printf("%lu\t%lu\t%f s\tTransfering from device\n", now.tv_sec, now.tv_nsec, (1000000000 * now.tv_sec + now.tv_nsec - reftime) * 0.000000001);

        t->dthTransfer();

        cudaDeviceSynchronize();
        if (t->checkResults())
            printf("Test passed\n");
        //      sleep(1);
        //nvmlAPIEnd();
        nvmlPowerEnd();

        // const auto event_names_all = cupti_profiler::available_events(deviceId);
        // const auto metric_names_all = cupti_profiler::available_metrics(deviceId);
        // std::cout << "Events: " << std::endl;
        // for (auto i = event_names_all.begin(); i != event_names_all.end(); ++i)
        //     std::cout << "\t" << *i << std::endl;
        // std::cout << "Metrics: " << std::endl;
        // for (auto i = metric_names_all.begin(); i != metric_names_all.end(); ++i)
        //     std::cout << "    " << *i << std::endl;

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
            break;
        }
        exit(0);
    }
}