/**
 * @file ckeTest.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief
 * @version 0.1
 * @date 2021-03-12
 *
 */
#include <stdio.h>          /* printf()                 */
#include <stdlib.h>         /* exit(), malloc(), free() */
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>      /* key_t, sem_t, pid_t      */
#include <sys/shm.h>        /* shmat(), IPC_RMID        */
#include <errno.h>          /* errno, ECHILD            */
#include <semaphore.h>      /* sem_open(), sem_destroy(), sem_wait().. */
#include <fcntl.h>          /* O_CREAT, O_EXEC          */
#include <pthread.h>
#include <time.h>

#include <cuda_profiler_api.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>   

#include "tasks/cuda_tasks.h"
#include "profile/nvmlPower.hpp"
#include "profile/cupti_profiler.h"
#include "memBench/memBench.h"
#include "dummy/dummy.h"

//#include "profile/cupti_profiler.h"

int main (int argc, char **argv)
{
	int it;                        /*      loop variables          */
    key_t shmkey;                 /*      shared memory key       */
    int shmid;                    /*      shared memory id        */
    sem_t *sem;                   /*      synch semaphore         *//*shared */
    pid_t pid;                    /*      fork pid                */
    int *p;                       /*      shared variable         *//*shared */

    int num_kernels = 1;
    if ( argc > 2 ) num_kernels = atoi(argv[2]);

    /* initialize a shared variable in shared memory */
    shmkey = ftok ("/dev/null", 5);       /* valid directory name and a number */
    printf ("shmkey for p = %d\n", shmkey);
    shmid = shmget (shmkey, sizeof (int), 0644 | IPC_CREAT);
    if (shmid < 0){                           /* shared memory error check */
        perror ("shmget\n");
        exit (1);
    }

    p = (int *) shmat (shmid, NULL, 0);   /* attach p to shared memory */
    *p = 0;

    /* initialize semaphores for shared processes */
    sem = sem_open ("pSem", O_CREAT | O_EXCL, 0644, 1); // Binary semaphore 
    /* name of semaphore is "pSem", semaphore is reached using this name */

     /* fork child processes */
    for (it = 0; it < num_kernels; it++){
        pid = fork ();
        if (pid < 0) {
        /* check for error      */
            sem_unlink ("pSem");   
            sem_close(sem);  
            /* unlink prevents the semaphore existing forever */
            /* if a crash occurs during the execution         */
            fprintf (stderr, "Fork error.\n");
        }
        else if (pid == 0)
            break;                  /* child processes */
    }

    /* PARENT PROCESS */
    if (pid != 0){
        /* wait for all children to exit */
        while (pid = waitpid (-1, NULL, 0)){
            if (errno == ECHILD)
                break;
        }

        /* shared memory detach */
        shmdt (p);
        shmctl (shmid, IPC_RMID, 0);

        /* cleanup semaphores */
        sem_unlink ("pSem");   
        sem_close(sem);  
        /* unlink prevents the semaphore existing forever */
        /* if a crash occurs during the execution         */
        exit (0);
    }
    /* CHILD PROCESS */
    else{
		// Select device
		cudaError_t err;
		int deviceId = 0;
		if ( argc > 1 ) deviceId = atoi(argv[1]);
		cudaSetDevice(deviceId);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, deviceId);	
		printf("Working on Device %s\n", deviceProp.name);

        memBench mb;
        mb.init(deviceId);
        mb.getChipAssignments();
        mb.getMemoryRanges();
        mb.writeAssignments();
        exit(0);

        nvmlAPIRun();

        // Timers
        double ProfilingTimeThreshold = 1.0; // Kernels are launched many times during this interval
        struct timespec now;
        double time0, time1, time2, elapsed_time = 0.0, exec_time = 0.0;

        // CUPTI
        vector<string> event_names {
 //           "active_cycles"
           };
        vector<string> metric_names {
             "l2_read_transactions",
//             "ipc"
            };
//        cupti_profiler::profiler profiler(event_names, metric_names);        
//        int passes = profiler.get_passes();
//        std::vector<std::string> curr_metric = init_cupti_profiler( deviceId );
//        std::string mis_metricas[2] = {"l2_read_transactions", "ipc"};
      
        // Create task
        ProfileMode prof = CUPTIProf;//CUPTIProf; //TimerProf;//EventsProf;
        CUDAtaskNames task_name = VA;
        CKEmode cke_mode = ASYNC; 
        MemoryRangeMode mr_mode = None;//None; //Shared;
        CUDAtask *t = createCUDATask(task_name);
        t->setPinned(true);
        t->setProfileMode(prof);
        t->setCKEMode(cke_mode);
        t->setMRMode(mr_mode);
        vectorAddTask *vt = dynamic_cast<vectorAddTask*>(t);
        vt->setNumElements(16*16*1024);

        t->allocHostMemory();
        t->dataGeneration();

        clock_gettime(CLOCK_MONOTONIC, &now);
        printf("%lu\t%lu\tAllocating device memory\n", now.tv_sec, now.tv_nsec);

        t->allocDeviceMemory();

        clock_gettime(CLOCK_MONOTONIC, &now);
        printf("%lu\t%lu\tTransfering to device\n", now.tv_sec, now.tv_nsec);

        t->htdTransfer();

        // Barrier
		*p += 1;
		while (*p < num_kernels); // Spin lock
		
        clock_gettime(CLOCK_MONOTONIC, &now);
        printf("%lu\t%lu\tExecuting kernel\n", now.tv_sec, now.tv_nsec);

		// Solo original profiling

        unsigned long int numLaunchs = 0;
        if ( prof == TimerProf ) {
            clock_gettime(CLOCK_MONOTONIC, &now);
            time0 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
            while ( elapsed_time < ProfilingTimeThreshold ) {
                clock_gettime(CLOCK_MONOTONIC, &now);
                time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
                t->kernelExec();
                clock_gettime(CLOCK_MONOTONIC, &now);
                time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
                exec_time += time2 - time1;
                elapsed_time = time2 - time0;
                numLaunchs++;
            }
        }
        else if ( prof == CUPTIProf ) {
//           FILE *fp = open_metric_file( "metric_values.log" );
//           fprintf(fp, "MetricName, EventName, Sum, TotalInstances, NumInstances, Normalized, Values, ...\n"); 
//            for ( int i = 0; i < curr_metric.size(); i++ )
for ( int j = 0; j < 1; j++ )
{
    vt->kk = j;
//            for ( int i = 0; i < 1; i++ )
            {
                printf("%d", j);
//                CUpti_EventGroupSets *passData = start_cupti_profiler( curr_metric[i].c_str() );
//               CUpti_EventGroupSets *passData = start_cupti_profiler( mis_metricas[i].c_str() );
//                int num_passes = passData->numSets ;
                // advance_cupti_profiler( passData, 0 );
                // if ( num_passes > 1 ) {
                //     stop_cupti_profiler( false );       
                //     printf("Ignoring metric %s because it needs %d passes\n", curr_metric[i].c_str(), num_passes);
                // } else {
                //     t->kernelExec();
                //     stop_cupti_profiler( true );
                //     printf("Max %llu\n", getMaxIdxEvent());
                // }
//                 profiler.start();
//                 for ( int i = 0; i < passes; ++i )
//                     t->kernelExec();
//                 profiler.stop();
//                 printf("Event Trace\n");
//   profiler.print_event_values(std::cout);
//   printf("Metric Trace\n");
//   profiler.print_metric_values(std::cout);
//   auto names = profiler.get_kernel_names();
//   for(auto name: names) {
//     printf("\n%s ", name.c_str());
//     std::vector<uint64_t> inst = profiler.get_event_instances(name.c_str());
//     printf("Instances:\n");
//     for (auto i = inst.begin(); i != inst.end(); ++i)
//       std::cout << "\t" << *i;
//     }
//     std::cout << "\n";
  
            }
        }
//            close_metric_file();
        }

        clock_gettime(CLOCK_MONOTONIC, &now);
        printf("%lu\t%lu\tTransfering from device\n", now.tv_sec, now.tv_nsec);

        t->dthTransfer();

		cudaDeviceSynchronize();
        t->checkResults();
        nvmlAPIEnd();

        // const auto event_names_all = cupti_profiler::available_events(deviceId);
        // const auto metric_names_all = cupti_profiler::available_metrics(deviceId);
        // std::cout << "Events: " << std::endl;
        // for (auto i = event_names_all.begin(); i != event_names_all.end(); ++i)
        //     std::cout << "\t" << *i << std::endl;
        // std::cout << "Metrics: " << std::endl;
        // for (auto i = metric_names_all.begin(); i != metric_names_all.end(); ++i)
        //     std::cout << "    " << *i << std::endl;

        switch(prof) {
        case NoProfile:
            break;
        case TimerProf:
        	printf("Child %lu Kernel %lu NumLaunchs %lu Time %f TimePerKernel %f us\n", (ulong) getpid(), (ulong) t->getName(), (ulong) numLaunchs, exec_time, 1000000*exec_time/numLaunchs);
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