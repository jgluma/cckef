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

        // Create task
        ProfileMode prof = EventsProf;
        CUDAtaskNames task_name = VA;
        CKEmode mode = SYNC;
        CUDAtask *t = createCUDATask(task_name);
        t->setPinned(true);
        t->setProfileMode(prof);
        t->setCKEMode(mode);
        vectorAddTask *vt = dynamic_cast<vectorAddTask*>(t);
        vt->setNumElements(16*1024*1024);

        t->allocHostMemory();
        t->dataGeneration();
        t->allocDeviceMemory();
        t->htdTransfer();

        // Barrier
		*p += 1;
		while (*p < num_kernels); // Spin lock
		
        printf("Starting profiling\n");
		// Solo original profiling
        double ProfilingTimeThreshold = 1.0; // Kernels are launched many times during this interval
        struct timespec now;
        double time0, time1, time2, elapsed_time = 0.0, exec_time = 0.0;
        unsigned long int numLaunchs = 0;

        if ( prof == TimerProf ) {
            clock_gettime(CLOCK_REALTIME, &now);
            while ( elapsed_time < ProfilingTimeThreshold ) {
                clock_gettime(CLOCK_REALTIME, &now);
                time0 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
                time1 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
                t->kernelExec();
                clock_gettime(CLOCK_REALTIME, &now);
                time2 = (double)now.tv_sec+(double)now.tv_nsec*1e-9;
                exec_time += time2 - time1;
                elapsed_time = time2 - time0;
                numLaunchs++;
            }
        }
        else
            t->kernelExec();

        t->dthTransfer();
        t->checkResults();

        switch(prof) {
        case NoProfile:
            break;
        case TimerProf:
        	printf("Child %lu Kernel %lu NumLaunchs %lu Time %f TimePerKernel %f\n", (ulong) getpid(), (ulong) t->getName(), (ulong) numLaunchs, exec_time, exec_time/numLaunchs);
            break;
        case EventsProf:
            float th = t->getHtDElapsedTime();
            float td = t->getDtHElapsedTime();
            float tk = t->getKernelElapsedTime();
            printf("HtD %f K %f DtH %f\n", th, tk, td);
            break;
        }
		cudaDeviceSynchronize();
		exit(0);
    }
}