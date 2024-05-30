#include <stdio.h>  /* printf()                 */
#include <stdlib.h> /* exit(), malloc(), free() */
#include <sys/wait.h>
#include <sys/types.h> /* key_t, sem_t, pid_t      */
#include <errno.h>     /* errno, ECHILD            */

#include <cupti.h>
#include <helper_functions.h> // helper functions for string parsing
#include <helper_cuda.h>

#include "profile/memory_profiler.h"

__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        const int numIter = 1000;
        float kk = 0;
        for (int j = 0; j < numIter; j++)
            kk += (A[i] + B[i])/(A[i]*B[i]);
        C[i] = kk;
    }
}

__global__ void
checkSMS(unsigned int *sm)
{
    if (threadIdx.x == 0)
    {
        unsigned int ret;
        asm("mov.u32 %0, %smid;"
            : "=r"(ret));
        unsigned int *ptr = sm + ret;
        atomicAdd(ptr, 1);
    }
}

void *launchVectorAdd(void *arg)
{
    float *h_ptrA;
    float *h_ptrB;
    float *h_ptrC;

    float *d_ptrA;
    float *d_ptrB;
    float *d_ptrC;

    int deviceId = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    printf("Device num SMs: %d\n", deviceProp.multiProcessorCount);

    int numElements = 256 * 1024 * 1024;
    int size = numElements * sizeof(float);
    cudaMallocHost((void **)&h_ptrA, size);
    cudaMallocHost((void **)&h_ptrB, size);
    cudaMallocHost((void **)&h_ptrC, size);

    cudaMalloc((void **)&d_ptrA, size);
    cudaMalloc((void **)&d_ptrB, size);
    cudaMalloc((void **)&d_ptrC, size);

    dim3 thr(256, 1, 1);
    dim3 blk((numElements + thr.x - 1) / thr.x, 1, 1);
    cudaStream_t m_stream;
    cudaStreamCreate(&m_stream);

    int numIter = 10;
    cudaMemcpyAsync(d_ptrA, h_ptrA, size, cudaMemcpyHostToDevice, m_stream);
    cudaMemcpyAsync(d_ptrB, h_ptrB, size, cudaMemcpyHostToDevice, m_stream);
    printf("Launching vectorAdd\n");
    for (int i = 0; i < numIter; i++)
    {
        vectorAdd<<<blk, thr, 0, m_stream>>>(d_ptrC, d_ptrA, d_ptrB, size);
    }
    cudaDeviceSynchronize();
    printf("Completed\n");
    cudaMemcpyAsync(h_ptrC, d_ptrC, size, cudaMemcpyDeviceToHost, m_stream);
    return NULL;
}

void *another_launch(void *arg)
{
    float *d_ptr = 0;
    MemoryProfiler memprof;
    int size = 256 * 1024;

    memprof.initMemoryProfiler(0);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Working on Device %s. Memory size %llu \n", deviceProp.name, deviceProp.totalGlobalMem);

    cudaMalloc((void **)&d_ptr, size);

    memprof.setPtr(d_ptr, size);
    memprof.profileMemory();
    memprof.writeAssignments();

    cuptiFinalize();

    return NULL;
}

void *sampling_func(void *arg)
{
    CUcontext m_context = 0;
    CUdevice m_device = 0;
    int deviceCount, deviceID = 0;
    char deviceName[32];
    int flag;

    // Init CUDA and set the device
    cuInit(0);
    cuDeviceGetCount(&deviceCount);
    if (deviceCount == 0)
    {
        printf("There is no device supporting CUDA.\n");
        exit(-1);
    }
    if (deviceID >= deviceCount)
    {
        printf("Device %d does not exist. Device count is %d\n", deviceID, deviceCount);
        exit(-1);
    }
    cuDeviceGet(&m_device, deviceID);
    cuDeviceGetName(deviceName, 32, m_device);
    cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, m_device);
    if (flag == 0)
    {
        fprintf(stderr, "Device %d (%s) does not support mapping CPU host memory!\n", deviceCount, deviceName);
        exit(EXIT_SUCCESS);
    }
    // Use the runtime to set the MapHostMemory flag
    cudaSetDeviceFlags(cudaDeviceMapHost);
    // Create a context if there isn't one
    cuCtxGetCurrent(&m_context);
    if (m_context == 0)
    {
        printf("There is no CUDA context, creating one\n");
        cuCtxCreate(&m_context, 0, m_device);
    }

    int *shOffset = (int *)malloc(sizeof(int));
    int *shStatus = (int *)malloc(sizeof(int));
    cudaHostRegister(shOffset, sizeof(int), CU_MEMHOSTALLOC_DEVICEMAP);
    cudaHostRegister(shStatus, sizeof(int), CU_MEMHOSTALLOC_DEVICEMAP);

    float *d_ptrA = 0, *d_ptrB = 0, *d_ptrC = 0;
    int size = 256 * 1024;
    cudaMalloc((void **)&d_ptrA, size * sizeof(float));
    cudaMalloc((void **)&d_ptrB, size * sizeof(float));
    cudaMalloc((void **)&d_ptrC, size * sizeof(float));

    dim3 thr(256, 1, 1);
    dim3 blk(1, 1, 1);
    cudaStream_t m_stream;
    cudaStreamCreate(&m_stream);
    printf("Launch kernel\n");
    vectorAdd<<<blk, thr, 0, m_stream>>>(d_ptrC, d_ptrA, d_ptrB, size);

    CUptiResult cuptiErr;
    CUpti_EventGroup eventGroup;
    CUpti_EventID eventId0, eventId1;
    const char m_eventName0[] = "l2_subp0_read_sector_misses"; // The even names must be accessible by the sampling function
    const char m_eventName1[] = "l2_subp1_read_sector_misses";
    uint32_t numInstances = 0, j = 0;
    size_t bytesRead0, bytesRead1, valueSize;
    uint64_t *eventValues0 = NULL, *eventValues1 = NULL, eventVal = 0;
    uint32_t profile_all = 1;

    printf("Set-up CUPTI\n");
    cuptiErr = cuptiSetEventCollectionMode(m_context,
                                           CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS);
    printf("1\n");

    cuptiErr = cuptiEventGroupCreate(m_context, &eventGroup, 0);
    printf("2\n");

    cuptiErr = cuptiEventGetIdFromName(m_device, m_eventName0, &eventId0);
    printf("3\n");

    cuptiErr = cuptiEventGroupAddEvent(eventGroup, eventId0);
    printf("4\n");

    cuptiErr = cuptiEventGetIdFromName(m_device, m_eventName1, &eventId1);
    printf("5\n");

    cuptiErr = cuptiEventGroupAddEvent(eventGroup, eventId1);
    printf("6\n");

    cuptiErr = cuptiEventGroupSetAttribute(eventGroup,
                                           CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                           sizeof(profile_all), &profile_all);
    printf("7\n");

    cuptiErr = cuptiEventGroupEnable(eventGroup);
    printf("8\n");

    valueSize = sizeof(numInstances);
    cuptiErr = cuptiEventGroupGetAttribute(eventGroup,
                                           CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                           &valueSize, &numInstances);
    printf("9\n");

    cuptiErr = cuptiEventGroupDisable(eventGroup);

    cuptiErr = cuptiEventGroupDestroy(eventGroup);

    cuptiFinalize();
}

int change_thread_percentage(int percentage)
{
    FILE *server_list = NULL;
    char server_string[256], command_string[256];
    int server_pid;
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
    //printf("%s\n", command_string);
    int status = system(command_string);
    return (status);
}

void *launchCheckSMS(void *arg)
{
    unsigned int *h_ptrSMS = 0;
    unsigned int *d_ptrSMS = 0;
    int numSMS = 80;
    int sizeSMS = numSMS * sizeof(unsigned int);
    dim3 thr(256, 1, 1);
    dim3 blk(1024, 1, 1);
    cudaStream_t m_stream;

    cudaStreamCreate(&m_stream);
    cudaMallocHost((void **)&h_ptrSMS, sizeSMS);
    cudaMalloc((void **)&d_ptrSMS, sizeSMS);

    for (int perc = 20; perc <= 100; perc += 20)
    {
        change_thread_percentage(perc);

        for (int i = 0; i < numSMS; i++)
            h_ptrSMS[i] = 0;

        cudaMemcpyAsync(d_ptrSMS, h_ptrSMS, sizeSMS, cudaMemcpyHostToDevice, m_stream);

        checkSMS<<<blk, thr, 0, m_stream>>>(d_ptrSMS);

        cudaDeviceSynchronize();
        cudaMemcpyAsync(h_ptrSMS, d_ptrSMS, sizeSMS, cudaMemcpyDeviceToHost, m_stream);
        printf("MPS perc %d:\t", perc);
        for (int i = 0; i < numSMS; i++)
            printf("%d\t", h_ptrSMS[i]);
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    /* initialize a shared variable in shared memory */
    key_t shmkey;                                         /*      shared memory key       */
    int shmid;                                            /*      shared memory id        */
    sem_t *sem, *semt; /*      synch semaphore         */ /*shared */
    pid_t pid;                                            /*      fork pid                */
    int *p; /*      shared variable         */            /*shared */
    unsigned int value = 1;                               /*      semaphore value         */

    shmkey = ftok("/dev/null", 5); /* valid directory name and a number */
    //printf ("shmkey for p = %d\n", shmkey);
    shmid = shmget(shmkey, sizeof(int), 0644 | IPC_CREAT);
    if (shmid < 0)
    { /* shared memory error check */
        perror("shmget\n");
        exit(1);
    }

    p = (int *)shmat(shmid, NULL, 0); /* attach p to shared memory */
    *p = 0;

    /* initialize semaphores for shared processes */
    sem = sem_open("pSem", O_CREAT | O_EXCL, 0644, value);
    /* name of semaphore is "pSem", semaphore is reached using this name */
    semt = sem_open("tSem", O_CREAT | O_EXCL, 0644, 0);

    int numProcs = 1;
    if (argc > 1)
        numProcs = atoi(argv[1]);
    for (int i = 0; i < numProcs; i++)
    {

        pid = fork();
        if (pid < 0)
        { /* check for error      */
            sem_unlink("pSem");
            sem_close(sem);
            sem_unlink("tSem");
            sem_close(semt);
            /* unlink prevents the semaphore existing forever */
            /* if a crash occurs during the execution         */
            printf("Fork error.\n");
        }
        else if (pid == 0)
            break;
    }

    if (pid > 0)
    {
        int retval;
        while (pid = waitpid(-1, &retval, 0))
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
        sem_unlink("tSem");
        sem_close(semt);
        /* unlink prevents the semaphore existing forever */
        /* if a crash occurs during the execution         */
    }
    else
    {
        int tmp;
        int pidt = getpid();
        sem_wait(sem);
        tmp = sem_getvalue(sem, &tmp);
        printf("%d 1: %d\n", pidt, tmp);
        change_thread_percentage(50);
        launchVectorAdd(NULL);
        sem_post(sem);
        tmp = sem_getvalue(sem, &tmp);
        printf("%d Post: %d\n", pidt, tmp);
        tmp = sem_getvalue(semt, &tmp);
        printf("%d Post t: %d\n", pidt, tmp);
        if (sem_post(semt) < 0)
            printf("Error\n");
        tmp = sem_getvalue(semt, &tmp);
        printf("%d Wait: %d\n", pidt, tmp);
        sem_wait(semt);
        tmp = sem_getvalue(sem, &tmp);
        printf("%d Wait 2: %d\n", pidt, tmp);
        sem_post(sem);
        printf("%d 2\n", pidt);
        change_thread_percentage(100);
        launchVectorAdd(NULL);
        exit(0);
    }
}
