/**
 * @file memory_profiler.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief 
 * @version 0.1
 * @date 2021-06-04
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "memory_profiler.h"

enum SharedVarValues
{
    FLUSH_SAMPLES,
    START_SAMPLES,
    WAIT_SAMPLES,
    END_SAMPLES,
    FLUSH_KERNEL,
    CONT_KERNEL
};

// Static variables
static CUcontext m_context = 0;
static CUdevice m_device = 0;
static long long numChipAssignments;
static int *chipAssignments = 0;
static const char m_eventName0[] = "l2_subp0_read_sector_misses"; // The even names must be accessible by the sampling function
static const char m_eventName1[] = "l2_subp1_read_sector_misses";
static dim3 thblock(256, 1, 1);
static dim3 blgrid(1, 1, 1);

// Communication between the host and the device
static int *shOffset = 0; // Current offset
static int *shStatus = 0;

// Kernel routine
__global__ void
memProfile(volatile float *ptr, unsigned long long numOffsets, volatile int *status, volatile int *current_offset, float *flush_ptr, int flush_size)
{
    unsigned long long i = threadIdx.x;
    int numrepeats = 0;
    // Flush L2 cache
    int j = i;
    while (j < flush_size)
    {
        flush_ptr[j] = flush_ptr[j] + 1;
        j += blockDim.x;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        *status = FLUSH_SAMPLES;
        //printf("th %d\n", *status);
        __threadfence_system(); // Host thread must be aware of this
        while (*status != CONT_KERNEL)
            ;//__nanosleep(1000);
        //printf("\tth %d\n", *status);
    }
    __syncthreads();

    for (unsigned long long offset = 0; offset < numOffsets; offset++)
    {
        ptr[i + offset * blockDim.x] = ptr[i + offset * blockDim.x] + 1;
        if (threadIdx.x == 0)
        {
            *current_offset = offset;
            *status = START_SAMPLES;
            //printf("th %d off %d\n", *status, *current_offset);
            __threadfence_system(); // Host thread must be aware of this

            while (*status == START_SAMPLES)
                ;//__nanosleep(1000);
            //printf("\tth %d off %d\n", *status, *current_offset);
        }
        __syncthreads();

        if (*status == FLUSH_KERNEL)
        {
            j = i;
            while (j < flush_size)
            {
                flush_ptr[j] = flush_ptr[j] + 1;
                j += blockDim.x;
            }
            numrepeats++;
            if (numrepeats < 10)
                offset--;
//            else if (threadIdx.x == 0)
//                printf("Flushed %d times for offset %llu\n", numrepeats, offset);
            __syncthreads();
            if (threadIdx.x == 0)
            {
                *status = FLUSH_SAMPLES;
                __threadfence_system(); // Host thread must be aware of this
                while (*status != CONT_KERNEL)
                    ;//__nanosleep(1000);
            }
            __syncthreads();
        }
        else
        {
            numrepeats = 0;
        }
    }
    *status = END_SAMPLES;
}

// Static routines

static CUptiResult cuptiErr;
static CUpti_EventGroup eventGroup;
static CUpti_EventID eventId0, eventId1;
static uint32_t numInstances = 0, j = 0;
static size_t valueSize;
static uint64_t eventVal = 0;
static uint32_t profile_all = 1;

static void *sampling_func(void *arg)
{
    size_t bytesRead0, bytesRead1;
    uint64_t *eventValues0 = NULL, *eventValues1 = NULL;

    bytesRead0 = sizeof(uint64_t) * numInstances;
    eventValues0 = (uint64_t *)malloc(bytesRead0);
    bytesRead1 = sizeof(uint64_t) * numInstances;
    eventValues1 = (uint64_t *)malloc(bytesRead1);
    if (eventValues0 == NULL || eventValues1 == NULL)
    {
        printf("%s:%d: Failed to allocate memory.\n", __FILE__, __LINE__);
        exit(-1);
    }
    float perc = 0.1;

    struct timespec now;
    double time0, time1, time2;
    int num_completed = 0;
    clock_gettime(CLOCK_MONOTONIC, &now);
    time0 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
    time1 = time0;
    printf("Profiling %d chunks of memory\n", numChipAssignments);

    while (*shStatus != END_SAMPLES)
    {
        if (*shStatus == START_SAMPLES || *shStatus == FLUSH_SAMPLES)
        {
            // printf("Status %d\n", *shStatus);

            cuptiErr = cuptiEventGroupReadEvent(eventGroup,
                                                CUPTI_EVENT_READ_FLAG_NONE,
                                                eventId0, &bytesRead0, eventValues0);
            CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupReadEvent");
            if (bytesRead0 != (sizeof(uint64_t) * numInstances))
            {
                printf("Failed to read value for \"%s\"\n", m_eventName0);
                exit(-1);
            }

            // printf("\tStatus %d - bytes read %d\n", *shStatus, bytesRead0);

            cuptiErr = cuptiEventGroupReadEvent(eventGroup,
                                                CUPTI_EVENT_READ_FLAG_NONE,
                                                eventId1, &bytesRead1, eventValues1);
            CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupReadEvent");
            if (bytesRead1 != (sizeof(uint64_t) * numInstances))
            {
                printf("Failed to read value for \"%s\"\n", m_eventName1);
                exit(-1);
            }

            if (*shStatus == FLUSH_SAMPLES)
            {
                // printf("Flushed\n");
                *shStatus = CONT_KERNEL;
                usleep(1);
            }
            else if (*shStatus == START_SAMPLES)
            {
                // printf("Started\n");
                // Check consistency
                eventVal = 0;
                unsigned long long expected = 0.8 * (4 * thblock.x / 32) * blgrid.x;
                int numMax = 0, idx = -1;
                int wrong = 0;
                //printf("shOffset %d: ", shOffset[0]);
                for (j = 0; j < numInstances; j++)
                {
                    //printf("%d - %d\t", eventValues0[j], eventValues1[j]);
                    eventVal = eventValues0[j] + eventValues1[j];
                    if (eventVal > expected)
                    {
                        numMax++;
                        idx = j;
                    }
                }
                //printf(" %d > %d - %f x %d\n", numMax, expected, perc, numChipAssignments);
                // if (numMax == 1000)
                { // Get assignments
                //    wrong = 1;
                    if (shOffset[0] < 0 || shOffset[0] >= numChipAssignments)
                    {
                        fprintf(stderr, "Warning: something went wrong, wrong offset %d", shOffset[0]);
                        break;
                    }
                    chipAssignments[shOffset[0]] = idx;
                    if (*shOffset > perc * numChipAssignments)
                    {
                        clock_gettime(CLOCK_MONOTONIC, &now);
                        time2 = (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
                        printf("Elapsed time %f s. Completed %d out of %d (%f samples/s)\r", time2 - time0, *shOffset, numChipAssignments, (*shOffset - num_completed) / (time2 - time1));
                        num_completed = *shOffset;
                        time1 = time2;
                        perc += 0.1;
                    }
                }

                if (wrong == 0)
                {
                    *shStatus = FLUSH_KERNEL;
                }
                else
                    *shStatus = CONT_KERNEL;
                usleep(1);
            }
        }
        else
        {
//            printf("Wtf!\n");
            usleep(1);
        }
    }

    printf("\33[2K\r");
    printf("Done. %d chunks were profiled in %f s (%f chunks/s)\n", numChipAssignments, time2 - time0, numChipAssignments / (time2 - time0));

    cuptiErr = cuptiEventGroupDisable(eventGroup);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDisable");

    cuptiErr = cuptiEventGroupDestroy(eventGroup);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupDestroy");

    free(eventValues0);
    free(eventValues1);

    return NULL;
}

// MemoryProfiler routines

void MemoryProfiler::initMemoryProfiler(const int device_num)
{
    int deviceCount;
    char deviceName[32];
    int flag;

    // Init CUDA and set the device
    DRIVER_API_CALL(cuInit(0));
    DRIVER_API_CALL(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0)
    {
        printf("There is no device supporting CUDA.\n");
        exit(-1);
    }
    if (device_num >= deviceCount)
    {
        printf("Device %d does not exist. Device count is %d\n", device_num, deviceCount);
        exit(-2);
    }
    DRIVER_API_CALL(cuDeviceGet(&m_device, device_num));
    DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, m_device));
    DRIVER_API_CALL(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, m_device));
    if (flag == 0)
    {
        fprintf(stderr, "Device %d (%s) does not support mapping CPU host memory!\n", deviceCount, deviceName);

        exit(EXIT_SUCCESS);
    }
    // Use the runtime to set the MapHostMemory flag
    RUNTIME_API_CALL(cudaSetDeviceFlags(cudaDeviceMapHost));
    // Create a context if there isn't one
    DRIVER_API_CALL(cuCtxGetCurrent(&m_context));
    if (m_context == 0)
    {
        printf("There is no CUDA context, creating one\n");
        DRIVER_API_CALL(cuCtxCreate(&m_context, 0, m_device));
    }

    // Allocate and register shared variables between kernel and sampling function
    m_ptr_size = 0;
    shOffset = (int *)malloc(sizeof(int));
    shStatus = (int *)malloc(sizeof(int));
    RUNTIME_API_CALL(cudaHostRegister(shOffset, sizeof(int), CU_MEMHOSTALLOC_DEVICEMAP));
    RUNTIME_API_CALL(cudaHostRegister(shStatus, sizeof(int), CU_MEMHOSTALLOC_DEVICEMAP));
    *shOffset = 0;
    *shStatus = FLUSH_SAMPLES;

    // Memory parameters
    m_numChips = 24;
}

void MemoryProfiler::initMemoryProfiler(const int device_num, float *ptr, const size_t ptr_size)
{
    initMemoryProfiler(device_num);
    setPtr(ptr, ptr_size);
}

void MemoryProfiler::setPtr(float *ptr, const size_t ptr_size)
{
    m_ptr = ptr;
    m_ptr_size = ptr_size;
    m_numAssignments = ptr_size / (256 * sizeof(float));
    m_numAssignmentsPerChip = (int *)calloc(m_numChips, sizeof(int));
    m_chipAssignments = (int *)malloc(m_numAssignments * sizeof(int));

    numChipAssignments = m_numAssignments;
    chipAssignments = (int *)malloc(m_numAssignments * sizeof(int));
}

void MemoryProfiler::profileMemory()
{
    // Start profiling

    // Init CUPTI
    cuptiErr = cuptiSetEventCollectionMode(m_context,
                                           CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiSetEventCollectionMode");
    cuptiErr = cuptiEventGroupCreate(m_context, &eventGroup, 0);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupCreate");
    cuptiErr = cuptiEventGetIdFromName(m_device, m_eventName0, &eventId0);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGetIdFromName");
    cuptiErr = cuptiEventGroupAddEvent(eventGroup, eventId0);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupAddEvent");
    cuptiErr = cuptiEventGetIdFromName(m_device, m_eventName1, &eventId1);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGetIdFromName");
    cuptiErr = cuptiEventGroupAddEvent(eventGroup, eventId1);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupAddEvent");
    cuptiErr = cuptiEventGroupSetAttribute(eventGroup,
                                           CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                           sizeof(profile_all), &profile_all);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupSetAttribute");
    cuptiErr = cuptiEventGroupEnable(eventGroup);
    CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupEnable");

    valueSize = sizeof(numInstances);
    cuptiErr = cuptiEventGroupGetAttribute(eventGroup,
                                           CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                           &valueSize, &numInstances);
    // CHECK_CUPTI_ERROR(cuptiErr, "cuptiEventGroupGetAttribute");

    setThreadsPerBlock(thblock);
    setBlocksPerGrid(blgrid);
    cudaMemset(m_ptr, 0, m_ptr_size);
    int flush_size = 4 * 1024 * 1024;
    float *flush_ptr;
    cudaMalloc((void **)&flush_ptr, flush_size * sizeof(float));
    cudaMemset(flush_ptr, 0, flush_size * sizeof(float));
    cudaStream_t m_stream;
    cudaStreamCreate(&m_stream);
    printf("Launching kernel with grid %d x %d to get %d assignments over %d instances and a flush size of %d\n", m_blocksPerGrid.x, m_threadsPerBlock.x, m_numAssignments, numInstances, flush_size);
    memProfile<<<m_blocksPerGrid, m_threadsPerBlock, 0, m_stream>>>(m_ptr, m_numAssignments, shStatus, shOffset, flush_ptr, flush_size);
    sampling_func(NULL);

    cudaDeviceSynchronize();

    cudaFree(flush_ptr);

    // Collect results
    // printf("\nNum %d\n", m_numAssignments);
    for (int i = 0; i < m_numAssignments; i++)
    {
        m_chipAssignments[i] = chipAssignments[i];
        // printf("%d - %d\n", i, m_chipAssignments[i]);
    }

    m_offsetMR = 0;

    for (int i = 0; i < m_numAssignments; i++)
        m_numAssignmentsPerChip[m_chipAssignments[i]]++;

    for (int i = 0; i < m_numChips; i++)
    {
        if (m_offsetMR < m_numAssignmentsPerChip[i])
            m_offsetMR = m_numAssignmentsPerChip[i];
    }

    m_chipMR = (int *)calloc(m_offsetMR * m_numChips, sizeof(int));

    int *c = (int *)calloc(m_numChips, sizeof(int));

    for (int i = 0; i < m_numAssignments; i++)
    {
        int a = m_chipAssignments[i];
        int idx = a * m_offsetMR + c[a];
        m_chipMR[idx] = i;
        c[a]++;
    }

    free(c);

    cuptiFinalize();
}

void MemoryProfiler::writeAssignments()
{
    sprintf(m_filenameA, "Assigns_%d_%p.txt", getpid(), m_ptr);
    sprintf(m_filenameM, "Ranges_%d_%p.txt", getpid(), m_ptr);
    writeAssignments(m_filenameA, m_filenameM);
}

void MemoryProfiler::writeAssignments(char *filenameA, char *filenameM)
{
    m_fpA = fopen(filenameA, "w");
    m_fpM = fopen(filenameM, "w");

    int a = fprintf(m_fpA, "%llu\n", m_numAssignments);

    for (int i = 0; i < m_numAssignments; i++)
        fprintf(m_fpA, "%d\n", m_chipAssignments[i]);

    fprintf(m_fpM, "%d\n", m_numChips);
    for (int i = 0; i < m_numChips; i++)
        fprintf(m_fpM, "%d\t", m_numAssignmentsPerChip[i]);
    fprintf(m_fpM, "\n");

    for (int i = 0; i < m_numChips; i++)
        for (int j = 0; j < m_numAssignmentsPerChip[i]; j++)
            fprintf(m_fpM, "%d\n", m_chipMR[i * m_offsetMR + j]);

    fclose(m_fpA);
    fclose(m_fpM);
}

void MemoryProfiler::readAssignments()
{
    sprintf(m_filenameA, "Assigns_%d_%p.txt", getpid(), m_ptr);
    sprintf(m_filenameM, "Ranges_%d_%p.txt", getpid(), m_ptr);
    readAssignments(m_filenameA, m_filenameM);
}

void MemoryProfiler::readAssignments(char *filenameA, char *filenameM)
{
    m_fpA = fopen(filenameA, "r");
    m_fpM = fopen(filenameM, "r");

    if (m_chipAssignments)
        free(m_chipAssignments);

    fscanf(m_fpA, "%llu\n", &m_numAssignments);
    m_chipAssignments = (int *)malloc(m_numAssignments * sizeof(int));
    for (int i = 0; i < m_numAssignments; i++)
        fscanf(m_fpA, "%d\n", &(m_chipAssignments[i]));

    if (m_numAssignmentsPerChip)
        free(m_numAssignmentsPerChip);

    fscanf(m_fpM, "%d\n", &m_numChips);
    m_numAssignmentsPerChip = (int *)calloc(m_numChips, sizeof(int));
    long long total = 0;
    for (int i = 0; i < m_numChips; i++)
    {
        fscanf(m_fpM, "%d\t", &(m_numAssignmentsPerChip[i]));
        if (total < m_numAssignmentsPerChip[i])
            total = m_numAssignmentsPerChip[i];
    }

    m_offsetMR = total;

    if (m_chipMR)
        free(m_chipMR);
    m_chipMR = (int *)calloc(m_offsetMR * m_numChips, sizeof(int));

    for (int i = 0; i < m_numChips; i++)
        for (int j = 0; j < m_numAssignmentsPerChip[i]; j++)
            fscanf(m_fpM, "%d\n", &(m_chipMR[i * m_offsetMR + j]));

    fclose(m_fpA);
    fclose(m_fpM);
}

static int mycmpfunc(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

/**
 * @brief Assignments are like page tables for virtual addresses. The mapping
 *        has two levels and the page size is 1024 bytes (256 entries). First
 *        level has just one page with pointers to the 256 tables of second
 *        level. This function returns the indices of the assigned page tables
 * 
 * @param numChips 
 * @param numChipAssignments 
 * @param chipMR 
 */
int *MemoryProfiler::getAssignmentsIdx(size_t bytes, int *modules, int *num_indices)
{
    int chunksize = 1024; // This should be a parameter
    int numAssignments = (bytes + chunksize - 1) / chunksize + 257;
    int i = 0, j = 0, left = 0;
    int *indices = NULL, *first = NULL;

    indices = (int *)malloc(numAssignments * sizeof(int));
    first = (int *)calloc(m_numChips, sizeof(int));

    // Check if there are enough assignments
    for (int i = 0; i < m_numChips; i++)
        if (modules[i] > 0)
            for (int j = 0; j < m_numAssignmentsPerChip[i]; j++)
                if (m_chipMR[i * m_offsetMR + j] >= 0)
                    left++;
    if (left < numAssignments)
    {
        fprintf(stderr, "Not enough assignments left (requested %d, available %d)\n", numAssignments, left);
        return (indices);
    }

    // Look for first valid assignment in each module
    for (int i = 0; i < m_numChips; i++)
    {
        j = 0;
        while (m_chipMR[i * m_offsetMR + j] < 0)
            j++;
        first[i] = j;
    }

    // Get the assignments
    while (1)
    {
        for (int k = 0; k < m_numChips; k++)
        {
            if (modules[k] > 0)
            {
                if (m_chipMR[k * m_offsetMR + first[k]] >= 0)
                {
                    indices[i] = m_chipMR[k * m_offsetMR + first[k]];
                    m_chipMR[k * m_offsetMR + first[k]] = -1;
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
