/**
 * @file cuda_task_class.cu
 * @author José María González Linares (jgl@uma.es)
 * @brief 
 * @version 0.1
 * @date 2021-04-06
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "cuda_task_class.h"

int *CUDAtask::selectModules(int init, int step, int n)
{
    int last = init + step * (n - 1);
    int *modules = NULL;
    if (init < 0 || init > 23 || last > 23)
    {
        fprintf(stderr, "Valid memory modules are 0 to 23: %d, %d, %d\n", init, step, n);
        return modules;
    }

    modules = (int *)calloc(24, sizeof(int));
    for (int i = init; i <= last; i += step)
    {
        modules[i] = 1;
    }
    return modules;
}

/**
 * @brief Send page tables from host to device
 * 
 * @param d_ptr Device memory pointer
 * @param h_ptr Host memory pointer with page tables entries
 * @param numEntries Number of valid entries (including entries for page tables)
 */

void CUDAtask::sendPageTables(void *d_ptr, void *h_ptr, int numEntries)
{
    int *d_iptr = (int *)d_ptr; // Cast to integer pointer to use pointer arithmetic
    int *h_iptr = (int *)h_ptr;

    // First level page table entries are stored from h_iptr[1] to h_iptr[256]
    int *entries = (int *)(&h_iptr[1]);
    if (cke_mode == SYNC)
        err = cudaMemcpy(d_iptr, entries, chunkSize, cudaMemcpyHostToDevice);
    else
        err = cudaMemcpyAsync(d_iptr, entries, chunkSize, cudaMemcpyHostToDevice, t_stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy first level page table from host to device (error code %s)!\n", cudaGetErrorString(err));
        fprintf(stderr, "\t%p <- %p, %d...\n", d_iptr, entries, entries[0]);
        for (int i = 0; i < 256; i++)
            fprintf(stderr, "%d\t", entries[i]);
        fprintf(stderr, "\nChunksize %d CKE mode %d\n", chunkSize, cke_mode);
        exit(EXIT_FAILURE);
    }
    // Allocate memory for second level page tables
    int *page = NULL;
    if (cke_mode == SYNC)
        page = (int *)malloc(256 * chunkInts * sizeof(int));
    else
    {
        err = cudaMallocHost((void **)&page, 256 * chunkInts * sizeof(int));
        if (err != cudaSuccess)
        {
            fprintf(stderr,
                    "Failed to allocate pinned host vector A (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
    // Init second level page tables
    for (int i = 0; i < numEntries-257; i++)
        page[i] = h_iptr[i + 257];
    for (int i = numEntries-257; i < 256 * chunkInts; i++)
        page[i] = 0;
    // Send second level page tables
    for (int i = 0; i < 256; i++)
    {
        entries = &(page[i * 256]);
        d_iptr = (int *)d_ptr + h_iptr[i + 1] * chunkInts;

        if (cke_mode == SYNC)
            err = cudaMemcpy(d_iptr, entries, chunkSize, cudaMemcpyHostToDevice);
        else
            err = cudaMemcpyAsync(d_iptr, entries, chunkSize, cudaMemcpyHostToDevice, t_stream);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy second level page table from host to device (error code %s)!\n", cudaGetErrorString(err));
            fprintf(stderr, "\t%p <- %p, %d...\n", d_iptr, entries, entries[0]);
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * @brief Send data from host to device. Size of host array must be (numEntries - 257) x chunkInts
 * 
 * @param d_ptr Device memory pointer
 * @param h_ptr Host memory pointer with data
 * @param ptEntries Page table entries
 * @param numEntries Number of valid entries (including entries for page tables)
 *
 */

void CUDAtask::sendData(void *d_ptr, void *h_ptr, int *ptEntries, int numEntries)
{
    for (int i = 257; i < numEntries; i++)
    {
        int *d_iptr = (int *)d_ptr + ptEntries[i] * chunkInts;
        int *h_iptr = (int *)h_ptr + (i - 257) * chunkInts;
        if (cke_mode == SYNC)
            err = cudaMemcpy(d_iptr, h_iptr, chunkSize, cudaMemcpyHostToDevice);
        else
            err = cudaMemcpyAsync(d_iptr, h_iptr, chunkSize, cudaMemcpyHostToDevice, t_stream);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy chunk %d of a vector from host %p to device %p (error code %s)!\n", i-257, h_iptr, d_iptr, cudaGetErrorString(err));
            fprintf(stderr, "\tNumber of entries %d Entry %d\n", numEntries, ptEntries[i]);
            fprintf(stderr, "\t%p <- %p\n", d_ptr, h_ptr);
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * @brief Send data from device to host. Size of host array must be numEntries x chunkInts
 * 
 * @param d_ptr Device memory pointer
 * @param h_ptr Host memory pointer with data
 * @param ptEntries Page table entries
 * @param numEntries Number of valid entries (including entries for page tables)
 */
void CUDAtask::receiveData(void *d_ptr, void *h_ptr, int *ptEntries, int numEntries)
{
    for (int i = 257; i < numEntries; i++)
    {
        int *d_iptr = (int *)d_ptr + ptEntries[i] * chunkInts;
        int *h_iptr = (int *)h_ptr + (i - 257) * chunkInts;
        if (cke_mode == SYNC)
            err = cudaMemcpy(h_iptr, d_iptr, chunkSize, cudaMemcpyDeviceToHost);
        else
            err = cudaMemcpyAsync(h_iptr, d_iptr, chunkSize, cudaMemcpyDeviceToHost, t_stream);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy chunk %d of a vector from device %p to host %p (error code %s)!\n", i, d_iptr, h_iptr, cudaGetErrorString(err));
            fprintf(stderr, "\tNumber of entries %d Entry %d\n", numEntries, ptEntries[i]);
            fprintf(stderr, "\t%p <- %p\n", h_ptr, d_ptr);
            exit(EXIT_FAILURE);
        }
    }
}

int mycmpfunc(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

/**
 * @brief Assignments are like page tables for virtual addresses. The mapping
 *        has two levels and the page size is 1024 bytes (256 entries). First
 *        level has just one page with pointers to the 256 tables of second
 *        level.
 * 
 * @param numChips 
 * @param numChipAssignments 
 * @param chipMR 
 */
void CUDAtask::setAssignments(int numChips, int *numChipAssignments, int *chipMR)
{

    maxChipsAssignments0 = 0; // Maximum number of valid entries
    maxChipsAssignments1 = 0;
    long long maxC = 0; // Maximum number of chunks in one chip

    // Compute the maximum number of second level page tables entries that can
    //   be obtained from numChipAssignments
    for (int i = 0; i < numChips; i++)
    {
        if (maxC < numChipAssignments[i])
            maxC = numChipAssignments[i];
        if (assig_mode == All)
        {
            maxChipsAssignments0 += numChipAssignments[i];
            maxChipsAssignments1 += numChipAssignments[i];
        }
        else if (assig_mode == Half)
        {
            if (i % 2)
                maxChipsAssignments0 += numChipAssignments[i];
            else
                maxChipsAssignments1 += numChipAssignments[i];
        }
        else if (assig_mode == ThreeQuarters)
        {
            if (i % 4)
                maxChipsAssignments0 += numChipAssignments[i];
            else
                maxChipsAssignments1 += numChipAssignments[i];
        }
    }

    // Store the assignments and sort them
    chipAssignments0 = (int *)calloc(maxChipsAssignments0, sizeof(int));
    chipAssignments1 = (int *)calloc(maxChipsAssignments1, sizeof(int));
    int currA = 0, currB = 0, j = 0;
    bool more = true;
    do
    {
        for (int i = 0; i < numChips; i++)
        {
            if (j < numChipAssignments[i])
            {
                if (assig_mode == All)
                {
                    chipAssignments0[currA] = chipMR[i * maxC + j];
                    currA++;
                    chipAssignments1[currB] = chipMR[i * maxC + j];
                    currB++;
                }
                else if (assig_mode == Half)
                {
                    if (i % 2)
                    {
                        chipAssignments0[currA] = chipMR[i * maxC + j];
                        currA++;
                    }
                    else
                    {
                        chipAssignments1[currB] = chipMR[i * maxC + j];
                        currB++;
                    }
                }
                else if (assig_mode == ThreeQuarters)
                {
                    if (i % 4)
                    {
                        chipAssignments0[currA] = chipMR[i * maxC + j];
                        currA++;
                    }
                    else
                    {
                        chipAssignments1[currB] = chipMR[i * maxC + j];
                        currB++;
                    }
                }
            }
        }
        j++;
        if (currA >= maxChipsAssignments0 && currB >= maxChipsAssignments1)
            more = false;
    } while (more);

    qsort(chipAssignments0, maxChipsAssignments0, sizeof(int), mycmpfunc);
    qsort(chipAssignments1, maxChipsAssignments1, sizeof(int), mycmpfunc);
}

/**
 * @brief Transfer header (page tables) from host to device in chunks
 *        This is necessary for output arrays
 * 
 * @param d_A device array address (aligned to the right chunk of memory)
 * @param h_A host array
 * @param count array size in bytes
 * @param chunks array with chunks assignments in device memory
 * @param idx index of first available chunk
 */
void CUDAtask::transferHeaderHtD(void *d_A, void *h_A, size_t count, int *chunks, int idx)
{
    int *d_ptr = (int *)d_A; // Cast to integer pointer to use pointer arithmetic
    if (count > 256 * 256 * chunkSize)
    {
        fprintf(stderr, "Error: Array size %zu larger than %u\n", count, 256 * 256 * chunkSize);
        exit(EXIT_FAILURE); // TODO: add a better exit mechanism
    }
    // Allocate memory for two levels page tables (1 + 256)
    int *header_ptr = (int *)malloc(257 * chunkSize);
    int base = chunks[idx];
    // Init first level and transfer it to the device
    for (int i = 0; i < chunkInts; i++)
        header_ptr[i] = chunks[i + idx + 1] - base;
    if (cke_mode == SYNC)
        err = cudaMemcpy(d_ptr, header_ptr, chunkSize, cudaMemcpyHostToDevice);
    else
        err = cudaMemcpyAsync(d_ptr, header_ptr, chunkSize, cudaMemcpyHostToDevice, t_stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy first level page table from host to device (error code %s)!\n", cudaGetErrorString(err));
        fprintf(stderr, "\t%p <- %d...\n", d_ptr, header_ptr[0]);
        exit(EXIT_FAILURE);
    }
    // Init and transfer second level page tables
    for (int i = 0; i < chunkInts; i++)
    {
        int *tmp = header_ptr + (i + 1) * chunkInts;
        d_ptr = (int *)d_A + header_ptr[i] * chunkInts;
        int chunkIdx = (i + 1) * chunkInts + idx + 1;
        if (chunkIdx > maxChipsAssignments0 || chunkIdx > maxChipsAssignments1)
            chunkIdx = 0;

        for (int j = 0; j < chunkInts; j++)
        {
            tmp[j] = chunks[chunkIdx + j] - base;
        }
        if (cke_mode == SYNC)
            err = cudaMemcpy(d_ptr, tmp, chunkSize, cudaMemcpyHostToDevice);
        else
            err = cudaMemcpyAsync(d_ptr, tmp, chunkSize, cudaMemcpyHostToDevice, t_stream);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy second level page table from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * @brief Transfer data from host to device in chunks
 * 
 * @param d_A device array address (aligned to the right chunk of memory)
 * @param h_A host array
 * @param count array size in bytes
 * @param chunks array with chunks assignments in device memory
 * @param idx index of first available chunk
 */
void CUDAtask::transferChunksHtD(void *d_A, void *h_A, size_t count, int *chunks, int idx)
{
    int *d_ptr = (int *)d_A; // Cast to integer pointer to use pointer arithmetic
    if (count > 256 * 256 * chunkSize)
    {
        fprintf(stderr, "Error: Array size %zu larger than %u\n", count, 256 * 256 * chunkSize);
        exit(EXIT_FAILURE); // TODO: add a better exit mechanism
    }
    // Allocate memory for two levels page tables (1 + 256)
    int *header_ptr = 0; // = (int *)malloc(257 * chunkSize);
    err = cudaMallocHost((void **)&header_ptr, 257 * chunkSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to malloc pinned host memory (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    int base = chunks[idx];
    // Init first level and transfer it to the device
    // printf("Level 1 Page Table\n");
    for (int i = 0; i < chunkInts; i++)
    {
        header_ptr[i] = chunks[i + idx + 1] - base;
        // printf("%d\t", header_ptr[i]);
    }
    // printf("\n");

    if (cke_mode == SYNC)
        err = cudaMemcpy(d_ptr, header_ptr, chunkSize, cudaMemcpyHostToDevice);
    else
    {
        err = cudaMemcpyAsync(d_ptr, header_ptr, chunkSize, cudaMemcpyHostToDevice, t_stream);
    }
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy first level page table from host to device (error code %s)!\n", cudaGetErrorString(err));
        fprintf(stderr, "\t%p <- %d bytes: %d... %d\n", d_ptr, chunkSize, header_ptr[0], header_ptr[chunkInts - 1]);
        exit(EXIT_FAILURE);
    }
    // Init and transfer second level page tables
    for (int i = 0; i < chunkInts; i++)
    {
        int *tmp = header_ptr + (i + 1) * chunkInts;
        d_ptr = (int *)d_A + header_ptr[i] * chunkInts;
        int chunkIdx = (i + 1) * chunkInts + idx + 1;
        if (chunkIdx > maxChipsAssignments0 || chunkIdx > maxChipsAssignments1)
            chunkIdx = 0;

        // if ( chunkIdx > 0 )
        //     printf("Level 2 Page Table %d\n", i);
        for (int j = 0; j < chunkInts; j++)
        {
            tmp[j] = chunks[chunkIdx + j] - base;
            // if ( chunkIdx > 0 )
            //     printf("%d\t", tmp[j]);
        }
        // if ( chunkIdx > 0 )
        //     printf("\n");

        if (cke_mode == SYNC)
            err = cudaMemcpy(d_ptr, tmp, chunkSize, cudaMemcpyHostToDevice);
        else
            err = cudaMemcpyAsync(d_ptr, tmp, chunkSize, cudaMemcpyHostToDevice, t_stream);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "transferChunksHtD: Failed to copy second level page table from host to device (error code %s)!\n", cudaGetErrorString(err));
            fprintf(stderr, "\t%p <- %d bytes: page %d in chunk %d : %d... %d\n", d_ptr, chunkSize, i, chunkIdx, tmp[0], tmp[chunkInts - 1]);
            fprintf(stderr, "\t%p (%d) -  %p - Max %d, %d\n", d_A, header_ptr[i], tmp, maxChipsAssignments0, maxChipsAssignments1);
            exit(EXIT_FAILURE);
        }
    }
    // Transfer data in host array to device memory
    int numChunks = ceil(count / chunkSize);
    for (int i = 0; i < numChunks; i++)
    {
        d_ptr = (int *)d_A + header_ptr[256 + i] * chunkInts;
        int *tmp = (int *)h_A + i * chunkInts;
        if (cke_mode == SYNC)
            err = cudaMemcpy(d_ptr, tmp, chunkSize, cudaMemcpyHostToDevice);
        else
            err = cudaMemcpyAsync(d_ptr, tmp, chunkSize, cudaMemcpyHostToDevice, t_stream);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy chunk %d of a vector from host %p to device %p (error code %s)!\n", i, tmp, d_ptr, cudaGetErrorString(err));
            fprintf(stderr, "Count %zu Chunks %d i %d header %d\n", count, numChunks, i, header_ptr[256 + i]);
            fprintf(stderr, "\t%p (%d) -  %p - Max %d, %d\n", d_A, header_ptr[i + 256], tmp, maxChipsAssignments0, maxChipsAssignments1);
            exit(EXIT_FAILURE);
        }
    }
}

/**
 * @brief Transfer data from device to host in chunks
 * 
 * @param d_A device array address (aligned to the memory chunk with the first level pages table)
 * @param h_A host array
 * @param count array size in bytes
 * @param chunks array with chunks assignments in device memory
 * @param idx index of first available chunk
 */
void CUDAtask::transferChunksDtH(void *d_A, void *h_A, size_t count, int *chunks, int idx)
{
    int *d_ptr = (int *)d_A; // Cast to integer pointer to use pointer arithmetic
    // Transfer data in device memory to host array
    int numChunks = ceil(count / chunkSize);
    int base = chunks[idx];
    for (int i = 0; i < numChunks; i++)
    {
        d_ptr = (int *)d_A + (chunks[257 + i + idx] - base) * chunkInts;
        int *tmp = (int *)h_A + i * chunkInts;

        if (cke_mode == SYNC)
            err = cudaMemcpy(tmp, d_ptr, chunkSize, cudaMemcpyDeviceToHost);
        else
            err = cudaMemcpyAsync(tmp, d_ptr, chunkSize, cudaMemcpyDeviceToHost, t_stream);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy chunk %d of a vector from device %p to host %p (error code %s)!\n", i, d_ptr, tmp, cudaGetErrorString(err));
            fprintf(stderr, "Count %zu Chunks %d i %d header %d base %d\n", count, numChunks, i, chunks[257 + i + idx], base);
            exit(EXIT_FAILURE);
        }
    }
}
