#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 

#include "kernels.cu.h"

typedef struct Interval interval_t;
typedef int data_t;
#define N_ELEMENTS 1048576
#define GPU_RUNS 100

struct Interval
{
    int offset0;
    int length0;
    int offset1;
    int length1;
};

#define N_THREADS 256
#define ELEMS_PER_THREAD 4

/*
Sorts sub-blocks of input data with REGULAR bitonic sort
*/
void runBitoicSortRegularKernel(data_t *d_keys, int arrayLength)
{
    int elemsPerThreadBlock, sharedMemSize;

    elemsPerThreadBlock = N_THREADS * ELEMS_PER_THREAD;
    sharedMemSize = elemsPerThreadBlock * sizeof(*d_keys);

    dim3 dimGrid(arrayLength / elemsPerThreadBlock, 1, 1);
    dim3 dimBlock(N_THREADS, 1, 1);

    bitonicSortRegularKernel
        <N_THREADS, ELEMS_PER_THREAD>
        <<<dimGrid, dimBlock, sharedMemSize>>>(
        d_keys, arrayLength
    );
}

/*
Initializes intervals and continues to evolve them until the end step.
*/
void runInitIntervalsKernel(
    data_t *d_keys, interval_t *intervals, int arrayLength, int phasesAll, int stepStart,
    int stepEnd
)
{
    int intervalsLen = 1 << (phasesAll - stepEnd);

    int threadBlockSize = min((intervalsLen - 1) / ELEMS_PER_THREAD + 1, N_THREADS);
    int numThreadBlocks = (intervalsLen - 1) / (ELEMS_PER_THREAD * threadBlockSize) + 1;
    // "2 *" because of BUFFER MEMORY for intervals
    int sharedMemSize = 2 * ELEMS_PER_THREAD * threadBlockSize * sizeof(interval_t);

    dim3 dimGrid(numThreadBlocks, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    initIntervalsKernel<ELEMS_PER_THREAD><<<dimGrid, dimBlock, sharedMemSize>>>(
        d_keys, intervals, arrayLength, stepStart, stepEnd
    );
}

/*
Evolves intervals from start step to end step.
*/
void runGenerateIntervalsKernel(
    data_t *d_keys, interval_t *inputIntervals, interval_t *outputIntervals, int arrayLength, int phasesAll,
    int phase, int stepStart, int stepEnd
)
{
    int intervalsLen = 1 << (phasesAll - stepEnd);

    int threadBlockSize = min((intervalsLen - 1) / ELEMS_PER_THREAD + 1, N_THREADS);
    int numThreadBlocks = (intervalsLen - 1) / (ELEMS_PER_THREAD * threadBlockSize) + 1;
    // "2 *" because of BUFFER MEMORY for intervals
    int sharedMemSize = 2 * ELEMS_PER_THREAD * threadBlockSize * sizeof(interval_t);

    dim3 dimGrid(numThreadBlocks, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    generateIntervalsKernel<ELEMS_PER_THREAD><<<dimGrid, dimBlock, sharedMemSize>>>(
        d_keys, inputIntervals, outputIntervals, arrayLength, phase, stepStart, stepEnd
    );
}

/*
Runs kernel, which performs bitonic merge from provided intervals.
*/
void runBitoicMergeIntervalsKernel(
    data_t *d_keys, data_t *d_keysBuffer, interval_t *intervals,
    int arrayLength, int phase
)
{
    // If table length is not power of 2, than table is padded to the next power of 2. In that case it is not
    // necessary for entire padded table to be merged. It is only necessary that table is merged to the next
    // multiple of phase stride.
    //int arrayLenRoundedUp = roundUp(arrayLength, 1 << phase);
    int arrayLenRoundedUp = arrayLength;
    int elemsPerThreadBlock, sharedMemSize;

    elemsPerThreadBlock = N_THREADS * ELEMS_PER_THREAD;
    sharedMemSize = elemsPerThreadBlock * sizeof(*d_keys);

    dim3 dimGrid(arrayLenRoundedUp / elemsPerThreadBlock, 1, 1);
    dim3 dimBlock(N_THREADS, 1, 1);

    bitonicMergeIntervalsKernel<N_THREADS, ELEMS_PER_THREAD>
        <<<dimGrid, dimBlock, sharedMemSize>>>(
        d_keys, d_keysBuffer, intervals, phase
    );
}

/*
Sorts data with parallel adaptive bitonic sort.
*/
void IBR_binotic_sort(
    data_t *&d_keys, data_t *&d_keysBuffer, interval_t *d_intervals, interval_t *d_intervalsBuffer, int arrayLength
)
{
    int elemsPerBlockBitonicSort, phasesBitonicMerge, phasesInitIntervals, phasesGenerateIntervals;

    elemsPerBlockBitonicSort = N_THREADS * ELEMS_PER_THREAD; // 512
    phasesBitonicMerge = log2((double)(N_THREADS * ELEMS_PER_THREAD)); // 10
    //phasesBitonicMerge = log2((double) arrayLength);
    phasesInitIntervals = log2((double)N_THREADS * ELEMS_PER_THREAD); // 8
    phasesGenerateIntervals = log2((double)N_THREADS * ELEMS_PER_THREAD); // 9 

    int phasesAll = log2((double)arrayLength);
    int phasesBitonicSort = log2((double)min(arrayLength, elemsPerBlockBitonicSort)); // 10 if arrlen > 1024

    if (phasesBitonicMerge < phasesBitonicSort)
    {
        printf(
            "\nNumber of phases executed in bitonic merge has to be lower than number of phases "
            "executed in initial bitonic sort. This is due to the fact, that regular bitonic sort is "
            "used (not normalized). This way the sort direction for entire thread block can be computed "
            "when executing bitonic merge, which is much more efficient.\n"
        );
        exit(EXIT_FAILURE);
    }

    // BS_firstStages
    // note that this does only phasesBitonicSort (log(512) = 9) phases 
    runBitoicSortRegularKernel(d_keys, arrayLength);
    
    for (int phase = phasesBitonicSort + 1; phase <= phasesAll; phase++)
    {
        int stepStart = phase;
        int stepEnd = max((double)phasesBitonicMerge, (double)phase - phasesInitIntervals);
        //printf("phase: %d, ", phase);
        //printf("stepStart: %d, ", stepStart);
        //printf("stepEnd: %d\n", stepEnd);

        // BS_2_IBR step 
        runInitIntervalsKernel(
            d_keys, d_intervals, arrayLength, phasesAll, stepStart, stepEnd
        );

        // IBR_stages
        // After initial intervals were generated intervals have to be evolved to the end step
        while (stepEnd > phasesBitonicMerge)
        {
            interval_t *tempIntervals = d_intervals;
            d_intervals = d_intervalsBuffer;
            d_intervalsBuffer = tempIntervals;

            stepStart = stepEnd;
            stepEnd = max((double)phasesBitonicMerge, (double)stepStart - phasesGenerateIntervals);
            runGenerateIntervalsKernel(
                d_keys, d_intervalsBuffer, d_intervals, arrayLength, phasesAll, phase, stepStart, stepEnd
            );
        }
        
        // BS_lastSteps
        // Global merge with intervals
        runBitoicMergeIntervalsKernel(
            d_keys, d_keysBuffer, d_intervals, arrayLength, phase
        );

        // Exchanges keys
        data_t *tempTable = d_keys;
        d_keys = d_keysBuffer;
        d_keysBuffer = tempTable;
        
    }
    
}

void randomInit(data_t* data, int size) {
    for (int i = 0; i < size; ++i)
    data[i] = rand() - (data_t)RAND_MAX/2;
 }
 
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
    unsigned int resolution=1000000;
    long int diff = (t2->tv_usec + resolution * t2->tv_sec) - (t1->tv_usec + resolution * t1->tv_sec);
    result->tv_sec = diff / resolution;
    result->tv_usec = diff % resolution;
    return (diff<0);
}

int main() {

    srand(2006);

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    int size_keys = N_ELEMENTS;
    int mem_size_keys = size_keys * sizeof(data_t);
    data_t* h_keys = (data_t*) malloc(mem_size_keys); 

    randomInit(h_keys, size_keys);
    /*
    printf("Random keys:\n");
    for(int i = 0; i<size_keys; i++ ){
        printf("%d, ", h_keys[i]);
    }
    printf("\n");
    */
    data_t* d_keys;
    cudaMalloc((void**) &d_keys, mem_size_keys);

    cudaMemcpy(d_keys, h_keys, mem_size_keys, cudaMemcpyHostToDevice);

    int phasesAll = log2((double)size_keys);
    int phasesBitonicMerge = log2((double)2 * N_THREADS);
    int intervalsLen = 1 << (phasesAll - phasesBitonicMerge);

    // Allocates buffer for keys
    data_t* d_keysBuffer;
    cudaMalloc((void **)&d_keysBuffer, size_keys * sizeof(*d_keysBuffer));

    // Memory needed for storing intervals
    interval_t* d_intervals;
    interval_t* d_intervalsBuffer;
    cudaMalloc((void **)&d_intervals, intervalsLen * sizeof(*d_intervals));
    cudaMalloc((void **)&d_intervalsBuffer, intervalsLen * sizeof(*d_intervalsBuffer));

    gettimeofday(&t_start, NULL); 

    for(int i=0; i<GPU_RUNS; i++){
        IBR_binotic_sort(d_keys, d_keysBuffer, d_intervals, d_intervalsBuffer, size_keys);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

    cudaMemcpy(h_keys, d_keys, mem_size_keys, cudaMemcpyDeviceToHost);

    printf("Bitonic search on %d elements runs in: %lu microsecs\n", size_keys, elapsed);

    /*
    printf("Sorted keys:\n");
    for(int i = 0; i<size_keys; i++ ){
        printf("%d, ", h_keys[i]);
    }
    printf("\n");
    */

    for(int i = 0; i<size_keys-1; i++) {
        if(h_keys[i] > h_keys[i+1]) {
            printf("INVALID!\n");
            return 1;
        }
    }
    printf("VALID!\n");
    return 0;

}