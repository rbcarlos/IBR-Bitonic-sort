#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 

#include "constants.h"
#include "kernels.cu.h"

/*
Sorts sub-blocks of input data with REGULAR bitonic sort
*/
void runBitoicSortRegularKernel(data_t *d_keys, int arrayLength)
{
    int elemsPerThreadBlock, sharedMemSize;

    elemsPerThreadBlock = THREADS_BITONIC_SORT * ELEMS_BITONIC_SORT;
    sharedMemSize = elemsPerThreadBlock * sizeof(*d_keys);

    dim3 dimGrid(arrayLength / elemsPerThreadBlock, 1, 1);
    dim3 dimBlock(THREADS_BITONIC_SORT, 1, 1);

    bitonicSortRegularKernel
        <arrayLength/ELEMS_BITONIC_SORT, ELEMS_BITONIC_SORT>
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

    int threadBlockSize = min((intervalsLen - 1) / ELEMS_INIT_INTERVALS + 1, THREADS_INIT_INTERVALS);
    int numThreadBlocks = (intervalsLen - 1) / (ELEMS_INIT_INTERVALS * threadBlockSize) + 1;
    // "2 *" because of BUFFER MEMORY for intervals
    int sharedMemSize = 2 * ELEMS_INIT_INTERVALS * threadBlockSize * sizeof(interval_t);

    dim3 dimGrid(numThreadBlocks, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    initIntervalsKernel<ELEMS_INIT_INTERVALS><<<dimGrid, dimBlock, sharedMemSize>>>(
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

    int threadBlockSize = min((intervalsLen - 1) / ELEMS_GEN_INTERVALS + 1, THREADS_GEN_INTERVALS);
    int numThreadBlocks = (intervalsLen - 1) / (ELEMS_GEN_INTERVALS * threadBlockSize) + 1;
    // "2 *" because of BUFFER MEMORY for intervals
    int sharedMemSize = 2 * ELEMS_GEN_INTERVALS * threadBlockSize * sizeof(interval_t);

    dim3 dimGrid(numThreadBlocks, 1, 1);
    dim3 dimBlock(threadBlockSize, 1, 1);

    generateIntervalsKernel<ELEMS_GEN_INTERVALS><<<dimGrid, dimBlock, sharedMemSize>>>(
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

    elemsPerThreadBlock = THREADS_LOCAL_MERGE * ELEMS_LOCAL_MERGE;
    sharedMemSize = elemsPerThreadBlock * sizeof(*d_keys);

    dim3 dimGrid(arrayLenRoundedUp / elemsPerThreadBlock, 1, 1);
    dim3 dimBlock(THREADS_LOCAL_MERGE, 1, 1);

    bitonicMergeIntervalsKernel<THREADS_LOCAL_MERGE, ELEMS_LOCAL_MERGE>
        <<<dimGrid, dimBlock, sharedMemSize>>>(
        d_keys, d_keysBuffer, intervals, phase
    );
}

/*
Sorts data with parallel adaptive bitonic sort.
*/
void bitonicSortAdaptiveParallel(
    data_t *&d_keys, data_t *&d_keysBuffer, interval_t *d_intervals, interval_t *d_intervalsBuffer, int arrayLength
)
{
    int arrayLenPower2 = arrayLength;
    int elemsPerBlockBitonicSort, phasesBitonicMerge, phasesInitIntervals, phasesGenerateIntervals;

    elemsPerBlockBitonicSort = THREADS_BITONIC_SORT * ELEMS_BITONIC_SORT; // 512
    phasesBitonicMerge = log2((double)(THREADS_LOCAL_MERGE * ELEMS_LOCAL_MERGE)); // 9
    //phasesBitonicMerge = log2((double) arrayLength);
    phasesInitIntervals = log2((double)THREADS_INIT_INTERVALS * ELEMS_INIT_INTERVALS); // 8
    phasesGenerateIntervals = log2((double)THREADS_GEN_INTERVALS * ELEMS_GEN_INTERVALS); // 9 

    int phasesAll = log2((double)arrayLenPower2);
    int phasesBitonicSort = log2((double)min(arrayLenPower2, elemsPerBlockBitonicSort)); // 10 if arrlen > 512

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
        printf("phase: %d, ", phase);
        printf("stepStart: %d, ", stepStart);
        printf("stepEnd: %d\n", stepEnd);

        if (phase > phasesBitonicMerge) {
            // BS_2_IBR step 
            runInitIntervalsKernel(
                d_keys, d_intervals, arrayLenPower2, phasesAll, stepStart, stepEnd
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
                    d_keys, d_intervalsBuffer, d_intervals, arrayLenPower2, phasesAll, phase, stepStart, stepEnd
                );
            }
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

main() {

    srand(2006);

    unsigned long int elapsed;
    struct timeval t_start, t_end, t_diff;

    int size_keys = N_ELEMENTS;
    int mem_size_keys = size_keys * sizeof(data_t);
    data_t* h_keys = (data_t*) malloc(mem_size_keys); 

    randomInit(h_keys, size_keys);

    printf("Random keys:\n");
    for(int i = 0; i<size_keys; i++ ){
        printf("%d, ", h_keys[i]);
    }
    printf("\n");

    data_t* d_keys;
    cudaMalloc((void**) &d_keys, mem_size_keys);

    cudaMemcpy(d_keys, h_keys, mem_size_keys, cudaMemcpyHostToDevice);

    int phasesAll = log2((double)size_keys);
    int phasesBitonicMerge = log2((double)2 * THREADS_LOCAL_MERGE);
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
        bitonicSortAdaptiveParallel(d_keys, d_keysBuffer, d_intervals, d_intervalsBuffer, size_keys);
    }
    cudaDeviceSynchronize();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

    cudaMemcpy(h_keys, d_keys, mem_size_keys, cudaMemcpyDeviceToHost);

    printf("Bitonic search on %d elements runs in: %lu microsecs\n", size_keys, elapsed);

    
    printf("Sorted keys:\n");
    for(int i = 0; i<size_keys; i++ ){
        printf("%d, ", h_keys[i]);
    }
    printf("\n");

}