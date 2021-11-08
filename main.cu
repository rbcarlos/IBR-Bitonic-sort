#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h> 

#include "constants.h"
#include "kernels.cu.h"

#define smaxof(t) (((0x1ULL << ((sizeof(t) * 8ULL) - 1ULL)) - 1ULL) | \
                    (0x7ULL << ((sizeof(t) * 8ULL) - 4ULL)))

/*
Sorts data with parallel adaptive bitonic sort.
*/
template<class OpTp>
void IBR_binotic_sort(
    typename OpTp::ElTp *&d_keys, typename OpTp::ElTp *&d_keysBuffer, interval_t *d_intervals, interval_t *d_intervalsBuffer, int arrayLength
)
{
    int numBlocks, numThreads, sharedMemSize;
    int elemsPerBlock = N_THREADS * ELEMS_PER_THREAD; //1024
    int stagesInMemory = log2((double)(elemsPerBlock));
    int stagesAll = log2((double)arrayLength);
    int stagesBitonicSort = min(stagesAll, stagesInMemory); // 10 if arrlen > 1024

    //=========================================================================================
    //=====================================BS_firstStages======================================
    //=========================================================================================

    // note that this does only stagesBitonicSort (log(1024) = 10) stages 
    // if arrlen <= 1024, only regular bitonic sort (BS_firstStages) is used
    sharedMemSize = elemsPerBlock * sizeof(*d_keys);

    numBlocks = arrayLength / elemsPerBlock;
    numThreads = N_THREADS;

    BS_firstStagesKernel<OpTp><<<numBlocks, numThreads, sharedMemSize>>>(d_keys);

    //=========================================================================================
    //=========================================================================================
    //=========================================================================================

    // picks up where BS_firstStages left of if any elements left
    for (int stage = stagesBitonicSort + 1; stage <= stagesAll; stage++)
    {
        //=========================================================================================
        //=================================BS_2_IBR + IBR_stages===================================
        //=========================================================================================

        int stepStart = stage;
        int stepEnd = max((double)stagesInMemory, (double)stepStart - stagesInMemory);

        int intervalsLen = 1 << (stagesAll - stepEnd);

        numThreads = min((intervalsLen - 1) / ELEMS_PER_THREAD + 1, N_THREADS);
        numBlocks = (intervalsLen - 1) / (ELEMS_PER_THREAD * numThreads) + 1;
        // "2 *" because of BUFFER MEMORY for intervals
        sharedMemSize = 2 * ELEMS_PER_THREAD * numThreads * sizeof(interval_t);

        // BS_2_IBR + IBR_stages
        IBRKernel<OpTp><<<numBlocks, numThreads, sharedMemSize>>>(d_keys, d_intervals, arrayLength, stepStart, stepEnd);

        // picks up where the previous call left off if it did not fully fit in  shared memory
        // with 1024 elements per block, this step is only going to be called after 20 stages
        while (stepEnd > stagesInMemory)
        {
            interval_t *tempIntervals = d_intervals;
            d_intervals = d_intervalsBuffer;
            d_intervalsBuffer = tempIntervals;

            stepStart = stepEnd;
            stepEnd = max((double)stagesInMemory, (double)stepStart - stagesInMemory);

            intervalsLen = 1 << (stagesAll - stepEnd);

            numThreads = min((intervalsLen - 1) / ELEMS_PER_THREAD + 1, N_THREADS);
            numBlocks = (intervalsLen - 1) / (ELEMS_PER_THREAD * numThreads) + 1;
            // "2 *" because of BUFFER MEMORY for intervals
            sharedMemSize = 2 * ELEMS_PER_THREAD * numThreads * sizeof(interval_t);
        
            // only IBR_stages
            IBRContKernel<OpTp><<<numBlocks, numThreads, sharedMemSize>>>(d_keys, d_intervalsBuffer, d_intervals, arrayLength, stage, stepStart, stepEnd);
        }

        //=========================================================================================
        //=========================================================================================
        //=========================================================================================
        
        //=========================================================================================
        //========================================IBR_2_BS=========================================
        //=========================================================================================

        // uses the intervals to merge the blocks for that phase
        // uses regular bitonic merge

        sharedMemSize = elemsPerBlock * sizeof(*d_keys);

        numBlocks = arrayLength/ elemsPerBlock;
        numThreads = N_THREADS;

        IBR_2_BSKernel<OpTp><<<numBlocks, numThreads, sharedMemSize>>>(d_keys, d_keysBuffer, d_intervals, stage);

        //=========================================================================================
        //=========================================================================================
        //=========================================================================================

        // Exchanges keys
        typename OpTp::ElTp *tempTable = d_keys;
        d_keys = d_keysBuffer;
        d_keysBuffer = tempTable;
        
    }
    
}

template<class T>
void randomInts(T* data, int size) {
    T maxVal = smaxof(T);
    T multiplier = maxVal/RAND_MAX;
    for (int i = 0; i < size; ++i)
    {
        data[i] = (rand() - (T)RAND_MAX/2) * multiplier;
    }
}

template<class T>
void randomFloats(T* data, int size) {
    for (int i = 0; i < size; ++i)
    {
        data[i] = (T)(rand() - (T)RAND_MAX/2) / (T)RAND_MAX;
    }
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

    // test
    for(int j=10; j<=20; j++)
    {
        int n_el = pow((double)2, (double)j);
        unsigned long int elapsed;
        struct timeval t_start, t_end, t_diff;

        int size_keys = n_el;
        int mem_size_keys = size_keys * sizeof(Single<long>::ElTp);
        Single<long>::ElTp* h_keys = (Single<long>::ElTp*) malloc(mem_size_keys); 

        //randomFloats<float>(h_keys, size_keys);
 
        // creating a FILE variable
        FILE *fptr;

        fptr = fopen("datasets/mostly_zeros.txt", "r");
        for (int i=0; i< n_el; i++)
        {
            fscanf(fptr, "%ld", &h_keys[i]);
        };
        fclose(fptr);

        /*
        printf("Unsorted keys:\n");
        for(int i = 0; i<size_keys; i++ ){
            printf("%ld, ", h_keys[i]);
        }
        printf("\n");
        */

        Single<long>::ElTp* d_keys;
        cudaMalloc((void**) &d_keys, mem_size_keys);

        cudaMemcpy(d_keys, h_keys, mem_size_keys, cudaMemcpyHostToDevice);

        int stagesAll = log2((double)size_keys);
        int stagesBitonicMerge = log2((double)2 * N_THREADS);
        int intervalsLen = 1 << (stagesAll - stagesBitonicMerge);

        // Allocates buffer for keys
        Single<long>::ElTp* d_keysBuffer;
        cudaMalloc((void **)&d_keysBuffer, size_keys * sizeof(*d_keysBuffer));

        // Memory needed for storing intervals
        interval_t* d_intervals;
        interval_t* d_intervalsBuffer;
        cudaMalloc((void **)&d_intervals, intervalsLen * sizeof(*d_intervals));
        cudaMalloc((void **)&d_intervalsBuffer, intervalsLen * sizeof(*d_intervalsBuffer));

        gettimeofday(&t_start, NULL); 

        for(int i=0; i<GPU_RUNS; i++){
            IBR_binotic_sort<Single<long> >(d_keys, d_keysBuffer, d_intervals, d_intervalsBuffer, size_keys);
        }
        cudaDeviceSynchronize();

        gettimeofday(&t_end, NULL);
        timeval_subtract(&t_diff, &t_end, &t_start);
        elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / GPU_RUNS; 

        cudaMemcpy(h_keys, d_keys, mem_size_keys, cudaMemcpyDeviceToHost);

        printf("Bitonic sort on %d elements (type int) runs in: %lu microsecs\n", size_keys, elapsed);

        /*
        printf("Sorted keys:\n");
        for(int i = 0; i<size_keys; i++ ){
            printf("%ld, ", h_keys[i]);
        }
        printf("\n");
        */

        for(int i = 0; i<size_keys-1; i++) {
            if(h_keys[i] > h_keys[i+1]) {
                printf("INVALID!\n");
                break;
                //return 1;
            }
        }
        printf("VALID!\n");
        //return 0;

        free(h_keys);
        cudaFree(d_keys);
        cudaFree(d_keysBuffer);
        cudaFree(d_intervals);
        cudaFree(d_intervalsBuffer);
    }
    

}