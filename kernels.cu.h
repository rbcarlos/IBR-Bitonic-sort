
#ifndef BITONIC_KERNELS
#define BITONIC_KERNELS

#include "constants.h"

template<class T>
class Single {
public:
    typedef T ElTp;
    static __device__ __host__ inline void compare(ElTp *elem1, ElTp *elem2, bool asc)
    { 
        if (asc ? (*elem1 > *elem) : (*elem1 < *elem2))
        {
            ElTp temp = *elem1;
            *elem1 = *elem2;
            *elem2 = temp;
        }    
    }
    static __device__ __host__ inline void compareQ(ElTp *elem1, ElTp *elem2, bool asc, int mid, int s, int e) {}
};

/*
Compares 2 elements and exchanges them according to asc.
*/
inline __device__ void compareExchange(data_t *elem1, data_t *elem2, bool asc)
{
    if (asc ? (*elem1 > *elem2) : (*elem1 < *elem2))
    {
        data_t temp = *elem1;
        *elem1 = *elem2;
        *elem2 = temp;
    }
}

/*
Sorts the elements using a regular bitonic sort until the subblocks are too large to be processed in shared memory
*/
__global__ void BS_firstStagesKernel(data_t *keys)
{
    // dynamically allocate the shared memory
    extern __shared__ data_t sortTile[];

    //calculate the offset and length of a block of data processed by the current block
    int elemsPerBlock = N_THREADS * ELEMS_PER_THREAD;
    int offset = blockIdx.x * elemsPerBlock;

    // if this is not the only kernel to be spawned, the blocks have to have an alternating direction 
    bool blockDirection = 1 ^ (blockIdx.x & 1);

    // Loads data into shared memory with coallesced access, each thread loading ELEMS_PER_THREAD elements.
    for (int tx = threadIdx.x; tx < elemsPerBlock; tx += N_THREADS)
    {
        sortTile[tx] = keys[offset + tx];
    }
    __syncthreads();

    // 2^stage
    for (int subBlockSize = 1; subBlockSize < elemsPerBlock; subBlockSize <<= 1)
    {
        // 2^step
        for (int stride = subBlockSize; stride > 0; stride >>= 1)
        {
            for (int tx = threadIdx.x; tx < elemsPerBlock >> 1; tx += N_THREADS)
            {
                bool direction = blockDirection ^ ((tx & subBlockSize) != 0);
                int index = 2 * tx - (tx & (stride - 1));

                if (direction)
                {
                    compareExchange(&sortTile[index], &sortTile[index + stride], true);
                }
                else
                {
                    compareExchange(&sortTile[index], &sortTile[index + stride], false);
                }
            }
            __syncthreads();
        }
    }

    // Stores sorted elements from shared to global memory
    for (int tx = threadIdx.x; tx < elemsPerBlock; tx += N_THREADS)
    {
        keys[offset + tx] = sortTile[tx];
    }
}

/*
returns an element from interval by index
*/
__device__ data_t get(data_t *keys, interval_t interval, int index)
{
    bool useInterval1 = index >= interval.length0;
    int offset = useInterval1 ? interval.offset1 : interval.offset0;

    index -= useInterval1 ? interval.length0 : 0;
    index -= useInterval1 && index >= interval.length1 ? interval.length1 : 0;

    return keys[offset + index];
}

/*
Finds q which is used to generate the intervals
*/
inline __device__ int findQ(data_t* keys, interval_t interval, int subBlockHalfLen, bool asc)
{
    // Depending which interval is longer, different start and end indexes are used
    int indexStart = interval.length0 <= interval.length1 ? 0 : subBlockHalfLen - interval.length1;
    int indexEnd = interval.length0 <= interval.length1 ? interval.length0 : subBlockHalfLen;

    while (indexStart < indexEnd)
    {
        int index = indexStart + (indexEnd - indexStart) / 2;
        data_t el0 = get(keys, interval, index);
        data_t el1 = get(keys, interval, index + subBlockHalfLen);

        if (asc ? (el0 > el1) : (el0 < el1))
        {
            indexStart = index + 1;
        }
        else
        {
            indexEnd = index;
        }
    }

    return indexStart;
}

/*
generates the intervals until the end block size is reached
*/
inline __device__ void generateIntervals(
    data_t *keys, int subBlockHalfSize, int subBlockSizeEnd, int stride, int activeThreadsPerBlock
)
{
    extern __shared__ interval_t intervalsTile[];
    interval_t interval;

    // only active threads generate intervals
    // buffer is used to minimize the number of thread syncs as per paper
    interval_t *intervals = intervalsTile;
    interval_t *intervalsBuffer = intervalsTile + blockDim.x * ELEMS_PER_THREAD;

    for (; subBlockHalfSize >= subBlockSizeEnd; subBlockHalfSize /= 2, stride *= 2, activeThreadsPerBlock *= 2)
    {
        for (int tx = threadIdx.x; tx < activeThreadsPerBlock; tx += blockDim.x)
        {
            interval = intervals[tx];

            int intervalIndex = blockIdx.x * activeThreadsPerBlock + tx;
            bool orderAsc = 0 ^ ((intervalIndex / stride) & 1);
            int q;

            // Finds q - an index, where exchanges begin in bitonic sequences being merged.
            if (orderAsc)
            {
                q = findQ(keys, interval, subBlockHalfSize, true);
            }
            else
            {
                q = findQ(keys, interval, subBlockHalfSize, false);
            }

            // Output indexes of newly generated intervals
            int index1 = 2 * tx;
            int index2 = index1 + 1;

            // L_E intervals
            intervalsBuffer[index1].offset0 = interval.offset0;
            intervalsBuffer[index1].length0 = q;
            intervalsBuffer[index1].offset1 = interval.offset1 + interval.length1 - subBlockHalfSize + q;
            intervalsBuffer[index1].length1 = subBlockHalfSize - q;

            // U_E intervals
            intervalsBuffer[index2].offset0 = interval.offset0 + q;
            intervalsBuffer[index2].length0 = interval.length0 - q;
            intervalsBuffer[index2].offset1 = interval.offset1;
            intervalsBuffer[index2].length1 = q + interval.length1 - subBlockHalfSize;
        }

        interval_t *temp = intervals;
        intervals = intervalsBuffer;
        intervalsBuffer = temp;

        __syncthreads();
    }
}

/*
Generates initial intervals and intervals for all steps
*/
__global__ void IBRKernel(
    data_t *keys, interval_t *intervals, int arrayLength, int stepStart, int stepEnd
)
{
    // dynamically allocate shared memory
    extern __shared__ interval_t intervalsTile[];
    // first step is also the index of phase
    int subBlockSize = 1 << stepStart;
    int activeThreadsPerBlock = arrayLength / subBlockSize / gridDim.x;

    // initialize the intervals 
    for (int tx = threadIdx.x; tx < activeThreadsPerBlock; tx += blockDim.x)
    {
        // creates the L_E and U_E sequences
        int intervalIndex = blockIdx.x * activeThreadsPerBlock + tx;
        int offset0 = intervalIndex * subBlockSize;
        int offset1 = intervalIndex * subBlockSize + subBlockSize / 2;

        intervalsTile[tx].offset0 = intervalIndex % 2 ? offset1 : offset0;
        intervalsTile[tx].offset1 = intervalIndex % 2 ? offset0 : offset1;
        intervalsTile[tx].length0 = subBlockSize / 2;
        intervalsTile[tx].length1 = subBlockSize / 2;
    }
    __syncthreads();

    // Evolves intervals in shared memory to end step
    generateIntervals(keys, subBlockSize / 2, 1 << stepEnd, 1, activeThreadsPerBlock);

    int elemsPerBlock = blockDim.x * ELEMS_PER_THREAD;
    // calculate offset in global intervals array
    interval_t *outputIntervalsGlobal = intervals + blockIdx.x * elemsPerBlock;
    // calculate offset in local intervals array
    interval_t *outputIntervalsLocal = intervalsTile + ((stepStart - stepEnd) % 2 != 0 ? elemsPerBlock : 0);

    for (int tx = threadIdx.x; tx < elemsPerBlock; tx += blockDim.x)
    {
        outputIntervalsGlobal[tx] = outputIntervalsLocal[tx];
    }
}

/*
Uses the generated intervals and evolves them further. This is only used for large arrays (>2^20 elements).
*/
__global__ void IBRContKernel(
    data_t *table, interval_t *inputIntervals, interval_t *outputIntervals, int tableLen, int phase,
    int stepStart, int stepEnd
)
{
    // dynamically allocate shared memory
    extern __shared__ interval_t intervalsTile[];
    // first step is also the index of phase
    int subBlockSize = 1 << stepStart;
    int activeThreadsPerBlock = tableLen / subBlockSize / gridDim.x;
    interval_t *inputIntervalsGlobal = inputIntervals + blockIdx.x * activeThreadsPerBlock;

    // read the generated intervals into shared memory
    for (int tx = threadIdx.x; tx < activeThreadsPerBlock; tx += blockDim.x)
    {
        intervalsTile[tx] = inputIntervalsGlobal[tx];
    }
    __syncthreads();

    // 1 << (phase - stepStart) picks up where it left off
    generateIntervals(table, subBlockSize / 2, 1 << stepEnd, 1 << (phase - stepStart), activeThreadsPerBlock);

    int elemsPerBlock = blockDim.x * ELEMS_PER_THREAD;
    // calculate offset in global intervals array
    interval_t *outputIntervalsGlobal = outputIntervals + blockIdx.x * elemsPerBlock;
    // calculate offset in local intervals array
    interval_t *outputIntervalsLocal = intervalsTile + ((stepStart - stepEnd) % 2 != 0 ? elemsPerBlock : 0);

    for (int tx = threadIdx.x; tx < elemsPerBlock; tx += blockDim.x)
    {
        outputIntervalsGlobal[tx] = outputIntervalsLocal[tx];
    }
}

/*
Merges the blocks using the intervals
*/
__global__ void IBR_2_BSKernel(data_t *keys, data_t *keysBuffer, interval_t *intervals, int phase)
{
    extern __shared__ data_t mergeTile[];
    interval_t interval = intervals[blockIdx.x];

    // Elements inside same sub-block have to be ordered in same direction
    int elemsPerBlock = N_THREADS * ELEMS_PER_THREAD;
    int offset = blockIdx.x * elemsPerBlock;
    bool orderAsc = 1 ^ ((offset >> phase) & 1);

    // Loads data from global to shared memory
    for (int tx = threadIdx.x; tx < elemsPerBlock; tx += N_THREADS)
    {
        mergeTile[tx] = get(keys, interval, tx);
    }
    __syncthreads();

    // Bitonic merge
    for (int stride = elemsPerBlock / 2; stride > 0; stride >>= 1)
    {
        for (int tx = threadIdx.x; tx < elemsPerBlock / 2; tx += N_THREADS)
        {
            int index = 2 * tx - (tx & (stride - 1));

            if (orderAsc)
            {
                compareExchange(&mergeTile[index], &mergeTile[index + stride], true);
            }
            else
            {
                compareExchange(&mergeTile[index], &mergeTile[index + stride], false);
            }
        }
        __syncthreads();
    }

    // Stores sorted data to buffer array
    for (int tx = threadIdx.x; tx < elemsPerBlock; tx += N_THREADS)
    {
        keysBuffer[offset + tx] = mergeTile[tx];
    }
}

#endif