#ifndef CONSTANTS_BITONIC_SORT_H
#define CONSTANTS_BITONIC_SORT_H

typedef struct Interval interval_t;
typedef double data_t;
#define DATA_PATH "datasets/floats/random_uniform.txt"
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

#endif