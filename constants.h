#ifndef CONSTANTS_BITONIC_SORT_H
#define CONSTANTS_BITONIC_SORT_H

typedef struct Interval interval_t;
typedef int data_t;
#define N_ELEMENTS 2048
#define GPU_RUNS 1

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