//#include "../../cub-1.8.0/cub/cub.cuh"   // or equivalently <cub/device/device_histogram.cuh>
#include "cub.cuh"
#include "helper.cu.h"

template<class Z>
bool validateZ(Z* A, uint32_t sizeAB) {
    for(uint32_t i = 1; i < sizeAB; i++)
      if (A[i-1] > A[i]){
        printf("INVALID RESULT for i:%d, (A[i-1]=%d > A[i]=%d)\n", i, A[i-1], A[i]);
        return false;
      }
    return true;
}

void randomInitNat(uint32_t* data, const uint32_t size, const uint32_t H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        data[i] = r % H;
    }
}

double sortRedByKeyCUB( uint64_t* data_keys_in
                      , uint64_t* data_keys_out
                      , const uint64_t N
) {
    int beg_bit = 0;
    int end_bit = 64;

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    { // sort prelude
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }
    cudaCheckError();

    { // one dry run
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaDeviceSynchronize();
    }
    cudaCheckError();

    // timing
    double elapsed;
    struct timeval t_start, t_end, t_diff;
    gettimeofday(&t_start, NULL);

    for(int k=0; k<GPU_RUNS; k++) {
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
    }
    cudaDeviceSynchronize();
    cudaCheckError();

    gettimeofday(&t_end, NULL);
    timeval_subtract(&t_diff, &t_end, &t_start);
    elapsed = (t_diff.tv_sec*1e6+t_diff.tv_usec) / ((double)GPU_RUNS);

    cudaFree(tmp_sort_mem);

    return elapsed;
}


int main (int argc, char * argv[]) {
    for (int i=10; i<=20; i++)
    {
    int n_el = pow((double)2, (double)i);
    //Allocate and Initialize Host data with random values
    uint64_t* h_keys  = (uint64_t*) malloc(n_el*sizeof(uint64_t));
    uint64_t* h_keys_res  = (uint64_t*) malloc(n_el*sizeof(uint64_t));
    //randomInitNat(h_keys, N, N/10);

    FILE *fptr;

    fptr = fopen("../datasets/ints/random_uniform.txt", "r");
    for (int j=0; j< n_el; j++)
    {
            fscanf(fptr, "%ld", &h_keys[j]);
    };
    fclose(fptr);

    //Allocate and Initialize Device data
    uint64_t* d_keys_in;
    uint64_t* d_keys_out;
    cudaSucceeded(cudaMalloc((void**) &d_keys_in,  n_el * sizeof(uint64_t)));
    cudaSucceeded(cudaMemcpy(d_keys_in, h_keys, n_el * sizeof(uint64_t), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMalloc((void**) &d_keys_out, n_el * sizeof(uint64_t)));

    double elapsed = sortRedByKeyCUB( d_keys_in, d_keys_out, n_el );

    cudaMemcpy(h_keys_res, d_keys_out, n_el*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaCheckError();

    bool success = validateZ<uint64_t>(h_keys_res, n_el);

    printf("CUB Sorting for N=%lu runs in: %.2f us, VALID: %d\n", n_el, elapsed, success);

    // Cleanup and closing
    cudaFree(d_keys_in); cudaFree(d_keys_out);
    free(h_keys); free(h_keys_res);
    }
    return 0;
}
