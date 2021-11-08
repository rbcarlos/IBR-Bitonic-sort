CUB=cub-1.8.0

all: cub-sort

cub-sort: sorting_test.cu helper.cu.h
	nvcc -O3 -arch=compute_35  -I$(CUB)/cub -o test-cub sorting_test.cu
	./test-cub

clean:
	rm -f test-cub

