CXX        = nvcc -O3 -arch=compute_35

SRC1 = main.cu
SOURCES_CPP = main.cu
HELPERS     =  kernels.cu.h constants.h
EXECUTABLE1 = bitonic-search

default: compile run

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE1)

$(EXECUTABLE1): $(SRC1) $(HELPERS)
	$(CXX) -o $(EXECUTABLE1) $(SRC1)


run: $(EXECUTABLE1)
	./$(EXECUTABLE1)

clean:
	rm -f $(EXECUTABLE1)