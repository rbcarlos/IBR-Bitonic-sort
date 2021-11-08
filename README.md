# IBR bitonic sort
Robert Bayer (tvb333), Neil Kim Nielsen (vln677)

## Instructions
To run the IBR bitonic sort compile the executable by running ```make```.

In the ```constants.h``` file, you can find variables, which define the datatype of the keys as well as the path to the dataset used for the test.

Note: When changing the data type the ```%lf``` on line 167 of the ```main.cu``` file has to be changed to the corresponding format specifier (```%d``` for 32-bit ints, ```%ld``` for 64-bit ints, ```%f``` for float and ```%lf``` for double (default)).

The datasets lie in the datasets folder and hold two subdirectories for integers (32 and 64 bit) and floating point numbers (32 and 64 bit).

The program will loop through different array sizes from $2^{10}$ to $2^{20}$.