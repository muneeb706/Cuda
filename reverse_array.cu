# include <iostream>
# include <stdlib.h>
using namespace std;

__global__ void reverseArray(int * array, int n) {
  int blockId_x = blockIdx.x;
  int threadId_x = threadIdx.x;

  int index_1 = blockDim.x * blockId_x + threadId_x;
  int index_2 =  (blockDim.x * gridDim.x) + index_1;
  
  int iter = index_1;
  int max_iter = n / 2;
  while (iter < max_iter){
    	int temp = array[index_1];
      int pair_index = n-index_1-1;
    	array[index_1] = array[pair_index];
    	array[pair_index] = temp;
      iter += index_2;
  }


}

int main() {
    
    int *host_array;
    int *host_array_reverse;
    int size = 16*1024*1024;

    int *device_array;

    // max kernel size
    int num_threads_per_block = 256;
    int num_blocks = size/num_threads_per_block;

    size_t mem_size = num_blocks * num_threads_per_block * sizeof(int);
    host_array = (int*) malloc(mem_size);
    host_array_reverse = (int*) malloc(mem_size);
    cudaMalloc((void **) &device_array, mem_size);

    for (int i = 0; i < size; i++)
    {
        host_array[i] = rand() % 100;
    }

    cudaMemcpy(device_array, host_array, mem_size, cudaMemcpyHostToDevice);

    dim3 dimGrid(num_blocks);
    dim3 dimBlock(num_threads_per_block);
    reverseArray<<< dimGrid, dimBlock >>>(device_array, size);

    cudaThreadSynchronize();

    cudaMemcpy(host_array_reverse, device_array, mem_size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < size; i++)
    {
        if (host_array_reverse[i] != host_array[size-1-i]) {
            correct = false;
            break;
        }
    }

    if (correct) {
      printf("Array Reversed Correctly!\n");
    } else {
        printf("Something wrong with array reverse operation.\n");
    }

    cudaFree(device_array);

    free(host_array);
    free(host_array_reverse);

    return 0;
}