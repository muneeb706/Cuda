# include <iostream>
# include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <chrono>
using namespace std;

__global__ void cudaMul(float * A, float * B, float * C, int n)
{
    int T = 8;
    __shared__ float smem_a[64][8];
    __shared__ float smem_b[8][64];
    __shared__ float smem_c[64][64];

    int c = blockIdx.x * 64;
    int r = blockIdx.y * 64;
    
    float ctemp1 = 0;
    float ctemp2 = 0;
    float ctemp3 = 0;
    float ctemp4 = 0;
 
    for (int kk=0; kk<n; kk+=T) {

      for (int i=threadIdx.x+blockDim.x*threadIdx.y;
        i<64*8; i+=blockDim.x*blockDim.y) {
        int k = kk + i / 64;
        int rt = r + i % 64;
        int ct = c + i % 64;
        smem_a[i%64][i/64] = A[rt*n+k];
        smem_b[i/64][i%64] = B[k*n+ct];
      }
      __syncthreads();

      //Multiplying Elements present in tile
      for (int k = 0; k < 8; ++k)
      {
        ctemp1 += smem_a[threadIdx.y*2][k] * smem_b[k][threadIdx.x*2];
        ctemp2 += smem_a[threadIdx.y*2][k] * smem_b[k][threadIdx.x*2+1];
        ctemp3 += smem_a[threadIdx.y*2+1][k] * smem_b[k][threadIdx.x*2];
        ctemp4 += smem_a[threadIdx.y*2+1][k] * smem_b[k][threadIdx.x*2+1];
      }
      __syncthreads();

      smem_c[threadIdx.y*2][threadIdx.x*2] = ctemp1;
      smem_c[threadIdx.y*2][threadIdx.x*2+1] = ctemp2;
      smem_c[threadIdx.y*2+1][threadIdx.x*2] = ctemp3;
      smem_c[threadIdx.y*2+1][threadIdx.x*2+1] = ctemp4;
      

    }
 
    if (r + threadIdx.y < n && c + threadIdx.x < n)//Saving Final result into Matrix C
    {
        C[(blockIdx.y*blockDim.y + threadIdx.y*2)*n + (blockIdx.x *blockDim.x)+threadIdx.x*2] = smem_c[threadIdx.y*2][threadIdx.x*2];
        C[(blockIdx.y*blockDim.y + threadIdx.y*2+1)*n + (blockIdx.x *blockDim.x)+threadIdx.x*2] = smem_c[threadIdx.y*2+1][threadIdx.x*2];
        C[(blockIdx.y*blockDim.y + threadIdx.y*2)*n + (blockIdx.x *blockDim.x)+threadIdx.x*2+1] = smem_c[threadIdx.y*2][threadIdx.x*2+1];
        C[(blockIdx.y*blockDim.y + threadIdx.y*2+1)*n + (blockIdx.x *blockDim.x)+threadIdx.x*2+1] = smem_c[threadIdx.y*2+1][threadIdx.x*2+1];
    }
}

int main() {
    
    cout<<"Results for N = 4096:"<<endl<<endl;
    
    int n1 = 4096;

    int a1[n1][n1], b1[n1][n1], c1[n1][n1];
    
    for (int i=0; i<n1; i++) {
        for (int j=0; j<n1; j++) {
          a1[i][j] = 1;
          b1[i][j] = 1;
          c1[i][j] = 0;
        }	
    }

	auto t1 = chrono::high_resolution_clock::now();

    for(int i = 0; i < n1; ++i)
    {
      for(int j = 0; j < n1; ++j)
      {
        for(int k=0; k<n1; ++k)
        {
          c1[i][j] += a1[i][k] * b1[k][j];
        }
      }
    }

	auto t2 = chrono::high_resolution_clock::now();

	auto duration_1 = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
    	
    cout <<"Execution time (microseconds) for serial implementation: " << duration_1 <<endl<<endl;

    float * A1; // The A matrix
    float * B1; // The B matrix
    float * C1; // The output C matrix
    float * computedC1;
    float * dA1;
    float * dB1;
    float * dC1;

    A1 = (float *) malloc(sizeof(float)*n1*n1);
    B1 = (float *) malloc(sizeof(float)*n1*n1);
    C1 = (float *) malloc(sizeof(float)*n1*n1);

	for (int i = 0; i < n1*n1; i++)//Matrix Initialization
    {
        A1[i]=1;
        B1[i]=1;
    }
    
	computedC1 = (float *) malloc(sizeof(float)*n1*n1);
    cudaMalloc((void **)&dA1, sizeof(float)*n1*n1);
    cudaMalloc((void **)&dB1, sizeof(float)*n1*n1);
    cudaMalloc((void **)&dC1, sizeof(float)*n1*n1);
    
    // Copy memory to the GPU
    cudaMemcpy(dA1, A1, sizeof(float)*n1*n1, cudaMemcpyHostToDevice);
    cudaMemcpy(dB1, B1, sizeof(float)*n1*n1, cudaMemcpyHostToDevice);

	
	dim3 dimGrid((n1/64), (n1/64));//Number of Blocks required
    dim3 dimBlock(32, 32);//Number of threads in each block

    t1 = chrono::high_resolution_clock::now();

	cudaMul<<<dimGrid, dimBlock>>>(dA1, dB1, dC1, n1);
    
    cudaDeviceSynchronize();//To synchronize the device

    // Copy the results in GPU memory back to the CPU
    cudaMemcpy(C1, dC1, sizeof(float)*n1*n1, cudaMemcpyDeviceToHost);

    t2 = chrono::high_resolution_clock::now();

    duration_1 = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
    	
    cout <<"Execution time (microseconds) for cuda implementation: " << duration_1 <<endl<<endl;


    cout<< "Product Matrix from serial implementation" <<endl<<endl;

    for (int i=0; i<n1; i++) {
        for (int j=0; j<n1; j++) {
          cout<<c1[i][j]<<" ";
        }
        cout<<endl;	
    }
    cout<<endl;

    cout<< "Product Matrix from cuda implementation" <<endl<<endl;
    
    
    for(int i=0;i<n1*n1;i++)
    {
      cout<<*(C1+i)<<" ";
     
      if((i%n1)==0 )
      {
        cout<<endl;
      }
    }
    cout<<endl;
    
	cudaFree(dA1);
    cudaFree(dB1);
    cudaFree(dC1);
    free(A1);
    free(B1);
    free(C1);
    free(computedC1);
 
    cout<<"Results for N = 8192:"<<endl<<endl;
    
    int n2 = 8192;

    int a2[n2][n2], b2[n2][n2], c2[n2][n2];
    
    for (int i=0; i<n2; i++) {
        for (int j=0; j<n2; j++) {
          a1[i][j] = 1;
          b1[i][j] = 1;
          c1[i][j] = 0;
        }	
    }

	t1 = chrono::high_resolution_clock::now();

    for(int i = 0; i < n2; ++i)
    {
      for(int j = 0; j < n2; ++j)
      {
        for(int k=0; k<n2; ++k)
        {
          c2[i][j] += a2[i][k] * b2[k][j];
        }
      }
    }

	t2 = chrono::high_resolution_clock::now();

	duration_1 = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
    	
    cout <<"Execution time (microseconds) for serial implementation: " << duration_1 <<endl<<endl;


    float * A2; // The A matrix
    float * B2; // The B matrix
    float * C2; // The output C matrix
    float * computedC2;
    float * dA2;
    float * dB2;
    float * dC2;

    A2 = (float *) malloc(sizeof(float)*n1*n1);
    B2 = (float *) malloc(sizeof(float)*n1*n1);
    C2 = (float *) malloc(sizeof(float)*n1*n1);

	for (int i = 0; i < n2*n2; i++)//Matrix Initialization
    {
        A2[i]=1;
        B2[i]=1;
    }
    
	computedC2 = (float *) malloc(sizeof(float)*n2*n2);
    cudaMalloc((void **)&dA2, sizeof(float)*n2*n2);
    cudaMalloc((void **)&dB2, sizeof(float)*n2*n2);
    cudaMalloc((void **)&dC2, sizeof(float)*n2*n2);
    
    // Copy memory to the GPU
    cudaMemcpy(dA2, A2, sizeof(float)*n2*n2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB2, B2, sizeof(float)*n2*n2, cudaMemcpyHostToDevice);

	
	dim3 dimGrid2((n2/64), (n2/64));//Number of Blocks required
    dim3 dimBlock2(32, 32);//Number of threads in each block

    t1 = chrono::high_resolution_clock::now();

	cudaMul<<<dimGrid2, dimBlock2>>>(dA2, dB2, dC2, n2);
    
    cudaDeviceSynchronize();//To synchronize the device

    // Copy the results in GPU memory back to the CPU
    cudaMemcpy(C2, dC2, sizeof(float)*n2*n2, cudaMemcpyDeviceToHost);

    t2 = chrono::high_resolution_clock::now();

    duration_1 = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
    	
    cout <<"Execution time (microseconds) for cuda implementation: " << duration_1 <<endl<<endl;

    cout<< "Product Matrix from serial implementation" <<endl<<endl;
    
    for (int i=0; i<n2; i++) {
        for (int j=0; j<n2; j++) {
          cout<<c2[i][j]<<" ";
        }
        cout<<endl;	
    }
    cout<<endl;

    cout<< "Product Matrix from cuda implementation" <<endl<<endl;

    for(int i=0;i<n2*n2;i++)
    {
      cout<<*(C2+i)<<" ";
     
      if((i%n1)==0 )
      {
        cout<<endl;
      }
    }

	cudaFree(dA2);
    cudaFree(dB2);
    cudaFree(dC2);
    free(A2);
    free(B2);
    free(C2);
    free(computedC2);

    return 0;
}
