#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define BLOCK_SIZE 1024
#define GRID_SIZE 4

__global__ void reduce_max_stage1(const float* x, float* partial, int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    __shared__ float s_data[BLOCK_SIZE];
    s_data[tid] = -1e9f;

    for(int i = idx; i < N; i+=stride){
        s_data[tid] = max(s_data[tid], x[i]);
    }
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride>>=1){
        if(tid < stride){
            s_data[tid] = max(s_data[tid], s_data[tid+stride]);
        }
        __syncthreads();
    }

    if(tid == 0) partial[blockIdx.x] = s_data[0];
}

__global__ void reduce_max_stage2(float* partial, float* out){
    int tid = threadIdx.x;
    __shared__ float s_data[BLOCK_SIZE];
    if(tid < GRID_SIZE) {
        s_data[tid] = partial[tid];
    }
    else s_data[tid] = -1e9f;

    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride>>=1){
        if(tid < stride){
            s_data[tid] = max(s_data[tid], s_data[tid+stride]);
        }
        __syncthreads();
    }

    if(tid == 0) *out = s_data[0];

}



int main(){
    int N = BLOCK_SIZE * GRID_SIZE + 123;


    float* h_x, *h_out;
    h_x = (float*)malloc(N*sizeof(float));
    h_out = (float*)malloc(sizeof(float));

    srand(time(nullptr));

    for(int i = 0; i < N; i++){
        h_x[i] = (float)(rand()%201) - 100;
    }

    h_x[999] = 999.0f;

    float* d_x, *d_out, *partial;
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&partial, GRID_SIZE * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);


    dim3 block(BLOCK_SIZE);
    dim3 grid(GRID_SIZE);

    reduce_max_stage1<<<grid, block>>>(d_x, partial, N);
    reduce_max_stage2<<<1, block>>>(partial, d_out);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Test code
    float cpu_out = 999.0f;

    printf("CPU output: %f, GPU output: %f, Error: %f\n", cpu_out, *h_out, fabsf(cpu_out-*h_out));

    free(h_x);
    free(h_out);

    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(partial);

    return 0;
}