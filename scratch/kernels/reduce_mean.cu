#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define BLOCK_SIZE 1024
#define GRID_SIZE 4

__global__ void reduce_mean_stage1(const float* x, float* partial, int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    __shared__ float s_data[BLOCK_SIZE];
    s_data[tid] = 0.0f;

    for(int i = idx; i < N; i+=stride){
        s_data[tid] += x[i];
    }
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride>>=1){
        if(tid < stride){
            s_data[tid] += s_data[tid+stride];
        }
        __syncthreads();
    }

    if(tid == 0) partial[blockIdx.x] = s_data[0];
}

__global__ void reduce_mean_stage2(float* partial, float* out, int N){
    int tid = threadIdx.x;
    __shared__ float s_data[BLOCK_SIZE];
    if(tid < GRID_SIZE) {
        s_data[tid] = partial[tid];
    }
    else s_data[tid] = 0.0f;

    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride>>=1){
        if(tid < stride){
            s_data[tid] += s_data[tid+stride];
        }
        __syncthreads();
    }

    if(tid == 0) *out = s_data[0]/N;
}

int main(){
    int N = BLOCK_SIZE * GRID_SIZE + 123;

    float* h_x = (float*)malloc(N * sizeof(float));
    float* h_out = (float*)malloc(sizeof(float));

    for(int i = 0; i < N; i++){
        h_x[i] = 1.0f;
    }

    float* d_x;
    float* d_partial;
    float* d_out;

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_partial, GRID_SIZE * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(GRID_SIZE);

    reduce_mean_stage1<<<grid, block>>>(d_x, d_partial, N);
    reduce_mean_stage2<<<1, block>>>(d_partial, d_out, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    float cpu_out = 0.0f;
    for(int i = 0; i < N; i++){
        cpu_out += h_x[i];
    }
    cpu_out /= N;

    float max_error = fabsf(h_out[0] - cpu_out);

    printf("CPU Mean: %f\n", cpu_out);
    printf("GPU Mean: %f\n", h_out[0]);
    printf("Max Error: %f\n", max_error);

    if(max_error < 1e-6f) printf("Test Passed\n");
    else printf("Test Failed\n");

    free(h_x);
    free(h_out);

    cudaFree(d_x);
    cudaFree(d_partial);
    cudaFree(d_out);

    return 0;
}