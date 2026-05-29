#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void relu(const float* x, float* out, int N){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    for(int i = idx; i < N; i+=stride){
        out[i] = fmaxf(x[i], 0.0f);
    }
}

#define BLOCK_SIZE 1024
#define GRID_SIZE 4

int main(){
    int N = BLOCK_SIZE * GRID_SIZE + 123;

    float* h_x = (float*)malloc(N * sizeof(float));
    float* h_out = (float*)malloc(N * sizeof(float));
    float* h_cpu = (float*)malloc(N * sizeof(float));

    for(int i = 0; i < N; i++){
        h_x[i] = (float)(i % 21 - 10);
    }

    float* d_x;
    float* d_out;

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(GRID_SIZE);

    relu<<<grid, block>>>(d_x, d_out, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0.0f;

    for(int i = 0; i < N; i++){
        h_cpu[i] = fmaxf(h_x[i], 0.0f);
        max_error = fmaxf(max_error, fabsf(h_cpu[i] - h_out[i]));
    }

    printf("ReLU Max Error: %f\n", max_error);

    if(max_error < 1e-6f) printf("Test Passed\n");
    else printf("Test Failed\n");

    free(h_x);
    free(h_out);
    free(h_cpu);

    cudaFree(d_x);
    cudaFree(d_out);

    return 0;
}