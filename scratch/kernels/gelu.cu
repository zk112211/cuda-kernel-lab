#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void gelu(const float* x, float* out, int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
     
    for(int i = idx; i < N; i+=stride){
        out[i] = 0.5*x[i]*(1+tanhf(sqrtf(2/3.1415926535f)*(x[i]+0.044715*x[i]*x[i]*x[i])));
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
        h_x[i] = (float)(i % 21 - 10) / 2.0f;
    }

    float* d_x;
    float* d_out;

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(GRID_SIZE);

    gelu<<<grid, block>>>(d_x, d_out, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0.0f;

    for(int i = 0; i < N; i++){
        float v = h_x[i];
        h_cpu[i] = 0.5f * v * (1.0f + tanhf(sqrtf(2.0f / 3.1415926535f) * (v + 0.044715f * v * v * v)));
        max_error = fmaxf(max_error, fabsf(h_cpu[i] - h_out[i]));
    }

    printf("GELU Max Error: %f\n", max_error);

    if(max_error < 1e-6f) printf("Test Passed\n");
    else printf("Test Failed\n");

    free(h_x);
    free(h_out);
    free(h_cpu);

    cudaFree(d_x);
    cudaFree(d_out);

    return 0;
}