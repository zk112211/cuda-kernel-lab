#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void sigmoid_forward(const float* x, float* out, int N){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    for(int i = idx; i < N; i+= stride) {
        out[i] = 1/(1+expf(-x[i]));
    }
}

__global__ void tanh_forward(const float* x, float* out, int N){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    for(int i = idx; i < N; i+= stride) {
        out[i] = tanhf(x[i]);
    }
}

#define BLOCK_SIZE 1024
#define GRID_SIZE 4

int main(){
    int N = BLOCK_SIZE * GRID_SIZE + 123;

    float* h_x = (float*)malloc(N * sizeof(float));
    float* h_sigmoid_out = (float*)malloc(N * sizeof(float));
    float* h_tanh_out = (float*)malloc(N * sizeof(float));
    float* h_sigmoid_cpu = (float*)malloc(N * sizeof(float));
    float* h_tanh_cpu = (float*)malloc(N * sizeof(float));

    for(int i = 0; i < N; i++){
        h_x[i] = (float)(i % 21 - 10) / 2.0f;
    }

    float* d_x;
    float* d_sigmoid_out;
    float* d_tanh_out;

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_sigmoid_out, N * sizeof(float));
    cudaMalloc(&d_tanh_out, N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(GRID_SIZE);

    sigmoid_forward<<<grid, block>>>(d_x, d_sigmoid_out, N);
    tanh_forward<<<grid, block>>>(d_x, d_tanh_out, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_sigmoid_out, d_sigmoid_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tanh_out, d_tanh_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error_sigmoid = 0.0f;
    float max_error_tanh = 0.0f;

    for(int i = 0; i < N; i++){
        h_sigmoid_cpu[i] = 1.0f / (1.0f + expf(-h_x[i]));
        h_tanh_cpu[i] = tanhf(h_x[i]);

        max_error_sigmoid = fmaxf(max_error_sigmoid, fabsf(h_sigmoid_cpu[i] - h_sigmoid_out[i]));
        max_error_tanh = fmaxf(max_error_tanh, fabsf(h_tanh_cpu[i] - h_tanh_out[i]));
    }

    printf("Sigmoid Max Error: %f\n", max_error_sigmoid);
    printf("Tanh Max Error: %f\n", max_error_tanh);

    if(max_error_sigmoid < 1e-6f && max_error_tanh < 1e-6f) printf("Test Passed\n");
    else printf("Test Failed\n");

    free(h_x);
    free(h_sigmoid_out);
    free(h_tanh_out);
    free(h_sigmoid_cpu);
    free(h_tanh_cpu);

    cudaFree(d_x);
    cudaFree(d_sigmoid_out);
    cudaFree(d_tanh_out);

    return 0;
}