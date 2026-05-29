#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void scalar_add(const float* x, float* out, float s, int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for(int i = idx; i < N; i+=stride){
        out[i] = x[i]+s;
    }
}

__global__ void scalar_mul(const float* x, float* out, float s, int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for(int i = idx; i < N; i+=stride){
        out[i] = x[i]*s;
    }
}

#define BLOCK_SIZE 1024
#define GRID_SIZE 4

int main() {

    int N = BLOCK_SIZE * GRID_SIZE + 123;

    float scalar_add_value = 3.0f;
    float scalar_mul_value = 0.5f;

    float* h_x = (float*)malloc(N * sizeof(float));

    float* h_add_out = (float*)malloc(N * sizeof(float));
    float* h_mul_out = (float*)malloc(N * sizeof(float));

    float* h_add_cpu = (float*)malloc(N * sizeof(float));
    float* h_mul_cpu = (float*)malloc(N * sizeof(float));

    srand(time(nullptr));

    for (int i = 0; i < N; i++) {
        h_x[i] = (float)(rand() % 201 - 100);
    }

    float* d_x;
    float* d_add_out;
    float* d_mul_out;

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_add_out, N * sizeof(float));
    cudaMalloc(&d_mul_out, N * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(GRID_SIZE);

    scalar_add<<<grid, block>>>(d_x, d_add_out, scalar_add_value, N);

    scalar_mul<<<grid, block>>>(d_x, d_mul_out, scalar_mul_value, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_add_out, d_add_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_mul_out, d_mul_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error_add = 0.0f;
    float max_error_mul = 0.0f;

    for (int i = 0; i < N; i++) {

        h_add_cpu[i] = h_x[i] + scalar_add_value;
        h_mul_cpu[i] = h_x[i] * scalar_mul_value;

        max_error_add = fmaxf(max_error_add, fabsf(h_add_cpu[i] - h_add_out[i]));

        max_error_mul = fmaxf(max_error_mul, fabsf(h_mul_cpu[i] - h_mul_out[i]));
    }

    printf("Scalar Add Max Error: %f\n", max_error_add);

    printf("Scalar Mul Max Error: %f\n", max_error_mul);

    if (max_error_add < 1e-6f && max_error_mul < 1e-6f) {

        printf("Test Passed\n");

    } else {

        printf("Test Failed\n");
    }

    free(h_x);

    free(h_add_out);
    free(h_mul_out);

    free(h_add_cpu);
    free(h_mul_cpu);

    cudaFree(d_x);
    cudaFree(d_add_out);
    cudaFree(d_mul_out);

    return 0;
}