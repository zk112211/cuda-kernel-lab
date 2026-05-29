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

__global__ void reduce_variance_stage1(const float* x, const float* mean, float* partial, int N){
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ float s_data[BLOCK_SIZE];
    s_data[tid] = 0.0f;

    for(int i = idx; i < N; i+= blockDim.x*gridDim.x){
        s_data[tid] += (x[i]-mean[0])*(x[i]-mean[0]);
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

__global__ void reduce_variance_stage2(float* partial, float* out, int N){
    int tid = threadIdx.x;
    __shared__ float s_data[BLOCK_SIZE];

    if(tid < GRID_SIZE){
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

    if(tid == 0) out[0] = s_data[0]/N;
}

int main(){
    int N = BLOCK_SIZE * GRID_SIZE + 123;

    float* h_x = (float*)malloc(N * sizeof(float));
    float* h_mean = (float*)malloc(sizeof(float));
    float* h_var = (float*)malloc(sizeof(float));

    for(int i = 0; i < N; i++){
        h_x[i] = (float)(i % 10);
    }

    float* d_x;
    float* d_partial_mean;
    float* d_mean;
    float* d_partial_var;
    float* d_var;

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_partial_mean, GRID_SIZE * sizeof(float));
    cudaMalloc(&d_mean, sizeof(float));
    cudaMalloc(&d_partial_var, GRID_SIZE * sizeof(float));
    cudaMalloc(&d_var, sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(GRID_SIZE);

    reduce_mean_stage1<<<grid, block>>>(d_x, d_partial_mean, N);
    reduce_mean_stage2<<<1, block>>>(d_partial_mean, d_mean, N);

    reduce_variance_stage1<<<grid, block>>>(d_x, d_mean, d_partial_var, N);
    reduce_variance_stage2<<<1, block>>>(d_partial_var, d_var, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_mean, d_mean, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_var, d_var, sizeof(float), cudaMemcpyDeviceToHost);

    float cpu_mean = 0.0f;
    for(int i = 0; i < N; i++){
        cpu_mean += h_x[i];
    }
    cpu_mean /= N;

    float cpu_var = 0.0f;
    for(int i = 0; i < N; i++){
        float diff = h_x[i] - cpu_mean;
        cpu_var += diff * diff;
    }
    cpu_var /= N;

    float mean_error = fabsf(h_mean[0] - cpu_mean);
    float var_error = fabsf(h_var[0] - cpu_var);

    printf("CPU Mean: %f\n", cpu_mean);
    printf("GPU Mean: %f\n", h_mean[0]);
    printf("Mean Error: %f\n", mean_error);

    printf("CPU Variance: %f\n", cpu_var);
    printf("GPU Variance: %f\n", h_var[0]);
    printf("Variance Error: %f\n", var_error);

    if(mean_error < 1e-5f && var_error < 1e-5f) printf("Test Passed\n");
    else printf("Test Failed\n");

    free(h_x);
    free(h_mean);
    free(h_var);

    cudaFree(d_x);
    cudaFree(d_partial_mean);
    cudaFree(d_mean);
    cudaFree(d_partial_var);
    cudaFree(d_var);

    return 0;
}