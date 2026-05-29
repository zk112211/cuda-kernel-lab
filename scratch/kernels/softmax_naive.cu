#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define BLOCK_SIZE 1024
#define GRID_SIZE 4

__global__ void row_max(const float* x, float* out, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float s_data[BLOCK_SIZE];

    for(int i = row; i < M; i+=gridDim.x){
        s_data[tid] = -1e9f;
        for(int j = tid; j < N; j+=blockDim.x){
            s_data[tid] = fmaxf(s_data[tid], x[i*N + j]);
        }
        __syncthreads();
        for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
            if(tid < stride) s_data[tid] = fmaxf(s_data[tid], s_data[tid+stride]);
        }
        __syncthreads();
        if(tid == 0){
            out[i] = s_data[0];
        }
        __syncthreads();
    }
}

__global__ void row_exp_sum(const float* x, const float* row_max, float* row_sum, int M, int N){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float s_data[BLOCK_SIZE];

    for(int i = row; i < M; i+=gridDim.x){
        s_data[tid] = 0.0f;
        for(int j = tid; j < N; j+=blockDim.x){
            s_data[tid] += expf(x[i*N + j]-row_max[i]);
        }
        __syncthreads();
        for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
            if(tid < stride) s_data[tid] += s_data[tid+stride];
        }
        __syncthreads();
        if(tid == 0){
            row_sum[i] = s_data[0];
        }
        __syncthreads();
    }
}

__global__ void softmax_norm(const float* x, const float* row_max, const float* row_sum, float* out, int M, int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int row = idx/N;
    if(idx < M*N) out[idx] = expf(x[idx] - row_max[row])/row_sum[row];
}

int main(){
    int M = 4;
    int N = 8;

    float* h_x = (float*)malloc(M * N * sizeof(float));
    float* h_out = (float*)malloc(M * N * sizeof(float));
    float* h_cpu = (float*)malloc(M * N * sizeof(float));

    srand(time(nullptr));

    for(int i = 0; i < M * N; i++){
        h_x[i] = (float)(rand() % 21 - 10);
    }

    float* d_x;
    float* d_out;
    float* d_row_max;
    float* d_row_sum;

    cudaMalloc(&d_x, M * N * sizeof(float));
    cudaMalloc(&d_out, M * N * sizeof(float));
    cudaMalloc(&d_row_max, M * sizeof(float));
    cudaMalloc(&d_row_sum, M * sizeof(float));

    cudaMemcpy(d_x, h_x, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 row_grid(4);
    dim3 row_block(BLOCK_SIZE);

    row_max<<<row_grid, row_block>>>(d_x, d_row_max, M, N);
    row_exp_sum<<<row_grid, row_block>>>(d_x, d_row_max, d_row_sum, M, N);

    int total = M * N;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    softmax_norm<<<grid_size, block_size>>>(d_x, d_row_max, d_row_sum, d_out, M, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0.0f;

    for(int i = 0; i < M; i++){
        float m = -1e9f;
        for(int j = 0; j < N; j++){
            m = fmaxf(m, h_x[i * N + j]);
        }

        float s = 0.0f;
        for(int j = 0; j < N; j++){
            s += expf(h_x[i * N + j] - m);
        }

        float row_sum_check = 0.0f;

        for(int j = 0; j < N; j++){
            h_cpu[i * N + j] = expf(h_x[i * N + j] - m) / s;

            float error = fabsf(h_cpu[i * N + j] - h_out[i * N + j]);
            max_error = fmaxf(max_error, error);

            row_sum_check += h_out[i * N + j];
        }

        printf("Row %d softmax sum: %f\n", i, row_sum_check);
    }

    if(max_error < 1e-5f) {
        printf("Test passed, max error: %f\n", max_error);
    } else {
        printf("Test failed, max error: %f\n", max_error);
    }

    free(h_x);
    free(h_out);
    free(h_cpu);

    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_row_max);
    cudaFree(d_row_sum);

    return 0;
}