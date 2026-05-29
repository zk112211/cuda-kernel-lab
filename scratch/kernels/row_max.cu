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

int main(){
    int M = BLOCK_SIZE * GRID_SIZE * 2 + 63;
    int N = BLOCK_SIZE * GRID_SIZE + 127;

    float* h_x, *h_out;

    h_x = (float*)malloc(M*N*sizeof(float));
    h_out = (float*)malloc(M*sizeof(float));

    srand(time(nullptr));

    for(int i = 0; i < M*N; i++){
        h_x[i] = (float)(rand()%201 - 100);
    }
    
    float* d_x, *d_out;

    cudaMalloc(&d_x, M*N*sizeof(float));
    cudaMalloc(&d_out, M*sizeof(float));

    cudaMemcpy(d_x, h_x, M*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(GRID_SIZE);
    dim3 block(BLOCK_SIZE);

    row_max<<<grid, block>>>(d_x, d_out, M, N);

    cudaMemcpy(h_out, d_out, M*sizeof(float), cudaMemcpyDeviceToHost);

    float cpu_out[M], max_error = 0.0f;

    for(int i = 0; i < M; i++){
        cpu_out[i] = -1e9f;
        for(int j = 0; j < N; j++){
            cpu_out[i] = fmaxf(cpu_out[i], h_x[i*N + j]);
        }
        max_error = max(max_error, fabsf(cpu_out[i]-h_out[i]));
    }

    if(max_error < 1e-4f) printf("Test passed\n");
    else printf("Test failed, max error: %f\n", max_error);

    free(h_x);
    free(h_out);

    cudaFree(d_x);
    cudaFree(d_out);

    return 0;
}