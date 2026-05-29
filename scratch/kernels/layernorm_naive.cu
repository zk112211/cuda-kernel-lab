#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define BLOCK_SIZE 1024

__global__ void layernorm_naive(const float* x, const float* gamma, const float* beta, float* out, int B, int T, int C, float eps){
    int b = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;

    int row = b*T + t;
    int base = row*C;

    __shared__ float mean, variance;

    __shared__ float s_data[BLOCK_SIZE];
    s_data[tid] = 0.0f;

    for(int c = tid; c < C; c+=blockDim.x){
        int idx = base+c;
        s_data[tid] += x[idx];
    }
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride>>=1){
        if(tid < stride) s_data[tid] += s_data[tid+stride];
        __syncthreads();
    }

    if(tid == 0) mean = s_data[0]/C;

    __syncthreads();

    s_data[tid] = 0.0f;

    for(int c = tid; c < C; c+= blockDim.x){
        int idx = base+c;
        s_data[tid] += (x[idx]-mean)*(x[idx]-mean);
    }
    __syncthreads();

    for(int stride = blockDim.x/2; stride > 0; stride>>=1){
        if(tid < stride){
            s_data[tid] += s_data[tid+stride];
        }
        __syncthreads();
    }

    if(tid == 0) variance = s_data[0]/C;

    __syncthreads();

    for(int c = tid; c < C; c+=blockDim.x){
        int idx = base+c;
        out[idx] = (x[idx]-mean)/sqrtf(variance+eps)*gamma[c] + beta[c];
    }
}

int main(){
    int B = 2;
    int T = 3;
    int C = 8;

    float eps = 1e-5f;

    int total = B * T * C;
    int rows = B * T;

    float* h_x = (float*)malloc(total * sizeof(float));
    float* h_gamma = (float*)malloc(C * sizeof(float));
    float* h_beta = (float*)malloc(C * sizeof(float));
    float* h_out = (float*)malloc(total * sizeof(float));
    float* h_cpu = (float*)malloc(total * sizeof(float));

    for(int i = 0; i < total; i++){
        h_x[i] = (float)(i % 17 - 8) / 4.0f;
    }

    for(int c = 0; c < C; c++){
        h_gamma[c] = 1.0f;
        h_beta[c] = 0.0f;
    }

    float* d_x;
    float* d_gamma;
    float* d_beta;
    float* d_out;

    cudaMalloc(&d_x, total * sizeof(float));
    cudaMalloc(&d_gamma, C * sizeof(float));
    cudaMalloc(&d_beta, C * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));

    cudaMemcpy(d_x, h_x, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, C * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(B, T);
    dim3 block(BLOCK_SIZE);

    layernorm_naive<<<grid, block>>>(d_x, d_gamma, d_beta, d_out, B, T, C, eps);

    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, total * sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0.0f;

    for(int row = 0; row < rows; row++){
        int base = row * C;

        float mean = 0.0f;
        for(int c = 0; c < C; c++){
            mean += h_x[base + c];
        }
        mean /= C;

        float variance = 0.0f;
        for(int c = 0; c < C; c++){
            float diff = h_x[base + c] - mean;
            variance += diff * diff;
        }
        variance /= C;

        for(int c = 0; c < C; c++){
            int idx = base + c;
            h_cpu[idx] = (h_x[idx] - mean) / sqrtf(variance + eps) * h_gamma[c] + h_beta[c];
            max_error = fmaxf(max_error, fabsf(h_cpu[idx] - h_out[idx]));
        }
    }

    printf("LayerNorm Max Error: %f\n", max_error);

    if(max_error < 1e-5f) printf("Test Passed\n");
    else printf("Test Failed\n");

    free(h_x);
    free(h_gamma);
    free(h_beta);
    free(h_out);
    free(h_cpu);

    cudaFree(d_x);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    cudaFree(d_out);

    return 0;
}