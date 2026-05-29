#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void vector_add(const float* a, const float* b, float* c, int N){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for(int i = idx; i < N; i+=stride){
        c[i] = a[i]+b[i];
    }
}

#define BLOCK_SIZE 1024
#define GRID_SIZE 4


int main(){
    int N = BLOCK_SIZE * GRID_SIZE + 123;


    float* h_a, *h_b, *h_c;
    h_a = (float*)malloc(N*sizeof(float));
    h_b = (float*)malloc(N*sizeof(float));
    h_c = (float*)malloc(N*sizeof(float));

    for(int i = 0; i < N; i++){
        h_a[i] = 1.0f*i;
        h_b[i] = 2.0f*i*i;
    }

    float* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));

    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);


    dim3 block(BLOCK_SIZE);
    dim3 grid(GRID_SIZE);

    vector_add<<<grid, block>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        float error = fabs(h_c[i] - expected);
        if (error > max_error) max_error = error;
    }

    if(max_error < 1e-4f) printf("Test pass, max error: %f\n", max_error);
    else printf("Test failed, max error: %f\n", max_error);

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}