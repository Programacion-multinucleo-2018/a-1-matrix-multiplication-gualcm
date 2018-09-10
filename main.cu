#include "common.h"
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <math.h>
#include <string>

using namespace std;

// Matrix Multiplication

// 1. In CPU without threads.
void inCPUWithoutThreads(int *A, int *B, int *C, const int nx, const int ny)
{
    int i, j, k;
    for(i = 0; i < ny; i++)
    {
        for(j = 0; j < nx; j++)
        {
            for(k = 0; k < ny; k++)
            {
                C[i * nx + j] += (A[i * nx + k] * B[k *nx + i]);
            }
        }

    }
    return;
}

void inCPUWithThreads(int *A, int *B, int *C, const int nx, const int ny)
{
    int i, j, k;
    #pragma omp parallel for private(i, j, k) shared(A, B, C)
    for(i = 0; i < ny; i++)
    {
        for(j = 0; j < nx; j++)
        {
            for(k = 0; k < ny; k++)
            {
                C[i * nx + j] += (A[i * nx + k] * B[k *nx + i]);
            }
        }

    }
    return;
}

__global__ void cudaWithBlocksAndThreads(int *MatA, int *MatB, int *MatC, const int nx, const int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < nx )
        for (int iy = 0; iy < ny; iy++)
        {
            int idx = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
}

void initialData(int *ip, const int size)
{
    int i;

    for(i = 0; i < size; i++)
    {
        ip[i] = i;
    }
}

int main(int argc, char const *argv[])
{
    printf("%s Starting...\n", argv[0]);

    // Set up data size of matrix
    int size = 0;
    if(argc < 2)
        size = 1000;
    else
        size = stoi(argv[1]);

    int nx = size;
    int ny = size;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // Malloc host memory
    int *h_A, *h_B, *hostRefNoThreads, *hostRefThreads, *gpuRefThreads;
    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    hostRefNoThreads = (int *)malloc(nBytes);
    hostRefThreads = (int *)malloc(nBytes);
    gpuRefThreads = (int *)malloc(nBytes);

    // Initialize data at host side
    initialData(h_A, nxy);
    initialData(h_B, nxy);

    // Start Matrix Multiplication and timer
    // CPU No Threads
    auto start =  chrono::high_resolution_clock::now();
    inCPUWithoutThreads(h_A, h_B, hostRefNoThreads, nx, ny);
    auto end =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end - start;
    printf("inCPUWithoutThreads elapsed %f ms\n", duration_ms.count());

    // CPU OMP Threads
    start = chrono::high_resolution_clock::now();
    inCPUWithThreads(h_A, h_B, hostRefThreads, nx, ny);
    end = chrono::high_resolution_clock::now();
    duration_ms = end - start;
    printf("inCPUWithThreads elapsed %f ms\n", duration_ms.count());

    // Malloc device global memory
    int *d_MatA, *d_MatB, *d_MatC;
    SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
    SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
    SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

    // transfer data from host to device
    SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
    SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

    // invoke kernel at host side
    int dimx = 256;
    dim3 block(dimx, 1);
    dim3 grid((nx + block.x - 1) / block.x, 1);

    start_cpu =  chrono::high_resolution_clock::now();
    sumMatrixOnGPU1D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
    end_cpu =  chrono::high_resolution_clock::now();

    duration_ms = end_cpu - start_cpu;

    printf("cudaWithBlocksAndThreads <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x,
           grid.y,
           block.x, block.y, duration_ms.count());

    // SAFE_CALL kernel error
    SAFE_CALL(cudaGetLastError(), "Error with last error");

    // copy kernel result back to host side
    /* SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC"); */

    // check device results
    /* checkResult(hostRef, gpuRef, nxy); */

    // free device global memory
    SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
    SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRefWithoutThreads);
    free(hostRefWithThreads);

    // reset device
    SAFE_CALL(cudaDeviceReset(), "Error reseting");

    return 0;
}
