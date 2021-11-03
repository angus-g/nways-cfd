#include <stdio.h>

#include <algorithm>
#include <execution>
#include <thrust/iterator/counting_iterator.h>

#include "jacobi.h"
#include "error.h"

__global__ void jacobistep(double *psinew, double *psi, int m, int n) {
  int i, j;
  i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  j = threadIdx.y + blockIdx.y * blockDim.y + 1;

  psinew[i * (m + 2) + j] =
    0.25 * (psi[(i - 1) * (m + 2) + j] + psi[(i + 1) * (m + 2) + j] +
	    psi[i * (m + 2) + j - 1] + psi[i * (m + 2) + j + 1]);
}

__global__ void jacobistepvort(double *zetnew, double *psinew, double *zet, double *psi,
                    int m, int n, double re) {
  int i, j;
  i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  j = threadIdx.y + blockIdx.y * blockDim.y + 1;

  psinew[i * (m + 2) + j] =
    0.25 * (psi[(i - 1) * (m + 2) + j] + psi[(i + 1) * (m + 2) + j] +
	    psi[i * (m + 2) + j - 1] + psi[i * (m + 2) + j + 1] -
	    zet[i * (m + 2) + j]);

  zetnew[i * (m + 2) + j] =
    0.25 * (zet[(i - 1) * (m + 2) + j] + zet[(i + 1) * (m + 2) + j] +
	    zet[i * (m + 2) + j - 1] + zet[i * (m + 2) + j + 1]) -
    re / 16.0 *
    ((psi[i * (m + 2) + j + 1] - psi[i * (m + 2) + j - 1]) *
     (zet[(i + 1) * (m + 2) + j] - zet[(i - 1) * (m + 2) + j]) -
     (psi[(i + 1) * (m + 2) + j] - psi[(i - 1) * (m + 2) + j]) *
     (zet[i * (m + 2) + j + 1] - zet[i * (m + 2) + j - 1]));
}

__device__ double d_error;

__global__ void deltasq_impl(double *newarr, double *oldarr, int m, int n) {
  int i, j;

  double tmp;

  i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  j = threadIdx.y + blockIdx.y * blockDim.y + 1;

  tmp = newarr[i * (m + 2) + j] - oldarr[i * (m + 2) + j];
  atomicAdd(&d_error, tmp * tmp);
}

__global__ void reset_error() {
  d_error = 0;
}

double deltasq(dim3 dimGrid, dim3 dimBlock, double *newarr, double *oldarr, int m, int n) {
  reset_error<<<1, 1>>>();

  deltasq_impl<<<dimGrid, dimBlock>>>(newarr, oldarr, m, n);

  double res;
  cudaMemcpyFromSymbol(&res, d_error, sizeof(res), 0, cudaMemcpyDeviceToHost);

  return res;
}
