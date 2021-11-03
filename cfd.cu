#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "arraymalloc.h"
#include "boundary.h"
#include "cfdio.h"
#include "jacobi.h"
#include "error.h"

int main(int argc, char **argv) {
  int printfreq = 1000; // output frequency
  double error, bnorm;
  double tolerance = 0; // tolerance for convergence. <=0 means do not check

  // main arrays
  double *psi, *zet;
  // temporary versions of main arrays
  double *psi_old, *psi_new;
  double *zet_old, *zet_new;

  // command line arguments
  int scalefactor, numiter;

  size_t sizef;

  double re; // Reynold's number - must be less than 3.7

  // simulation sizes
  int bbase = 10;
  int hbase = 15;
  int wbase = 5;
  int mbase = 32;
  int nbase = 32;

  int irrotational = 1, checkerr = 0;

  int m, n, b, h, w;
  int iter;
  int i, j;

  int device;

  double tstart, tstop, ttot, titer;

  // do we stop because of tolerance?
  if (tolerance > 0) {
    checkerr = 1;
  }

  // check command line parameters and parse them

  if (argc < 3 || argc > 4) {
    printf("Usage: cfd <scale> <numiter> [reynolds]\n");
    return 0;
  }

  scalefactor = atoi(argv[1]);
  numiter = atoi(argv[2]);

  if (argc == 4) {
    re = atof(argv[3]);
    irrotational = 0;
  } else {
    re = -1.0;
  }

  if (!checkerr) {
    printf("Scale Factor = %i, iterations = %i\n", scalefactor, numiter);
  } else {
    printf("Scale Factor = %i, iterations = %i, tolerance= %g\n", scalefactor,
           numiter, tolerance);
  }

  if (irrotational) {
    printf("Irrotational flow\n");
  } else {
    printf("Reynolds number = %f\n", re);
  }

  // Calculate b, h & w and m & n
  b = bbase * scalefactor;
  h = hbase * scalefactor;
  w = wbase * scalefactor;
  m = mbase * scalefactor;
  n = nbase * scalefactor;

  re = re / (double)scalefactor;

  printf("Running CFD on %d x %d grid in cuda\n", m, n);

  device = 0;
  HANDLE_ERROR(cudaSetDevice(device));

  // allocate arrays

  sizef = (m + 2) * (n + 2) * sizeof(double);
  HANDLE_ERROR(cudaHostAlloc((void **)&psi, sizef, cudaHostAllocDefault));
  HANDLE_ERROR(cudaMalloc((void **)&psi_old, sizef));
  HANDLE_ERROR(cudaMalloc((void **)&psi_new, sizef));

  nvtxRangePush("Initialization");
  // zero the psi array
  for (i = 0; i < m + 2; i++) {
    for (j = 0; j < n + 2; j++) {
      psi[i * (m + 2) + j] = 0.0;
    }
  }
  nvtxRangePop(); // pop

  if (!irrotational) {
    // allocate arrays
    HANDLE_ERROR(cudaHostAlloc((void **)&zet, sizef, cudaHostAllocDefault));
    HANDLE_ERROR(cudaMalloc((void **)&zet_old, sizef));
    HANDLE_ERROR(cudaMalloc((void **)&zet_new, sizef));

    // zero the zeta array
    nvtxRangePush("Initialization");

    for (i = 0; i < m + 2; i++) {
      for (j = 0; j < n + 2; j++) {
        zet[i * (m + 2) + j] = 0.0;
      }
    }
    nvtxRangePop(); // pop for reading file
  }

  // set the psi boundary conditions
  nvtxRangePush("Boundary_PSI");

  boundarypsi(psi, m, n, b, h, w);
  nvtxRangePop(); // pop

  // compute normalisation factor for error

  bnorm = 0.0;
  nvtxRangePush("Compute_Normalization");

  for (i = 0; i < m + 2; i++) {
    for (j = 0; j < n + 2; j++) {
      bnorm += psi[i * (m + 2) + j] * psi[i * (m + 2) + j];
    }
  }
  nvtxRangePop(); // pop

  if (!irrotational) {
    // update zeta BCs that depend on psi
    boundaryzet(zet, psi, m, n);

    // update normalisation
    nvtxRangePush("Compute_Normalization");
    for (i = 0; i < m + 2; i++) {
      for (j = 0; j < n + 2; j++) {
        bnorm += zet[i * (m + 2) + j] * zet[i * (m + 2) + j];
      }
    }
    nvtxRangePop(); // pop
  }

  bnorm = sqrt(bnorm);

  // copy initialised field to device
  HANDLE_ERROR(cudaMemcpy(psi_old, psi, sizef, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(psi_new, psi, sizef, cudaMemcpyHostToDevice));

  // begin iterative Jacobi loop
  printf("\nStarting main loop...\n\n");

  tstart = gettime();
  nvtxRangePush("Overall_Iteration");

  dim3 dimBlock(32, 32);
  dim3 dimGrid(64, 64);

  for (iter = 1; iter <= numiter; iter++) {
    // calculate psi for next iteration
    nvtxRangePush("JacobiStep");
    if (irrotational) {
      jacobistep<<<dimGrid, dimBlock>>>(psi_new, psi_old, m, n);
      HANDLE_ERROR(cudaPeekAtLastError());
    } else {
      jacobistepvort<<<dimGrid, dimBlock>>>(zet_new, psi_new, zet_old, psi_old, m, n, re);
    }
    nvtxRangePop(); // pop
    nvtxRangePush("Calculate_Error");
    // calculate current error if required

    if (checkerr || iter == numiter) {
      error = deltasq(dimGrid, dimBlock, psi_new, psi_old, m, n);

      if (!irrotational) {
        error += deltasq(dimGrid, dimBlock, zet_new, zet_old, m, n);
      }

      error = sqrt(error);
      error = error / bnorm;
    }
    nvtxRangePop(); // pop

    // quit early if we have reached required tolerance

    if (checkerr) {
      if (error < tolerance) {
        printf("Converged on iteration %d\n", iter);
        break;
      }
    }

    // copy back
    nvtxRangePush("Switch_Array");

    // swap old/new arrays
    //HANDLE_ERROR(cudaMemcpy(psi_d, psitmp_d, sizef, cudaMemcpyDeviceToDevice));
    //HANDLE_ERROR(cudaPeekAtLastError());

    std::swap(psi_old, psi_new);
    /*
    for (i = 1; i <= m; i++) {
      for (j = 1; j <= n; j++) {
        psi[i * (m + 2) + j] = psitmp[i * (m + 2) + j];
      }
    }

    if (!irrotational) {
      for (i = 1; i <= m; i++) {
        for (j = 1; j <= n; j++) {
          zet[i * (m + 2) + j] = zettmp[i * (m + 2) + j];
        }
      }
    }
    */
    nvtxRangePop(); // pop

    if (!irrotational) {
      // update zeta BCs that depend on psi
      boundaryzet(zet, psi, m, n);
    }

    // print loop information

    if (iter % printfreq == 0) {
      if (!checkerr) {
        printf("Completed iteration %d\n", iter);
      } else {
        printf("Completed iteration %d, error = %g\n", iter, error);
      }
    }
  }
  nvtxRangePop(); // pop

  if (iter > numiter)
    iter = numiter;

  tstop = gettime();

  ttot = tstop - tstart;
  titer = ttot / (double)iter;

  // print out some stats

  printf("\n... finished\n");
  printf("After %d iterations, the error is %g\n", iter, error);
  printf("Time for %d iterations was %g seconds\n", iter, ttot);
  printf("Each iteration took %g seconds\n", titer);

  // output results

  HANDLE_ERROR(cudaMemcpy(psi, psi_new, sizef, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaPeekAtLastError());

  writedatafiles(psi, m, n, scalefactor);
  writeplotfile(m, n, scalefactor);

  // free un-needed arrays
  HANDLE_ERROR(cudaFree(psi_old));
  HANDLE_ERROR(cudaFree(psi_new));
  HANDLE_ERROR(cudaFreeHost(psi));

  if (!irrotational) {
    free(zet);
  }

  printf("... finished\n");

  return 0;
}
