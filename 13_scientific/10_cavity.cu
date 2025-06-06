#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>

using namespace std;

// Defined matrix struct since vector struct can't directly be sent to gpu
// memory
struct matrix {
  int rows;
  int cols;
  double *elems;
};

// ------------------- Functions for computing b -------------------
__global__ void compute_b(matrix *b, const matrix *u, const matrix *v,
                          const double rho, const double dt, const double dx,
                          const double dy, const int nx, const int ny) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int total = nx * ny;

  if (i >= total)
    return;

  int row = i / nx;
  int col = i % nx;

  if (row <= 0 || row >= ny - 1 || col <= 0 || col >= nx - 1)
    return;

  b->elems[i] =
      rho * (1.0 / dt *
                 ((u->elems[i + 1] - u->elems[i - 1]) / (2.0 * dx) +
                  (v->elems[i + nx] - v->elems[i - nx]) / (2.0 * dy)) -
             ((u->elems[i + 1] - u->elems[i - 1]) / (2.0 * dx)) *
                 ((u->elems[i + 1] - u->elems[i - 1]) / (2.0 * dx)) -
             2.0 * ((u->elems[i + nx] - u->elems[i - nx]) / (2.0 * dy) *
                    (v->elems[i + 1] - v->elems[i - 1]) / (2.0 * dx)) -
             ((v->elems[i + nx] - v->elems[i - nx]) / (2.0 * dy)) *
                 ((v->elems[i + nx] - v->elems[i - nx]) / (2.0 * dy)));
}

// ------------------- Functions for computing p -------------------

__device__ void copy_pn(const matrix *p, matrix *pn, const int nx, const int ny,
                        int idx) {
  pn->elems[idx] = p->elems[idx];
}

__device__ void calc_p(matrix *p, const matrix *pn, const matrix *b,
                       const double dx, const double dy, const int nx,
                       const int ny, const int idx) {

  p->elems[idx] = (dy * dy * (pn->elems[idx + 1] + pn->elems[idx - 1]) +
                   dx * dx * (pn->elems[idx + nx] + pn->elems[idx - nx]) -
                   b->elems[idx] * dx * dx * dy * dy) /
                  (2 * (dx * dx + dy * dy));
}

__device__ void pad_p(matrix *p, const int nx, const int ny, const int idx) {
  if (idx % nx == 0)
    p->elems[idx] = p->elems[idx + 1];
  else if (idx % nx == nx - 1)
    p->elems[idx] = p->elems[idx - 1];
  else if (idx < nx)
    p->elems[idx] = p->elems[idx + nx];
  else if (idx > nx * ny - nx)
    p->elems[idx] = p->elems[idx - nx];
  else
    printf("thread index not on edge of p");
}

__global__ void compute_p(matrix *pn, matrix *p, const matrix *b,
                          const double dx, const double dy, const int nit,
                          const int nx, const int ny) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total = nx * ny;

  int row = idx / nx;
  int col = idx % nx;

  for (int it = 0; it < nit; it++) {

    if (idx < total)
      copy_pn(p, pn, nx, ny, idx);

    __syncthreads();
    if (idx < total) {
      if (row > 0 && row < ny - 1 && col > 0 && col < nx - 1)
        calc_p(p, pn, b, dx, dy, nx, ny, idx);
      else if (row == 0 || row == ny - 1 || col == 0 || col == nx - 1)
        pad_p(p, nx, ny, idx);
    }
    __syncthreads();
  }
}

// ------------------- Functions for computing u and v -------------------
__device__ void copy_un_vn(const matrix *u, const matrix *v, matrix *un,
                           matrix *vn, const int nx, const int ny,
                           const int idx) {

  un->elems[idx] = u->elems[idx];
  vn->elems[idx] = v->elems[idx];
}

__device__ void calc_u_v(matrix *u, matrix *v, const matrix *un,
                         const matrix *vn, const matrix *p, const double dx,
                         const double dy, const double dt, const double rho,
                         const double nu, const int nx, const int ny,
                         const int idx) {

  u->elems[idx] =
      un->elems[idx] -
      un->elems[idx] * dt / dx * (un->elems[idx] - un->elems[idx - 1]) -
      un->elems[idx] * dt / dy * (un->elems[idx] - un->elems[idx - nx]) -
      dt / (2 * rho * dx) * (p->elems[idx + 1] - p->elems[idx - 1]) +
      nu * dt / (dx * dx) *
          (un->elems[idx + 1] - 2 * un->elems[idx] + un->elems[idx - 1]) +
      nu * dt / (dy * dy) *
          (un->elems[idx + nx] - 2 * un->elems[idx] + un->elems[idx - nx]);
  v->elems[idx] =
      vn->elems[idx] -
      vn->elems[idx] * dt / dx * (vn->elems[idx] - vn->elems[idx - 1]) -
      vn->elems[idx] * dt / dy * (vn->elems[idx] - vn->elems[idx - nx]) -
      dt / (2 * rho * dx) * (p->elems[idx + nx] - p->elems[idx - nx]) +
      nu * dt / (dx * dx) *
          (vn->elems[idx + 1] - 2 * vn->elems[idx] + vn->elems[idx - 1]) +
      nu * dt / (dy * dy) *
          (vn->elems[idx + nx] - 2 * vn->elems[idx] + vn->elems[idx - nx]);
}

__device__ void pad_u_v(matrix *u, matrix *v, const int nx, const int ny,
                        const int idx) {
  u->elems[idx] = 0;
  v->elems[idx] = 0;
  if (idx / nx == ny - 1)
    u->elems[idx] = 1;
}

__global__ void compute_u_v(matrix *u, matrix *v, matrix *un, matrix *vn,
                            const matrix *p, const double dt, const double dx,
                            const double dy, const double rho, const double nu,
                            const int nx, const int ny) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int total = nx * ny;
  int row = idx / nx;
  int col = idx % nx;

  if (idx < total)
    copy_un_vn(u, v, un, vn, nx, ny, idx);

  __syncthreads();
  if (idx < total) {

    if (row > 0 && row < ny - 1 && col > 0 && col < nx - 1)
      calc_u_v(u, v, un, vn, p, dx, dy, dt, rho, nu, nx, ny, idx);
    else if (row == 0 || row == ny - 1 || col == 0 || col == nx - 1)
      pad_u_v(u, v, nx, ny, idx);
  }
  __syncthreads();
}

void init_matrix(matrix *&mat, const int nx, const int ny) {
  // Allocate the matrix struct
  cudaMallocManaged(&mat, sizeof(matrix));
  mat->rows = ny;
  mat->cols = nx;

  // Allocate for elements
  cudaError_t err =
      cudaMallocManaged(&(mat->elems), mat->rows * mat->cols * sizeof(double));
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  for (int i = 0; i < ny * nx; ++i)
    mat->elems[i] = 0.0;
}

int main() {
  constexpr int nx = 41;
  constexpr int ny = 41;
  const int nt = 500;
  const int nit = 50;
  const double dx = 2. / (nx - 1);
  const double dy = 2. / (ny - 1);
  const double dt = .01;
  const double rho = 1.;
  const double nu = .02;

  constexpr int THREADS_PER_BLOCK = 256; // maybe higher? 512? 1024?
  constexpr int BLOCKS = (nx * ny + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  matrix *u;
  matrix *v;
  matrix *p;
  matrix *b;
  matrix *un;
  matrix *vn;
  matrix *pn;
  init_matrix(u, nx, ny);
  init_matrix(v, nx, ny);
  init_matrix(p, nx, ny);
  init_matrix(b, nx, ny);
  init_matrix(un, nx, ny);
  init_matrix(vn, nx, ny);
  init_matrix(pn, nx, ny);

  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");

  auto tic = std::chrono::steady_clock::now();

  for (int n = 0; n < nt; n++) {
    compute_b<<<BLOCKS, THREADS_PER_BLOCK>>>(b, u, v, rho, dt, dx, dy, nx, ny);

    cudaDeviceSynchronize();

    compute_p<<<BLOCKS, THREADS_PER_BLOCK>>>(pn, p, b, dx, dy, nit, nx, ny);

    cudaDeviceSynchronize();

    compute_u_v<<<BLOCKS, THREADS_PER_BLOCK>>>(u, v, un, vn, p, dt, dx, dy, rho,
                                               nu, nx, ny);

    cudaDeviceSynchronize();

    // leave this for now, only one host thread should write to file
    if (n % 10 == 0) {
      for (int i = 0; i < ny * nx; i++)
        ufile << u->elems[i] << " ";
      ufile << "\n";
      for (int i = 0; i < ny * nx; i++)
        vfile << v->elems[i] << " ";
      vfile << "\n";
      for (int i = 0; i < ny * nx; i++)
        pfile << p->elems[i] << " ";
      pfile << "\n";
    }
  }
  auto toc = std::chrono::steady_clock::now();
  double time = std::chrono::duration<double>(toc - tic).count();
  printf("time = %lf\n", time);

  ufile.close();
  vfile.close();
  pfile.close();

  cudaFree(u);
  cudaFree(v);
  cudaFree(p);
  cudaFree(b);
  cudaFree(un);
  cudaFree(vn);
  cudaFree(pn);
}
