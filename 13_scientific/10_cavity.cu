#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <chrono>

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

  // Skip boundary points
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

__global__ void copy_pn(const matrix *p, matrix *pn, const int nx,
                        const int ny) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int total = nx * ny;

  if (i >= total)
    return;

  pn->elems[i] = p->elems[i];
}

__global__ void calc_p(matrix *p, const matrix *pn, const matrix *b,
                       const double dx, const double dy, const int nx,
                       const int ny) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int total = nx * ny;

  if (i >= total)
    return;

  int row = i / nx;
  int col = i % nx;

  // Skip boundary points
  if (row <= 0 || row >= ny - 1 || col <= 0 || col >= nx - 1)
    return;

  p->elems[i] = (dy * dy * (pn->elems[i + 1] + pn->elems[i - 1]) +
                 dx * dx * (pn->elems[i + nx] + pn->elems[i - nx]) -
                 b->elems[i] * dx * dx * dy * dy) /
                (2 * (dx * dx + dy * dy));
}

__host__ void pad_p(matrix *p, const int nx,
                    const int ny) // leave on host for now
{
  // pad left most and rightmost columns
  for (int i = 0; i < nx * ny; i += nx) {
    p->elems[i] = p->elems[i + 1];
    p->elems[i + nx - 1] = p->elems[i + nx - 2];
  }

  // pad top and bottom row
  for (int i = 0; i < nx; ++i) {
    p->elems[i] = p->elems[i + nx];
    p->elems[ny * nx - nx + i] = 0.0;
  }
}

__host__ void compute_p(matrix *pn, matrix *p, const matrix *b, const double dx,
                        const double dy, const int nit, const int nx,
                        const int ny, const int BLOCKS,
                        const int THREADS_PER_BLOCK) {

  for (int it = 0; it < nit; it++) {

    copy_pn<<<BLOCKS, THREADS_PER_BLOCK>>>(p, pn, nx, ny);

    cudaDeviceSynchronize();

    calc_p<<<BLOCKS, THREADS_PER_BLOCK>>>(p, pn, b, dx, dy, nx, ny);

    cudaDeviceSynchronize();

    pad_p(p, nx, ny);
  }
}

// ------------------- Functions for computing u and v -------------------
__global__ void copy_un_vn(const matrix *u, const matrix *v, matrix *un,
                           matrix *vn, const int nx, const int ny) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int total = nx * ny;

  if (i >= total)
    return;

  un->elems[i] = u->elems[i];
  vn->elems[i] = v->elems[i];
}

__global__ void calc_u_v(matrix *u, matrix *v, const matrix *un,
                         const matrix *vn, const matrix *p, const double dx,
                         const double dy, const double dt, const double rho,
                         const double nu, const int nx, const int ny) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int total = nx * ny;

  if (i >= total)
    return;

  int row = i / nx;
  int col = i % nx;

  // Skip boundary points
  if (row <= 0 || row >= ny - 1 || col <= 0 || col >= nx - 1)
    return;

  // Compute u[j][i] and v[j][i]
  u->elems[i] = un->elems[i] -
                un->elems[i] * dt / dx * (un->elems[i] - un->elems[i - 1]) -
                un->elems[i] * dt / dy * (un->elems[i] - un->elems[i - nx]) -
                dt / (2 * rho * dx) * (p->elems[i + 1] - p->elems[i - 1]) +
                nu * dt / (dx * dx) *
                    (un->elems[i + 1] - 2 * un->elems[i] + un->elems[i - 1]) +
                nu * dt / (dy * dy) *
                    (un->elems[i + nx] - 2 * un->elems[i] + un->elems[i - nx]);
  v->elems[i] = vn->elems[i] -
                vn->elems[i] * dt / dx * (vn->elems[i] - vn->elems[i - 1]) -
                vn->elems[i] * dt / dy * (vn->elems[i] - vn->elems[i - nx]) -
                dt / (2 * rho * dx) * (p->elems[i + nx] - p->elems[i - nx]) +
                nu * dt / (dx * dx) *
                    (vn->elems[i + 1] - 2 * vn->elems[i] + vn->elems[i - 1]) +
                nu * dt / (dy * dy) *
                    (vn->elems[i + nx] - 2 * vn->elems[i] + vn->elems[i - nx]);
}

__host__ void pad_u_v(matrix *u, matrix *v, const int nx, const int ny) {
  for (int y = 0; y < ny; ++y) {
    int rowStart = y * nx;
    u->elems[rowStart] = 0;
    u->elems[rowStart + nx - 1] = 0;
    v->elems[rowStart] = 0;
    v->elems[rowStart + nx - 1] = 0;
  }

  for (int x = 0; x < nx; x++) {
    // pad top and bottom
    u->elems[x] = 0;
    u->elems[nx * ny - nx + x] = 1;
    v->elems[x] = 0;
    v->elems[nx * ny - nx + x] = 0;
  }
}

__host__ void compute_u_v(matrix *u, matrix *v, matrix *un, matrix *vn,
                          const matrix *p, const double dt, const double dx,
                          const double dy, const double rho, const double nu,
                          const int nx, const int ny, const int BLOCKS,
                          const int THREADS_PER_BLOCK) {
  copy_un_vn<<<BLOCKS, THREADS_PER_BLOCK>>>(u, v, un, vn, nx, ny);

  cudaDeviceSynchronize();

  calc_u_v<<<BLOCKS, THREADS_PER_BLOCK>>>(u, v, un, vn, p, dx, dy, dt, rho, nu,
                                          nx, ny);

  cudaDeviceSynchronize();

  pad_u_v(u, v, nx, ny); // on host for now
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

    compute_p(pn, p, b, dx, dy, nit, nx, ny, BLOCKS, THREADS_PER_BLOCK);

    cudaDeviceSynchronize();

    compute_u_v(u, v, un, vn, p, dt, dx, dy, rho, nu, nx, ny, BLOCKS,
                THREADS_PER_BLOCK);

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

  double time = std::chrono::duration<double>(toc - tic).count();
  printf("time = %lf\n", time);
}