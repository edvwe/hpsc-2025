#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>
#include <chrono>

int main()
{

  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i = 0; i < n; i++)
  {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range, 0);
  for (int i = 0; i < n; i++)
    bucket[key[i]]++;

  std::vector<int> offset(range, 0);
  for (int i = 1; i < range; i++)
    offset[i] = offset[i - 1] + bucket[i - 1];
  auto tic = std::chrono::steady_clock::now();

  for (int i = 0; i < range; i++)
  {
    const int start_pos = offset[i];
    const int end_pos = start_pos + bucket[i];

#pragma omp parallel for
    for (int j = start_pos; j < end_pos; j++)
    {
      key[j] = i;
    }
  }
  auto toc = std::chrono::steady_clock::now();

  for (int i = 0; i < n; i++)
  {
    printf("%d ", key[i]);
  }
  printf("\n");

  double time = std::chrono::duration<double>(toc - tic).count();
  printf("time = %lf\n", time);
}
