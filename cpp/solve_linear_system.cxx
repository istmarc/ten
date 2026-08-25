#include <ten/tensor>
#include <ten/linalg>
#include <ten/io>

int main() {
  using namespace ten;

  tensor<double> b({3}, {4., 9., 2.});
  std::cout << b << std::endl;

  tensor<double> A({3, 3}, {3., 1., 3., 4., 3., 5., 5., 1., 9.});
  std::cout << A << std::endl;

  auto x = linalg::solve(A, b);
  std::cout << x << std::endl;
}
