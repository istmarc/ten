#include <ten/tensor>
#include <ten/io>
#include <ten/random>
#include <ten/linalg>

int main() {
  using namespace ten;
  set_seed(123);
  constexpr std::size_t n = 10;
  auto A = rand_norm<float>({n, n});
  auto b = rand_norm<float>({n});

  auto xqr = linalg::solve(A, b);
  std::cout << "QR\n";
  std::cout << xqr << std::endl;

  auto xlu = linalg::solve(A, b, linalg::ls_method::lu);
  std::cout << "LU\n";
  std::cout << xlu << std::endl;

  auto xsvd = linalg::solve(A, b, linalg::ls_method::svd);
  std::cout << "SVD\n";
  std::cout << xsvd << std::endl;
}
