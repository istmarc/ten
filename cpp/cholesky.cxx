#include <ten/tensor>
#include <ten/io>
#include <ten/linalg>

int main() {
  using namespace ten;
  auto x = range<float>({4, 4});
  std::cout << x << std::endl;
  auto [l, u] = linalg::cholesky(x);
  std::cout << l << std::endl;
  std::cout << u << std::endl;
}
