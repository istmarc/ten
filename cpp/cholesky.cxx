#include <ten/tensor>
#include <ten/io>
#include <ten/linalg>

int main() {
  using namespace ten;
  auto x = range<float>({4, 4});
  std::cout << x << std::endl;
  auto [q, r] = linalg::cholesky(x);
  std::cout << q << std::endl;
  std::cout << r << std::endl;
}
