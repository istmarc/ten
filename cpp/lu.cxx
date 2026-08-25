#include <ten/tensor>
#include <ten/io>
#include <ten/linalg>

int main() {
  using namespace ten;
  auto x = range<float>({4, 4});
  std::cout << x << std::endl;
  auto [p, l, u] = linalg::lu(x);
  std::cout << p << std::endl;
  std::cout << l << std::endl;
  std::cout << u << std::endl;
}
