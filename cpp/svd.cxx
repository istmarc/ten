#include <ten/tensor>
#include <ten/io>
#include <ten/linalg>
#include <ten/random>

int main() {
  using namespace ten;
  auto x = rand_norm<float>({4, 4});
  std::cout << x << std::endl;
  auto [u, sigma, vt] = linalg::svd(x);
  std::cout << u << std::endl;
  std::cout << dense(sigma) << std::endl;
  std::cout << vt << std::endl;
}
