#include <ten/tensor>
#include <ten/io>
#include <ten/linalg>

int main() {
  using namespace ten;
  auto x = range<float>({3, 3});
  std::cout << x << std::endl;
  auto [q, r] = linalg::qr(x);
  std::cout << q << std::endl;
  std::cout << r << std::endl;
}
