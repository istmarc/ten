#include <ten/tensor>
#include <ten/io>

int main() {
  using namespace ten;

  auto a = zeros<float>({3, 3});
  std::cout << a << std::endl;

  auto b = ones<float>({3, 3});
  std::cout << b << std::endl;

  auto c = fill<float>({3, 3}, 9.9);
  std::cout << c << std::endl;

  auto x = range<float>({3, 3}, 1.);
  std::cout << x << std::endl;

  auto y = linear<float>({3,3}, 1., 10.);
  std::cout << y << std::endl;

}
