#include <ten/tensor>
#include <ten/io>
#include <ten/random>

int main() {
  using namespace ten;
  auto x = range<float>({2, 3});
  std::cout << x << std::endl;

  tensor<float> m0 = mean(x, 0);
  std::cout << m0 << std::endl;

  tensor<float> m1 = mean(x, 1);
  std::cout << m1 << std::endl;

  tensor<float> s0 = sum(x, 0);
  std::cout << s0 << std::endl;

  ten::tensor<float> s1 = sum(x, 1);
  std::cout << s1 << std::endl;
}
