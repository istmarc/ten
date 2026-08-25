#include <ten/tensor>
#include <ten/io>

int main() {
  using namespace ten;
  auto x = range<float>({3, 3});
  auto y = sqrt(x);
  tensor<float> z = y;
  std::cout << x << std::endl;
  std::cout << z << std::endl;

  tensor<float> t = sqrt(x);
  std::cout << t << std::endl;

}
