#include <ten/tensor>
#include <ten/io>

int main() {
  using namespace ten;
  auto x = range<float>({4, 4});
  std::cout << x << std::endl;

  auto slice_0 = x(seq(0,1), seq(0,4));
  slice_0 = .5;
  std::cout << x << std::endl;

  auto slice_2 = x(seq(2,3), seq(1,4));
  slice_2 = .3;

  std::cout << x << std::endl;

  auto slice_3 = x(seq(3), seq(1,2));
  slice_3 = .9;
  std::cout << x << std::endl;
}
