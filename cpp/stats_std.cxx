#include <ten/tensor>
#include <ten/io>

int main() {
  auto x = ten::range<double>({2, 3});
  std::cout << x << std::endl;

  ten::scalar<double> s = ten::std(x);
  std::cout << s << std::endl;

  ten::scalar<double> s_biased = ten::std(x, true);
  std::cout << s_biased << std::endl;
}
