#include <ten/tensor>
#include <ten/io>

int main() {
  auto x = ten::range<double>({2, 3});
  std::cout << x << std::endl;

  ten::scalar<double> v = ten::var(x);
  std::cout << v << std::endl;

  ten::scalar<double> v_biased = ten::var(x, true);
  std::cout << v_biased << std::endl;
}
