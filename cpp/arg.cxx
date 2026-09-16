#include <ten/tensor>
#include <ten/random>
#include <ten/io>

int main() {
  ten::set_seed(123);
  auto x = ten::rand_norm<float>({10});

  std::cout << x << std::endl;
  std::cout << ten::argmin(x).eval() << std::endl;
  std::cout << ten::argmax(x).eval() << std::endl;
}
