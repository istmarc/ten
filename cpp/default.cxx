#include <ten/tensor>
#include <ten/io>

int main() {
  constexpr std::size_t n = 3;
  ten::tensor x({n, n});
  for (std::size_t i = 0; i < n*n; i++) {
    x[i] = i + 1;
  }
  std::cout << x << std::endl;

  auto y = ten::range({n});
  std::cout << y << std::endl;
}
