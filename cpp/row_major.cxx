#include <ten/tensor>
#include <ten/io>

int main() {
  auto x = ten::row_major<float>({3, 3});
  std::cout << x << std::endl;
}
