#include <ten/tensor>
#include <ten/io>


int main() {
  constexpr std::size_t m = 3;
  constexpr std::size_t n = 4;
  auto x = ten::range({m, n});
  std::cout << x << std::endl;

  for (size_t idx = 0; idx < n; idx++) {
    auto col = x.col(idx);
    std::cout << "Col " << idx << ": ";
    for (std::size_t i = 0; i < m; i++) {
      std::cout << col[i] << " ";
    }
    std::cout << std::endl;
  }

  for (size_t idx = 0; idx < m; idx++) {
    auto row = x.row(idx);
    std::cout << "Row " << idx << ": ";
    for (std::size_t i = 0; i < n; i++) {
      std::cout << row[i] << " ";
    }
    std::cout << std::endl;
  }

  std::cout << "Print rows and columns\n";

  std::cout << x.row(0) << std::endl;
  std::cout << x.col(0) << std::endl;


}

