#include <ten/tensor>
#include <ten/io>

int main() {
  using namespace ten;
  {
    auto x = range({2, 3});
    auto y = range({3, 4});
    tensor z = x * y;
    std::cout << "x @ y\n";
    std::cout << x << std::endl;
    std::cout << y << std::endl;
    std::cout << z << std::endl;
  }

  {
    auto x = range({2, 3});
    auto y = range({4, 3});
    tensor z = x * transposed(y);
    std::cout << "x @ t(y)\n";
    std::cout << x << std::endl;
    std::cout << y << std::endl;
    std::cout << z << std::endl;
  }

  {
    auto x = range({3, 2});
    auto y = range({3, 4});
    tensor z = transposed(x) * y;
    std::cout << "t(x) @ y\n";
    std::cout << x << std::endl;
    std::cout << y << std::endl;
    std::cout << z << std::endl;
  }

  {
    auto x = range({3, 2});
    auto y = range({4, 3});
    tensor z = transposed(x) * transposed(y);
    std::cout << "t(x) @ t(y)\n";
    std::cout << x << std::endl;
    std::cout << y << std::endl;
    std::cout << z << std::endl;
  }

  {
    auto x = range({2, 3});
    auto y = range({3});
    tensor z = x * y;
    std::cout << "matvector x @ y\n";
    std::cout << x << std::endl;
    std::cout << y << std::endl;
    std::cout << z << std::endl;
  }

  {
    auto x = range({3, 2});
    auto y = range({3});
    tensor z = transposed(x) * y;
    std::cout << "matvector t(x) @ y\n";
    std::cout << x << std::endl;
    std::cout << y << std::endl;
    std::cout << z << std::endl;
  }

}
