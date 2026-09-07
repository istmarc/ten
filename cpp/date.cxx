#include <ten/tensor>
#include <ten/io>

int main() {
  auto d = ten::date(2026, ten::month::aug, 23);
  std::cout << d.year << std::endl;
  std::cout << d.month << std::endl;
  std::cout << d.day << std::endl;

  std::cout << d << std::endl;

  ten::tensor<ten::date> x({3}, {ten::date(2025, ten::month::aug, 23),
      ten::date(2026, ten::month::jan, 2), ten::date(2020, ten::month::mar, 10)});
  std::cout << "x size = " << x.size() << std::endl;
  std::cout << x << std::endl;

}
