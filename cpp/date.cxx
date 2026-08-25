#include <ten/datetime.hxx>
#include <iostream>

int main() {
  auto d = ten::date(2026, ten::month::aug, 23);
  std::cout << d.year << std::endl;
  std::cout << d.month << std::endl;
  std::cout << d.day << std::endl;
}
