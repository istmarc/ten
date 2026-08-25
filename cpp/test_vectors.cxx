#include <iostream>
#include <vector>

auto foo(size_t index, auto...tail) {
   std::vector<size_t> v = {index, static_cast<size_t>(tail)...};
   return v;
}

void print(const std::vector<size_t>& v) {
   std::cout << "vector<size_t>(";
   std::cout << v.size() << ")\n";
   for (size_t i=0; i< v.size(); i++) {
      std::cout << v[i] << " ";
   }
   std::cout << std::endl;
}

int main() {
   auto v = foo(2, 3, 5, 7, 11);
   print(v);
}
