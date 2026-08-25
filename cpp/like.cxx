#include <ten/tensor>
#include <ten/io>

int main() {
   auto x = ten::range<float>({3, 3});
   auto y = ten::like(x);
   std::cout << y << std::endl;
}
