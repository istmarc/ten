#include <ten/tensor>
#include <ten/io>

int main() {
   auto x = ten::range<ten::matrix<float>>({2, 3});
   std::cout << x.shape() << std::endl;
   std::cout << x.strides() << std::endl;
}
