#include <ten/tensor>
#include <ten/io>

int main() {
   {
      auto a = ten::range<float>({2, 3});
      auto b = a.copy();
      std::cout << a << std::endl;
      std::cout << b << std::endl;
   }

   {
      std::cout << "with copy grad" << std::endl;
      auto x = ten::range<float>({2, 3}, 1.0f, true);
      auto y = x.copy(true);
      std::cout << x << std::endl;
      std::cout << x.grad() << std::endl;
      std::cout << y << std::endl;
      std::cout << y.grad() << std::endl;
   }
}
