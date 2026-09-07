#include <ten/tensor>
#include <ten/io>

int main() {

   {
      auto x = ten::range({3, 3});
      ten::tensor y = ten::reshape(x, {9});
      std::cout << y << std::endl;
      y[0] = .5;
      std::cout << x << std::endl;
   }

   /*{
      auto x = ten::range<float>({3, 3});
      std::cout << x << std::endl;
      auto y = ten::transpose(x);
      std::cout << y.eval() << std::endl;
   }*/

   {
      auto x = ten::range({3, 3});
      ten::tensor y = ten::flatten(x);
      std::cout << y << std::endl;
      y[0] = .5;
      std::cout << x << std::endl;
   }

}
