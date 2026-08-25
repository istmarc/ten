#include <ten/tensor>
#include <ten/random>
#include <ten/io>

int main(){
   using namespace ten;
   set_seed(1234);

   auto a = rand_norm<tensor<float>>({3, 3});
   std::cout << a << std::endl;

   set_seed(1234);
   auto x = rand_norm<float>({3, 3});
   std::cout << x << std::endl;
}
