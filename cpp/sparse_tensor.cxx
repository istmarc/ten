#include <ten/tensor>
#include <ten/io>

int main() {
   {
      ten::sparse_tensor<float> x({3, 3});
      x.set({0, 0}, 1.);
      x.set({1, 1}, 1.);
      std::cout << x(0,0) << std::endl;
      std::cout << x(1,1) << std::endl;
   }

   {
      //ten::coo_matrix<float> x({3, 3});//, {{0,0}, {1, 1}}, {1.0f, 2.0f});
      std::vector<std::tuple<std::vector<std::size_t>, float>> values = {{{0, 0}, 1.}, {{1, 1}, 1.}};
      auto x = ten::sparse_tensor<float>::coo({3, 3}, values);
      std::cout << x(0, 0) << std::endl;
      std::cout << x(1, 1) << std::endl;
   }
}
