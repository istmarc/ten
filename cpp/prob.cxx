#include "ten/ios/mtx.hxx"
#include <iostream>

#include <ten/tensor>
#include <ten/io>

int main() {
   ten::tensor<float> x({3, 3});

   for (size_t i = 0; i < 3*3; i++) {
      x[i] = i + 1;
   }

   std::cout << "tensor" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for(size_t j = 0; j < 3; j++) {
         std::cout << x(i, j) << " ";
      }
      std::cout << std::endl;
   }

   ten::diagonal<float> y({4, 4});
   for (size_t i = 0; i < 4; i++) {
      y[i] = i + 1;
   }

   std::cout << "diagonal"<< std::endl;
   for (size_t i = 0; i < 4; i++) {
      std::cout << y(i, i) << " ";
   }
   std::cout << std::endl;

   auto a = ten::sqrt(x);
   auto b = a.eval();

    std::cout << "sqrt tensor" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for(size_t j = 0; j < 3; j++) {
         std::cout << b(i, j) << " ";
      }
      std::cout << std::endl;
   }

   auto d = ten::sqrt(ten::sqrt(ten::sqrt(x)));
   auto e = d.eval();
    std::cout << "sqrt sqrt sqrt tensor" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for(size_t j = 0; j < 3; j++) {
         std::cout << e(i, j) << " ";
      }
      std::cout << std::endl;
   }

   auto f = ten::sqr(x);
   auto g = f.eval();
  std::cout << "sqr tensor" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for(size_t j = 0; j < 3; j++) {
         std::cout << g(i, j) << " ";
      }
      std::cout << std::endl;
   }

   ten::normal<float> dist;
   auto h = dist.sample({3, 3});
  std::cout << "random tensor" << std::endl;
   for (size_t i = 0; i < 3; i++) {
      for(size_t j = 0; j < 3; j++) {
         std::cout << h(i, j) << " ";
      }
      std::cout << std::endl;
   }

   std::cout << h << std::endl;

   ten::io::save_mtx(h, "random_matrix.mtx");

   return 0;
}

