#include <ten/tensor>
#include <ten/io>

int main() {
   {
      ten::matrix<float> x({2, 2}, {1.0f, 0.0f, 0.0f, 1.0f});
      auto y = ten::symmetric(x);
      std::cout << std::boolalpha << y.is_symmetric() << std::endl;
      std::cout << y << std::endl;
   }

   {
      ten::matrix<std::complex<float>> x({2, 2}, {1.0f, 1.0f-1.0i, 1.0f + 1.0i, 2.0f});
      auto y = ten::hermitian(x);
      std::cout << std::boolalpha << y.is_hermitian() << std::endl;
      std::cout << y << std::endl;
   }

   {
      ten::matrix<float> x = ten::range<ten::matrix<float>>({2, 3}, 1.0f);
      ten::matrix<float> a = 1.0f + x;
      ten::matrix<float> b = 1.0f - x;
      ten::matrix<float> c = 2.0f * x;
      ten::matrix<float> d = 1.0f / x;
   }
}
