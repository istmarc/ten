#include <ten/tensor>
#include <ten/io>
#include <ten/random>

int main() {

  auto x = ten::range<float>({3, 3});
  auto y = ten::range<float>({3, 3});

  std::cout << "Input tensors\n";
  std::cout << x << std::endl;
  std::cout << y << std::endl;

  {
    std::cout << "Add two tensors.\n";
    ten::tensor<float> z = x + y;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Add a value to tensors.\n";
    ten::tensor<float> z = 1.0f + x;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Add a value to tensors.\n";
    ten::tensor<float> z = x + 1.0f;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Add a scalar to tensors.\n";
    ten::tensor<float> z = ten::scalar<float>(1.0f) + x;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Add a scalr to tensors.\n";
    ten::tensor<float> z = x + ten::scalar<float>(1.0f);
    std::cout << z << std::endl;
  }

  {
    std::cout << "Sub two tensor.\n";
    auto x = ten::range<float>({3, 3}, 1.);
    auto y = ten::range<float>({3, 3});
    ten::tensor<float> z = x - y;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Sub a tensor and a scalar.\n";
    auto x = ten::range<float>({3, 3}, 1.);
    ten::tensor<float> z = 1.0f - x;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Sub a tensor and a scalar.\n";
    auto x = ten::range<float>({3, 3}, 1.);
    ten::tensor<float> z = x - 1.0f;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Sub a tensor and a scalar.\n";
    auto x = ten::range<float>({3, 3}, 1.);
    ten::tensor<float> z = ten::scalar<float>(1.0f) - x;
    std::cout << z << std::endl;
  }
  {
    std::cout << "Sub a tensor and a scalar.\n";
    auto x = ten::range<float>({3, 3}, 1.);
    ten::tensor<float> z = x - ten::scalar<float>(1.0f);
    std::cout << z << std::endl;
  }

  {
    std::cout << "Elementwise vector multiplication.\n";
    auto x = ten::range<float>({10});
    auto y = ten::range<float>({10});
    ten::tensor<float> z = x * y;
    std::cout << z.rank() << std::endl;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Elementwise tensor multiplication.\n";
    auto x = ten::range<float>({2, 3, 4});
    auto y = ten::range<float>({2, 3, 4});
    ten::tensor<float> z = x * y;
    //std::cout << ten::print_shape(std::cout, z.shape()) << std::endl;
    std::cout << z[0] << "..." << z[z.size()-1] << std::endl;
  }

  {
    std::cout << "Matrix vector mul.\n";
    auto x = ten::range<float>({3, 3});
    auto v = ten::range<float>({3});
    ten::tensor<float> z = x * v;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Matrix multiplication.\n";
    auto x = ten::range<float>({3, 3});
    auto y = ten::range<float>({3, 3});
    ten::tensor<float> z = x * y;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Multiply by a scalar left.\n";
    ten::tensor<float> z = 2.0f * x;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Multiply by a scalar right.\n";
    ten::tensor<float> z = x * 2.0f;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Multiply by a scalar left.\n";
    ten::tensor<float> z = ten::scalar<float>(2.0f) * x;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Multiply by a scalar right.\n";
    ten::tensor<float> z = x * ten::scalar<float>(2.0f);
    std::cout << z << std::endl;
  }

  {
    std::cout << "Divide two tensors.\n";
    auto x = ten::range<float>({3, 3}, 1.);
    auto y = ten::range<float>({3, 3}, 1.);
    ten::tensor<float> z = x / y;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Divide by a scalar left.\n";
    auto x = ten::range<float>({3, 3}, 1.);
    ten::tensor<float> z = 2.0f / x;
    std::cout << z << std::endl;
  }

  {
    std::cout << "Divide by a scalar right.\n";
    auto x = ten::range<float>({3, 3}, 1.);
    ten::tensor<float> z = x / 2.0f;
    std::cout << z << std::endl;
  }
  {
    std::cout << "Divide by a scalar left.\n";
    auto x = ten::range<float>({3, 3}, 1.);
    ten::tensor<float> z = ten::scalar<float>(2.0f) / x;
    std::cout << z << std::endl;
  }
  {
    std::cout << "Divide by a scalar right.\n";
    auto x = ten::range<float>({3, 3}, 1.);
    ten::tensor<float> z = x / ten::scalar<float>(2.0f);
    std::cout << z << std::endl;
  }

}
