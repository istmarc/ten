#include <ten/tensor>
#include <ten/io>
#include <ten/random>

int main() {
  ten::set_seed(1234);

  {
    std::cout << "min\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::scalar<float> y = ten::min(x);
    std::cout << y << std::endl;
  }

  {
    std::cout << "max\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::scalar<float> y = ten::max(x);
    std::cout << y << std::endl;
  }
  {
    std::cout << "mean\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::scalar<float> y = ten::mean(x);
    std::cout << y << std::endl;
  }
  {
    std::cout << "sum\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::scalar<float> y = ten::sum(x);
    std::cout << y << std::endl;
  }

  {
    std::cout << "cum_sum\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::tensor<float> y = ten::cum_sum(x);
    std::cout << y << std::endl;
  }

  {
    std::cout << "prod\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::scalar<float> y = ten::prod(x);
    std::cout << y << std::endl;
  }

  {
    std::cout << "abs\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::tensor<float> y = ten::abs(x);
    std::cout << y << std::endl;
  }



  {
    std::cout << "sqrt\n";
    auto x = ten::range<float>({4, 4});
    std::cout << x << std::endl;
    ten::tensor<float> y = ten::sqrt(x);
    std::cout << y << std::endl;
  }

  {
    std::cout << "sqr\n";
    auto x = ten::range<float>({4, 4});
    std::cout << x << std::endl;
    ten::tensor<float> y = ten::sqr(x);
    std::cout << y << std::endl;
  }

  {
    std::cout << "sin\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::tensor<float> y = ten::sin(x);
    std::cout << y << std::endl;
  }


  {
    std::cout << "sinh\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::tensor<float> y = ten::sinh(x);
    std::cout << y << std::endl;
  }


  {
    std::cout << "asin\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::tensor<float> y = ten::asin(x);
    std::cout << y << std::endl;
  }

  {
    std::cout << "cos\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::tensor<float> y = ten::cos(x);
    std::cout << y << std::endl;
  }


  {
    std::cout << "cosh\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::tensor<float> y = ten::cosh(x);
    std::cout << y << std::endl;
  }

  {
    std::cout << "acos\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::tensor<float> y = ten::acos(x);
    std::cout << y << std::endl;
  }


  {
    std::cout << "tan\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::tensor<float> y = ten::tan(x);
    std::cout << y << std::endl;
  }


  {
    std::cout << "tanh\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::tensor<float> y = ten::tanh(x);
    std::cout << y << std::endl;
  }

  {
    std::cout << "atan\n";
    auto x = ten::rand_norm<float>({4, 4});
    std::cout << x << std::endl;
    ten::tensor<float> y = ten::atan(x);
    std::cout << y << std::endl;
  }

}
