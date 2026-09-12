#include <ten/tensor>
#include <ten/io>
#include <ten/linalg>


int main() {
  ten::tensor X({4, 2}, {1.f, 1.f, 1.f, 1.f, 1.f, 2.f, 3.f, 4.f});
  ten::tensor y({4}, {6.f, 5.f, 7.f, 10.f});

  std::cout << "X =\n" << X << std::endl;
  std::cout << "y = \n" << y << std::endl;

  auto beta = ten::linalg::lsqr(X, y);
  std::cout << "Solution beta =\n";
  std::cout << beta << std::endl;

  ten::tensor yhat({4});
  for (std::size_t i = 0; i < 4; i++) {
    yhat[i] = beta[0] * X(i,0) + X(i, 1) * beta[1];
  }
  std::cout << "Fitted values yhat =\n" << yhat << std::endl;
}
