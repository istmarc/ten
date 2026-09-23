#include <ten/io>
#include <ten/linalg>
#include <ten/tensor>

ten::tensor<float> f(ten::tensor<float> x, ten::tensor<float> beta) {
  std::size_t m = x.size(0);
  ten::tensor<float> y({m});
  for (std::size_t i = 0; i < m; i++) {
    y[i] = beta[0] * x[i] / (beta[1] + x[i]);
  }
  return y;
}

ten::tensor<float> Jr(ten::tensor<float> &x, ten::tensor<float> &beta) {
  std::size_t m = x.size();
  std::size_t n = beta.size();
  ten::tensor<float> J({m, n});
  for (std::size_t i = 0; i < m; i++) {
    J(i, 0) = -x[i] / (beta[1] + x[i]);
    float c = beta[1] + x[i];
    J(i, 1) = beta[0] * x[i] / (c * c);
  }
  return J;
}

int main() {
  using namespace ten;
  tensor s({7}, {0.038f, 0.194f, 0.425f, 0.626f, 1.253f, 2.500f, 3.740f});
  tensor rate({7},
              {0.050f, 0.127f, 0.094f, 0.2122f, 0.2729f, 0.2665f, 0.3317f});

  auto beta = linalg::nls_newton_gauss(
      f, s, rate, Jr,
      linalg::nls_options<tensor<float>>{
          .n = 2, .beta0 = tensor<float>({2}, {0.9f, 0.2f}), .itermax = 100});

  std::cout << "beta = \n";
  std::cout << beta << std::endl;

  // True beta is {0.362f, 0.556f}
  auto yhat = f(s, beta);
  std::cout << "true y=\n";
  std::cout << rate << std::endl;
  std::cout << "yhat =\n";
  std::cout << yhat << std::endl;

}
