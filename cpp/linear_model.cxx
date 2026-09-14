#include <ten/tensor>
#include <ten/ml>
#include <ten/io>

int main() {
  using namespace ten;

  auto x = range({4}, 1.0f);

  auto y = tensor({4}, {6.0f, 5.0f, 7.0f, 10.0f});

  ml::linear_model model;
  model.fit(x, y);
  auto beta = model.coeff();
  std::cout << "Coefficients:\n";
  std::cout << beta << std::endl;
  auto yhat = model.fitted();
  std::cout << "Fitted values:\n";
  std::cout << yhat << std::endl;
}
