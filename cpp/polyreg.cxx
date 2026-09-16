#include <ten/tensor>
#include <ten/ml>
#include <ten/io>

int main() {
  using namespace ten;
  auto x = linear({10}, 0.f, 4.0f * 3.14f);
  tensor y = sin(x);

  ml::polyreg model(7);
  model.fit(x, y);
  auto beta = model.coef();

  auto yhat = model.fitted();

  ten::io::save_mtx(x, "polyreg_x.mtx");
  ten::io::save_mtx(y, "polyreg_y.mtx");
  ten::io::save_mtx(yhat, "polyreg_yhat.mtx");
}
