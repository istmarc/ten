#include <ten/process>
#include <ten/io>

int main() {
  ten::set_seed(123);

  std::size_t n = 1000;
  auto s = ten::random_walk(n);

  auto [t, w] = ten::brownian_motion(s, n);
  ten::io::save_mtx(t, "brownian_motion_1000_t.mtx");
  ten::io::save_mtx(w, "brownian_motion_1000.mtx");

  auto [t500, w500] = ten::brownian_motion(s, 500);
  ten::io::save_mtx(t500, "brownian_motion_500_t.mtx");
  ten::io::save_mtx(w500, "brownian_motion_500.mtx");

  auto [t250, w250] = ten::brownian_motion(s, 250);
  ten::io::save_mtx(t250, "brownian_motion_250_t.mtx");
  ten::io::save_mtx(w250, "brownian_motion_250.mtx");
}
