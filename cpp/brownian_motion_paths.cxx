#include <ten/process>
#include <ten/io>

int main() {
  ten::set_seed(1234);

  auto [t, W] = ten::brownian_motion(1000, 500);
  ten::io::save_mtx(t, "brownian_motion_paths_t.mtx");
  ten::io::save_mtx(W, "brownian_motion_paths.mtx");
}
