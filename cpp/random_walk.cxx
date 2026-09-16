#include "ten/processes/random_walk.hxx"
#include "ten/ios/mtx.hxx"
#include <ten/tensor>
#include <ten/process>
#include <ten/io>

int main() {
  ten::set_seed(123);

  {
    auto s = ten::random_walk(1000);
    ten::io::save_mtx(s, "random_walk.mtx");
    std::cout << s << std::endl;
  }

  {
    ten::normal dist;
    ten::random_walk_options options{.prob = 0.};
    auto s = ten::random_walk(1000, dist, options);
    ten::io::save_mtx(s, "random_walk_norm.mtx");
    std::cout << s << std::endl;
  }


  {
    std::cout << "Continuous random walk\n";
    std::size_t n = 1000;
    auto s = ten::random_walk(n);
    auto [t1, x1] = ten::continuous_random_walk(s, n, 1000);
    auto [t2, x2] = ten::continuous_random_walk(s, n, 750);
    auto [t3, x3] = ten::continuous_random_walk(s, n, 500);
    auto [t4, x4] = ten::continuous_random_walk(s, n, 250);
    ten::io::save_mtx(s, "crw.mtx");
    ten::io::save_mtx(t1, "crw_1000_t.mtx");
    ten::io::save_mtx(x1, "crw_1000.mtx");
    ten::io::save_mtx(t2, "crw_750_t.mtx");
    ten::io::save_mtx(x2, "crw_750.mtx");
    ten::io::save_mtx(t3, "crw_500_t.mtx");
    ten::io::save_mtx(x3, "crw_500.mtx");
    ten::io::save_mtx(t4, "crw_250_t.mtx");
    ten::io::save_mtx(x4, "crw_250.mtx");
  }
}
