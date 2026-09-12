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
}
