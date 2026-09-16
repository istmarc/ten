#include <ten/tensor>
#include <ten/io>

int main() {
  using namespace ten;
  auto x = range<float>({3, 3});
  std::cout << x << std::endl;
  mdseq s = mdseq(seq(0), seq(0, 3));
  std::cout << s._start.size() << std::endl;
  std::cout << s._end.size() << std::endl;
  auto slice = x[s];
  slice = .5;
  std::cout << x << std::endl;
}
