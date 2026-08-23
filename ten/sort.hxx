#ifndef TEN_SORT_HXX
#define TEN_SORT_HXX

#include <type_traits>

#include <ten/tensor.hxx>
#include <ten/types.hxx>

namespace ten {

enum class sort_method { quick = 1, merge = 2 };

/// Sort inplace
template <Tensor T>
void sort_inplace(T &x, sort_method method = sort_method::quick) {
  if (!x.is_dense()) {
    std::cerr << "ten::sort_inplace, storage format not supported.\n";
    return;
  }
  std::sort(x.data(), x.data() + x.size());
}

/// Sort a tensor, be default using std::sort (quicksort)
template <Tensor T>
T sort(const T &x, sort_method method = sort_method::quick) {
  if (!x.is_dense()) {
    std::cerr << "ten::sort, storage format not supported.\n";
  }
  // Create a dense tensor
  T y(x.shape(), ten::storage_format::dense, false, x.storage_order());
  // Copy
  for (std::size_t i = 0; i < x.size(); i++) {
    y[i] = x[i];
  }
  // Sort
  std::sort(y.data(), y.data() + y.size());
  return y;
}

} // namespace ten

#endif
