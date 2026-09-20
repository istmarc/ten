#ifndef TEN_IO_SSHAPE_HXX
#define TEN_IO_SSHAPE_HXX

#include <iostream>

#include <ten/sshape.hxx>
#include <ten/types.hxx>

namespace ten {

namespace details {
template <std::size_t N, std::size_t... Ts>
void print_shape(std::ostream &os, const sshape<Ts...> &s) {
  using shape_type = sshape<Ts...>;

  os << shape_type::template size<N>();

  if constexpr (N + 1 < shape_type::rank()) {
    os << "x";
    print_shape<N + 1, Ts...>(os, s);
  }
}
} // namespace details

template <std::size_t... Ts>
std::ostream &operator<<(std::ostream &os, const sshape<Ts...> &s) {
  details::print_shape<0, Ts...>(os, s);
  return os;
}

namespace details {
template <std::size_t N, class SShape, storage_order Order>
void print_stride(std::ostream &os, const sstride<SShape, Order> &strides) {
  using stride_type = sstride<SShape, Order>;

  os << stride_type::template size<N>();

  if constexpr (N + 1 < SShape::rank()) {
    os << "x";
    print_stride<N + 1, SShape, Order>(os, strides);
  }
}
} // namespace details

template <class SShape, storage_order Order>
std::ostream &operator<<(std::ostream &os,
                         const sstride<SShape, Order> &strides) {
  details::print_stride<0, SShape, Order>(os, strides);
  return os;
}

} // namespace ten

#endif
