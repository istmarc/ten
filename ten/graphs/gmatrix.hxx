#ifndef TEN_GRAPHS_GMATRIX_HXX
#define TEN_GRAPHS_GMATRIX_HXX

#include <ten/graphs/types.hxx>
#include <ten/io>
#include <ten/tensor>

namespace ten {
namespace graph {

template <class T = float> class gmatrix {
private:
  graph_type _gtype;
  std::size_t _vertices;
  ten::tensor<T> _m = ten::tensor<T>::make_default();

public:
  gmatrix(const std::size_t vertices,
          const graph_type gtype = graph_type::undirected)
      : _gtype(gtype), _vertices(vertices),
        _m(ten::zeros<T>({vertices, vertices})) {}

  void add_edge(std::size_t src, std::size_t dest, const T value = T(1)) {
    _m(src, dest) = value;
    if (_gtype == graph_type::undirected) {
      _m(dest, src) = value;
    }
  }

  bool has_edge(std::size_t src, std::size_t dest) const {
    return _m(src, dest) != T(0);
  }

  template <class _T>
  friend std::ostream &operator<<(std::ostream &, const gmatrix<_T> &);
};

template <class T>
std::ostream &operator<<(std::ostream &os, const gmatrix<T> &g) {
  os << g._m;
  return os;
}

} // namespace graph
} // namespace ten

#endif
