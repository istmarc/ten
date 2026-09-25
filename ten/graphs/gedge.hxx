#ifndef TEN_GRAPHS_GEDGE_HXX
#define TEN_GRAPHS_GEDGE_HXX

#include <cstdint>
#include <ten/graphs/types.hxx>

namespace ten::graph {

template <typename T = std::size_t> class gedge {
private:
  graph_type _type;
  std::vector<std::pair<T, T>> _list;

public:
  gedge(graph_type type = graph_type::undirected) : _type(type) {}

  bool empty() const { return _list.empty(); }

  void add_edge(const T &src, const T &dest) {
    _list.push_back(std::make_pair(src, dest));
    if (_type == graph_type::undirected) {
      _list.push_back(std::make_pair(dest, src));
    }
  }
};

} // namespace ten::graph

#endif
