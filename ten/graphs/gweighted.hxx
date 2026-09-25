#ifndef TEN_GRAPHS_GWEIGHTED_HXX
#define TEN_GRAPHS_GWEIGHTED_HXX

#include <limits>
#include <ten/tensor>

#include <map>
#include <ten/graphs/types.hxx>
#include <type_traits>

namespace ten {
namespace graph {

/// Graph as a weighted ajanceny list
template <class T = std::size_t, class W = float> class gweighted {
private:
  ten::graph::graph_type _type;
  std::map<T, std::vector<std::pair<T, W>>> _graph;

public:
  gweighted(
      ten::graph::graph_type graph_type = ten::graph::graph_type::undirected)
      : _type(graph_type) {}

  bool empty() { return _graph.empty(); }

  void add_vertex(const T &vertex) {
    _graph[vertex] = std::vector<std::pair<T, W>>();
  }

  void add_edge(const T &src, const T &dest, W weight) {
    _graph[src].push_back(std::make_pair(dest, weight));
    if (_type == graph_type::undirected) {
      _graph[dest].push_back(std::make_pair(src, weight));
    }
  }

  bool has_edge(const T &src, const T &dest) {
    auto it = _graph[src].begin();
    while (it < _graph[src].end()) {
      auto [d, w] = *it;
      if (d == dest)
        return true;
      it++;
    }
    return false;
  }

  // Get the weight of the edge src->dest
  // Return +inf or max if there's no edge from src to dest
  W weight(const T &src, const T &dest) {
    auto it = _graph[src].begin();
    while (it < _graph[src].end()) {
      auto [d, w] = *it;
      if (d == dest)
        return w;
      it++;
    }
    static_assert(std::is_floating_point_v<W> || std::is_integral_v<W>,
                  "W must be flaoting point or integral.");
    if constexpr (std::is_floating_point_v<W>)
      return std::numeric_limits<W>::infinity();
    if constexpr (std::is_integral_v<W>)
      return std::numeric_limits<W>::max();
  }

  // Get the adjacency matrix
  auto to_matrix() -> ten::tensor<W> {
    std::size_t n = _graph.size();
    ten::tensor<W> m = ten::zeros<W>({n, n});
    std::map<T, std::size_t> map;
    std::size_t i = 0;
    for (auto it = _graph.begin(); it != _graph.end(); it++) {
      map[it->first] = i;
      i++;
    }
    for (auto const &[src, value] : _graph) {
      for (auto const &[dest, weight] : value) {
        m(map[src], map[dest]) = weight;
      }
    }
    return m;
  }
};

} // namespace graph
} // namespace ten

#endif
