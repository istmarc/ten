#include <ten/graph>
#include <iostream>

void print_node(const std::size_t n) {
  std::cout << n << std::endl;
}

int main() {
  using namespace ten;
  graph::glist g;
  for (std::size_t i = 0; i < 6; i++)
    g.add_vertex(i);
  g.add_edge(0, 1);
  g.add_edge(0, 2);
  g.add_edge(0, 4);
  g.add_edge(1, 3);
  g.add_edge(1, 4);
  g.add_edge(3, 4);
  g.add_edge(2, 5);
  auto m = g.matrix();
  std::cout << m << std::endl;

  std::cout << "DFS\n";
  g.dfs(0, print_node);

}
