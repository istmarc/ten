#include "ten/graphs/glist.hxx"
#include <ten/tensor>
#include <ten/graph>

int main() {
   {
      ten::graph::gmatrix<> g(5);
      g.add_edge(0, 1);
      g.add_edge(1, 0);
      g.add_edge(0, 4);
      g.add_edge(4, 0);
      g.add_edge(1, 4);
      g.add_edge(4, 1);
      g.add_edge(1, 2);
      g.add_edge(2, 1);
      g.add_edge(2, 3);
      g.add_edge(3, 2);
      g.add_edge(4, 3);
      g.add_edge(3, 4);
      std::cout << g << std::endl;
   }

   {
      ten::graph::gmatrix<double> g(5);
      g.add_edge(0, 1);
      g.add_edge(1, 0);
      g.add_edge(0, 4);
      g.add_edge(4, 0);
      g.add_edge(1, 4);
      g.add_edge(4, 1);
      g.add_edge(1, 2);
      g.add_edge(2, 1);
      g.add_edge(2, 3);
      g.add_edge(3, 2);
      g.add_edge(4, 3);
      g.add_edge(3, 4);
      std::cout << g << std::endl;
   }

  {
      ten::graph::glist<std::size_t> g;
      g.add_edge(0, 1);
      g.add_edge(1, 0);
      g.add_edge(0, 4);
      g.add_edge(4, 0);
      g.add_edge(1, 4);
      g.add_edge(4, 1);
      g.add_edge(1, 2);
      g.add_edge(2, 1);
      g.add_edge(2, 3);
      g.add_edge(3, 2);
      g.add_edge(4, 3);
      g.add_edge(3, 4);
      std::cout << g.to_matrix() << std::endl;

      std::cout << "DFS\n";
      g.dfs(0, [](std::size_t x){ std::cout << "Node " << x << std::endl;});

      std::cout << "BFS\n";
      g.bfs(0, [](std::size_t x){std::cout << "Node " << x << std::endl;});
  }

}
