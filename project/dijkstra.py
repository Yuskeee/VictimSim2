from dijkstar import Graph, find_path
from dijkstar.algorithm import (
    single_source_shortest_paths as calc_shortest_paths,
    extract_shortest_path_from_predecessor_list as extract_shortest_path,
    find_path
)

class Dijkstra:
    def __init__(self, start=(0,0)):
        self._graph = Graph()
        self.start = start
        self._graph.add_node(start)
        self._directions = {}
    
    def add_edge(self, node1, node2, cost1_2, cost2_1):
        self._graph.add_edge(node1, node2, cost1_2)
        self._graph.add_edge(node2, node1, cost2_1)
        self._directions[(node1, node2)] = self._get_direction(node1, node2)
        self._directions[(node2, node1)] = self._get_direction(node2, node1)
    
    def check_edge(self, node1, node2):
        try:
            if self._graph.get_edge(node1, node2):
                return True
            return False
        except:
            return False
    
    def _get_direction(self, node1, node2):
        x1, y1 = node1
        x2, y2 = node2
        dx = x2 - x1
        dy = y2 - y1
        return (dx, dy)

    def calc_shortest_path_back(self, node):
        path = find_path(self._graph, node, self.start)
        return path[0], path[3]

    def get_shortest_cost_back(self, node):
        return self.calc_shortest_path_back(node)[1]

    def get_shortest_path_back(self, node):
        return self.calc_shortest_path_back(node)[0]
    
    def calc_backtrack(self, node):
        backtrack = []
        path = self.calc_shortest_path_back(node)
        nodes = path[0]
        cost = path[1]
        print(nodes)
        last = nodes.pop()
        while nodes:
            tmp = nodes.pop()
            backtrack.insert(0, self._directions[(tmp, last)])
            last = tmp
        return backtrack, cost

if __name__ == "__main__":
    dijkstra = Dijkstra()
    dijkstra.add_edge((0,0), (1,0), 1, 1)
    dijkstra.add_edge((0,0), (0,1), 1, 1)
    dijkstra.add_edge((0,0), (1,1), 1.5, 1.5)

    dijkstra.add_edge((1,0), (2,0), 1, 1)
    dijkstra.add_edge((1,0), (1,1), 1, 1)
    dijkstra.add_edge((1,0), (0,1), 1.5, 1.5)

    dijkstra.add_edge((2,0), (3,0), 1, 1)
    dijkstra.add_edge((2,0), (1,1), 1.5, 1.5)

    dijkstra.add_edge((0,1), (1,1), 1, 1)
    dijkstra.add_edge((0,1), (0,2), 1, 1)
    dijkstra.add_edge((0,1), (1,2), 1.5, 1.5)

    dijkstra.add_edge((1,1), (0,2), 1.5, 1.5)
    dijkstra.add_edge((1,1), (1,2), 1, 1)
    dijkstra.add_edge((1,1), (2,2), 1.5, 1.5)

    dijkstra.add_edge((0,2), (1,2), 1, 1)

    dijkstra.add_edge((1,2), (2,2), 1, 1)

    dijkstra.add_edge((2,2), (3,2), 1, 1)

    print(dijkstra._graph)
    dijkstra.add_edge((0,0), (1,0), 1, 1)
    print(dijkstra._graph)

    # print(dijkstra.check_edge((0,0), (3,2)))

    # print(dijkstra.calc_backtrack((3,2)))