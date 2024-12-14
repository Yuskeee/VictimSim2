from dijkstar import Graph, find_path
from dijkstar.algorithm import (
    single_source_shortest_paths as calc_shortest_paths,
    extract_shortest_path_from_predecessor_list as extract_shortest_path,
    find_path
)

class Dijkstra:
    def __init__(self, start=(0,0)):
        self._graph = Graph()
        # self._shortest_paths = {}
        # self._curr_shortest_path_back = []
        # self._curr_shortest_path_back_cost = float('inf')
        # self._curr_last_node = start
        self.start = start
        self._graph.add_node(start)
        # self._calculate_all_shortest_paths()
        self._directions = {}
        # self._backtrack = list()
    
    def add_edge(self, node1, node2, cost1_2, cost2_1):
        self._graph.add_edge(node1, node2, cost1_2)
        self._graph.add_edge(node2, node1, cost2_1)
        self._directions[(node1, node2)] = self._get_direction(node1, node2)
        self._directions[(node2, node1)] = self._get_direction(node2, node1)

        # self._curr_last_node = node2
        # self._calc_shortest_path_back(self._curr_last_node)
    
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
            # print(self._directions[(tmp, last)])
            # print(tmp, last)
            backtrack.insert(0, self._directions[(tmp, last)])
            last = tmp
        return backtrack, cost
    
    # def _calculate_all_shortest_paths(self, start=(0,0)):
    #     for u in self._graph._data:
    #         print(u)
    #         # path = calc_shortest_paths(self._graph, s=u, d=(0,0))
    #         path = find_path(self._graph, u, (0,0))
    #         print(path)
    #         # self._shortest_paths.update({u: path[u]})
    #         # print(self._shortest_paths)

    # def _get_shortest_path(self, node):
    #     return extract_shortest_path(self._shortest_paths, node)
    
    # def get_cost(self, node):
    #     try:
    #         return self._get_shortest_path(node)[3]
    #     except:
    #         return float('inf')
    


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

    

    print(dijkstra.calc_backtrack((3,2)))

    # for u in dijkstra._graph._data:
    #     print(u)

    # print(dijkstra._shortest_paths)
    # print(dijkstra._directions)
    # dijkstra.add_edge((0,1), (0,0), 1)
    # print(dijkstra._graph)
    # print(dijkstra.calc_backtrack((3,2)))
    # print(dijkstra.get_cost((2,1)))