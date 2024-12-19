from math import sqrt
from dijkstar import Graph, find_path
from dijkstar.algorithm import (
    single_source_shortest_paths as calc_shortest_paths,
    extract_shortest_path_from_predecessor_list as extract_shortest_path,
    find_path
)
from vs.constants import VS
from map import Map

class Dijkstra:
    def __init__(self, base=(0,0), map=None, line_cost=1, diag_cost=1.5):
        self._graph = Graph()
        self.base = base
        self._graph.add_node(base)
        self._directions = {}
        self.map = map
        if self.map:
            self.map_to_graph(self.map, line_cost, diag_cost)

    def map_to_graph(self, map, line_cost, diag_cost):
        for coord in map.data:
            self._graph.add_node(coord)
            diff, vic_id, actions_res = map.get(coord)
            x, y = coord
            if actions_res[0] == VS.CLEAR:  # up
                up = (x, y-1)
                if map.in_map(up):
                    neighbor_diff = map.get_difficulty(up)
                    self.add_edge((x, y), (x, y-1), line_cost*neighbor_diff, line_cost*diff)
            if actions_res[1] == VS.CLEAR:  # up right
                up_right = (x+1, y-1)
                if map.in_map(up_right):
                    neighbor_diff = map.get_difficulty(up_right)
                    self.add_edge((x, y), (x+1, y-1), diag_cost*neighbor_diff, diag_cost*diff)
            if actions_res[2] == VS.CLEAR:  # right
                right = (x+1, y)
                if map.in_map(right):
                    neighbor_diff = map.get_difficulty(right)
                    self.add_edge((x, y), (x+1, y), line_cost*neighbor_diff, line_cost*diff)
            if actions_res[3] == VS.CLEAR:  # down right
                down_right = (x+1, y+1)
                if map.in_map(down_right):
                    neighbor_diff = map.get_difficulty(down_right)
                    self.add_edge((x, y), (x+1, y+1), diag_cost*neighbor_diff, diag_cost*diff)
            if actions_res[4] == VS.CLEAR:  # down
                down = (x, y+1)
                if map.in_map(down):
                    neighbor_diff = map.get_difficulty(down)
                    self.add_edge((x, y), (x, y+1), line_cost*neighbor_diff, line_cost*diff)
            if actions_res[5] == VS.CLEAR:  # down left
                down_left = (x-1, y+1)
                if map.in_map(down_left):
                    neighbor_diff = map.get_difficulty(down_left)
                    self.add_edge((x, y), (x-1, y+1), diag_cost*neighbor_diff, diag_cost*diff)
            if actions_res[6] == VS.CLEAR:  # left
                left = (x-1, y)
                if map.in_map(left):
                    neighbor_diff = map.get_difficulty(left)
                    self.add_edge((x, y), (x-1, y), line_cost*neighbor_diff, line_cost*diff)
            if actions_res[7] == VS.CLEAR:  # up left
                up_left = (x-1, y-1)
                if map.in_map(up_left):
                    neighbor_diff = map.get_difficulty(up_left)
                    self.add_edge((x, y), (x-1, y-1), diag_cost*neighbor_diff, diag_cost*diff)

    
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
    
    def _estimate_heuristics(self, node1, node2, edge, prev_edge):
        x1, y1 = node1
        x2, y2 = node2
        heuristics = sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if not prev_edge and edge < heuristics:
            return edge
        if prev_edge and edge + prev_edge < heuristics:
            return edge + prev_edge
        return heuristics

    def calc_shortest_path(self, node1, node2, tlim=float('inf')):
        path = find_path(self._graph, node1, node2, heuristic_func=self._estimate_heuristics)
        if path[3] > tlim:
            return [], -1
        return path[0], path[3]
    
    def get_shortest_cost(self, node1, node2):
        return self.calc_shortest_path(node1, node2)[1]

    def get_shortest_path(self, node1, node2):
        return self.calc_shortest_path(node1, node2)[0]


    # def calc_shortest_path_back(self, node):
    #     path = find_path(self._graph, node, self.base, heuristic_func=self._estimate_heuristics)
    #     return path[0], path[3]

    # def get_shortest_cost_back(self, node):
    #     return self.calc_shortest_path_back(node)[1]

    # def get_shortest_path_back(self, node):
    #     return self.calc_shortest_path_back(node)[0]
    
    def calc_plan(self, node1, node2, tlim=float('inf')):
        path, cost = self.calc_shortest_path(node1, node2, tlim)
        if not path:
            return [], -1
        plan = []
        start = path.pop(0)
        for node in path:
            plan.append(self._directions[(start, node)])
            start = node
        return plan, cost

    def calc_backtrack(self, node):
        backtrack = []
        # path = self.calc_shortest_path_back(node)
        nodes, cost = self.calc_shortest_path(node, self.base)
        last = nodes.pop()
        while nodes:
            tmp = nodes.pop()
            backtrack.insert(0, self._directions[(tmp, last)])
            last = tmp
        return backtrack, cost

if __name__ == "__main__":
    map = Map()
    map.data = {
        (0, 0): (10, VS.NO_VICTIM, [VS.END, VS.END, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.END, VS.END,   VS.END]),
        (1, 0): (5, VS.NO_VICTIM, [VS.END, VS.END, VS.CLEAR, VS.WALL, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.END]),
        (2, 0): (8, VS.NO_VICTIM, [VS.END, VS.END, VS.CLEAR, VS.WALL,  VS.CLEAR,  VS.CLEAR, VS.CLEAR, VS.END]),
        (3, 0): (3, VS.NO_VICTIM, [VS.END, VS.END, VS.END,   VS.END,   VS.WALL,  VS.WALL,  VS.CLEAR, VS.END]),   
        (0, 1): (5, 1,            [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.END,   VS.END,   VS.END]),
        (1, 1): (0.1, 2,            [VS.CLEAR, VS.CLEAR, VS.WALL,  VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR]),
        (0, 2): (3, VS.NO_VICTIM, [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.END, VS.END, VS.END, VS.END,   VS.END]),
        (1, 2): (1, VS.NO_VICTIM, [VS.CLEAR, VS.WALL,  VS.CLEAR, VS.END, VS.END, VS.END, VS.CLEAR, VS.CLEAR]),
        (2, 2): (1, VS.NO_VICTIM, [VS.WALL,  VS.WALL,  VS.CLEAR, VS.END, VS.END, VS.END, VS.CLEAR, VS.CLEAR]),
        (3, 2): (1, 3,            [VS.WALL,  VS.END,   VS.END,   VS.END, VS.END, VS.END, VS.CLEAR, VS.WALL]),
    }
    map.draw()  
    print(map.get((1,0)))
    dijkstra = Dijkstra((0,0), map)
    # print(dijkstra._graph)
    graph = dijkstra._graph
    for node in graph:
        print(node, graph[node])
    prev_back_plan, prev_back_time = dijkstra.calc_plan((0,0), (0,0))
    print(prev_back_plan, prev_back_time)
    # dijkstra.add_edge((0,0), (1,0), 1, 1)
    # dijkstra.add_edge((0,0), (0,1), 1, 1)
    # dijkstra.add_edge((0,0), (1,1), 1.5, 1.5)

    # dijkstra.add_edge((1,0), (2,0), 1, 1)
    # dijkstra.add_edge((1,0), (1,1), 1, 1)
    # dijkstra.add_edge((1,0), (0,1), 1.5, 1.5)

    # dijkstra.add_edge((2,0), (3,0), 1, 1)
    # dijkstra.add_edge((2,0), (1,1), 1.5, 1.5)

    # dijkstra.add_edge((0,1), (1,1), 1, 1)
    # dijkstra.add_edge((0,1), (0,2), 1, 1)
    # dijkstra.add_edge((0,1), (1,2), 1.5, 1.5)

    # dijkstra.add_edge((1,1), (0,2), 1.5, 1.5)
    # dijkstra.add_edge((1,1), (1,2), 1, 1)
    # dijkstra.add_edge((1,1), (2,2), 1.5, 1.5)

    # dijkstra.add_edge((0,2), (1,2), 1, 1)

    # dijkstra.add_edge((1,2), (2,2), 1, 1)

    # dijkstra.add_edge((2,2), (3,2), 1, 1)

    # print(dijkstra._graph)
    # dijkstra.add_edge((0,0), (1,0), 1, 1)
    # print(dijkstra._graph)

    # print(dijkstra.check_edge((0,0), (3,2)))

    # print(dijkstra.calc_shortest_path_back((3,2)))
    # print(dijkstra.calc_plan((3,2), (0,0)))

    # print(dijkstra.calc_backtrack((3,2)))