# EXPLORER AGENT
### It walks in the environment looking for victims.

import sys
import os
import random
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from bfs import BFS
from numpy.random import choice
from dijkstra import Dijkstra

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    """ class attribute """
    MAX_DIFFICULTY = 3             # the maximum degree of difficulty to enter into a cell
    # Increments for walk actions
    # EX_INCR = {
    #     0: (0, -1),  #  u: Up
    #     1: (1, -1),  # ur: Upper right diagonal
    #     2: (1, 0),   #  r: Right
    #     3: (1, 1),   # dr: Down right diagonal
    #     4: (0, 1),   #  d: Down
    #     5: (-1, 1),  # dl: Down left left diagonal
    #     6: (-1, 0),  #  l: Left
    #     7: (-1, -1)  # ul: Up left diagonal
    # }
    def __init__(self, env, config_file, resc, initial_direction = None):
        """ Construtor do agente [inserir algoritimo de busca]
        @param env: a reference to the environment
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements (for returning to the base)
        self.walk_time = 0         # time consumed to walk when exploring (to decide when to come back)
        self.set_state(VS.ACTIVE)  # explorer is active since the beginning
        self.visited = set()       # to keep track of visited positions
        self.resc = resc           # reference to the rescuer agent that will be invoked when exploration finishes
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim (the victim id),(x,y) the position, <vs> the list of vital signals

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        self.visited.add((self.x, self.y))
        self.is_coming_back = False
        self.back_plan = []        # the plan to come back to the base
        self.back_plan_cost = 0    # the cost of the plan to come back to the base
        self.dijkstra = Dijkstra((0,0))
        self.visited_cells_with_unvisited_neighbors = Stack()
        self.initial_direction = initial_direction

        # An array that is a permutation of [0,1,2,3,4,5,6,7] -> movements
        if initial_direction == None:
            initial_direction = random.randint(0, 7)
        other_directions = [i for i in range(8) if i != initial_direction]
        self.movements = [initial_direction] + other_directions
        aux_movements = self.movements[1:]
        random.shuffle(aux_movements)   # shuffle the array to avoid always trying the same direction
        self.movements = [self.movements[0]] + aux_movements
        self.n_movements_before_shuffle = 1  # number of movements before shuffling the array
        self.movements_counter = 0           # counter to shuffle the array

    def get_next_position(self):
        """ Gets the next position that can be explored (no wall and inside the grid)
            There must be at least one CLEAR position in the neighborhood, otherwise it loops forever.
        """
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()
        tried_directions = set()  # to keep track of attempted directions
        cost_current = self.map.get_difficulty((self.x, self.y))

        # Shuffle the movements to avoid always trying the same direction
        if self.movements_counter > self.n_movements_before_shuffle:
            # Shuffle only the last 7 elements of the array, the first one is always the same
            aux_movements = self.movements[1:]
            random.shuffle(aux_movements)
            self.movements = [self.movements[0]] + aux_movements
            self.movements_counter = 0
        else:
            self.movements_counter += 1

        for movement in self.movements:
            direction = movement

            dx, dy = Explorer.AC_INCR[direction]
            if obstacles[direction] == VS.CLEAR and (self.x + dx, self.y + dy) not in self.visited:
                return (dx, dy)
            
            if obstacles[direction] == VS.CLEAR and (self.x + dx, self.y + dy) in self.visited and not self.dijkstra.check_edge((self.x, self.y), (self.x + dx, self.y + dy)):
                cost_neighbor = self.map.get_difficulty((self.x + dx, self.y + dy))
                if dx == 0 or dy == 0:
                    cost_current = cost_current * self.COST_LINE
                    cost_neighbor = cost_neighbor * self.COST_LINE
                else:
                    cost_current = cost_current * self.COST_DIAG
                    cost_neighbor = cost_neighbor * self.COST_DIAG

                self.dijkstra.add_edge((self.x, self.y), (self.x + dx, self.y + dy), cost_current, cost_neighbor)

                

        # If all directions have been tried, return a random valid direction
        direction = random.randint(0, 7)
        return Explorer.AC_INCR[direction]

    def explore(self):
        # get an increment for x and y
        dx, dy = self.get_next_position()

        # checks whether the agent should backtrack due to all neighbors being visited
        # if all neighbors are visited or bumps into a wall or limit, return to the previous position
        neighbors_not_worth_visiting = [(self.x + incr[0], self.y + incr[1]) in self.visited or self.check_walls_and_lim()[i] == VS.WALL or self.check_walls_and_lim()[i] == VS.END for i, incr in Explorer.AC_INCR.items()]
        if all(neighbors_not_worth_visiting):
            # bfs_for_backtrack = BFS(self.map)
            start = (self.x, self.y)
            if self.visited_cells_with_unvisited_neighbors.is_empty():
                # goal = (0, 0)
                # backtrack_plan, backtrack_plan_cost = bfs_for_backtrack.search(start, goal)
                backtrack_plan, _ = self.dijkstra(start)
            else:
                goal = self.visited_cells_with_unvisited_neighbors.pop()
                # backtrack_plan, backtrack_plan_cost = bfs_for_backtrack.search(start, goal)
                backtrack_plan, _ = self.dijkstra.calc_backtrack(start, goal)

            self.walk_stack = Stack()
            for action in backtrack_plan[::-1]:
                self.walk_stack.push(action)
            return
        # if two or more neighbors are worth visiting, add the current position to the set of visited cells with unvisited neighbors
        elif len([neighbor for neighbor in neighbors_not_worth_visiting if not neighbor]) >= 2:
            self.visited_cells_with_unvisited_neighbors.push((self.x, self.y))

        # Moves the body to another position
        rtime_bef = self.get_rtime()    # previous remaining time
        result = self.walk(dx, dy)      # walk to the new position
        rtime_aft = self.get_rtime()    # remaining time after the walk



        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # # store the new step (displacement) in the stack
            # if (self.x + dx,self.y + dy) not in self.visited:
            #     self.walk_stack.push((dx, dy))

            self.visited.add((self.x + dx,self.y + dy))
            # check for victim, returns -1 if there is no victim or the sequential
            # the sequential number of a found victim

            prev_x = self.x
            prev_y = self.y
            prev_diff = self.map.get_difficulty((prev_x, prev_y))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy

            # update the walk time
            self.walk_time = self.walk_time + (rtime_bef - rtime_aft)
            #print(f"{self.NAME} walk time: {self.walk_time}")

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM and seq not in self.victims:
                vs = self.read_vital_signals()
                # add the victim to the dictionary (vs[0] = victim id)
                self.victims[vs[0]] = ((self.x, self.y), vs)

                #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")

            # Calculates the difficulty (cost) of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                prev_diff = prev_diff * self.COST_LINE
                self.dijkstra.add_edge((prev_x, prev_y), (self.x, self.y), difficulty, prev_diff)
                difficulty = difficulty / self.COST_LINE
            else:
                prev_diff = prev_diff * self.COST_DIAG
                self.dijkstra.add_edge((prev_x, prev_y), (self.x, self.y), difficulty, prev_diff)
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        """ Do the steps that are in the walk_stack to come back to the base """
        dx, dy = self.walk_stack.pop()
        # dx = dx * -1
        # dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return

        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            # print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")

    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        # TODO: IMPLEMENT LOGIC TO DECIDE THE NEXT ACTION

        # forth and back: go, read the vital signals and come back to the position

        time_tolerance = 2* self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ

        # keeps exploring while there is enough time
        if  self.back_plan_cost + time_tolerance < self.get_rtime():
            if not self.walk_stack.is_empty():
                self.come_back()
            else:
                self.explore()

            # start = (self.x, self.y)
            # goal = (0, 0)
            # bfs = BFS(self.map)
            # self.back_plan, self.back_plan_cost = bfs.search(start, goal)
            # self.back_plan_cost = self.dijkstra.get_shortest_cost_back((self.x, self.y))
            self.back_plan_cost = self.dijkstra.get_shortest_cost((self.x, self.y), (0,0))

            # with open(f"{self.NAME}.csv",'a') as f:
            #     f.write(f"{self.back_plan_cost},{self.get_rtime()}\n")
            
            return True

        if not self.is_coming_back:
            # self.map.draw
            # time to come back
            self.is_coming_back = True

            # calculates with BFS the path to the base
            # start = (self.x, self.y)
            # goal = (0, 0)
            # bfs = BFS(self.map)
            # self.back_plan, self.back_plan_cost = bfs.search(start, goal)
            # print(f"{self.NAME}: starting position: {start}")
            # print(f"{self.NAME}: back plan: {self.back_plan}, cost: {self.back_plan_cost}")
            self.back_plan, self.back_plan_cost = self.dijkstra.calc_backtrack((self.x, self.y))
            # updates walk_stack with the back_plan
            self.walk_stack = Stack()
            for action in self.back_plan[::-1]:
                self.walk_stack.push(action)
            # print (f"{self.NAME}: walk_stack: {self.walk_stack.items}")

        # no more come back walk actions to execute or already at base
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            # time to pass the map and found victims to the master rescuer
            self.resc.sync_explorers(self.map, self.victims)
            # prints position and time of the explorer when finishes
            print(f"{self.NAME}: at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
            # finishes the execution of this agent
            return False

        # proceed to the base
        self.come_back()
        return True
