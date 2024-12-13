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
    MAX_DIFFICULTY = 1             # the maximum degree of difficulty to enter into a cell
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
    def __init__(self, env, config_file, resc):
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

    def get_next_position(self):
        """ Gets the next position that can be explored (no wall and inside the grid)
            There must be at least one CLEAR position in the neighborhood, otherwise it loops forever.
        """
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()
        tried_directions = set()  # to keep track of attempted directions

        while len(tried_directions) < 8:
            direction = random.randint(0, 7)
            if direction in tried_directions:
                continue
            tried_directions.add(direction)

            dx, dy = Explorer.AC_INCR[direction]
            if obstacles[direction] == VS.CLEAR and (self.x + dx, self.y + dy) not in self.visited:
                return (dx, dy)

        # If all directions have been tried, return a random valid direction
        direction = random.choice(list(tried_directions))
        return Explorer.AC_INCR[direction]

    def explore(self):
        # get an increment for x and y
        dx, dy = self.get_next_position()

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
            self.visited.add((self.x + dx,self.y + dy))
            # check for victim, returns -1 if there is no victim or the sequential
            # the sequential number of a found victim

            # store the new step (displacement) in the stack
            self.walk_stack.push((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy

            # update the walk time
            self.walk_time = self.walk_time + (rtime_bef - rtime_aft)
            #print(f"{self.NAME} walk time: {self.walk_time}")

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                # add the victim to the dictionary (vs[0] = victim id)
                self.victims[vs[0]] = ((self.x, self.y), vs)

                #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")

            # Calculates the difficulty (cost) of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
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
            #print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")

    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        # TODO: IMPLEMENT LOGIC TO DECIDE THE NEXT ACTION

        # forth and back: go, read the vital signals and come back to the position

        time_tolerance = 2* self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ

        # keeps exploring while there is enough time
        if  self.walk_time < (self.get_rtime() - time_tolerance):
            self.explore()
            return True

        if not self.is_coming_back:
            self.map.draw
            # time to come back
            self.is_coming_back = True

            # calculates with BFS the path to the base
            start = (self.x, self.y)
            goal = (0, 0)
            bfs = BFS(self.map)
            self.back_plan, self.back_plan_cost = bfs.search(start, goal)
            print(f"{self.NAME}: starting position: {start}")
            print(f"{self.NAME}: back plan: {self.back_plan}, cost: {self.back_plan_cost}")
            # updates walk_stack with the back_plan
            self.walk_stack = Stack()
            for action in self.back_plan[::-1]:
                self.walk_stack.push(action)
            print (f"{self.NAME}: walk_stack: {self.walk_stack.items}")

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
