## EXPLORER AGENT
### @Author: Tacla, UTFPR
### It walks randomly in the environment looking for victims.

import sys
import os
import random
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc):
        """ Construtor do agente random on-line
        @param env referencia o ambiente
        @config_file: the absolute path to the explorer's config file
        @param resc referencia o rescuer para poder acorda-lo
        """
        super().__init__(env, config_file)
        self.set_state(VS.ACTIVE)
        
        # Specific initialization for the rescuer
        self.resc = resc           # reference to the rescuer agent   

        # List of visited coordinates
        self.visited = []

        # List of backtracked coordinates
        self.backtracked = []

        # List of coordinates with obstacles
        self.obstacles = []

        # Current coordinates
        self.x = 0
        self.y = 0
    
    # Neighbor Index to Position
    def neighbor_index_to_position(self, current_coordinates, index):
        if index == 0:
            return (current_coordinates[0], current_coordinates[1] - 1)
        elif index == 1:
            return (current_coordinates[0] + 1, current_coordinates[1] - 1)
        elif index == 2:
            return (current_coordinates[0] + 1, current_coordinates[1])
        elif index == 3:
            return (current_coordinates[0] + 1, current_coordinates[1] + 1)
        elif index == 4:
            return (current_coordinates[0], current_coordinates[1] + 1)
        elif index == 5:
            return (current_coordinates[0] -1, current_coordinates[1] + 1)
        elif index == 6:
            return (current_coordinates[0] - 1, current_coordinates[1])
        elif index == 7:
            return (current_coordinates[0] - 1, current_coordinates[1] - 1)

    # Exploration: DFS
    def explore(self):
        self.visited.append((self.x, self.y))  # Adds the current position to visited
        self.backtracked.append((self.x, self.y))  # Adds the current position to backtracked

        neighbors = self.check_walls_and_lim()
        valid_neighbors = []

        for i in range(len(neighbors)):
            if neighbors[i] == VS.CLEAR: # Adds valid neighbors to local list
                if self.neighbor_index_to_position((self.x, self.y), i) not in self.visited: # Checks if the neighbor is already visited
                    valid_neighbors.append(self.neighbor_index_to_position((self.x, self.y), i))
            else:
                # It does not differentiate between walls and end of map
                self.obstacles.append(self.neighbor_index_to_position((self.x, self.y), i))
        
        # Checks if should backtrack
        if len(valid_neighbors) == 0:
            if len(self.backtracked) <= 1:
                print(f"{self.NAME} No more places to explore... invoking the rescuer")
                self.resc.go_save_victims([],[])
                return (self.x, self.y)
            else:
                # Backtracks to the last position
                self.backtracked.pop()
                next_position = self.backtracked.pop()
                return next_position

        # Pops the next position to explore (maybe with random choice)
        next_position = valid_neighbors.pop(random.randint(0, len(valid_neighbors) - 1))
        return next_position

    def return_to_base(self):
        # Returns to the base using A* algorithm
        pass

    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        print(f"\n{self.NAME} deliberate:")
        # No more actions, time almost ended
        if self.get_rtime() <= 1.0:
            # time to wake up the rescuer
            # pass the walls and the victims (here, they're empty)
            print(f"{self.NAME} No more time to explore... invoking the rescuer")
            self.resc.go_save_victims([],[])
            return False
        
        dx = 0
        dy = 0

        (next_x, next_y) = self.explore()

        # Update dx and dy based on choice made through explore()
        dx = next_x - self.x
        dy = next_y - self.y
        self.x = next_x
        self.y = next_y

        print(f"{self.NAME} exploring starts")
        
        # Moves the body to another position
        result = self.walk(dx, dy)

        # Test the result of the walk action
        if result == VS.BUMPED:
            walls = 1  # build the map- to do
            print(f"{self.NAME}: wall or grid limit reached")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            print(f"{self.NAME} walk executed, rtime: {self.get_rtime()}")
            seq = self.check_for_victim()
            if seq >= 0:
                vs = self.read_vital_signals()
                print(f"{self.NAME} Vital signals read, rtime: {self.get_rtime()}")
                print(f"{self.NAME} Vict: {vs[0]}\n     pSist: {vs[1]:.1f} pDiast: {vs[2]:.1f} qPA: {vs[3]:.1f}")
                print(f"     pulse: {vs[4]:.1f} frResp: {vs[5]:.1f}")  
               
        return True

