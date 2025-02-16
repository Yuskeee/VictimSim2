import sys
import os
import pygame   # type: ignore
import random
import csv
import time
from .constants import VS

## Class PhysAgent
""" It is the representation of an agent in the environment
It MUST NOT be used by the rescuer or explorer """
class PhysAgent:
    def __init__(self, mind, env, x_base, y_base, state=VS.ACTIVE):
        """Instatiates a physical agent
        @param self: the physical agent
        @param mind: the mind of the physical agent (abstract agent)
        @param env: the environment
        @param x_base: x coordinate of the base
        @param y_base: y coordinate of the base"""

        self.mind = mind              # it is the agent's mind
        self.env = env                # it is the environment
        self.x_base = x_base          # x coordinate of the base (initial)
        self.y_base = y_base          # y coordinate of the base (initial)
        self.x = x_base               # current x coordinate
        self.y = y_base               # current y coordinate
        self._rtime = mind.TLIM       # current remaining time
        self._state = state           # -1=dead  0=successfully ended 1=alive
       

    def _end_of_time(self):
        """ This protected method allows the enviroment to check if time limit was reached and if the agent is at the base.
        @return: True - time exceeded
                 False - time not exceeded"""
        if self._rtime < 0.0:
           return True
        
        return False

    def _at_base(self):
        """ This protected method allows the enviroment to check if the agent is at the base.
        @return: True - the agent is at the base position
                 False - the agent is not at the base position"""
   
        if self.x == self.env.dic["BASE"][0] and self.y == self.env.dic["BASE"][1]:
           return True
       
        return False

    def _walk(self, dx, dy):
        """ Public method for moving the agent's body one cell to any direction (if possible)
        @param dx: an int value corresponding to deplacement in the x axis
        @param dy: an int value corresponding to deplacement in the y axis
        @returns -1 = the agent bumped into a wall or reached the end of grid
        @returns -2 = the agent has no enough time to execute the action
        @returns 1 = the action is succesfully executed
        In every case, action's executing time is discounted from time limit"""
        
        ## base time to be consumed
        if dx != 0 and dy != 0:   # diagonal
            base = self.mind.COST_DIAG 
        else:                     # walk vertical or horizontal
            base = self.mind.COST_LINE
        
        # new presumed coordinates
        new_x = self.x + dx
        new_y = self.y + dy

        # if it's within the boundaries and there is no wall
        if (new_x >= 0 and new_x < self.env.dic["GRID_WIDTH"] and
            new_y >= 0 and new_y < self.env.dic["GRID_HEIGHT"] and
            self.env.obst[new_x][new_y] != 100):
            #print(f"{self.mind.NAME}: obstacle difficulty {self.env.obst[new_x][new_y]}")
            self._rtime -= base * self.env.obst[new_x][new_y]   # consume time
            
            ## agent is dead: not enough time left
            if self._rtime < 0:
                return VS.TIME_EXCEEDED
            # execute action normally
            else:
                # update coordinates
                self.x = new_x
                self.y = new_y
                # update visited coordinate list
                if self not in self.env.visited[new_x][new_y]:
                    self.env.visited[new_x][new_y].append(self)
                return VS.EXECUTED
        else:
            ## when the agent bumps, we penalize the agent subtracting only the base time from the remaing time 
            self._rtime -= base
            return VS.BUMPED

    def _check_walls_and_lim(self):
        """ Protected method for checking walls and the grid limits in the neighborhood of the current position of the agent.
        @returns a vector of eight integers indexed in a clockwise manner. The first position in the vector is
        above the current position of the agent, the second is in the upper right diagonal direction, the third is to the right, and so on."        
        Each vector position containg one of the following values: {CLEAR, WALL, END}
        CLEAR means that there is no obstacle (value = 0)
        WALL means that there is a wall (value = 1)
        END means the end of the grid (value = 2)
        """
        
        # x -> agent
        # 7 0 1
        # 6 x 2
        # 5 4 3

        delta = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
        obstacles = [VS.CLEAR] * 8
        i = 0

        for d in delta:
            new_x = self.x + d[0]
            new_y = self.y + d[1]

            # check if it's out of the boundaries
            if new_x < 0 or new_x >= self.env.dic["GRID_WIDTH"] or new_y < 0 or new_y >= self.env.dic["GRID_HEIGHT"]:
                obstacles[i] = VS.END
            # check if there is a wall
            elif self.env.obst[new_x][new_y] == 100:
                obstacles[i] = VS.WALL

            i += 1

        #print(f"({self.x},{self.y}): obstacles={obstacles}")
  
        return obstacles 


    def _check_for_victim(self):
        """ Protected method for testing if there is a victim at the current position of the agent
        @returns: the id number of the victim - an integer starting from zero that corresponds to the position of
        the victim in the data files victims.txt and vital_signals.txt or VS.NO_VICTIMif there is no victim at the current position of the agent"""

        vic_id = VS.NO_VICTIM

        # check if there's a victim in the current coordinates
        if (self.x, self.y) in self.env.victims:
            vic_id = self.env.victims.index((self.x, self.y))

        return vic_id

    # explorer agent
    def _read_vital_signals(self):
        """ Protected method for reading the vital signals and marking a victim as found. The agent can only
        successfully execute this method if it is in the same position of the victim.
        Every tentative of reading the vital signal out of position consumes time.
        @returns:
        - VS.TIME_EXCEEDED if the agent has not enough time to read, or
        - the list of vital signals, removing the severity label and value 
        - an empty list if theres is no victim at the current agent's position."""

        ## Consume time
        self._rtime -= self.mind.COST_READ
    
        ## Agent is dead
        if self._rtime < 0:
           return VS.TIME_EXCEEDED

        ## check if there's a victim
        vic_id = self._check_for_victim()

        if vic_id == VS.NO_VICTIM:
            return []
        
        # Mark the victim as found by this agent.
        # More than one agent can found the same victim, so it's a list
        self.env.found[vic_id].append(self)
        return self.env.signals[vic_id][:-2] # remove the last two elements: label and value of severity (tbd by the rescuer)

    # rescuer agent
    def _first_aid(self):
        """ Protected method for dropping the first aid package to the victim located at the same position of the agent.
        This method marks the victim as saved.
        @returns:
        - VS.TIME_EXCEEDED when the agent has no enough battery time to execute the operation
        - True when the first aid is succesfully delivered
        - False when there is no victim at the current position of the agent"""

        ## Consume time
        self._rtime -= self.mind.COST_FIRST_AID

        ## Agent is dead
        if self._rtime < 0:
           return VS.TIME_EXCEEDED

        ## check if there's a victim
        vic_id = self._check_for_victim()
        if vic_id == VS.NO_VICTIM:
            return False
        
        # Mark the victim as found by this agent.
        # More than one agent can drop a first-aid package to the same victim, so it's a list
        self.env.saved[vic_id].append(self)
        return True

    # explorer agent
    def _get_found_victims(self):
        """ Protected method for returning the list of found victims by the agent
        @returns a list with the id number of the found victims """

        victims = []

        v = 0
        for finders in self.env.found:
            if self in finders:
                victims.append(v)
            v = v + 1
  
        return victims

    # rescuer agent
    def _get_saved_victims(self):
        """ Protected method for returning the list of of saved victims by the agent
        @returns a list with the id number of the saved victims """

        victims = []

        v = 0
        for rescuers in self.env.saved:
            if self in rescuers:
                victims.append(v)
            v = v + 1
  
        return victims 
                
            
