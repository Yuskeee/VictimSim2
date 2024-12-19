##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### This rescuer version implements:
### - clustering of victims by quadrants of the explored region
### - definition of a sequence of rescue of victims of a cluster
### - assigning one cluster to one rescuer
### - calculating paths between pair of victims using breadth-first search
###
### One of the rescuers is the master in charge of unifying the maps and the information
### about the found victims.

import os
import random
import math
import csv
import sys
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from bfs import BFS
from abc import ABC, abstractmethod
from dijkstra import Dijkstra


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1,clusters=[]):
        """
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file
        @param nb_of_explorers: number of explorer agents to wait for
        @param clusters: list of clusters of victims in the charge of this agent"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.nb_of_explorers = nb_of_explorers       # number of explorer agents to wait for start
        self.received_maps = 0                       # counts the number of explorers' maps
        self.map = Map()                             # explorer will pass the map
        self.victims = {}            # a dictionary of found victims: [vic_id]: ((x,y), [<vs>])
        self.plan = []               # a list of planned actions in increments of x and y
        self.plan_x = 0              # the x position of the rescuer during the planning phase
        self.plan_y = 0              # the y position of the rescuer during the planning phase
        self.plan_visited = set()    # positions already planned to be visited
        self.plan_rtime = self.TLIM  # the remaing time during the planning phase
        self.plan_walk_time = 0.0    # previewed time to walk during rescue
        self.x = 0                   # the current x position of the rescuer when executing the plan
        self.y = 0                   # the current y position of the rescuer when executing the plan
        self.clusters = clusters     # the clusters of victims this agent should take care of - see the method cluster_victims
        self.sequences = clusters    # the sequence of visit of victims for each cluster
        self.is_returning = False    # flag to indicate the agent is returning to the base
        self.last_victim = None      # the last victim rescued

        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    # save a calculated cluster in a csv file
    def save_cluster_csv(self, cluster, cluster_id):
        filename = f"./clusters/cluster{cluster_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for vic_id, values in cluster.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([vic_id, x, y, vs[6], vs[7]])

    # save the calculated sequence of rescue for a cluster in a csv file
    def save_sequence_csv(self, sequence, sequence_id):
        filename = f"./clusters/seq{sequence_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, values in sequence.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([id, x, y, vs[6], vs[7]])

    def cluster_victims(self):
        """ @TODO: IMPLEMENT A CLUSTERING METHOD
            This method divides the victims in four quadrants of the explored region.
            @returns: a list of clusters where each cluster is a dictionary in the format [vic_id]: ((x,y), [<vs>])
                      such as vic_id is the victim id, (x,y) is the victim's position, and [<vs>] the list of vital signals
                      including the severity value and the corresponding label"""
        # pass
        # Find the upper and lower limits for x and y
        lower_xlim = sys.maxsize
        lower_ylim = sys.maxsize
        upper_xlim = -sys.maxsize - 1
        upper_ylim = -sys.maxsize - 1

        vic = self.victims

        for key, values in self.victims.items():
            x, y = values[0]
            lower_xlim = min(lower_xlim, x)
            upper_xlim = max(upper_xlim, x)
            lower_ylim = min(lower_ylim, y)
            upper_ylim = max(upper_ylim, y)

        # Calculate midpoints
        mid_x = lower_xlim + (upper_xlim - lower_xlim) / 2
        mid_y = lower_ylim + (upper_ylim - lower_ylim) / 2
        print(f"{self.NAME} ({lower_xlim}, {lower_ylim}) - ({upper_xlim}, {upper_ylim})")
        print(f"{self.NAME} cluster mid_x, mid_y = {mid_x}, {mid_y}")

        # Divide dictionary into quadrants
        upper_left = {}
        upper_right = {}
        lower_left = {}
        lower_right = {}

        for key, values in self.victims.items():  # values are pairs: ((x,y), [<vital signals list>])
            x, y = values[0]
            if x <= mid_x:
                if y <= mid_y:
                    upper_left[key] = values
                else:
                    lower_left[key] = values
            else:
                if y <= mid_y:
                    upper_right[key] = values
                else:
                    lower_right[key] = values

        return [upper_left, upper_right, lower_left, lower_right]

    def predict_severity_and_class(self):
        """ @TODO to be replaced by a classifier and a regressor to calculate the class of severity and the severity values.
            This method should add the vital signals(vs) of the self.victims dictionary with these two values.
        """
        # pass
        for vic_id, values in self.victims.items():
            severity_value = random.uniform(0.1, 99.9)          # to be replaced by a regressor
            severity_class = random.randint(1, 4)               # to be replaced by a classifier
            values[1].extend([severity_value, severity_class])  # append to the list of vital signals; values is a pair( (x,y), [<vital signals list>] )


    def sequencing(self):
        """ Currently, this method sort the victims by the x coordinate followed by the y coordinate
            @TODO It must be replaced by a Genetic Algorithm that finds the possibly best visiting order """

        """ We consider an agent may have different sequences of rescue. The idea is the rescuer can execute
            sequence[0], sequence[1], ...
            A sequence is a dictionary with the following structure: [vic_id]: ((x,y), [<vs>]"""
        # pass
        new_sequences = []

        for seq in self.sequences:   # a list of sequences, being each sequence a dictionary
            seq = dict(sorted(seq.items(), key=lambda item: item[1]))
            new_sequences.append(seq)
            #print(f"{self.NAME} sequence of visit:\n{seq}\n")

        self.sequences = new_sequences


    def fitness(self, individual):
        """Calculate the fitness as the inverse of total time to rescue."""
        total_time = 0
        current_pos = (0, 0)
        dijkstra = Dijkstra((0,0), self.map, self.COST_LINE, self.COST_DIAG)

        for victim_id in individual:
            coord, _ = self.victims[victim_id]
            _, cost = dijkstra.calc_plan(current_pos, coord)
            cost += self.COST_FIRST_AID
            total_time += cost
            current_pos = coord
            # dx = abs(coord[0] - current_pos[0])
            # dy = abs(coord[1] - current_pos[1])
            # step_cost = dx * self.COST_LINE + dy * self.COST_DIAG
            # total_time += step_cost
            # current_pos = coord

        # Add time to return to base
        # total_time += abs(current_pos[0]) * self.COST_LINE + abs(current_pos[1]) * self.COST_DIAG
        total_time += dijkstra.calc_plan(current_pos, (0,0))[1]
        return 1 / (total_time + 1)  # Inverse of total time

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        crossover_point = random.randint(1, len(parent1) - 2)
        child = parent1[:crossover_point] + [v for v in parent2 if v not in parent1[:crossover_point]]
        return child

    def mutate(self, individual):
        """Mutate an individual randomly."""
        if random.random() < 0.1:  # 10% mutation probability
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    # def planner(self):
    #     """Use a genetic algorithm to define the rescue sequence."""
    #     if not self.victims:
    #         return
        
    #     dijkstra = Dijkstra((0,0), self.map, self.COST_LINE, self.COST_DIAG)

    #     # Generate initial population
    #     population = [random.sample(list(self.victims.keys()), len(self.victims)) for _ in range(10)]

    #     # Evolve population over generations
    #     for _ in range(100):
    #         # Evaluate fitness
    #         population = sorted(population, key=self.fitness, reverse=True)

    #         # Select top individuals for reproduction
    #         new_population = population[:5]

    #         # Create offspring through crossover and mutation
    #         for _ in range(5):
    #             parent1, parent2 = random.sample(new_population, 2)
    #             child = self.crossover(parent1, parent2)
    #             self.mutate(child)
    #             new_population.append(child)

    #         population = new_population

    #     # Use the best individual as the rescue sequence
    #     best_individual = population[0]
    #     self.plan = []
    #     current_pos = (0, 0)

    #     for victim_id in best_individual:
    #         coord, _ = self.victims[victim_id]
    #         tmp_plan, _ = dijkstra.calc_plan(current_pos, coord)
    #         plan = [(step[0], step[1], False) for step in tmp_plan]
    #         plan[-1] = (plan[-1][0], plan[-1][1], True)
    #         self.plan.extend(plan)
    #         current_pos = coord


    #         # dx = coord[0] - current_pos[0]
    #         # dy = coord[1] - current_pos[1]
    #         # self.plan.append((dx, dy, True))
    #         # current_pos = coord

    #     # Add return to base
    #     # self.plan.append((-current_pos[0], -current_pos[1], False))
    #     tmp_plan, _ = dijkstra.calc_plan(current_pos, (0,0))
    #     plan = [(step[0], step[1], False) for step in tmp_plan]
    #     self.plan.extend(plan)


    def planner(self):
        """ A method that calculates the path between victims: walk actions in a OFF-LINE MANNER (the agent plans, stores the plan, and
            after it executes. Eeach element of the plan is a pair dx, dy that defines the increments for the the x-axis and  y-axis."""


        # let's instantiate the breadth-first search
        # bfs = BFS(self.map, self.COST_LINE, self.COST_DIAG)
        dijkstra = Dijkstra((0,0), self.map, self.COST_LINE, self.COST_DIAG)

        # for each victim of the first sequence of rescue for this agent, we're going go calculate a path
        # starting at the base - always at (0,0) in relative coords
        
        if not self.sequences:   # no sequence assigned to the agent, nothing to do
            return

        # we consider only the first sequence (the simpler case)
        # The victims are sorted by x followed by y positions: [vic_id]: ((x,y), [<vs>]

        sequence = self.sequences[0]
        if not sequence:
            return
        
        start = (0,0) # always from starting at the base
        prv_plan = []
        prv_victim = None
        prv_time = 0
        prv_back_plan = []
        prv_back_time = 0
        time_tolerance = 3*self.COST_DIAG + self.COST_FIRST_AID
        
        for vic_id in sequence:
            goal = sequence[vic_id][0]
            plan, time = dijkstra.calc_plan(start, goal)
            plan = plan + [(0,0)]  # add the first aid action
            # back_plan, back_time = dijkstra.calc_plan(goal, (0,0))
            back_plan, back_time = dijkstra.calc_backtrack(goal)
            total_time = prv_time + time + back_time + self.COST_FIRST_AID + time_tolerance
            if total_time > self.TLIM:
                self.plan = prv_plan + prv_back_plan
                self.last_victim = prv_victim
                self.plan_rtime = self.TLIM - (prv_time + prv_back_time)
                return
            else:
                prv_plan = prv_plan + plan
                prv_victim = vic_id
                prv_time = prv_time + time + self.COST_FIRST_AID + time_tolerance
                prv_back_plan = back_plan
                prv_back_time = back_time
                start = goal
        
        self.plan = prv_plan + prv_back_plan
        self.last_victim = prv_victim
        self.plan_rtime = self.TLIM - (prv_time + prv_back_time)


        

        # # Plan to come back to the base
        # goal = (0,0)
        # # plan, time = bfs.search(start, goal, self.plan_rtime)
        # plan, time = dijkstra.calc_plan(start, goal)
        # self.plan = self.plan + plan
        # # self.plan_rtime = self.plan_rtime - time

    def sync_explorers(self, explorer_map, victims):
        """ This method should be invoked only to the master agent

        Each explorer sends the map containing the obstacles and
        victims' location. The master rescuer updates its map with the
        received one. It does the same for the victims' vital signals.
        After, it should classify each severity of each victim (critical, ..., stable);
        Following, using some clustering method, it should group the victims and
        and pass one (or more)clusters to each rescuer """

        self.received_maps += 1

        # Receive map from explorer
        print(f"{self.NAME} Map received from the explorer")
        self.map.update(explorer_map)
        self.victims.update(victims)

        # check if all maps were received
        if self.received_maps == self.nb_of_explorers:
            print(f"{self.NAME} all maps received from the explorers")
            #self.map.draw()
            #print(f"{self.NAME} found victims by all explorers:\n{self.victims}")

            #@TODO predict the severity and the class of victims' using a classifier
            self.predict_severity_and_class()

            #@TODO cluster the victims possibly using the severity and other criteria
            # Here, there 4 clusters
            clusters_of_vic = self.cluster_victims()

            # Save the clusters in csv files
            for i, cluster in enumerate(clusters_of_vic):
                self.save_cluster_csv(cluster, i+1)    # file names start at 1

            # Instantiate the other rescuers
            rescuers = [None] * 4
            rescuers[0] = self                    # the master rescuer is the index 0 agent

            # Assign the cluster the master agent is in charge of (first one)
            self.clusters = [clusters_of_vic[0]]

            # Instantiate the other rescuers and assign the clusters to them
            for i in range(1, 4):
                #print(f"{self.NAME} instantianting rescuer {i+1}, {self.get_env()}")
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                # each rescuer receives one cluster of victims
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, [clusters_of_vic[i]])
                rescuers[i].map = self.map     # each rescuer have the map


            # Calculate the sequence of rescue for each agent
            # In this case, each agent has just one cluster and one sequence
            self.sequences = self.clusters

            # For each rescuer, we calculate the rescue sequence
            for i, rescuer in enumerate(rescuers):
                rescuer:Rescuer
                rescuer.sequencing()         # the sequencing will reorder the cluster

                for j, sequence in enumerate(rescuer.sequences):
                    if j == 0:
                        self.save_sequence_csv(sequence, i+1)              # primeira sequencia do 1o. cluster 1: seq1
                    else:
                        self.save_sequence_csv(sequence, (i+1)+ j*10)      # demais sequencias do 1o. cluster: seq11, seq12, seq13, ...


                rescuer.planner()            # make the plan for the trajectory
                rescuer.set_state(VS.ACTIVE) # from now, the simulator calls the deliberation method


    # def deliberate(self) -> bool:
    #     """Choose the next action to execute."""
    #     if not self.plan:
    #         print(f"{self.NAME} has finished the plan.")
    #         return False

    #     dx, dy, there_is_vict = self.plan.pop(0)
    #     walked = self.walk(dx, dy)

    #     if walked == VS.EXECUTED:
    #         self.x += dx
    #         self.y += dy

    #         if there_is_vict and self.map.in_map((self.x, self.y)):
    #             vic_id = self.map.get_vic_id((self.x, self.y))
    #             if vic_id != VS.NO_VICTIM:
    #                 rescued = self.first_aid()
    #                 # if rescued:
    #                     # print(f"{self.NAME} rescued victim at ({self.x}, {self.y}).")

    #     return True

    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           print(f"{self.NAME} has finished the plan [ENTER]")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)
        #print(f"{self.NAME} pop dx: {dx} dy: {dy} ")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")
            
            if dx == 0 and dy == 0:  # first aid action
                # check if there is a victim at the current position
                if not self.is_returning and self.map.in_map((self.x, self.y)):
                    vic_id = self.map.get_vic_id((self.x, self.y))
                    if vic_id != VS.NO_VICTIM:
                        self.first_aid()
                        if vic_id == self.last_victim:
                            self.is_returning = True
                            print(f"{self.NAME} returning to base")
                        #if self.first_aid(): # True when rescued
                            #print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")                    
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        return True