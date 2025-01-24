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
from collections import OrderedDict
import pickle


## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    """ class attribute """
    MAX_DIFFICULTY = 3             # the maximum degree of difficulty to enter into a cell
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
        self.victims_to_be_saved = [] # list of victims to be saved

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
        """
        This method uses K-Means to calculate the clusters of victims using (x,y) positions and gravity classes
        """
        # Number of clusters
        k = 4

        # Calculate random initial centroids (within the limits of the explored region)
        x_positions = []
        y_positions = []
        gravity_classes = []
        for _, values in self.victims.items():
            x_positions.append(values[0][0])
            y_positions.append(values[0][1])
            gravity_classes.append(values[1][7])
        lower_xlim = min(x_positions)
        lower_ylim = min(y_positions)
        upper_xlim = max(x_positions)
        upper_ylim = max(y_positions)
        lower_gravity = min(gravity_classes)
        upper_gravity = max(gravity_classes)
        centroids = []
        for i in range(k):
            x = random.uniform(lower_xlim, upper_xlim)
            y = random.uniform(lower_ylim, upper_ylim)
            gravity = random.uniform(lower_gravity, upper_gravity)
            centroids.append((x, y, gravity))

        # K-Means algorithm
        centroid_changed = True
        MAX_ITER = 100
        iter = 0
        # {vic_id: cluster_id}
        clusters = {}

        while centroid_changed and iter < MAX_ITER:
            centroid_changed = False

            # Calculate distance of each victim to the current centroids
            for vic_id, values in self.victims.items():
                x, y = values[0]
                gravity = values[1][7]
                min_dist = float('inf')
                cluster_id = -1
                for i in range(k):
                    x_centroid, y_centroid, gravity_centroid = centroids[i]
                    dist = (x - x_centroid)**2 + (y - y_centroid)**2 + (gravity - gravity_centroid)**2
                    # Store closest centroid/cluster to current victim
                    if dist < min_dist:
                        min_dist = dist
                        cluster_id = i
                
                # Reassign the closest cluster to the current victim
                clusters[vic_id] = cluster_id
            
            # Calculate new centroids for each cluster
            for i in range(k):
                x_sum = 0
                y_sum = 0
                gravity_sum = 0
                count = 0
                for vic_id, values in self.victims.items():
                    if clusters[vic_id] == i:
                        x, y = values[0]
                        gravity = values[1][7]
                        x_sum += x
                        y_sum += y
                        gravity_sum += gravity
                        count += 1
                # Calculate average of each coordinate of the centroid
                x_centroid = x_sum / count
                y_centroid = y_sum / count
                gravity_centroid = gravity_sum / count
                # Check if the centroid has changed
                if (x_centroid, y_centroid, gravity_centroid) != centroids[i]:
                    centroids[i] = (x_centroid, y_centroid, gravity_centroid)
                    centroid_changed = True

            iter += 1
            
        # list[cluster_id] = {vic_id: values}
        final_clusters = [{} for _ in range(k)]
        # Assign each victim to its cluster
        for vic_id, values in self.victims.items():
            cluster_id = clusters[vic_id]
            final_clusters[cluster_id][vic_id] = values

        print(f"{self.NAME} Clusters of victims: {final_clusters}")
        
        return final_clusters

        # """ @TODO: IMPLEMENT A CLUSTERING METHOD
        #     This method divides the victims in four quadrants of the explored region.
        #     @returns: a list of clusters where each cluster is a dictionary in the format [vic_id]: ((x,y), [<vs>])
        #               such as vic_id is the victim id, (x,y) is the victim's position, and [<vs>] the list of vital signals
        #               including the severity value and the corresponding label"""
        # # pass
        # # Find the upper and lower limits for x and y
        # lower_xlim = sys.maxsize
        # lower_ylim = sys.maxsize
        # upper_xlim = -sys.maxsize - 1
        # upper_ylim = -sys.maxsize - 1

        # vic = self.victims

        # for key, values in self.victims.items():
        #     x, y = values[0]
        #     lower_xlim = min(lower_xlim, x)
        #     upper_xlim = max(upper_xlim, x)
        #     lower_ylim = min(lower_ylim, y)
        #     upper_ylim = max(upper_ylim, y)

        # # Calculate midpoints
        # mid_x = lower_xlim + (upper_xlim - lower_xlim) / 2
        # mid_y = lower_ylim + (upper_ylim - lower_ylim) / 2

        # # Divide dictionary into quadrants
        # upper_left = {}
        # upper_right = {}
        # lower_left = {}
        # lower_right = {}

        # for key, values in self.victims.items():  # values are pairs: ((x,y), [<vital signals list>])
        #     x, y = values[0]
        #     if x <= mid_x:
        #         if y <= mid_y:
        #             upper_left[key] = values
        #         else:
        #             lower_left[key] = values
        #     else:
        #         if y <= mid_y:
        #             upper_right[key] = values
        #         else:
        #             lower_right[key] = values

        # return [upper_left, upper_right, lower_left, lower_right]

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
        dijkstra = Dijkstra((0,0), self.map, self.COST_LINE, self.COST_DIAG)

        for seq in self.sequences:   # a list of sequences, being each sequence a dictionary
            start = (0,0)
            tmp_seq = seq.copy()
            new_sequence = OrderedDict()

            while tmp_seq:
                shortest_time = float('inf')
                closest_vic = None
                for vic_id in tmp_seq.keys():
                    goal = tmp_seq[vic_id][0]
                    time = dijkstra.get_shortest_cost(start, goal)
                    if time < shortest_time:
                        shortest_time = time
                        closest_vic = vic_id

                new_sequence[closest_vic] = seq[closest_vic]
                start = tmp_seq[closest_vic][0]
                tmp_seq.pop(closest_vic)
            
            new_sequences.append(new_sequence)

            # seq = dict(sorted(seq.items(), key=lambda item: item[1]))
            # new_sequences.append(seq)
            # #print(f"{self.NAME} sequence of visit:\n{seq}\n")


        self.sequences = new_sequences

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
        # print(f"{self.NAME} sequence of visit:\n{sequence}\n")
        start = (0,0) # always from starting at the base
        time_tolerance = self.COST_FIRST_AID
        for vic_id in sequence:
            # Plan to come back to the base from the next victim position
            goal = sequence[vic_id][0]
            # plan_back, time_back = bfs.search(goal, (0,0))
            plan_back, time_back = dijkstra.calc_plan(goal, (0,0))
            
            # plan, time = bfs.search(start, goal, self.plan_rtime - time_back - time_tolerance) 
            plan, time = dijkstra.calc_plan(start, goal, self.plan_rtime - time_back - time_tolerance)
            
            # Check whether the agent has to come back to the base
            if time == -1:
                print(f"{self.NAME} Plan incomplete - not enough time to rescue all victims")
                break

            # Victim should be rescued
            self.victims_to_be_saved.append(vic_id)
            self.plan = self.plan + plan
            self.plan_rtime = self.plan_rtime - time - time_tolerance
            start = goal

            # print the remaining time to rescue each victim
            # print(f"{self.NAME} Remaining Time to rescue victim {vic_id}: {self.plan_rtime}")

        # Plan to come back to the base from the last victim position
        # plan_back, time_back = bfs.search(start, (0,0), self.plan_rtime)
        plan_back, time_back = dijkstra.calc_plan(start, (0,0), self.plan_rtime)
        self.plan = self.plan + plan_back
        self.plan_rtime = self.plan_rtime - time_back

        return

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
            # save map in a file using pickle
            mapfile = open('map_pickle', 'ab')
            
            # source, destination
            pickle.dump(self.map, mapfile)                    
            mapfile.close()


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
                rescuer.sequencing()         # the sequencing will reorder the cluster

                for j, sequence in enumerate(rescuer.sequences):
                    if j == 0:
                        self.save_sequence_csv(sequence, i+1)              # primeira sequencia do 1o. cluster 1: seq1
                    else:
                        self.save_sequence_csv(sequence, (i+1)+ j*10)      # demais sequencias do 1o. cluster: seq11, seq12, seq13, ...


                rescuer.planner()            # make the plan for the trajectory
                rescuer.set_state(VS.ACTIVE) # from now, the simulator calls the deliberation method


    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           print(f"{self.NAME} has finished the plan [ENTER]")
           # Print remaining plan time
           print(f"{self.NAME} Remaining time: {self.plan_rtime}")
           print(f"{self.NAME} True remaining time: {self.get_rtime()}")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)
        #print(f"{self.NAME} pop dx: {dx} dy: {dy} ")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:   # walk action was a success
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")

            # check if there is a victim at the current position
            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                # if there's a victim and the victim position is on the rescuers sequence, drop first aid
                if vic_id != VS.NO_VICTIM and vic_id in self.victims_to_be_saved:
                    self.first_aid()
                    self.victims_to_be_saved.remove(vic_id)
                    # Print remaining time
                    # print(f"{self.NAME} True remaining time to rescue victim {vic_id}: {self.get_rtime()}")
                    #if self.first_aid(): # True when rescued
                        #print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x}) due to {walked}")
            # Print remaining plan time
            print(f"{self.NAME} Remaining time: {self.plan_rtime}")
            print(f"{self.NAME} True remaining time: {self.get_rtime()}")
            # Print the plan
            # for p in self.plan:
            #     print(f"{self.NAME} Plan: {p}")

        return True
