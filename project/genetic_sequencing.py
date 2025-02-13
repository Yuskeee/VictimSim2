import random
import numpy as np
from collections import defaultdict
from dijkstra import Dijkstra
from vs.constants import VS  # Import VS constants for wall and clear values

class GeneticSequencer:
    def __init__(self, cluster_victims, map, line_cost=1.0, diag_cost=1.5,
                 population_size=100, generations=200, mutation_rate=0.2,
                 Ts=300.0, COST_FIRST_AID=5.0):  # Add Ts and COST_FIRST_AID
        """
        Initialize the GA sequencer.
        :param cluster_victims: Dictionary of victims in the cluster {id: (pos, vitals)}.
        :param map: Map instance for pathfinding.
        :param Ts: Time limit for the rescuer.
        :param COST_FIRST_AID: Time cost to administer first aid to a victim.
        """
        self.victims = cluster_victims
        self.map = map
        self.line_cost = line_cost
        self.diag_cost = diag_cost
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.Ts = Ts  # Time limit
        self.COST_FIRST_AID = COST_FIRST_AID  # First aid cost
        self.base = (0, 0)

        # Precompute distance matrix
        self.dijkstra = Dijkstra(self.base, self.map, line_cost, diag_cost)
        self.vic_ids = list(cluster_victims.keys())
        self.distance_matrix = self._precompute_distances()

    def _precompute_distances(self):
        # Use A* with Manhattan distance heuristic
        nodes = [self.base] + [self.victims[vid][0] for vid in self.vic_ids]
        n = len(nodes)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i != j:
                    path, cost = self.dijkstra.calc_plan(nodes[i], nodes[j])
                    distance_matrix[i][j] = cost * 1.1 if cost != -1 else float('inf')
        return distance_matrix

    def _get_vic_index(self, position):
        """Get index in distance matrix for a victim's position."""
        if position == self.base:
            return 0
        return [self.victims[vid][0] for vid in self.vic_ids].index(position) + 1

    def _calculate_sequence_cost(self, sequence):
        """Calculate total cost for a sequence starting and ending at base."""
        total_cost = 0
        current = 0  # Base index

        for vic_id in sequence:
            pos = self.victims[vic_id][0]
            next_idx = self._get_vic_index(pos)
            total_cost += self.distance_matrix[current][next_idx]
            current = next_idx

        # Return to base
        total_cost += self.distance_matrix[current][0]
        return total_cost

    def _fitness(self, sequence):
        total_time = 0
        saved_severity = 0
        saved_ids = []
        current_pos = self.base

        for vic_id in sequence:
            pos = self.victims[vic_id][0]
            idx_current = self._get_vic_index(current_pos)
            idx_next = self._get_vic_index(pos)

            path_time = self.distance_matrix[idx_current][idx_next]
            rescue_time = self.COST_FIRST_AID

            # Check if it's possible to save and return to base
            return_time = self.distance_matrix[idx_next][0]
            if (total_time + path_time + rescue_time + return_time) > self.Ts:
                continue  # Skip unfeasible victim

            total_time += path_time + rescue_time
            saved_severity += self.victims[vic_id][1][6] * (1 + (4 - self.victims[vic_id][1][7])/4)  # Weight for critical classes
            saved_ids.append(vic_id)
            current_pos = pos

        # Add return time to base
        total_time += self.distance_matrix[self._get_vic_index(current_pos)][0]

        # Exponential penalty for exceeding time
        if total_time > self.Ts:
            time_over = total_time - self.Ts
            saved_severity *= max(0, 1 - (time_over/self.Ts)**2)  # Progressive penalty

        return saved_severity

    def _generate_initial_population(self):
        population = []

        # Heuristic 1: Priority by severity/time
        victims = sorted(self.vic_ids,
                         key=lambda x: self.victims[x][1][6]/self.distance_matrix[0][self._get_vic_index(self.victims[x][0])],
                         reverse=True)
        population.append(victims)

        # Heuristic 2: Modified Nearest Neighbor
        nn_seq = self._nearest_neighbor(self.vic_ids)
        population.append(nn_seq)

        # Heuristic 3: Geographic clustering
        clustered = sorted(self.vic_ids,
                           key=lambda x: self.victims[x][0][0] + self.victims[x][0][1])
        population.append(clustered)

        # Fill with random combinations
        for _ in range(self.pop_size - 3):
            population.append(random.sample(self.vic_ids, len(self.vic_ids)))

        return population

    def _nearest_neighbor(self, victims_list):
        """Nearest neighbor algorithm starting from base."""
        unvisited = victims_list.copy()
        sequence = []
        current_pos = self.base

        while unvisited:
            nearest = None
            min_cost = float('inf')

            for vic_id in unvisited:
                pos = self.victims[vic_id][0]
                cost = self.distance_matrix[self._get_vic_index(current_pos)][self._get_vic_index(pos)]
                if cost < min_cost:
                    min_cost = cost
                    nearest = vic_id

            if nearest is None:
                break  # No path found

            sequence.append(nearest)
            current_pos = self.victims[nearest][0]
            unvisited.remove(nearest)

        return sequence

    def _select_parents(self, population, fitnesses):
        # Tournament selection with variable selective pressure
        tournament_size = max(2, int(len(population)*0.1))
        selected = []
        for _ in range(2):
            candidates = random.sample(list(zip(population, fitnesses)), tournament_size)
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected.append(candidates[0][0])
        return selected

    def _edge_recombination_crossover(self, parent1, parent2):
        # Implement edge recombination (better for TSP)
        edge_table = defaultdict(set)
        for seq in [parent1, parent2]:
            for i, vic in enumerate(seq):
                prev = seq[i-1] if i > 0 else None
                next = seq[i+1] if i < len(seq)-1 else None
                edge_table[vic].update({prev, next})

        child = [parent1[0]]  # Start with first element
        current = child[0]
        while len(child) < len(parent1):
            neighbors = edge_table[current]
            best_neighbor = None
            for neighbor in neighbors:
                if neighbor not in child and neighbor is not None:
                    best_neighbor = neighbor
                    break
            if best_neighbor is None:
                remaining = [vic for vic in parent1 if vic not in child]
                best_neighbor = random.choice(remaining)
            child.append(best_neighbor)
            current = best_neighbor
        return child

    def _mutate(self, sequence):
        # Adaptive mutation based on progress
        if random.random() < self.mutation_rate:
            # 50% chance for 2-opt
            if random.random() < 0.5:
                i, j = sorted(random.sample(range(len(sequence)), 2))
                sequence[i:j+1] = reversed(sequence[i:j+1])
            # 50% for swap of critical elements
            else:
                criticals = [i for i, vid in enumerate(sequence) if self.victims[vid][1][7] == 1]
                if len(criticals) >= 2:
                    i, j = random.sample(criticals, 2)
                    sequence[i], sequence[j] = sequence[j], sequence[i]
        return sequence

    def run(self):
        """Execute the GA and return the best sequence."""
        population = self._generate_initial_population()
        best_sequence = None
        best_fitness = -float('inf')

        for _ in range(self.generations):
            fitnesses = [self._fitness(seq) for seq in population]

            # Track best sequence
            current_best_idx = np.argmax(fitnesses)
            if fitnesses[current_best_idx] > best_fitness:
                best_fitness = fitnesses[current_best_idx]
                best_sequence = population[current_best_idx]

            # Generate next generation
            new_population = [best_sequence.copy()]  # Elitism

            while len(new_population) < self.pop_size:
                parent1, parent2 = self._select_parents(population, fitnesses)
                child = self._edge_recombination_crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

        return best_sequence