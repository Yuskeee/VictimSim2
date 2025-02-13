import random
import numpy as np
from collections import defaultdict 
from dijkstra import Dijkstra
from vs.constants import VS  # Import VS constants for wall and clear values

class MockMap:
    """Mock Map class for testing the GeneticSequencer."""
    def __init__(self):
        # Mock data for the map
        self.data = {
            (0, 0): (1, VS.NO_VICTIM, [VS.END, VS.END, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.END, VS.END]),
            (10, 10): (1, VS.NO_VICTIM, [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR]),
            (20, 20): (1, VS.NO_VICTIM, [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR]),
        }

        # Ensure connectivity between points
        self._ensure_connectivity()

    def _ensure_connectivity(self):
        """Ensure that all points in the map are connected."""
        # Add intermediate points to connect (0, 0) to (10, 10) and (10, 10) to (20, 20)
        for x in range(1, 10):
            self.data[(x, 0)] = (1, VS.NO_VICTIM, [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR])
            self.data[(x, 10)] = (1, VS.NO_VICTIM, [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR])
        for y in range(1, 10):
            self.data[(10, y)] = (1, VS.NO_VICTIM, [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR])
        for x in range(11, 20):
            self.data[(x, 10)] = (1, VS.NO_VICTIM, [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR])
            self.data[(x, 20)] = (1, VS.NO_VICTIM, [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR])
        for y in range(11, 20):
            self.data[(20, y)] = (1, VS.NO_VICTIM, [VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR, VS.CLEAR])

    def in_map(self, pos):
        """Check if a position is within the map."""
        return pos in self.data

    def get_difficulty(self, pos):
        """Get the difficulty of a position."""
        return self.data[pos][0] if pos in self.data else VS.OBST_WALL

    def get_actions_results(self, pos):
        """Get the results of actions from a position."""
        return self.data[pos][2] if pos in self.data else [VS.END] * 8

    def get(self, coord):
        """Get the difficulty, victim ID, and action results for a coordinate."""
        if coord in self.data:
            return self.data[coord]
        else:
            return (VS.OBST_WALL, VS.NO_VICTIM, [VS.END] * 8)

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
        """Precompute distances between all pairs of victims and the base."""
        nodes = [self.base] + [self.victims[vid][0] for vid in self.vic_ids]
        n = len(nodes)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    path, cost = self.dijkstra.calc_plan(nodes[i], nodes[j])
                    distance_matrix[i][j] = cost if cost != -1 else float('inf')
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
        current_pos = self.base
        saved_severity = 0
        COST_FIRST_AID = 5.0  # Example value; replace with actual from rescuer

        for vic_id in sequence:
            next_pos = self.victims[vic_id][0]
            idx_current = self._get_vic_index(current_pos)
            idx_next = self._get_vic_index(next_pos)
            path_time = self.distance_matrix[idx_current][idx_next]
            rescue_time = COST_FIRST_AID

            if total_time + path_time + rescue_time > self.Ts:
                break  # Cannot save this victim

            total_time += path_time + rescue_time
            saved_severity += self.victims[vic_id][1][6]  # Severity value
            current_pos = next_pos

        # Add return to base time if possible
        return_time = self.distance_matrix[self._get_vic_index(current_pos)][0]
        if total_time + return_time <= self.Ts:
            total_time += return_time
        else:
            saved_severity *= 0.5  # Penalize for not returning

        return saved_severity

    def _generate_initial_population(self):
        # Greedy heuristic: Prioritize victims with highest severity/time ratio
        victims = []
        for vic_id in self.vic_ids:
            pos = self.victims[vic_id][0]
            time_to_vic = self.distance_matrix[0][self._get_vic_index(pos)]
            severity = self.victims[vic_id][1][6]
            if time_to_vic > 0:
                ratio = severity / time_to_vic
            else:
                ratio = severity
            victims.append((vic_id, ratio))
        
        sorted_victims = sorted(victims, key=lambda x: x[1], reverse=True)
        heuristic_seq = [vic[0] for vic in sorted_victims]
        
        population = [heuristic_seq]
        # Add random sequences for diversity
        for _ in range(self.pop_size - 1):
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
        """Tournament selection."""
        selected = []
        for _ in range(2):
            candidates = random.sample(list(zip(population, fitnesses)), 3)
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
        # Apply 2-opt local search
        if random.random() < self.mutation_rate:
            i, j = sorted(random.sample(range(len(sequence)), 2))
            sequence[i:j+1] = reversed(sequence[i:j+1])
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

# Example usage (test in isolation)
if __name__ == "__main__":
    # Mock data (replace with actual data from rescuer)
    mock_cluster = {
        1: ((10, 10), [0, 0, 0, 0, 0, 0, 0, 1]),  # (pos, vitals including class 1)
        2: ((20, 20), [0, 0, 0, 0, 0, 0, 0, 3]),  # class 3
    }
    
    sequencer = GeneticSequencer(mock_cluster, MockMap(), population_size=20, generations=50)
    best_seq = sequencer.run()
    print("Best sequence:", best_seq)