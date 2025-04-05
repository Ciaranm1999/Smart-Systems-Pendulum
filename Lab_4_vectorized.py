import numpy as np
import random
import time
import multiprocessing
from Digital_twin import DigitalTwin
from hyperparameters import hyperparameters

hp = hyperparameters()

class InvertedPendulumGA:
    def __init__(self, population_size, num_actions, simulation_duration, action_resolution, simulation_delta_t):
        self.action_resolution = simulation_duration / (num_actions + 1)
        self.digital_twin = DigitalTwin()
        self.population_size = population_size
        self.parent_pool_size = 20
        self.num_actions = num_actions
        self.simulation_duration = simulation_duration
        self.action_resolution = action_resolution
        self.simulation_delta_t = simulation_delta_t
        self.simulation_steps = int(simulation_duration / simulation_delta_t)
        self.num_steps = int(simulation_duration / action_resolution)
        self.step_resolution = int(action_resolution / simulation_delta_t)
        self.population = [self.create_individual() for _ in range(population_size)]
        fitness_scores = self.evaluate_population()
        print(fitness_scores, "at start")

    def create_individual(self):
        actions = np.zeros(self.num_steps, dtype=int)
        net_movement = 0
        for i in range(self.num_steps):
            if abs(net_movement) < 100:
                action = np.random.randint(1, self.num_actions)
            elif net_movement >= 100:
                action = np.random.choice([1, 2, 3, 4])
            else:
                action = np.random.choice([5, 6, 7, 8])
            actions[i] = action
            if action in [1, 2, 3, 4]:
                net_movement -= self.digital_twin.action_map[action][1]
            else:
                net_movement += self.digital_twin.action_map[action - 4][1]
        return actions

    def simulate(self, actions):
        self.digital_twin.theta = 0.0
        self.digital_twin.theta_dot = 0.0
        self.digital_twin.x_pivot = 0.0
        self.digital_twin.steps = 0.0
        max_score = 0.0

        for step in range(self.simulation_steps):
            if step % self.step_resolution == 0 and step // self.step_resolution < len(actions):
                action = actions[step // self.step_resolution]
                direction, duration = self.digital_twin.action_map[action]
                self.digital_twin.perform_action(direction, duration)
            theta, theta_dot, x_pivot = self.digital_twin.step()
            max_score = max(max_score, abs(theta))
            if abs(self.digital_twin.x_pivot) > 0.135:
                return -100
        return max_score

    def evaluate_population(self):
        with multiprocessing.Pool() as pool:
            fitness_scores = pool.map(self.simulate, self.population)
        return fitness_scores

    def select_parents(self, fitness_scores):
        pool_size = min(self.parent_pool_size, len(fitness_scores))
        top_performers_indices = np.argsort(fitness_scores)[-pool_size:]
        return [self.population[i] for i in top_performers_indices]

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.num_steps - 1)
        offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        return offspring

    def mutate(self, individual, mutation_rate=0.3):
        mask = np.random.rand(self.num_steps) < mutation_rate
        individual[mask] = np.random.randint(0, self.num_actions, size=np.sum(mask))
        return individual

    def run_generation(self, num_elites=2):
        fitness_scores = self.evaluate_population()
        parents_pool = self.select_parents(fitness_scores)
        new_population = [parents_pool[-(i + 1)] for i in range(num_elites)]
        np.random.shuffle(parents_pool)
        while len(new_population) < self.population_size:
            for i in range(0, len(parents_pool) - 1, 2):
                offspring1 = self.crossover(parents_pool[i], parents_pool[i + 1])
                offspring2 = self.crossover(parents_pool[i + 1], parents_pool[i])
                new_population.extend([self.mutate(offspring1), self.mutate(offspring2)])
                if len(new_population) >= self.population_size:
                    break
        self.population = new_population[:self.population_size]

    def optimize(self, num_generations, fitness_threshold):
        for i in range(num_generations):
            self.run_generation()
            fitness_scores = self.evaluate_population()
            best_index = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_index]
            print(f"Generation: {i}, Best Fitness: {best_fitness}")
            if best_fitness >= fitness_threshold:
                print(f"Stopping early: Individual found with fitness {best_fitness} meeting the threshold at generation {i}.")
                return self.population[best_index]
        print(f"No individual met the fitness threshold. Best fitness after {num_generations} generations is {best_fitness}.")
        return self.population[best_index]

    def inject_elite(self, elite):
        self.population[0] = np.array(elite)
        self.evaluate_population()
if __name__ == '__main__':
    ga = InvertedPendulumGA(population_size=2, num_actions=9, simulation_duration=6, action_resolution=0.4, simulation_delta_t=hp.DELTA_T)
    best_solution = ga.optimize(num_generations=1, fitness_threshold=np.pi)
    print("Best Solution:", ", ".join(map(str, best_solution)))