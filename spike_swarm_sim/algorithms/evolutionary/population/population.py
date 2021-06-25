import numpy as np
from spike_swarm_sim.register import evo_operators

def exp_schedule(value, min_value, tau=20, dt=1):
    return value - (dt/tau)*(value-min_value)

class Population:
    def __init__(self, pop_size, min_vals, max_vals, objects,
                 encoding='real', selection_operator='roulette',
                 crossover_operator='multipoint', mutation_operator='gaussian',
                 mating_operator='random', mutation_prob=0.05, crossover_prob=0.9, num_elite=5,):
        self.population = []
        self.segment_lengths = []
        self.best = None
        self.objects = objects
        self.pop_size = pop_size
        self.max_vals = max_vals if isinstance(max_vals, list) else [max_vals for _ in objects]
        self.min_vals = min_vals if isinstance(min_vals, list) else [min_vals for _ in objects]
        self.encoding = encoding
        self.selection_operator = evo_operators[selection_operator + '_selection']
        self.crossover_operator = evo_operators[crossover_operator + '_crossover']
        self.mutation_operator = evo_operators[mutation_operator + '_mutation']
        self.mating_operator = evo_operators[mating_operator + '_mating']
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.num_elite = num_elite
    
    def step(self, fitness_vector, generation):
        #* --- Apply Selection operator ---
        parents, parents_fitness = self.selection_operator(self.population.copy(), fitness_vector.copy(),\
                    len(self) - self.num_elite)
        #* --- Apply Mating operator ---
        parents = self.mating_operator(parents, parents_fitness)
         
        #* --- Apply Crossover operator ---
        offspring = self.crossover_operator(parents, crossover_prob=self.crossover_prob)
        
        #* --- Apply mutation operator ---
        mutation_sigma = 0.1 #(self.max_vector - self.min_vector).copy() / 6
        offspring = self.mutation_operator(offspring, mutation_prob=self.mutation_prob, sigma=mutation_sigma,\
                            max_vals=1, min_vals=0)

        #* --- Constrain values ---
        if self.encoding == 'real':
            offspring = [np.clip(v, a_min=0, a_max=1) for v in offspring]

        #* --- Save elite based on highest fitness ---
        fitness_order = np.argsort(fitness_vector.copy())[::-1]
        elite = [self.population[idx].copy() for idx in fitness_order[:self.num_elite]]

        #* --- Update new population ---
        self.population = elite + offspring

        # Dynamic Mutation Prob.
        # self.mutation_prob = exp_schedule(self.mutation_prob, 0.01)

    def initialize(self, interface):
        self.segment_lengths = [interface.submit_query(query, primitive='LEN') for query in self.objects]
        for _ in range(self.pop_size):
            interface.initGenotype(self.objects, self.min_vals, self.max_vals)
            self.population.append(interface.toGenotype(self.objects, self.min_vals, self.max_vals))

    def __len__(self):
        return len(self.population)

    @property
    def genotype_length(self):
        return self.population[0].shape[0]
    
    @property
    def min_vector(self):
        return np.hstack([min_val * np.ones(seg_len) for seg_len, min_val in zip(self.segment_lengths, self.min_vals)])

    @property
    def max_vector(self):
        return  np.hstack([max_val * np.ones(seg_len) for seg_len, max_val in zip(self.segment_lengths, self.max_vals)])

  