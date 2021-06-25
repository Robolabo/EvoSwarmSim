
import numpy as np
from scipy.linalg import expm
from .population import Population
from spike_swarm_sim.utils import eigendecomposition, normalize

class OpenAI_ES_Population(Population):
    def __init__(self, *args, **kwargs):
        super(OpenAI_ES_Population, self).__init__(*args, **kwargs)
        self.learning_rate = None
        self.sigma = None

        self.mu = None
        self.z_samples = None
        
    def sample(self):
        # sample = np.random.multivariate_normal(np.zeros_like(self.mu), np.eye(len(self.mu)), size=self.pop_size)
        sample = np.array([np.random.randn(len(self.mu)) for _ in range(self.pop_size)])
        return (self.mu + self.sigma * sample, sample)
        
    def step(self, fitness_vector, generation):
        fitness_order = np.argsort(fitness_vector.copy())[::-1]
        ord_samples = [self.z_samples[idx].copy() for idx in fitness_order]
        ord_fitness = np.array([fitness_vector[idx] for idx in fitness_order])

        utilities = np.array([((max(0, np.log(1 + 0.5 * len(self.population)) - np.log(i + 1)))\
                    / np.sum([max(0, np.log(1 + 0.5 * len(self.population)) - np.log(j + 1))\
                    for j in range(len(self.population))]))\
                    for i in range(len(self.population))])
        utilities -= 1 / len(self.population)

        # utilities = (ord_fitness / ord_fitness[0]) - 0.5 #! IF no utilities
        # # import pdb; pdb.set_trace()
        # if any(np.array(fitness_vector) > 1.):
        #     ord_fitness = normalize(ord_fitness)
        # utilities = ord_fitness.copy() #! Quitar si usamos utitilies
        # import pdb; pdb.set_trace()

        #* --- Update distribution -- *#
        self.mu += (self.learning_rate / (self.sigma * len(ord_fitness))) * np.sum([ui * sample \
                for ui, sample in zip(utilities, ord_samples)], 0)
        self.mu = np.clip(self.mu, a_min=0, a_max=1)
        # self.learning_rate = max(5e-3, 0.999 * self.learning_rate) #* decay learning rate with gens
        # self.sigma = max(0.01, 0.999 * self.sigma) #* decay sigma with gens

        #* --- Sample New population -- *#
        self.population, self.z_samples = self.sample()
        # self.population = [np.clip(v, a_min=self.min_vector, a_max=self.max_vector) for v in self.population]
        self.population = [np.clip(v, a_min=0, a_max=1) for v in self.population]

    def initialize(self, interface):
        # import pdb; pdb.set_trace()
        self.segment_lengths = [interface.submit_query(query, primitive='LEN') for query in self.objects]
        genotype_length = interface.toGenotype(self.objects, self.min_vector, self.max_vector).shape[0]
        # self.mu = np.random.uniform(low=self.min_vector, high=self.max_vector, size=genotype_length)
        np.random.seed()

        self.sigma = 0.05
        self.learning_rate = 1
        self.mu = 0.5 * np.ones(genotype_length)
        # import pdb; pdb.set_trace()
        #* Use larger sigma at first for better initialization
        d = self.mu.shape[0]
        self.population, self.z_samples = self.sample()
        self.population = [np.clip(v, a_min=self.min_vector, a_max=self.max_vector) for v in self.population]
