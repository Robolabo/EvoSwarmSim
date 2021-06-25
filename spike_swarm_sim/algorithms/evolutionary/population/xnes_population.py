import numpy as np
from scipy.linalg import expm
from .population import Population
from spike_swarm_sim.utils import eigendecomposition, normalize

class xNES_Population(Population):
    def __init__(self, *args, **kwargs):
        super(xNES_Population, self).__init__(*args, **kwargs)
        self.eta_mu = None
        self.eta_s = None
        self.eta_B = None

        self.mu = None
        self.B = None
        self.sigma = None
        self.z_samples = None

    # def _sample(self):
    #     # self.sigma = 0.001
    #     sample = np.random.multivariate_normal(np.zeros_like(self.mu), np.eye(*self.B.shape))
    #     return self.mu + self.sigma * self.B.dot(sample), sample
        
    # def sample(self):
    #     return [self._sample() for _ in range(self.pop_size)]
    def sample(self):
        # self.sigma = 0.0001
        sample = np.random.multivariate_normal(np.zeros_like(self.mu), np.eye(len(self.mu)), size=self.pop_size)
        return (self.mu + self.sigma * np.array([self.B.dot(xx) for xx in sample]), sample)

    def step(self, fitness_vector):
        fitness_order = np.argsort(fitness_vector.copy())[::-1]
        ord_samples = [self.z_samples[idx].copy() for idx in fitness_order]
        ord_fitness = np.array([fitness_vector[idx] for idx in fitness_order])

        utilities = np.array([((max(0, np.log(1 + 0.5 * len(self.population)) - np.log(i+1)))\
                    / np.sum([max(0, np.log(1 + 0.5 * len(self.population)) - np.log(j+1))\
                    for j in range(len(self.population))]))\
                    for i in range(len(self.population))]) - 1/len(self.population)
        # if any(np.array(fitness_vector) > 1.):
        #     ord_fitness = normalize(ord_fitness)
        # utilities = ord_fitness.copy() #! Quitar si usamos utitilies

        #! --- quitar ----
        # self.sigma = 0.1
        # self.eta_mu = 1
        # self.eta_s = 1e-3
        # self.eta_B = 1e-3
        #! ---------------

        # G_delta = np.sum([u_i * sample for u_i, sample in zip(utilities, ord_samples)], 0)
        G_delta = np.dot(utilities, ord_samples)
        G_M = np.sum([u_i * (sample[np.newaxis].T.dot(sample[np.newaxis]) - np.eye(len(sample))) \
                    for u_i, sample in zip(utilities, ord_samples)], 0)
        G_s = np.trace(G_M) / self.genotype_length
        G_B = G_M - G_s * np.eye(G_M.shape[0])
        
        #* --- Update distribution -- *#
        self.mu += self.eta_mu * self.sigma * self.B.dot(G_delta)
        self.mu = np.clip(self.mu, a_min=0., a_max=1.)

        self.sigma *= np.exp(.5 * self.eta_s * G_s)
        self.B = self.B.dot(expm(.5 * self.eta_B * G_B))

        #* --- Sample New population -- *#
        self.population, self.z_samples = self.sample()
        self.population = [np.clip(v, a_min=0., a_max=1.) for v in self.population]

    def initialize(self, interface):
        
        self.segment_lengths = [interface.submit_query(query, primitive='LEN') for query in self.objects]
        genotype_length = interface.toGenotype(self.objects, self.min_vector, self.max_vector).shape[0]
        # self.mu = np.random.uniform(low=self.min_vector, high=self.max_vector, size=genotype_length)
        np.random.seed()
        #* Use larger sigma at first for better initialization
        self.mu = 0.5 * np.ones(genotype_length) #(self.max_vector + self.min_vector).copy() + np.random.randn(genotype_length) * 1.
        self.mu = np.clip(self.mu, a_min=0., a_max=1.)
        
        self.B = np.eye(genotype_length) # np.diag(self.max_vector - self.min_vector)
        d = self.mu.shape[0]
        self.eta_mu = 1.
        self.eta_s = 0.6 * (3 + np.log(d)) / (d * np.sqrt(d))
        self.eta_B = self.eta_s
        
        #* sample initial pop
        self.sigma = 0.1
        self.population, self.z_samples = self.sample()
        self.population = [np.clip(v, a_min=0., a_max=1.) for v in self.population]
        self.sigma = 0.1
