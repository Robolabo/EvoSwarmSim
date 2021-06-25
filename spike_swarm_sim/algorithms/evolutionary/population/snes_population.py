import numpy as np
from scipy.linalg import expm
from .population import Population
from spike_swarm_sim.utils import eigendecomposition, normalize

class SNES_Population(Population):
    """ Class of Separable Natural Evolution Strategy (SNES) Population defining 
    all the underlying algorithm steps.  
    """
    def __init__(self, *args, **kwargs):
        super(SNES_Population, self).__init__(*args, **kwargs)
        self.eta_mu = None
        self.eta_s = None
        self.mu = None
        self.sigma = None
        self.z_samples = None

    def sample(self):
        """ Sample all the genotypes of the population using the mu and sigma parameters of SNES and 
        a multivariate gaussian distribution.
        ==============================================================================================
        - Args: None
        - Returns:
            sampled genotypes [np.ndarray]
            genotypes in local coords [np.ndarray]
        ==============================================================================================
        """
        sample = np.array([np.random.randn(len(self.mu)) for _ in range(self.pop_size)])
        # sample = np.random.multivariate_normal(np.zeros_like(self.mu), np.eye(len(self.mu)), size=self.pop_size)
        return (self.mu + self.sigma * sample, sample)

    def step(self, fitness_vector, generation):
        """ SNES Evolution step applied at the end of each generation to update the population.
        ==================================================================================
        - Args:
            fitness_vector [np.ndarray or list]: array of computed fitness values.
        - Returns: None
        ==================================================================================
        """
        fitness_order = np.argsort(fitness_vector.copy())[::-1]
        ord_samples = [self.z_samples[idx].copy() for idx in fitness_order]
        ord_fitness = np.array([fitness_vector[idx] for idx in fitness_order])

        #* --- Compute utilities -- *#
        utilities = np.array([((max(0, np.log(1 + 0.5 * len(self.population)) - np.log(i+1)))\
                    / np.sum([max(0, np.log(1 + 0.5 * len(self.population)) - np.log(j+1))\
                    for j in range(len(self.population))]))\
                    for i in range(len(self.population))])
        utilities -= 1 / len(self.population)
        # utilities = ord_fitness
        #* --- Compute gradients -- *#
        grad_mu = np.dot(utilities, ord_samples)
        grad_sigma = np.dot(utilities, [sample ** 2 - 1  for sample in ord_samples])
        #* --- Update distribution -- *#
        self.mu += self.eta_mu * self.sigma * grad_mu
        self.sigma *= np.exp(.5 * self.eta_s * grad_sigma)
        self.mu = np.clip(self.mu, a_min=0., a_max=1.)
        self.sigma = np.clip(self.sigma, a_min=0, a_max=1.5)

        #* --- Sample New population -- *#
        self.population, self.z_samples = self.sample()
        self.population = [np.clip(v, a_min=0., a_max=1.) for v in self.population]

    def initialize(self, interface):
        """ Initializes the parameters and population of SNES.
        =====================================================================
        - Args:
            interface [GeneticInterface] : Phenotype to genotype interface of 
                Evolutionary algs.
        - Returns: None
        =====================================================================
        """
        self.segment_lengths = [interface.submit_query(query, primitive='LEN') for query in self.objects]
        genotype_length = interface.toGenotype(self.objects, self.min_vector, self.max_vector).shape[0]
        np.random.seed()
        #* Use larger sigma at first for better initialization
        self.mu = 0.5 * np.ones(genotype_length)
        self.mu = np.clip(self.mu, a_min=0., a_max=1.)
        self.sigma = 0.2 * np.ones(genotype_length) #0.2 * np.ones(genotype_length)
        d = self.mu.shape[0]
        n_expected = int(4 + np.floor(3 * np.log(d)))
        self.eta_mu = 1e-2
        self.eta_s = (3 + np.log(d)) / (5 * np.sqrt(d)) + 0.2 #!
        
        #* sample initial pop
        self.population, self.z_samples = self.sample()
        self.population = [np.clip(v, a_min=0., a_max=1.) for v in self.population]
        self.sigma = 0.1 * np.ones(genotype_length) # 0.1 * np.ones(genotype_length)
