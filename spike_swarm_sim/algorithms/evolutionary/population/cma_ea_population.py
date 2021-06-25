import numpy as np
from .population import Population
from spike_swarm_sim.utils import eigendecomposition, normalize

class CMA_EA_Population(Population):
    def __init__(self, *args, **kwargs):
        super(CMA_EA_Population, self).__init__(*args, **kwargs)
        self.mu = int(.5 * self.pop_size)
        self.weights = np.log(self.mu+.5) - np.log(np.arange(1, self.mu)) 
        self.weights /= self.weights.sum()
        self.mu_eff = 1/np.linalg.norm(self.weights)**2
        # Step sizes 
        self.sigma = 1 
        self.cc, self.cs, self.mu_cov, self.c_cov, self.ds = None, None, None, None, None


        # Strategy Parameters
        self.strategy_m = []
        self.strategy_C = [] 
        self.ps = []
        self.evo_path = []
        self.B, self.D, self.Bt = [], [], []
        self.num_evals = 0 #! MUCHO OJO CON LOAD

    def _sample(self):
        sample = np.random.multivariate_normal(np.zeros_like(self.strategy_m), np.eye(*self.strategy_C.shape))
        return self.strategy_m + self.sigma * self.B.dot(self.D).dot(sample)
        
    def sample(self):
        return [self._sample() for _ in range(self.pop_size)]
        
    def step(self, fitness_vector):
        self.num_evals += len(fitness_vector)
        selected, selected_fitness = self.selection_operator(self.population.copy(),\
                                fitness_vector.copy(), self.mu)
        fitness_order = np.argsort(fitness_vector.copy())[::-1]     
        selected = [self.population[idx].copy() for idx in fitness_order[:self.mu]]
        
        # Update mean vector
        old_mean = self.strategy_m.copy()
        self.strategy_m = np.sum([w_i * x_i for w_i, x_i in zip(self.weights, selected)], 0)
        # if self.num_evals % (self.pop_size / (self.c1 + self.cmu)/self.strategy_m.shape[0]/10) == 0:
        self.ps = (1-self.cs)*self.ps + np.sqrt(self.cs*(2-self.cs)*self.mu_eff)\
                * (self.B.dot(self.D).dot(self.B.T)) * ((self.strategy_m-old_mean)/self.sigma)
                
        hsig = np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*self.num_evals/self.pop_size))\
               < 1.4 + 2 / ((self.strategy_m.shape[0]+1)) * self.chiN

        self.evo_path = (1-self.cc) * self.evo_path +\
                        hsig*np.sqrt(self.cc*(2-self.cc)*self.mu_eff)\
                        *  ((self.strategy_m-old_mean)/self.sigma)
        # Update Covariance matrix
        y_vec = [(v - old_mean) / self.sigma for v in selected]
        # self.strategy_C = (1-self.c_cov) * self.strategy_C\
        #                 + self.c_cov *((1 / self.mu_cov) * (np.outer(self.evo_path, self.evo_path))\
        #                 + (1/(1-self.mu_cov))*np.sum([w_i*np.outer(y_i, y_i)\
        #                 for w_i, y_i in zip(self.weights, y_vec)], 0))
        self.strategy_C = (1-self.c1-self.c_mu) * self.strategy_C\
                        + self.c1 * np.outer(self.evo_path, self.evo_path)\
                        + self.c_mu*np.sum([w_i*np.outer(y_i, y_i)\
                        for w_i, y_i in zip(self.weights, y_vec)], 0)
        # Update sigma
        self.sigma = self.sigma * np.exp(\
                min(0, (self.cs/self.ds) * (np.linalg.norm(self.ps)/self.chiN-1))) 

        self.strategy_C = np.triu(self.strategy_C) + np.triu(self.strategy_C).T    
        self.D, self.B = np.linalg.eig(self.strategy_C)
        self.D = np.diag(self.D**-.5)
        # Finally sample new population
        self.population = self.sample() 
        self.population = [np.clip(v, a_min=self.min_vector, a_max=self.max_vector) for v in self.population] 

    def initialize(self, interface):
        self.segment_lengths = [interface.submit_query(query, primitive='LEN') for query in self.objects]
        genotype_length = interface.toGenotype(self.objects).shape[0]
        self.strategy_m = self.min_vector + np.random.random(size=genotype_length) * (self.max_vector-self.min_vector)
        self.strategy_C = np.eye(genotype_length)
        n = self.strategy_m.shape[0]

         
        self.cc = (4 + self.mu_eff/n) / (4 + n + 2*self.mu_eff/n)#4/(n+4) #
        self.cs = (self.mu_eff + 2) / (5+n+self.mu_eff)
        self.c1 = 2 / (self.mu_eff + (n+1.3)**2)  #2 / n ** 2 
        self.c_mu = 1e-4 # 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((n+2)**2 + 2*self.mu_eff/2)
        self.ds = 2 * self.mu_eff/self.pop_size + 0.3 + self.cs #
        self.c_cov = 1e-3 #2 / (n + 2**.5)**2
        self.mu_cov = self.mu
        self.chiN = np.sqrt(n) * (1- 1/(4*n) + 1/(21*n**2)) 


        # Dynamics Initialization
        self.evo_path = np.zeros_like(self.strategy_m)
        self.ps = np.zeros_like(self.strategy_m)
        self.B = np.eye(self.strategy_m.shape[0])
        self.Bt = np.eye(self.strategy_m.shape[0])
        self.D = np.eye(self.strategy_m.shape[0])
        self.population = [self._sample() for _ in range(self.pop_size)]   
        self.population = [np.clip(v, a_min=self.min_vector, a_max=self.max_vector) for v in self.population] 