import logging
import numpy as np
from .population import CMA_EA_Population  
from .evolutionary_algorithm import EvolutionaryAlgorithm
from spike_swarm_sim.register import algorithm_registry
from spike_swarm_sim.utils import save_pickle, load_pickle

@algorithm_registry(name='CMA-ES')
class CMA_ES(EvolutionaryAlgorithm):
    """ Class of the CMA-ES.
    The evolution step is defined in the CMA_EA_Population class.
    """
    def __init__(self, populations, *args, **kwargs):
        populations = {name : CMA_EA_Population(kwargs['population_size'],\
                            pop['min_vals'], pop['max_vals'], pop['objects'], **pop['params'])\
                            for name, pop in populations.items()}
        super(CMA_ES, self).__init__(populations, *args, **kwargs)

    def save_population(self, generation):
        """ Saves the checkpoint with the necessary information to resume the evolution. 
        """
        pop_checkpoint = {
            'populations' : {name : np.stack(pop.population) for name, pop in self.populations.items()},
            'generation' : generation,
            'mutation_prob' : {name : pop.mutation_prob for name, pop in self.populations.items()},
            'evolution_hist' : self.evolution_history,
            'mu' : {name : pop.strategy_m for name, pop in self.populations.items()},
            'C' :{name : pop.strategy_C for name, pop in self.populations.items()},
            'cc' :{name : pop.cc for name, pop in self.populations.items()},
            'cs' :{name : pop.cs for name, pop in self.populations.items()},
            'c_cov' :{name : pop.c_cov for name, pop in self.populations.items()},
            'mu_cov':{name : pop.mu_cov for name, pop in self.populations.items()},
            'ds':{name : pop.ds for name, pop in self.populations.items()},
            'evo_path':{name : pop.evo_path for name, pop in self.populations.items()},
            'ps':{name : pop.ps for name, pop in self.populations.items()},
            'B':{name : pop.B for name, pop in self.populations.items()},
            'Bt' :{name : pop.Bt for name, pop in self.populations.items()},
            'D' : {name : pop.D for name, pop in self.populations.items()},
            'sigma' : {name : pop.sigma for name, pop in self.populations.items()},
            'num_evals' :{name : pop.num_evals for name, pop in self.populations.items()},
        }
        file_name = 'spike_swarm_sim/checkpoints/populations/' + self.checkpoint_name
        save_pickle(pop_checkpoint, file_name)
        logging.info('Successfully saved evolution checkpoint.')
        
    def load_population(self):
        """ Loads a previously saved checkpoint to resume evolution.
        """
        checkpoint = load_pickle('spike_swarm_sim/checkpoints/populations/' + self.checkpoint_name)
        logging.info('Resuming CMA-ES evolution using checkpoint ' +  self.checkpoint_name)
        key = tuple(self.populations.keys())[0]
        for key, pop in checkpoint['populations'].items():
            self.populations[key].strategy_m = checkpoint['mu'][key]
            self.populations[key].strategy_C = checkpoint['C'][key]
            self.populations[key].cc = checkpoint['cc'][key]
            self.populations[key].cs = checkpoint['cs'][key]
            self.populations[key].mu_cov = checkpoint['mu_cov'][key]
            self.populations[key].c_cov = checkpoint['c_cov'][key]
            self.populations[key].ds = checkpoint['ds'][key]
            self.populations[key].evo_path = checkpoint['evo_path'][key]
            self.populations[key].ps = checkpoint['ps'][key]
            self.populations[key].B = checkpoint['B'][key]
            self.populations[key].Bt = checkpoint['Bt'][key]
            self.populations[key].D = checkpoint['D'][key]
            self.populations[key].sigma = checkpoint['sigma'][key]
            self.populations[key].num_evals = checkpoint['num_evals'][key]
            self.populations[key].population = self.populations[key].sample()
        self.init_generation = checkpoint['generation']
        self.evolution_history = checkpoint['evolution_hist']