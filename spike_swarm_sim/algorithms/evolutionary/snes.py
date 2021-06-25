import copy
import logging
import numpy as np
from .population import SNES_Population  
from .evolutionary_algorithm import EvolutionaryAlgorithm
from spike_swarm_sim.algorithms.interfaces import GeneticInterface
from spike_swarm_sim.register import algorithm_registry
from spike_swarm_sim.globals import global_states
from spike_swarm_sim.utils import save_pickle, load_pickle

@algorithm_registry(name='SNES')
class SNES(EvolutionaryAlgorithm):
    """ Class of the Separable Natural Evolution Strategy (SNES).  
    The evolution step is defined in the SNES_Population class.
    """
    def __init__(self, populations, *args, **kwargs):
        populations = {name : SNES_Population(kwargs['population_size'],\
                pop['min_vals'], pop['max_vals'], pop['objects'], **pop['params'])\
                for name, pop in populations.items()}
        super(SNES, self).__init__(populations, *args, **kwargs)

    def save_population(self, generation):
        """ Saves the checkpoint with the necessary information to resume the evolution. 
        """
        pop_checkpoint = {
            'populations' : {name : np.stack(pop.population) for name, pop in self.populations.items()},
            'generation' : generation,
            'mutation_prob' : {name : pop.mutation_prob for name, pop in self.populations.items()},
            'evolution_hist' : self.evolution_history,
            'mu' : {name : pop.mu for name, pop in self.populations.items()},
            'sigma' : {name : pop.sigma for name, pop in self.populations.items()},
            'eta_mu' :{name : pop.eta_mu for name, pop in self.populations.items()},
            'eta_s':{name : pop.eta_s for name, pop in self.populations.items()},
        }
        file_name = 'spike_swarm_sim/checkpoints/populations/' + self.checkpoint_name
        save_pickle(pop_checkpoint, file_name)
        logging.info('Successfully saved evolution checkpoint.')
        
    def load_population(self):
        """ Loads a previously saved checkpoint to resume evolution.
        """
        checkpoint = load_pickle('spike_swarm_sim/checkpoints/populations/' + self.checkpoint_name)
        logging.info('Resuming SNES evolution using checkpoint ' +  self.checkpoint_name)
        key = tuple(self.populations.keys())[0]
        for key, pop in checkpoint['populations'].items():
            self.populations[key].mu = checkpoint['mu'][key]
            self.populations[key].sigma = checkpoint['sigma'][key]
            self.populations[key].eta_mu = checkpoint['eta_mu'][key]
            self.populations[key].eta_s = checkpoint['eta_s'][key]
            robots = [copy.deepcopy(robot) for robot in self.world.robots.values()]
            interface = GeneticInterface(robots[0].controller.neural_network)
            self.populations[key].segment_lengths = [interface.submit_query(query, primitive='LEN')\
                        for query in self.populations[key].objects]
            if global_states.EVAL:
                self.populations[key].sigma = 1e-3 * np.ones_like(self.populations[key].sigma)
            # self.populations[key].mu = (self.populations[key].mu - self.populations[key].min_vector) / (self.populations[key].max_vector - self.populations[key].min_vector)
            self.populations[key].population, self.populations[key].z_samples = self.populations[key].sample()
            self.populations[key].population = [np.clip(v, a_min=0, a_max=1) for v in self.populations[key].population]
        self.init_generation = checkpoint['generation']
        self.evolution_history = checkpoint['evolution_hist']