import logging
import copy
import numpy as np
from .population import OpenAI_ES_Population 
from .evolutionary_algorithm import EvolutionaryAlgorithm
from spike_swarm_sim.algorithms.interfaces import GeneticInterface
from spike_swarm_sim.register import algorithm_registry
from spike_swarm_sim.utils import save_pickle, load_pickle

@algorithm_registry(name='openaiES')
class OpenAI_ES(EvolutionaryAlgorithm):
    def __init__(self, populations, *args, **kwargs):
        populations = {name : OpenAI_ES_Population(kwargs['population_size'], \
                pop['min_vals'], pop['max_vals'], pop['objects'], **pop['params'])\
                for name, pop in populations.items()}
        super(OpenAI_ES, self).__init__(populations, *args, **kwargs)

    def save_population(self, generation):
        pop_checkpoint = {
            'populations' : {name : np.stack(pop.population) for name, pop in self.populations.items()},
            'generation' : generation,
            'mutation_prob' : {name : pop.mutation_prob for name, pop in self.populations.items()},
            'evolution_hist' : self.evolution_history,
            'mu' : {name : pop.mu for name, pop in self.populations.items()},
            'sigma' : {name : pop.sigma for name, pop in self.populations.items()},
            'learning_rate' :{name : pop.learning_rate for name, pop in self.populations.items()},
        }
        file_name = 'spike_swarm_sim/checkpoints/populations/' + self.checkpoint_name
        save_pickle(pop_checkpoint, file_name)
        logging.info('Successfully saved evolution checkpoint.')

    def load_population(self):
        checkpoint = load_pickle('spike_swarm_sim/checkpoints/populations/' + self.checkpoint_name)
        logging.info('Resuming OpenaiES evolution using checkpoint ' +  self.checkpoint_name)
        key = tuple(self.populations.keys())[0]
        for key, pop in checkpoint['populations'].items():
            self.populations[key].mu = checkpoint['mu'][key]
            self.populations[key].sigma = checkpoint['sigma'][key]
            self.populations[key].learning_rate = checkpoint['learning_rate'][key]
            robots = [copy.deepcopy(robot) for robot in self.world.robots.values()]
            interface = GeneticInterface(robots[0].controller.neural_network)
            self.populations[key].segment_lengths = [interface.submit_query(query, primitive='LEN')\
                        for query in self.populations[key].objects]
            # import pdb; pdb.set_trace()
            self.populations[key].mu = (self.populations[key].mu - self.populations[key].min_vector) / (self.populations[key].max_vector - self.populations[key].min_vector)
            # import pdb; pdb.set_trace()
            # self.sigma = 1e-3
            self.populations[key].population, self.populations[key].z_samples = self.populations[key].sample()
            self.populations[key].population = [np.clip(v, a_min=0, a_max=1) for v in self.populations[key].population]
        self.init_generation = checkpoint['generation']
        self.evolution_history = checkpoint['evolution_hist']