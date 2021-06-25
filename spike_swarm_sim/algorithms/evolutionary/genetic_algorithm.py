import copy
import logging
import numpy as np
from .population import Population
from .evolutionary_algorithm import EvolutionaryAlgorithm
from spike_swarm_sim.algorithms.interfaces import GeneticInterface
from spike_swarm_sim.register import algorithm_registry
from spike_swarm_sim.utils import save_pickle, load_pickle

@algorithm_registry(name='GA')
class GeneticAlgorithm(EvolutionaryAlgorithm):
    """ Class of the Canonical Genetic Algorithm. The evolution step is defined in the Population class.
    """
    def __init__(self, populations, *args, **kwargs):
        populations = {name : Population(kwargs['population_size'],\
                pop['min_vals'], pop['max_vals'], pop['objects'],\
                **pop['params']) for name, pop in populations.items()}
                # encoding=pop['encoding'], selection_operator=pop['selection_operator'],\
                # crossover_operator=pop['crossover_operator'], mutation_operator=pop['mutation_operator'],\
                # mating_operator=pop['mating_operator'], mutation_prob=pop['mutation_prob'],\
                # crossover_prob=pop['crossover_prob'], num_elite=pop['num_elite'],)\
                # for name, pop in populations.items()}
        super(GeneticAlgorithm, self).__init__(populations, *args, **kwargs)

    def save_population(self, generation):
        """ Saves the checkpoint with the necessary information to resume the 
        evolution.
        """
        pop_checkpoint = {
            'populations' : {name : np.stack(pop.population) for name, pop in self.populations.items()},
            'generation' : generation,
            'mutation_prob' : {name : pop.mutation_prob for name, pop in self.populations.items()},
            'evolution_hist' : self.evolution_history,
        }
        file_name = 'spike_swarm_sim/checkpoints/populations/' + self.checkpoint_name
        save_pickle(pop_checkpoint, file_name)
        logging.info('Successfully saved evolution checkpoint.')
        
    def load_population(self):
        """ Loads a previously saved checkpoint to resume evolution.
        """
        checkpoint = load_pickle('spike_swarm_sim/checkpoints/populations/' + self.checkpoint_name)
        logging.info('Resuming GA evolution using checkpoint ' +  self.checkpoint_name)
        #! for robot in self.world.robots:
        #!    robot.controller.neural_net.graph = checkpoint['ann_graph']
        for name, pop in checkpoint['populations'].items():
            self.populations[name].population = [v.copy() for v in checkpoint['populations'][name]]
            self.populations[name].mutatation_prob = checkpoint['mutation_prob'][name]
            robots = [copy.deepcopy(robot) for robot in self.world.robots.values()]
            interface = GeneticInterface(robots[0].controller.neural_network)
            self.populations[name].segment_lengths = [interface.submit_query(query, primitive='LEN')\
                                            for query in self.populations[name].objects]
        self.init_generation = checkpoint['generation']
        self.evolution_history = checkpoint['evolution_hist']