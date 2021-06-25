import copy
import logging
import numpy as np
from .population import NEAT_Population  
from .evolutionary_algorithm import EvolutionaryAlgorithm
from spike_swarm_sim.algorithms.interfaces import NEATInterface
from spike_swarm_sim.register import algorithm_registry
from spike_swarm_sim.globals import global_states
from spike_swarm_sim.utils import save_pickle, load_pickle
from .species import Species

@algorithm_registry(name='NEAT')
class NEAT(EvolutionaryAlgorithm):
    """ 
    """
    def __init__(self, populations, *args, **kwargs):
        populations = {name : NEAT_Population(kwargs['population_size'],\
                pop['min_vals'], pop['max_vals'], pop['objects'], **pop['params'])\
                for name, pop in populations.items()}
        super(NEAT, self).__init__(populations, *args, **kwargs)

    def save_population(self, generation):
        #!!!!OJO SAVE TOPOLOGY
        """ Saves the checkpoint with the necessary information to resume the evolution. 
        """
        pop_checkpoint = {
            'populations' : {name : {
                'best' : pop.best,
                'genotypes' : pop.population, # List of dicts
                'current_innovation' : pop.current_innovation,
                'innovation_history' : pop.innovation_history,
                'input_nodes' : pop.input_nodes,
                'species_count' : pop.species_count,
                'species' : [{
                    'id' : spc.id,
                    'creation_generation' : spc.creation_generation,
                    'history' : copy.deepcopy(spc.history),
                    'representative' : spc.representative,
                    'thresh' : spc.compatib_thresh,
                    'c1' : spc.c1, 'c2' : spc.c2, 'c3' : spc.c3}
                for spc in pop.species],
                'species_hist' : None
            } for name, pop in self.populations.items()},
            'generation' : generation,
            'p_weight_mut' : {name : pop.p_weight_mut for name, pop in self.populations.items()},
            'p_node_mut' : {name : pop.p_node_mut for name, pop in self.populations.items()},
            'p_conn_mut' : {name : pop.p_conn_mut for name, pop in self.populations.items()},
            'evolution_hist' : self.evolution_history,
        }
        file_name = 'spike_swarm_sim/checkpoints/populations/' + self.checkpoint_name
        save_pickle(pop_checkpoint, file_name)
        logging.info('Successfully saved evolution checkpoint.')
        
    def load_population(self):
        """ Loads a previously saved checkpoint to resume evolution.
        """
        #!!!!OJO SAVE TOPOLOGY MAL AHORA
        checkpoint = load_pickle('spike_swarm_sim/checkpoints/populations/' + self.checkpoint_name)
        logging.info('Resuming NEAT evolution using checkpoint ' +  self.checkpoint_name)
        key = tuple(self.populations.keys())[0]
        for key, pop in checkpoint['populations'].items():
            self.populations[key].p_weight_mut = checkpoint['p_weight_mut'][key]
            self.populations[key].p_node_mut = checkpoint['p_node_mut'][key]
            self.populations[key].p_conn_mut = checkpoint['p_conn_mut'][key]
            self.populations[key].population = pop['genotypes']
            self.populations[key].best = pop.get('best', None)
            self.populations[key].current_innovation = pop['current_innovation']
            self.populations[key].innovation_history = pop['innovation_history']
            self.populations[key].input_nodes = pop['input_nodes']
            self.populations[key].species_count = pop['species_count']
            self.populations[key].species = []
            for spc_chk in pop['species']:
                spc = Species(spc_chk['id'], spc_chk['creation_generation'],
                        compatib_thresh=spc_chk['thresh'], c1=spc_chk['c1'], c2=spc_chk['c2'], c3=spc_chk['c3'])
                spc.history = copy.deepcopy(spc_chk['history'])
                self.populations[key].species.append(spc)
                spc.representative = spc_chk['representative']
                spc.num_genotypes = np.sum([genotype['species'] == spc.id  for genotype in pop['genotypes']])
            robots = [copy.deepcopy(robot) for robot in self.world.robots.values()]
            interface = NEATInterface(robots[0].controller.neural_network)
            # #!
            # self.populations[key].segment_lengths = [interface.submit_query(query, primitive='LEN')\
            #             for query in self.populations[key].objects]         
        self.init_generation = checkpoint['generation']
        self.evolution_history = checkpoint['evolution_hist']
