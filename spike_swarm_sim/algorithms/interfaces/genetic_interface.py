import copy
import numpy as np
from functools import wraps
from .interpreter import language_dict

class GeneticInterface:
    """ Interface used by all the evolutionary algorithms to manage phenotype to 
    genotype transactions (bijectively), genotype initialization and so on.
    ================================================================================
    - Params:
        neural_net [NeuralNetwork] : ANN of a robot.
    ================================================================================
    """
    def __init__(self, neural_net):
        self.neural_net = neural_net 

    def submit_query(self, query, primitive='GET', **kwargs):
        """
        Submits a query to read or write the ANN. The operation is of the syntax 
        "PRIMITIVE query data" (data if primitive is write). 
        - Args:
            query [str] : query subject to the primitive. The query addresses some 
                    variable of the ANN and it has the following syntax:
                        "ANN_part:variable:name"
                    Some examples are:
                        "synapses:weights:all" -> weights of all the synapses.
                        "synapses:weights:S1" -> weights of synapse with name S1 (must be)
                                                 defined
                        "neurons:tau:all" -> time constants of all neuron models.
                        "neurons:gain:sensory" -> gains of sensory neurons.
                        "decoding:weights:all" -> weights of all decoders (in SNN).
            primitive [str]: action or primitive of the query. Possible primitives 
                    are GET, SET, LEN, INIT.
            data [np.ndarray]: genotype array only supplied when the primitive is SET.
            min_vals [np.ndarray]: minimum bound of the search space. In GET it is None.
            max_vals [np.ndarray]: max. bound of the search space. In GET it is None.
        """
        query_hierarchy = [primitive] + query.split(':')
        query_status = language_dict
        for query_elem in query_hierarchy[:-1]:
            assert query_elem in query_status
            query_status = query_status[query_elem]
        func = getattr(getattr(self.neural_net, query_hierarchy[1]), query_status)\
               if query_hierarchy[1] not in ['decoding', 'encoding']\
               else getattr(self.neural_net, query_status)
        return func(query_hierarchy[-1], self.neural_net.graph, **kwargs)

    def toGenotype(self, queries, min_vals, max_vals):
        """ Converts a phenotype or structured ANN into a vector genotype. 
        It performs a series of queries (depending on the population segments) 
        and gathers the results as the final genotype.
        """
        genotype = np.hstack([self.submit_query(query, primitive='GET',\
                   min_val=min_val, max_val=max_val)\
                   for query, max_val, min_val in zip(queries, max_vals, min_vals)])
        return genotype

    def fromGenotype(self, queries, genotype, min_vals, max_vals):
        """ Converts a genotype into a phenotype or, in this case, structured ANN. 
        It iterates across population opt. vars. with the corresponding queries and 
        submits a SET operation towards the ANN.
        """
        counter = 0
        for query, max_val, min_val in zip(queries, max_vals, min_vals):
            segment_len = self.submit_query(query, primitive='LEN')
            genotype_segment = genotype[counter:counter+segment_len].copy()
            self.neural_net.graph = self.submit_query(query, primitive='SET',\
                    data=genotype_segment, min_val=min_val, max_val=max_val)
            counter += segment_len
        self.neural_net.build() #* Compile changes.
    
    def initGenotype(self, queries, min_vals, max_vals):
        """ Method for initializing the values of the genotype.
        """
        for query, max_val, min_val in zip(queries, max_vals, min_vals):
            self.neural_net.graph = self.submit_query(query, primitive='INIT', min_val=min_val, max_val=max_val)
        self.neural_net.build()

class NEATInterface:
    def __init__(self, neural_net):
        self.neural_net = neural_net

    def submit_query(self, query, primitive='GET', **kwargs):
        """
        Submits a query to read or write the ANN. The operation is of the syntax 
        "PRIMITIVE query data" (data if primitive is write). 
        - Args:
            query [str] : query subject to the primitive. The query addresses some 
                    variable of the ANN and it has the following syntax:
                        "ANN_part:variable:name"
                    Some examples are:
                        "synapses:weights:all" -> weights of all the synapses.
                        "synapses:weights:S1" -> weights of synapse with name S1 (must be)
                                                defined
                        "neurons:tau:all" -> time constants of all neuron models.
                        "neurons:gain:sensory" -> gains of sensory neurons.
                        "decoding:weights:all" -> weights of all decoders (in SNN).
            primitive [str]: action or primitive of the query. Possible primitives 
                    are GET, SET, LEN, INIT.
            data [np.ndarray]: genotype array only supplied when the primitive is SET.
            min_vals [np.ndarray]: minimum bound of the search space. In GET it is None.
            max_vals [np.ndarray]: max. bound of the search space. In GET it is None.
        """
        # if query == "learning_rules:params:all": import pdb; pdb.set_trace()
        query_hierarchy = [primitive] + query.split(':')
        query_status = language_dict
        for query_elem in query_hierarchy[:-1]:
            assert query_elem in query_status
            query_status = query_status[query_elem]
        func = getattr(getattr(self.neural_net, query_hierarchy[1]), query_status)\
            if query_hierarchy[1] not in ['decoding', 'encoding']\
            else getattr(self.neural_net, query_status)
        return func(query_hierarchy[-1], self.neural_net.graph, **kwargs)

    def toGenotype(self, queries, min_vals, max_vals):
        #! CHECK
        """ Converts a phenotype or structured ANN into a vector genotype. 
        It performs a series of queries (depending on the population segments)
        and gathers the results as the final genotype. 
        """
        genotype = np.hstack([self.submit_query(query, primitive='GET',\
                min_val=min_val, max_val=max_val) \
                for query, max_val, min_val in zip(queries, max_vals, min_vals)])
        return genotype

    def fromGenotype(self, queries, genotype, min_vals, max_vals):
        """ Converts a genotype into a phenotype or, in this case, structured ANN.
        """
        effective_genotype = copy.deepcopy(genotype)
        # disabled_nodes = []
        # for key, node in effective_genotype['nodes'].items():
        #     if not any(conn['post'] == key and conn['enabled'] for conn in effective_genotype['connections'].values()):
        #         disabled_nodes.append(key)
        # for key in disabled_nodes:
        #     del effective_genotype['nodes'][key]

        disabled_connections = []
        for key, conn in effective_genotype['connections'].items():
            if not conn['enabled']:
                disabled_connections.append(key)
        for key in disabled_connections:
            del effective_genotype['connections'][key]

        #* Clean previous architecture
        for name in list(self.neural_net.graph['neurons']):
            if name not in effective_genotype['nodes']: #* residual from prev genotype
                self.neural_net.delete_neuron(name)
        for name in list(self.neural_net.graph['synapses']):
            if name not in effective_genotype['connections']: #* residual from prev genotype
                self.neural_net.delete_synapse(name)
        #* Add new neurons 
        for i, key in enumerate(effective_genotype['nodes']):
            #* Reallocate node indices
            effective_genotype['nodes'][key]['idx'] = i
            if key not in self.neural_net.graph['neurons']:
                self.neural_net.add_neuron(key)
            self.neural_net.graph['neurons'].update({key : effective_genotype['nodes'][key].copy()})
        #* Add new synapses 
        for key in effective_genotype['connections']:
            if key not in self.neural_net.graph['synapses']:
                syn = effective_genotype['connections'][key]
                self.neural_net.add_synapse(key, syn['pre'], syn['post'], syn['weight'], conn_prob=1.)
            self.neural_net.graph['synapses'].update({key : effective_genotype['connections'][key].copy()})
        #* Update parameters (Decoders and encoders not supported yet).
        # counter = 0
        for query, max_val, min_val in zip(queries, max_vals, min_vals):
            # segment_len = self.submit_query(query, primitive='LEN')
            gene_type = {'synapses' : 'connections', 'neurons' : 'nodes'}.get(query.split(':')[0], 'connections')
            variable = {'weights' : 'weight'}.get(query.split(':')[1], query.split(':')[1])
            if 'learning_rule' in query:
                genotype_segment = np.array([[gnt['learning_rule'][v] for gnt in effective_genotype['connections'].values()] for v in ['A', 'B', 'C', 'D']]).flatten()
            else:
                genotype_segment = np.array([gnt[variable] for gnt in effective_genotype[gene_type].values()])
            # genotype_segment = np.array([g[variable] for g in [*genotype[gene_type].values()][counter:counter+segment_len]])
            self.neural_net.graph = self.submit_query(query, primitive='SET',\
                        data=genotype_segment, min_val=min_val, max_val=max_val)
            # counter += segment_len
        self.neural_net.build() #* Compile changes.

    def initGenotype(self, queries, min_vals, max_vals):
        """ Method for initializing the values of the genotype.
        """
        for query, max_val, min_val in zip(queries, max_vals, min_vals):
            self.neural_net.graph = self.submit_query(query, primitive='INIT', min_val=min_val, max_val=max_val)
        self.neural_net.build()

class InterfaceFactory:
    def create(self, algorithm, neural_net):
        return {
            'GeneticAlgorithm' : GeneticInterface(neural_net),
            'SNES' : GeneticInterface(neural_net),
            'xNES' : GeneticInterface(neural_net),
            'CMA_ES' : GeneticInterface(neural_net),
            'OpenAI_ES' : GeneticInterface(neural_net),
            'NEAT' : NEATInterface(neural_net)
        }[algorithm]
