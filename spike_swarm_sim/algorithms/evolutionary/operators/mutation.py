from itertools import product
import numpy as np
from spike_swarm_sim.register import evo_operator_registry
from spike_swarm_sim.utils import ShapeMismatchException


def assign_unique_key(keys, base_name):
    unique_id = len(keys)
    while '{}_{}'.format(base_name, unique_id) in keys:
        unique_id += 1
    return '{}_{}'.format(base_name, unique_id)

def add_node(genotype, current_innovation, innovation_history, node_variables, **kwargs):
    """ Add a new node in between an existing connection. The exisiting connection 
    is disabled and two new synapses are included.
    """
    
    #* Randomly select an enabled connection
    sel_conn = np.random.choice([*zip(*filter(lambda x: x[1]['enabled'], genotype['connections'].items()))][0])
    node_name = 'Node_' + str(genotype['connections'][sel_conn]['innovation'])

    genotype['nodes'][node_name] = {
            'ensemble' : node_name,
            'idx' : len(genotype['nodes']),
            'is_motor' : False
    }
    #* Initialize randomly node parameters
    for var in node_variables:
        genotype['nodes'][node_name].update({var : np.random.random()})
    #* Add the new connections
    genotype['connections'][sel_conn]['enabled'] = False
    conn_name = node_name + '-' + genotype['connections'][sel_conn]['post']
    genotype['connections'].update({
        conn_name : {
            'pre' : node_name,
            'post' : genotype['connections'][sel_conn]['post'],
            'weight': genotype['connections'][sel_conn]['weight'],
            'group' : conn_name,
            'enabled' : True,
            'trainable' : True,
            'learning_rule' : genotype['connections'][sel_conn].get('learning_rule', None),
            'innovation' : innovation_history.get((node_name,\
                    genotype['connections'][sel_conn]['post']), current_innovation),
            'idx' : len(genotype['connections']),#!
            'p' : 1.}
    })
    if (node_name, genotype['connections'][sel_conn]['post']) not in innovation_history:
        innovation_history.update({
            (node_name, genotype['connections'][sel_conn]['post']) : current_innovation
        })
        current_innovation += 1
    conn_name = genotype['connections'][sel_conn]['pre'] + '-' + node_name
    genotype['connections'].update({
        conn_name : {
            'pre' : genotype['connections'][sel_conn]['pre'],
            'post' : node_name,
            # Random weight but very close to zero (in the paper the authors propose w=0.5 fixed).
            'weight': np.clip(0.5 + np.random.randn() * 0.1, a_min=0, a_max=1),
            'group' : conn_name,
            'enabled' : True,
            'trainable':True,
            'learning_rule' : {v : np.clip(0.5 + np.random.randn() * 0.05, a_min=0, a_max=1) for v in ['A', 'B', 'C', 'D']},
            'innovation' : innovation_history.get((genotype['connections'][sel_conn]['pre'],\
                    node_name), current_innovation),
            'idx' : len(genotype['connections']),#!
            'p' : 1.}
    })
    if (genotype['connections'][sel_conn]['pre'], node_name) not in innovation_history:
        innovation_history.update({
            (genotype['connections'][sel_conn]['pre'], node_name): current_innovation
        })
        current_innovation += 1
    return genotype, current_innovation, innovation_history

def add_connection(genotype, input_nodes, current_innovation, innovation_history, **kwargs):
    """ Add a new gene connection to the genotype. The pre and post 
    synaptic nodes are selected randomly (validating that the connection does not 
    exist).
    """
    pos_conns = set([*product(input_nodes, genotype['nodes'].keys())]
                    + [*product(genotype['nodes'].keys(), repeat=2)])
    existing_conns = set([(conn['pre'], conn['post']) for conn in genotype['connections'].values()])
    allowed_conns = list(pos_conns - existing_conns)
    if len(allowed_conns) == 0:
        return genotype, current_innovation, innovation_history
    new_conn = allowed_conns[np.random.choice(range(len(allowed_conns)))]

    #* Name connection is "pre-post"
    conn_name = '-'.join(new_conn)
    genotype['connections'].update({
        conn_name : {
            'pre' : new_conn[0],
            'post' : new_conn[1],
            'weight': np.clip(0.1 * np.random.randn() + 0.5, a_min=0, a_max=1), # Random weight in [0,1] (denormalized later).
            'group' : conn_name,
            'enabled' : True,
            'trainable' : True,
            'learning_rule' : {v : np.random.random() for v in ['A', 'B', 'C', 'D']},
            'innovation' : innovation_history.get((new_conn[0], new_conn[1]), current_innovation),
            'idx' : len(genotype['connections']),#!
            'p' : 1.}
    })
    if (new_conn[0], new_conn[1]) not in innovation_history:
        innovation_history.update({(new_conn[0], new_conn[1]): current_innovation})
        current_innovation += 1
    return genotype, current_innovation, innovation_history


def delete_node(genotype, input_nodes, **kwargs):
    #TODO: to be implemented (not ready yet)
    node = np.random.choice([*genotype['nodes']])
    if node in input_nodes or genotype['nodes'][node]['is_motor']:
        return
    for conn_name, conn in genotype['connections'].items():
        if node in [conn['pre'], conn['post']]:
            del genotype['connections'][conn_name]
    del genotype['nodes'][node]

    import pdb; pdb.set_trace()

def delete_connection(genotype, input_nodes, **kwargs):
    #TODO: to be implemented (not ready yet)
    conn = np.random.choice([*genotype['connections']])
    pre_node = genotype['connections'][conn]['pre']
    post_node = genotype['connections'][conn]['post']
    condition = any(conn2['pre'] == pre_node and conn2['post'] == post_node\
        for conn2_name, conn2 in genotype['connections'].items() if conn2_name != conn)
    if condition:
        return
    del genotype['connections'][conn]

def neat_mutation(population, input_nodes, current_innovation, innovation_history,
            mutable_variables, p_weight_mut=0.75, p_node_mut=0.03, p_conn_mut=0.5):
    #* Parameter Mutations
    for param in mutable_variables:
        gene_type = {'synapses' : 'connections', 'neurons' : 'nodes'}.get(param.split(':')[0], 'connections')
        variable = {'weights' : 'weight'}.get(param.split(':')[1], param.split(':')[1])
        for i, genotype in filter(lambda x: np.random.random() < p_weight_mut, enumerate(population)):
            for conn in genotype[gene_type].values(): #!optimize
                if np.random.random() < 0.02:
                    if 'learning_rule' in param:
                        conn['learning_rule'] = {v : np.random.random() for v in ['A', 'B', 'C', 'D']}
                    else:
                        conn[variable] = np.random.random()
                else:
                    if 'learning_rule' in param:
                        for v in ['A', 'B', 'C', 'D']:
                            conn['learning_rule'][v] += np.random.randn() * .05
                            conn['learning_rule'][v] = np.clip(conn['learning_rule'][v], a_min=0, a_max=1)
                    else:
                        conn[variable] += np.random.randn() * 0.05
                        conn[variable] = np.clip(conn[variable], a_min=0, a_max=1)
    #* Connnections mutations
    for i, genotype in filter(lambda x: np.random.random() < p_conn_mut, enumerate(population)):
        genotype, current_innovation, innovation_history = add_connection(genotype, input_nodes, current_innovation, innovation_history)
    #* Node mutations
    for i, genotype in filter(lambda x: np.random.random() < p_node_mut, enumerate(population)):
        genotype, current_innovation, innovation_history = add_node(genotype, current_innovation,
                    innovation_history, [var.split(':')[1] for var in mutable_variables if var.split(':')[0] == 'neurons'])
    return population, current_innovation, innovation_history

@evo_operator_registry(name='gaussian_mutation')
def gaussian_mutation(population, **kwargs):
    """ Gaussian mutation operator for GA. Each genotype gene is mutated with a 
    probability mutation_prob. Mutation is accomplished by sampling a new gene value 
    from a gaussian dist. centered at the gene and with a std. dev. sigma.
    ================================================================================
    - Args:
        mutation_prob [float]: probability of mutating a gene.
        sigma [float]: std. dev. of the gaussian mutation.
    - Returns
        new_pop [list of np.ndarray]: list of mutated genotypes.
    ================================================================================
    """
    new_pop = []
    for indiv in population:
        mutation_mask = np.random.random(size=indiv.shape) < kwargs['mutation_prob']
        mutated = indiv + mutation_mask * np.random.randn(indiv.shape[0]) * kwargs['sigma']
        new_pop.append(mutated)
    return new_pop

@evo_operator_registry(name='uniform_mutation')
def uniform_mutation(population, **kwargs):
    """ Uniform mutation operator for GA. Each genotype gene is mutated with a 
    probability mutation_prob. Mutation is accomplished by uniformly resampling the
    gene within the interval [min_vals[g], max_vals[g]], where min_vals and max_vals 
    are the bounds of each gene and g is the gene index.
    ================================================================================
    - Args:
        mutation_prob [float]: probability of mutating a gene.
        min_vals [float or np.ndarray]: minimum values of uniform mutation. It can be 
                either a numpy array of same length as the genotype or a float. If float, 
                it is assumed that all genes have the same min value. 
        max_vals [float or np.ndarray]: maximum values of uniform mutation. It can be 
                either a numpy array of same length as the genotype or a float. If float, 
                it is assumed that all genes have the same max value. 
    - Returns
        new_pop [list of np.ndarray]: list of mutated genotypes.
    ================================================================================
    """
    min_vals = {
        'int' : np.repeat(kwargs['min_vals'], len(population[0])),
        'float' : np.repeat(kwargs['min_vals'], len(population[0])),
        'list' : np.array(kwargs['min_vals']),
        'ndarray' : kwargs['min_vals']
    }[type(kwargs['min_vals']).__name__]
    max_vals = {
        'int' : np.repeat(kwargs['max_vals'], len(population[0])),
        'float' : np.repeat(kwargs['max_vals'], len(population[0])),
        'list' : np.array(kwargs['max_vals']),
        'ndarray' : kwargs['max_vals']
    }[type(kwargs['max_vals']).__name__]
    if len(min_vals) != len(population[0]):
        raise ShapeMismatchException('Dimension of minimum values of uniform mutation '\
            'did not match with the genotype length.')
    if len(max_vals) != len(population[0]):
        raise ShapeMismatchException('Dimension of maximum values of uniform mutation '\
            'did not match with the genotype length.')
    new_pop = []
    for indiv in population:
        mutation_mask = np.random.random(size=indiv.shape) < kwargs['mutation_prob']
        mutated = (1. - mutation_mask) * indiv + mutation_mask * np.random.uniform(low=min_vals, high=max_vals)
        new_pop.append(mutated)
    return new_pop

@evo_operator_registry(name='bitflip_mutation')
def bitFlip_mutation(population, **kwargs):
    """ Bit-Flip mutation operator for binary coded GA. Each genotype gene is mutated 
    with a probability mutation_prob. Mutation is accomplished by flipping the gene 
    bit (0 -> 1 or 1 -> 0).
    ================================================================================
    - Args:
        mutation_prob [float]: probability of mutating a gene.
        min_vals [float or np.ndarray]: minimum values of uniform mutation. It can be 
                either a numpy array of same length as the genotype or a float. If float, 
                it is assumed that all genes have the same min value. 
        max_vals [float or np.ndarray]: maximum values of uniform mutation. It can be 
                either a numpy array of same length as the genotype or a float. If float, 
                it is assumed that all genes have the same max value. 
    - Returns
        new_pop [list of np.ndarray]: list of mutated genotypes.
    ================================================================================
    """
    new_pop = []
    for indiv in population:
        mutation_mask = np.random.random(size=indiv.shape) < kwargs['mutation_prob']
        indiv[mutation_mask] = 1 - indiv[mutation_mask]
        new_pop.append(indiv.copy())
    return new_pop

@evo_operator_registry(name='categorical_mutation')
def categorical_mutation(population, **kwargs):
    new_pop = []
    for indiv in population:
        mutation_mask = np.random.random(size=indiv.shape) < kwargs['mutation_prob']
        indiv[mutation_mask] = np.random.randint(kwargs['min_vals'][mutation_mask], kwargs['max_vals'][mutation_mask])
        new_pop.append(indiv)
    return new_pop
