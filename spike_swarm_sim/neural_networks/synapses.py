from itertools import chain, product
from collections import deque
from functools import wraps
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from spike_swarm_sim.neural_networks.utils.builder import SynapsesBuilder
from spike_swarm_sim.algorithms.interfaces import GET, SET, LEN, INIT
from spike_swarm_sim.utils import merge_dicts
from spike_swarm_sim.register import synapse_registry

def delay(func):
    @wraps(func)
    def wrapper(self, spikes, voltages, **kwargs):
        spikes = spikes.astype(bool)
        delayed_spikes = np.array([v.pop() for v in self.spike_buffer])
        any([v.appendleft(bool(s)) for s, v in zip(spikes, self.spike_buffer)])
        ret = func(self, delayed_spikes.copy(), voltages, **kwargs)
        return ret
    return wrapper  

class Synapses(ABC):
    def __init__(self, dt):
        self.dt = dt
        #* Weighted adjacency matrix.
        self.weights = None
        #* Unweighted boolean adjacency matrix (w=0 != no connection)
        self.mask = None
        #* Mask of connection that can be optimized.
        self.trainable_mask = None
    
    @abstractmethod
    def step(self, spikes, voltages):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def build(self, ann_graph):
        mask = np.full((len(ann_graph['neurons']), len(ann_graph['inputs']) + len(ann_graph['neurons'])), False)
        trainable_mask = mask.copy()
        weights = mask.copy().astype(float)
        for name, node in ann_graph['neurons'].items():
            in_connections = [syn for syn in ann_graph['synapses'].values() if syn['post'] == name]
            for syn in in_connections:
                if syn['enabled']:
                    pre_idx = ann_graph['inputs'][syn['pre']]['idx'] if syn['pre'] in ann_graph['inputs']\
                                else ann_graph['neurons'][syn['pre']]['idx'] + len(ann_graph['inputs'])
                    mask[node['idx'], pre_idx] = True
                    trainable_mask[node['idx'], pre_idx] = True
                    weights[node['idx'], pre_idx] = syn['weight']
        self.weights = weights
        self.mask = mask
        self.trainable_mask = trainable_mask

    @GET("synapses:weights")
    def get_weights(self, conn_name, ann_graph, min_val=0., max_val=1., only_trainable=True):
        """ Given a connection name the method returns the flattened array of synapse strengths in
        that connection. If only_trainable is active, only weights in train mode are returned.
        """
        #* Return scaled in [0,1]
        if conn_name == 'all':
            weights = np.array([syn['weight'] for syn in ann_graph['synapses'].values() if syn['trainable']])
            return (weights - min_val) / (max_val - min_val)
        #* Special queries of synapses
        conn_name = {
            'sensory' : [key for key, syn in ann_graph['synapses'].items()\
                            if syn['pre'] in ann_graph['inputs']],
            'hidden' : [key for key, syn in ann_graph['synapses'].items()\
                        if syn['pre'] in ann_graph['neurons']\
                        and not ann_graph['neurons'][syn['pre']]['is_motor']],
            'motor' : [key for key, syn in ann_graph['synapses'].items()\
                        if syn['pre'] in ann_graph['neurons']\
                        and ann_graph['neurons'][syn['pre']]['is_motor']]
        }.get(conn_name, [conn_name])
        weights = np.array([ann_graph['synapses'][name]['weight'] for name in conn_name\
                    if not only_trainable or ann_graph['synapses'][name]['trainable']])
        return (weights - min_val) / (max_val - min_val)

    @SET("synapses:weights")
    def set_weights(self, conn_name, ann_graph, data, min_val=0., max_val=1.,):
        """
        """
        #* rescale genotype segment to weight range
        data = min_val + data * (max_val - min_val)
        if conn_name == 'all':
            for w, syn in zip(data, filter(lambda x: x['trainable'], ann_graph['synapses'].values())):
                syn['weight'] = w
            return ann_graph
        else:
            #* Special queries of synapses
            conn_name = {
                'sensory' : [key for key, syn in ann_graph['synapses'].items()\
                                if syn['pre'] in ann_graph['inputs']],
                'hidden' : [key for key, syn in ann_graph['synapses'].items()\
                            if syn['pre'] in ann_graph['neurons']\
                            and not ann_graph['neurons'][syn['pre']]['is_motor']],
                'motor' : [key for key, syn in ann_graph['synapses'].items()\
                            if syn['pre'] in ann_graph['neurons']\
                            and ann_graph['neurons'][syn['pre']]['is_motor']]
            }.get(conn_name, [conn_name])
            for w, syn_name in zip(data, conn_name):
                if ann_graph['synapses'][syn_name]['trainable']:
                    ann_graph['synapses'][syn_name]['weight'] = w
            return ann_graph

    @INIT('synapses:weights')
    def init_weights(self, conn_name, ann_graph, min_val=0., max_val=1., only_trainable=True):
        """
        """
        weights_len = self.len_weights(conn_name, ann_graph)
        random_weights = 0.5 + np.random.randn(weights_len) * 0.3 #between 0 and 1 (denormalized in set)
        # random_weights = np.random.random(size=weights_len)
        random_weights = np.clip(random_weights, a_min=0, a_max=1)
        return self.set_weights(conn_name, ann_graph, random_weights,\
                            min_val=min_val, max_val=max_val)

    @LEN('synapses:weights')
    def len_weights(self, conn_name, ann_graph, only_trainable=True):
        """
        """
        return self.get_weights(conn_name, ann_graph, only_trainable=True).shape[0]



@synapse_registry(name='static_synapse')
class StaticSynapses(Synapses):

    def step(self, spikes, voltages):
        return self.weights.dot(spikes)

    def reset(self):
        pass

@synapse_registry(name='dynamic_synapse')
class DynamicSynapses(Synapses):
    def __init__(self, dt):
        super(DynamicSynapses, self).__init__(dt)
        # masks and matrices
        self.gaba_mask = None
        self.ampa_mask = None
        self.ndma_mask = None
        self.delays = None
        self.spike_buffer = None
        
        # synapse variables (initialized in reset)
        self.s_ampa, self.s_gaba, self.s_ndma = None, None, None
        self.x_ndma = None
        
        #* Synapse Parameters (default)
        #* Changed in build method
        self.ampa_gain = 0.6
        self.ampa_tau = 5.
        self.ampa_E = 0.0
        self.gaba_gain = 1.
        self.gaba_tau = 10
        self.gaba_E = -70.
        self.ndma_gain = 0.1
        self.ndma_tau_rise = 6
        self.ndma_tau_decay = 100
        self.ndma_Mg2 = 0.5
        self.ndma_E = 0.0

    @delay
    def step(self, spikes, voltages):
        Iampa = self.step_ampa(spikes, voltages)
        Igaba = self.step_gaba(spikes, voltages)
        Indma = self.step_nmda(spikes, voltages)
        I = Iampa + Igaba + Indma
        return I

    def step_ampa(self, spikes, voltages):
        self.s_ampa[spikes] = 1
        self.s_ampa[~spikes] += self.dt * (-self.s_ampa[~spikes] / self.ampa_tau) 
        PSP = self.ampa_gain * (self.ampa_mask * self.weights).dot(self.s_ampa)# * (self.ampa_E - voltages) 
        if self.ampa_E is not None: PSP *= (self.ampa_E - voltages)
        return PSP

    def step_gaba(self, spikes, voltages):
        self.s_gaba[spikes] = 1
        self.s_gaba[~spikes] += self.dt * (-self.s_gaba[~spikes]/self.gaba_tau)
        PSP = self.gaba_gain * (self.gaba_mask * self.weights).dot(self.s_gaba)#* (self.gaba_E - voltages) 
        if self.gaba_E is not None: PSP *= (self.gaba_E - voltages)
        return PSP

    def step_nmda(self, spikes, voltages):
        self.x_ndma[spikes] = 1
        self.x_ndma[~spikes] += self.dt * (-self.x_ndma[~spikes] / self.ndma_tau_rise)
        self.s_ndma += self.dt * (.5*self.x_ndma*(1-self.s_ndma) - self.s_ndma/self.ndma_tau_decay) 
        PSP = self.ndma_gain * (self.ndma_mask * self.weights).dot(self.s_ndma)\
                * ( 1 / (1 + self.ndma_Mg2 * np.exp(-0.062*voltages)/3.57))
        if self.ndma_E is not None: PSP *= (self.ndma_E - voltages)
        return PSP
    
    def build(self, ann_graph):
        mask = np.full((len(ann_graph['neurons']), len(ann_graph['inputs']) + len(ann_graph['neurons'])), False)
        trainable_mask = mask.copy()
        weights = mask.copy().astype(float)
        ampa_mask = mask.copy()
        gaba_mask = mask.copy()
        ndma_mask = mask.copy()
        
        for name, node in ann_graph['neurons'].items():
            #* List of input connections to node.
            in_connections = [syn for syn in ann_graph['synapses'].values() if syn['post'] == name]
            for syn in in_connections:
                if syn['enabled']:
                    pre_idx = ann_graph['inputs'][syn['pre']]['idx'] if syn['pre'] in ann_graph['inputs']\
                            else ann_graph['neurons'][syn['pre']]['idx'] + len(ann_graph['inputs'])
                    mask[node['idx'], pre_idx] = True
                    trainable_mask[node['idx'], pre_idx] = True
                    weights[node['idx'], pre_idx] = syn['weight']
                    ampa_mask[node['idx'], pre_idx] = 'AMPA' in syn['neuroTX']
                    gaba_mask[node['idx'], pre_idx] = 'GABA' in syn['neuroTX']
                    ndma_mask[node['idx'], pre_idx] = 'NDMA' in syn['neuroTX']
        self.weights = weights
        self.mask = mask
        self.trainable_mask = trainable_mask
        self.ampa_mask = ampa_mask
        self.gaba_mask = gaba_mask
        self.ndma_mask = ndma_mask

        #* Set up synapse delays
        #! Not implemented yet.
        self.delays = np.ones(len(ann_graph['inputs'])+len(ann_graph['neurons'])) 
        # np.random.choice(range(min_delay, max_delay),\
        # size=len(ann_graph['inputs'])+len(ann_graph['neurons']))  
        self.spike_buffer = [deque([False for _ in range(int(d))]) for d in self.delays]
        for d in self.delays:
            self.spike_buffer.append(deque([False for _ in range(int(d))]))

        #! Pasar a matrices mediante kwargs en ann_graph
        self.ampa_gain = 6.
        self.ampa_tau = 6.
        self.ampa_E =  None
        self.gaba_gain = 5.
        self.gaba_tau = 10.
        self.gaba_E = None
        self.ndma_gain = 1.
        self.ndma_tau_rise = 10.
        self.ndma_tau_decay = 50.
        self.ndma_Mg2 = 0.5
        self.ndma_E = None
        if self.gaba_E is None:
            self.gaba_gain *= -1

    def reset(self):
        self.s_ampa = np.zeros(self.weights.shape[1])
        self.s_gaba = np.zeros(self.weights.shape[1])
        self.s_ndma = np.zeros(self.weights.shape[1])
        self.x_ndma = np.zeros(self.weights.shape[1])
        self.spike_buffer = [deque([False for _ in range(int(d))]) for d in self.delays]

    @GET('synapses:delays')
    def get_delays(self, neuron_name):
        return self.delays.copy()

    @SET('synapses:delays')
    def set_delays(self, neuron_name, data):
        self.delays = data.copy()

    @LEN('synapses:delays')
    def len_delays(self, neuron_name):
        return self.delays.shape[0]

    @INIT('synapses:delays')
    def init_delays(self, neuron_name, min_val=0., max_val=1.):
        self.delays = np.random.randint(min_val, max_val, size=self.delays.shape[0])
    
    def trainable_weights(self, shared=False):
        return self.weights[self.trainable_mask]
       
    def normalize_weights(self):
        for i, w_row in enumerate(self.weights):
            self.weights[i] /= np.linalg.norm(w_row)
