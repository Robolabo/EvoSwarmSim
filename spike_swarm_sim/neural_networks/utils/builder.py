from abc import ABC, abstractmethod
from itertools import chain, product
from collections import deque
import numpy as np









class SynapsesBuilder(object):
    """
    Builder devoted to construct the ANN graph arch. from a config. dict.
    """
    #! NO ME GUSTA
    def __init__(self, synapses, ensembles, stimuli, encoding=None):
        # self.topology = topology
        self.synapses = synapses
        self.ensembles = ensembles
        self.stimuli = stimuli
        for k in self.stimuli.keys():
            if 'encoding' in topology.keys():
                if 'n_neurons' in topology['encoding'][k]['receptive_field']['params']:
                    self.stimuli[k]['n'] *= topology['encoding'][k]['receptive_field']['params']['n_neurons']
        self.n_inputs = sum([stim['n'] for stim in self.stimuli.values()])
        self.n_neurons = sum([ens['n'] for ens in self.ensembles.values()])
        self.overall_topology = {name : v for name, v in chain(self.stimuli.items(), self.ensembles.items())}
        self.ensemble_pointers = {name: v for name, v in zip(self.overall_topology.keys(), \
                                    np.cumsum([v['n'] for v in self.overall_topology.values()]))}
        self.connection_dict = {}

    def _build(self, build_func, *args, **kwargs):
        matrix = np.zeros([self.n_neurons, self.n_inputs + self.n_neurons])
        conn_dict_filled = bool(len(self.connection_dict))
        for name, synapse in self.synapses.items():
            pre_lst = synapse['pre']
            post_lst = synapse['post']
            if not isinstance(pre_lst, list): pre_lst = [pre_lst]
            if not isinstance(post_lst, list): post_lst = [post_lst]
            for (pre, post) in tuple(product(pre_lst, post_lst)):
                post_range = (self.ensemble_pointers[post] - self.overall_topology[post]['n']\
                           - self.n_inputs, self.ensemble_pointers[post] - self.n_inputs)
                pre_range = (self.ensemble_pointers[pre] - self.overall_topology[pre]['n'], self.ensemble_pointers[pre])
                matrix[post_range[0]:post_range[1], pre_range[0]:pre_range[1]] = build_func(\
                        pre_range, post_range, synapse_params=synapse, *args, **kwargs)
                if not conn_dict_filled:
                    self.update_connection_dict(name, post_range, pre_range)
        return matrix

    def build_connections(self):
        def _build_connections(node_pre_indices, node_post_indices, synapse_params=None):
            conn_submask = np.random.choice(2, p=[1 - synapse_params['p'], synapse_params['p']],\
                    size=(len(range(*node_post_indices)), len(range(*node_pre_indices)))).astype(bool)
            return conn_submask.astype(int)
        return self._build(_build_connections)

    def build_neurotransmitters(self, mask):
        def _build_neurotransmitter(node_pre_indices, node_post_indices, synapse_params=None, neurotransmitter='AMPA', mask=None):
            submask = mask[range(*node_post_indices), :][:, range(*node_pre_indices)]
            if 'ntx_fixed' in synapse_params and synapse_params['ntx_fixed']:
                if neurotransmitter in synapse_params['neuroTX'].split('+'):
                    return submask.copy()
                else:
                    return np.zeros_like(submask).astype(bool)
        ampa_mask = self._build(_build_neurotransmitter, neurotransmitter='AMPA', mask=mask)
        gaba_mask = self._build(_build_neurotransmitter, neurotransmitter='GABA', mask=mask)
        ndma_mask = self._build(_build_neurotransmitter, neurotransmitter='NDMA', mask=mask)
        return {'AMPA' : ampa_mask, 'GABA' : gaba_mask, 'NDMA' : ndma_mask}

    def build_trainable_mask(self, mask, weights):
        def _build_trainable_mask(node_pre_indices, node_post_indices, synapse_params=None, mask=None):
            submask = mask[range(*node_post_indices), :][:, range(*node_pre_indices)]
            if 'trainable' in synapse_params.keys():
                return submask if synapse_params['trainable'] else np.zeros_like(submask).astype(bool)
            else:
                return np.zeros_like(submask).astype(bool)
        trainable_mask = self._build(_build_trainable_mask, mask=mask).astype(bool)
        weights[~trainable_mask&mask]=(np.random.random(size=np.sum(~trainable_mask&mask))) *.6
        return trainable_mask, weights

    def build_delays(self, min_delay=1, max_delay=10):
        delays = np.random.choice(range(min_delay, max_delay), size=self.n_inputs+self.n_neurons)  
        spike_buffer = [deque([False for _ in range(int(d))]) for d in delays]
        for d in delays:
            spike_buffer.append(deque([False for _ in range(int(d))]))
        return delays, spike_buffer

    def update_connection_dict(self, conn_name, post_range, pre_range):
        if conn_name in self.connection_dict:
            self.connection_dict[conn_name]['post'].append(post_range)
            self.connection_dict[conn_name]['pre'].append(pre_range)
        else:
            self.connection_dict.update({conn_name : {'post' : [post_range], 'pre' : [pre_range]}})


    def build_synapse_params(self):
       pass 

    def build_shared_connections(self):
        raise NotImplementedError

