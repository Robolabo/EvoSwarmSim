import copy
import logging
from itertools import product
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
# Own imports
from spike_swarm_sim.register import neuron_models, synapse_models, learning_rules
from spike_swarm_sim.utils import increase_time, merge_dicts, remove_duplicates
from .neuron_models import NonSpikingNeuronModel, SpikingNeuronModel
from .decoding import DecodingWrapper
from .encoding import EncodingWrapper
from .utils.monitor import NeuralNetMonitor
try:
    from .utils.visualization import *
except:
    pass

def monitor(func):
    """ Decorator for recording and monitoring the relevant neuronal variables. 
    The records are stored in the monitor attribute of the NeuralNetwork class.
    """
    @wraps(func)
    def wrapper(self, encoded_stimuli, **kwargs):
        spikes, Isynapses, voltages = func(self, encoded_stimuli, **kwargs)
        #* Debugging and Monitoring (debug option must be enabled)
        if self.monitor is not None:
            monitor_vars = {
                # 'encoded_inputs' : encoded_stimuli.copy(),
                'stimuli' : np.hstack(tuple(self.stimuli.values())).copy(),
                'voltages' : voltages.copy(),
                'currents' : Isynapses.copy(),
                'outputs' : spikes.copy()
            }
            if issubclass(type(self.neurons), SpikingNeuronModel):
                monitor_vars.update({
                    'encoded_inputs' : encoded_stimuli.copy(),
                    'spikes' : spikes.copy(),
                    'recovery' : self.neurons.recovery.copy(),
                    'neuron_theta' : self.neurons.theta.copy(),
                    'activities' : np.hstack([v.activities.copy() for v in self.decoders.all.values()])#!
                })
            self.monitor.update(**monitor_vars)
        return spikes, Isynapses, voltages
    return wrapper

class NeuralNetwork:
    """ Class for the artificial neural networks. This class is mainly a wrapper that creates and executes 
    the main building blocks of ANNs. These blocks are encoding, synapses, neurons and decoding, albeit there 
    are other functionalities such as learning rules, monitors, and so on. This class encompasses any kind 
    of neural network, the precise architecture and dynamics will be fixed by the neuron and synapses models 
    throughout the topology dictionary.
    ==========================================================================================================
    - Params:
        topology [dict] : dictionary specifying the ANN architecture (see configuration files for more details).
    - Attributes:
        dt [float] : Euler step of the ANN.
        t [int] : time counter.
        time_scale [int] : ratio between neuronal and environment dynamics. This means that every time step of 
                    the env., the ANN performs time_scale updates.
        synapses [Synapses] : object storing the synapse models.
        stim_encoding [dict] : dict of sensor_name : Encoding object storing all the neural encoders.
        pointers [dict] : dict mapping ensembles to the index in the ANN adjacency matrix. The index is only 
                    the index of the last neuron of the ensemble.
        subpop_neurons [dict] : dict mapping ensembles to number of neurons per ensemble.
        n_inputs [int] : number of ANN inputs (after decoding). 
        stimuli_order [list of str]: ordered list with the sensor order as specified in the ANN config.
        neurons [SpikingNeuronModel or NonSpikingNeuronModel] : object storing the neurons of the ANN.
        update_rule : #TODO
        output_neurons [list] : list with the name of the motor/output ensembles.
        monitor [NeuralNetMonitor or None]: Monitor to record neuronal variables if mode is DEBUG.
        spikes [np.ndarray of shape=num_neurons]: current generated spikes.
        stimuli [dict] : current supplied stimuli. Dict mapping sensor name to stimuli values. 
        action_decoding [dict] :  dict of action_name : Decoding object storing all the neural decoders.
    ==========================================================================================================
    """
    def __init__(self, dt, neuron_model='rate_model', synapse_model='static_synapse', time_scale=1):
        self.t = 0
        self.dt = dt #* Euler Step
        self.time_scale = time_scale #* ANN steps per world step.
        self.neuron_model = neuron_model
        self.synapse_model = synapse_model

        #* Flag indicating if the ANN is built and functional. 
        #* The ANN cannot be used if this flag is False.
        self.is_built = False #! NO SE USA

        #* Submodules of the neural network distributing its functioning
        #* and computations.
        if synapse_model == 'dynamic_synapse' and issubclass(neuron_models[neuron_model], NonSpikingNeuronModel):
            raise Exception(logging.error('The combination of dynamic synapses and '\
                'non-spiking neuron models is not currently implemented.'))
        self.synapses = synapse_models[self.synapse_model](self.dt)
        self.neurons = neuron_models[self.neuron_model](self.dt)
        self.encoders = EncodingWrapper(self.time_scale)
        self.decoders = None
        self.learning_rule = None
        #* Monitor that, if in DEBUG mode, will store all the relevant neural 
        #* variables.
        self.monitor = None

        #* Overall ANN directed graph description.
        self.graph = {'inputs' : {}, 'neurons' : {}, 'synapses' : {}}
        self.ensemble_names = []
        self.input_ensemble_names = []
        self.motor_ensemble_names = []
        #* Ordered list of stimuli names (not input nodes)
        self.stimuli_names = []
        #* Variables storing the previous stim and spikes.
        self.stimuli, self.spikes, self.prev_input = None, None, None
        self.weight_registry = None
    
    def build(self):
        #! BUILD NEURONS
        self.synapses.build(self.graph)
        if self.learning_rule is not None:
            self.learning_rule.build(self.graph)
        #TODO --- Create Monitor (DEBUG MODE) ---
        # self.output_neurons = remove_duplicates([out['ensemble'] for out in topology['outputs'].values()])
        if logging.root.level == logging.DEBUG:
            self.monitor = NeuralNetMonitor({ens : self.num_ensemble_neurons(ens)\
                        for ens in self.ensemble_names},\
                        {name : self.encoders.get(name).n_stimuli for name in self.stimuli_names},\
                        {name : self.num_input_nodes(name) for name in self.input_ensemble_names},\
                        self.motor_ensemble_names)
        else:
            self.monitor = None
        # #* --- Reset dynamics ---
        # self.reset()

    def build_from_dict(self, topology):
        #* Add neurons
        for name, ensemble in topology['ensembles'].items():
            self.add_ensemble(name, ensemble['n'], **ensemble['params'])
        #* Add stimuli
        for name, stim in topology['stimuli'].items():
            self.add_stimuli(name, stim['n'], stim['sensor'])
        #* Add motor ensembles
        for out in topology['outputs'].values():
            self.set_motor(out['ensemble'])
        #* Add encoders
        for input_name, encoder in topology['encoding'].items():
            self.add_encoder(encoder['scheme'], topology['stimuli'][input_name]['sensor'],\
                receptive_field=encoder['receptive_field']['name'], receptive_field_params=encoder['receptive_field']['params'])
        #* Add Learning Rule
        if topology.get('learning_rule', {}).get('rule') is not None:
            self.learning_rule = learning_rules.get(topology.get('learning_rule', {}).get('rule'))() #TODO decouple, improve.
        #* Add Synapses
        for name, syn in topology['synapses'].items():
            syn_params = {key : val for key, val in syn.items()\
                    if key not in ['pre', 'post', 'p', 'trainable']}
            self.add_synapse(name, syn['pre'], syn['post'], conn_prob=syn['p'], **syn_params)
        #* Add Decoders
        self.decoders = DecodingWrapper(topology)
        #* Build ANN
        self.build()

    def set_motor(self, ensemble_name):
        if ensemble_name not in self.ensemble_names:
            raise Exception(logging.error('Ensemble "{}" does not exist').format(ensemble_name))
        if ensemble_name in self.motor_ensemble_names:
            return
        self.motor_ensemble_names.append(ensemble_name)
        for neuron in self.graph['neurons'].values():
            if neuron['ensemble'] == ensemble_name:
                neuron['is_motor'] = True

    def add_stimuli(self, name, num_nodes, sensor):
        for n in range(num_nodes):
            self.graph['inputs'].update({
                '{}_{}'.format(name, n) : {'ensemble' : name, 'sensor' : sensor, 'idx': len(self.graph['inputs'])}
            })
        self.input_ensemble_names.append(name)
        self.stimuli_names.append(sensor)

    def add_ensemble(self, name, num_neurons, **kwargs):
        self.ensemble_names.append(name)
        for n in range(num_neurons):
            self.add_neuron('{}_{}'.format(name, n), ensemble=name, **kwargs)
        

    def add_neuron(self, name, ensemble=None, **kwargs):
        self.neurons.add(**kwargs)#!
        ensemble = ensemble if ensemble is not None else name
        if ensemble not in self.ensemble_names:
            self.ensemble_names.append(ensemble)
        self.graph['neurons'].update({name : merge_dicts([{'ensemble' : ensemble,
                'idx' : len(self.neurons)-1, 'is_motor' : False}, kwargs])})

    def delete_neuron(self, name):
        neuron_index = self.graph['neurons'][name]['idx']
        ensemble = self.graph['neurons'][name]['ensemble']
        self.neurons.delete(neuron_index)
        self.graph['neurons'].pop(name, None)
        #! Ojo index of other neurons?
        for neuron in self.graph['neurons'].values():
            if neuron['idx'] >= neuron_index:
                neuron['idx'] -= 1
        #* Remove ensemble if neuron was the only unit.
        if not any([neuron['ensemble'] == ensemble for neuron in self.graph['neurons'].values()]):
            self.ensemble_names.remove(ensemble)
        #* Remove any synapse with the neuron as pre or post
        for syn_name, syn in [*self.graph['synapses'].items()]:
            if syn['pre'] == name or syn['post'] == name:
                self.delete_synapse(syn_name)

    def add_synapse(self, name, pre, post, weight=1., conn_prob=1., trainable=True, **kwargs):
        """ Adds synapses between pre and post ensembles. """
        if post in self.graph['inputs'] or post in self.input_ensemble_names:
            raise Exception(logging.error('An input node or ensemble cannot be '\
                'a postsynaptic neuron or ensemble.'))
        #* Check if pre is neuron or ensemble.
        if pre not in merge_dicts([self.graph['inputs'], self.graph['neurons']]):
            if pre not in self.input_ensemble_names + self.ensemble_names:
                raise Exception(logging.error('Connection presynaptic neuron or ensemble '\
                    '"{}" does not exist').format(pre))
            pre = [name for name, node in merge_dicts([self.graph['inputs'], self.graph['neurons']]).items() if node['ensemble'] == pre]
        else:
            pre = [pre]
        #* Check if post is neuron or ensemble.
        if post not in merge_dicts([self.graph['inputs'], self.graph['neurons']]):
            if post not in self.input_ensemble_names + self.ensemble_names:
                raise Exception(logging.error('Connection postsynaptic neuron or ensemble '\
                    '"{}" does not exist').format(post))
            post = [name for name, node in merge_dicts([self.graph['inputs'], self.graph['neurons']]).items() if node['ensemble'] == post]
        else:
            post = [post]
        #* Add connections (note: not compatible with previous implementation checkpoints).
        #! REVISAR SEED
        np.random.seed(44 + len(self.graph['synapses']))
        for i, (pre_node, post_node) in enumerate(product(pre, post)):
            if np.random.random() < conn_prob:
                synapse_config = merge_dicts([{
                    'pre' : pre_node, 'post' : post_node,
                    'weight': weight, 'trainable' : trainable,
                    'group' : name, 'idx' : len(self.graph['synapses']), 'enabled' : True}, kwargs])
                if self.learning_rule is not None:
                    synapse_config.update({'learning_rule' : {p : 0. for p in ['A', 'B', 'C', 'D']}})
                if self.synapse_model == 'dynamic_synapse':
                    #! Add min and max possible delays?
                    synapse_config.update({'delay' : np.random.randint(1, 10)})
                syn_name = "{}_{}".format(name, i) if len(pre + post) > 2 else name
                self.graph['synapses'].update({syn_name : synapse_config})
        np.random.seed()

    def delete_synapse(self, name):
        self.graph['synapses'].pop(name, None)
        
    def add_encoder(self, scheme, sensor, receptive_field=None, receptive_field_params={}):
        raw_inputs = [inp for inp in self.graph['inputs'].values() if inp['sensor'] == sensor]
        self.encoders.add(scheme, sensor, len(raw_inputs), receptive_field=receptive_field,\
                receptive_field_params=receptive_field_params)
        if receptive_field is not None:
            if 'n_neurons' in receptive_field_params and receptive_field_params['n_neurons'] > 1:
                #* Correct the input nodes if the encoding augments their dimension.
                ensemble_name = tuple(raw_inputs)[0]['ensemble']
                for n in range(len(raw_inputs), receptive_field_params['n_neurons'] * len(raw_inputs)):
                    prev_idx = self.graph['inputs'][ensemble_name+'_'+str(n-1)]['idx']
                    for inp_node in filter(lambda x: x['idx'] >= prev_idx + 1, self.graph['inputs'].values()):
                        inp_node['idx'] += 1
                    self.graph['inputs'].update({'{}_{}'.format(ensemble_name, n) :\
                        {'ensemble' : ensemble_name, 'sensor' : sensor, 'idx': prev_idx + 1}})
    @increase_time
    @monitor
    def _step(self, stimuli):
        """ Private method devoted to step the synapses and neurons sequentially. 
        ====================================================================================
        - Args:
            stimuli [dict]: dict mapping stimuli name and numpy array containing its values.
        - Returns:
            spikes [np.ndarray]: boolean vector with the generated spikes.
            soma_currents [np.ndarray]: vector of currents injected to the neurons.
            voltages [np.ndarray]: vector of membrane voltages after neurons step.
        ====================================================================================
        """
        soma_currents = self.synapses.step(np.r_[stimuli, self.spikes], self.voltages)
        spikes, voltages = self.neurons.step(soma_currents)
        return spikes, soma_currents, voltages

    def step(self, stimuli, reward=None):
        """ Simulation step of the neural network.
        It is composed by four main steps:
            1) Encoding of stimuli to spikes (if SNN used).
            2) Synapses step.
            3) Neurons step.
            4) Decoding of spikes or activities into actions.
        ===============================================================
        - Args:
            stimuli [dict]: dict mapping stimuli name and numpy array 
                    containing its values.
        - Returns:
            actions [dict]: dict mapping output names and actions.
        ===============================================================
        """  
        #* --- Convert stimuli into spikes (Encoders Step) ---
        if len(stimuli) == 0:
            raise Exception(logging.error('The ANN received empty stimuli.'))
        stimuli = {s : stimuli[s].copy() for s in self.stimuli_names}
        inputs = self.encoders.step(stimuli)
        self.stimuli = stimuli.copy()
        if self.time_scale == 1:
            inputs = inputs[np.newaxis]
        #* --- Apply update rules to synapses ---
        if self.t > 1 and self.learning_rule is not None:
            # If reward is None  while learning rule is not, then 
            # assume that it is a non modulated learning rule.
            if reward is None:
                reward = 1.
            # Use inputs and neuron outputs of previous time step.
            self.synapses.weights += self.learning_rule.step(self.prev_input, self.spikes, reward=reward)
            self.synapses.weights = np.clip(self.synapses.weights, a_min=-6, a_max=6)
            
        #* --- Step synapses and neurons ---
        spikes_window = []
        for tt, stim in enumerate(inputs):
            spikes, _, _ = self._step(stim)
            self.spikes = spikes.copy()
            spikes_window.append(spikes.copy())
        spikes_window = np.stack(spikes_window)

        #* --- Convert spikes into actions (Decoding Step) ---
        actions = self.decoders.step(spikes_window[:, self.motor_neurons])
        self.prev_input = inputs[-1].copy()
        #* --- Debugging stuff (DEBUG MODE) --- #
        if self.t == self.time_scale * 4000 and self.monitor is not None:
            vv = np.stack(tuple(self.monitor.get('outputs').values()))
            ii = np.stack(tuple(self.monitor.get('stimuli').values()))
            II = np.stack(tuple(self.monitor.get('currents').values()))
            aa = np.stack(tuple(self.monitor.get('voltages').values()))
            # grasp0 = self.monitor.get('outputs')['OUT_GRASP_0']
            # plot_spikes(self)
            import pdb; pdb.set_trace()
        return actions
    
    @property
    def num_neurons(self):
        """ Number of neurons in the ANN (non-input). """
        return len(self.neurons)

    @property
    def num_inputs(self):
        return len(self.graph['inputs'])
    
    @property
    def num_motor(self):
        return len(self.motor_neurons)
    
    @property
    def num_hidden(self):
        return self.num_neurons - self.num_motor

    @property
    def motor_neurons(self):
        """ Indices of motor neurons without counting input nodes. When addressing the 
        weight matrix or any kind of ANN adj. mat., the number of inputs MUST be added.
        """
        return np.hstack([self.ensemble_indices(motor) for motor in self.motor_ensemble_names])
    
    def num_ensemble_neurons(self, ensemble):
        return len(self.ensemble_indices(ensemble))

    def num_input_nodes(self, ensemble):
        return len(self.input_ensemble_indices(ensemble))

    def ensemble_indices(self, ens_name, consider_inputs=False):
        """ Indices of the neurons of the requested ensemble. """
        if ens_name not in self.ensemble_names:
            raise Exception(logging.error('Requested ensemble "{}" does not exist.'.format(ens_name)))
        
        indices = np.array([neuron['idx'] for neuron in self.graph['neurons'].values() if neuron['ensemble'] == ens_name])
        if consider_inputs:
            indices += self.num_inputs #!
        return indices

    def input_ensemble_indices(self, input_name):
        """ Indices of the neurons of the requested ensemble. """
        if input_name not in self.input_ensemble_names:
            raise Exception(logging.error('Requested input ensemble "{}" does not exist.'.format(input_name)))
        indices = np.array([node['idx'] for node in self.graph['inputs'].values() if node['ensemble'] == input_name])
        return indices

    @property
    def is_spiking(self):
        """ Whether the neural network is a spiking neural network or not. """
        return issubclass(type(self.neurons), SpikingNeuronModel)

    @property
    def voltages(self):
        """Getter instantaneous voltage vector (membrane voltage of each neuron membrane)
        at current simulation timestep."""
        return self.neurons.voltages

    @property
    def weights(self):
        "Getter of the numpy weight matrix."
        return self.synapses.weights


    def reset(self):
        """ Reset process of all the neural network dynamics. """
        self.t = 0
        self.build()
        self.neurons.reset()
        self.synapses.reset()
        self.encoders.reset()
        self.decoders.reset()
        if self.learning_rule is not None:
            self.learning_rule.reset()
        if self.monitor is not None:
            self.monitor.reset()
        self.spikes = np.zeros(self.weights.shape[0])
        self.stimuli = None
        self.prev_input = None
        
