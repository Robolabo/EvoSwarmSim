import logging
import numpy as np
import matplotlib.pyplot as plt
from spike_swarm_sim.register import encoding_registry, encoders, receptive_fields
from spike_swarm_sim.utils import increase_time
from spike_swarm_sim.algorithms.interfaces import GET, SET, LEN, INIT
from .neuron_models import LIFModel
from spike_swarm_sim.neural_networks.receptive_field import *

class EncodingWrapper:
    """ Wrapper for gathering all the encoders of the different stimulus. 
    It contains a dict mapping sensor names to Encoder instances.
    =====================================================================
    - Params:
        topology [dict] : configuration dict of the overall topology.
    """
    def __init__(self, time_scale=1):
        self.time_scale = time_scale
        self._encoders = {}
        # if topology is not None:
        #     self.build(topology)
        # else:
        #     logging.warning('No topology config. provided to the encoder wrapper.')

    def step(self, stimuli):
        """ Steps all the encoders with the corresponding stimulus. """
        return np.hstack([self._encoders[key].step(stim)\
                for key, stim in stimuli.items()])

    def add(self, scheme, sensor, stimuli_dim, receptive_field=None, receptive_field_params={}):
        if receptive_field is not None:
            rf_cls = receptive_fields[receptive_field]
            receptive_field = rf_cls(n_inputs=stimuli_dim, **receptive_field_params)
        self._encoders.update({sensor : encoders[scheme](stimuli_dim,\
                self.time_scale, receptive_field, dt=1e-3)})


    # def build(self, topology):
    #     """ Builds the encoders using the topology config. dict. """
    #     for name, enc in topology['encoding'].items():
    #         receptive_field = None
    #         if 'receptive_field' in enc and enc['receptive_field'] is not None and\
    #             len(enc['receptive_field']) > 0:
    #             receptive_field_cls = receptive_fields[enc['receptive_field']['name']]
    #             receptive_field = receptive_field_cls(n_inputs=topology['stimuli'][name]['n'],\
    #                         **enc['receptive_field']['params'])
    #         self._encoders.update({topology['stimuli'][name]['sensor'] :
    #                 encoders[enc['scheme']](topology['stimuli'][name]['n'],\
    #                 topology['time_scale'], receptive_field, dt=1e-3)})

    def reset(self):
        """ Resets all the encoders. """
        for encoder in self._encoders.values():
            encoder.reset()

    @property
    def all(self):
        """ Return all the encoders. """
        return self._encoders

    def get(self, key):
        """ Return the encoder corresponding to key."""
        if key not in self._encoders:
            raise Exception(logging.error('Encoder corresponding to key '\
                '"{}" does not exist.'.format(key)))
        return self._encoders[key]

    @GET('encoders:weights')
    def get_encoding_weights(self, enc_name, min_val=0., max_val=1., only_trainable=True):
        if enc_name == 'all':
            weights = np.hstack([encoder.receptive_field.weights.copy()\
                for encoder in self._encoders.values() if encoder.receptive_field.trainable])
            return (weights - min_val) / (max_val - min_val)

    @SET('encoders:weights')
    def set_encoding_weights(self, enc_name, data, min_val=0., max_val=1.):
        data = min_val + data * (max_val - min_val)
        if enc_name == 'all':
            pointer = 0
            for encoder in self._encoders.values():
                if encoder.receptive_field.trainable:
                    weights_len = encoder.receptive_field.weights.shape[0]
                    encoder.receptive_field.weights = data[pointer : pointer + weights_len].copy()
                    pointer += weights_len

    @INIT('encoders:weights')
    def init_encoding_weights(self, enc_name, min_val=0., max_val=1., only_trainable=True):
        weights_len = self.len_encoding_weights(enc_name)
        random_weights = np.random.random(size=weights_len)
        self.set_encoding_weights(enc_name, random_weights, min_val=min_val, max_val=max_val)

    @LEN('encoders:weights')
    def len_encoding_weights(self, enc_name, only_trainable=True):
        return self.get_encoding_weights(enc_name, only_trainable=only_trainable).shape[0]

class Encoder:
    """ Base (abstract) class for neural encoding processes.
    =============================================================
    - Params:
        n_stimuli [int]: dimension of the stumuli vector to encode.
        time_scale [int]: ratio between neuronal and environment time scales.
        receptive_field [ReceptiveField or None]: receptive field instance that, 
            normally, encodes stimulus into firing rates or delays to produce
            spikes. RFs also augment the input dim. If the encoder doesn't employ
            receptive fields, the argument is None
    =============================================================
    """
    def __init__(self, n_stimuli, time_scale, receptive_field):
        self.n_stimuli = n_stimuli
        self.time_scale = time_scale
        self.receptive_field = receptive_field
        self.t = 0

    def step(self, stimuli):
        """ Step the encoder by transforming input stimuli into spikes. """
        raise NotImplementedError

    def reset(self):
        """ Reset the encoding states (if any). """
        self.t = 0
    
    @property
    def num_neurons(self):
        """ Returns the number of nodes encoding stimuli. It is computed 
        as dim(stimuli) * N_RF, where N_RF is the number of nodes or "neurons" 
        encoding each stimulus in the receptive field. 
        """
        return int(self.n_stimuli * (self.receptive_field.n_neurons\
                if self.receptive_field is not None else 1))

    def plot(self, stimuli):
        """ Plots a visual example of encoding of the provided stimuli. 
        The subplot shows in the left side the receptive fields of the neurons 
        and the corresponding firing rate or delay in the y-axis. The 
        right suplot shows the corresponding encoded spike trains for 1000 time steps.
        """
        if self.receptive_field is None:
            raise Exception(logging.error('Unable to plot Encoding if the '\
            'receptive field is None.'))
        stimuli = np.array([stimuli]) if type(stimuli) not in [list, np.ndarray] else stimuli
        f, axes = plt.subplots(1, 2, figsize=(15, 7))
        axes[0] = self.receptive_field.plot(stimuli, ax=axes[0])
        spikes = [self.step(stimuli) for _ in range(1000)]
        spikes = np.vstack(spikes).T
        colors = [l.get_color() for l in axes[0].get_lines()]
        axes[1].eventplot([np.where(v)[0] for v in spikes], lineoffsets=1,\
            linelengths=.5, linewidths=1, color=colors[:spikes.shape[0]])
        axes[1].set_ylabel('Neuron'); axes[1].set_xlabel('Time')
        axes[1].set_xlim(0, 1000)
        axes[1].set_yticks(range(spikes.shape[0]))
        self.reset()
        plt.show()


    def stim_rate_curve(self):
        """ Plots the Firing Rate - Stimulus curve depicting the approx. spiking rate 
        of the encoded stimulus.
        """
        if self.receptive_field is None:
            raise Exception(logging.error('Unable to plot Rate-Stimulus curve if the '\
            'receptive field is None.'))
        f, ax = plt.subplots()
        xx = np.linspace(self.receptive_field.min_stim, self.receptive_field.max_stim, 200)
        firing_rates = np.empty((0, self.receptive_field.n_neurons))
        for x_i in xx:
            self.reset()
            spikes = np.vstack([self.step(np.array([x_i])) for _ in range(1000 // self.time_scale)])
            Fi = np.mean([np.sum(chunk,0) / chunk.shape[0] for chunk in np.split(spikes, 50)], 0)
            firing_rates = np.vstack((firing_rates, Fi))
        plt.plot(xx, firing_rates)
        plt.ylabel('Firing Rate')
        plt.xlabel('Stimulus')
        self.reset()
        plt.show()

@encoding_registry
class IdentityEncoding(Encoder):
    """ Identity or  dummy encoding that outputs the untouched input. 
    If time_scale > 1 then the encoding returns a numpy matrix with time_scale 
    times the stimuli (stim) vector repeated (shape of time_scale x dim(stim)).
    """
    def __init__(self, *args, **kwargs):
        super(IdentityEncoding, self).__init__(*args,)

    def step(self, stim):
        """ Step of the encoder. It simply returns the same value as the input, 
        repeated time_scale times.
        """
        if self.time_scale == 1:
            return stim
        return np.stack([stim for _ in range(self.time_scale)])

    def stim_rate_curve(self):
       raise Exception(logging.error('Encoding Firing Rate - Stimulus is not '\
            'available in Identity Encoding.'))

    def plot(self, stimuli):
        raise Exception(logging.error('Encoding plot is not '\
            'available in Identity Encoding.'))


@encoding_registry
class LIF_Encoding(Encoder):
    """ Maps stimulus to the input current of LIF neurons that produce 
    spikes. The mapping to currents is generated by means of a receptive 
    field.
    """
    def __init__(self, *args, **kwargs):
        super(LIF_Encoding, self).__init__(*args)
        self.lif_model = None
        self.create_lif(self.n_stimuli * self.receptive_field.n_neurons)

    def create_lif(self, num_neurons):
        self.lif_model = LIFModel(1.)
        for n in range(num_neurons):
            self.lif_model.add()

    @increase_time
    def step(self, stimuli):
        """ Encoding Step of LIF enc. It firstly converts stimuli into currents. 
        Subsequently the currents are applied to an ensemble of LIF neurons that 
        return the corresponding spikes.
        """
        stimuli = stimuli.flatten()
        currents = np.hstack([self.receptive_field(stim) for stim in stimuli])
        spikes = np.stack([self.lif_model.step(currents.flatten())[0]\
                    for _ in range(self.time_scale)]).squeeze()
        return spikes

    def reset(self):
        """ Resets the encoding. In this case, it resets the LIF neurons' dynamics.
        """
        self.t = 0
        self.lif_model.reset()

    def stim_rate_curve(self):
        self.create_lif(self.receptive_field.n_neurons)
        super().stim_rate_curve()
        self.create_lif(self.n_stimuli * self.receptive_field.n_neurons)

    def plot(self, stimuli):
        self.create_lif(self.receptive_field.n_neurons)
        self.reset()
        super().plot(stimuli)
        self.create_lif(self.n_stimuli * self.receptive_field.n_neurons)

@encoding_registry
class PoissonRateCoding(Encoder):
    """ Maps stimulus value to firing rate of a Poisson process of 
    spike generation. The specified receptive field is used to generate the 
    rates.
    """
    def __init__(self, *args, **kwargs):
        super(PoissonRateCoding, self).__init__(*args,)
        self.dt = kwargs['dt']
        self.s = np.ones(self.num_neurons) # survive prob
        self.prev_output = np.zeros(self.num_neurons) # previous output sample (spikes)

    def step(self, stimuli):
        """ Step the dynamics of the Poisson process and stochastically sample a spike 
        based on the survival probability.
        """
        stimuli = stimuli.flatten()
        rates = np.hstack([self.receptive_field(stim).astype(int) for stim in stimuli])
        self.s += np.array([self.dt * (-rate / 10 * s) for s, rate in zip(self.s, rates)])
        output = np.array([np.random.choice(2, p=[prob, 1-prob]) for prob in self.s])
        self.s[output.astype(bool)] = 1.
        return output

    def reset(self):
        """ Reset the dynamics of the Poisson process. """
        self.t = 0
        self.s = np.ones(self.num_neurons) # survive prob
        self.prev_output = np.zeros(self.num_neurons)

    def stim_rate_curve(self):
        """ Plots the Firing Rate - Stimulus curve depicting the approx. spiking rate 
        of the encoded stimulus.
        """
        logging.info('Ploting the Firing Rate - Stimulus curve. It may take a while.')
        f, ax = plt.subplots()
        xx = np.linspace(self.receptive_field.min_stim, self.receptive_field.max_stim, 1000)
        firing_rates = np.empty((0, self.receptive_field.n_neurons))
        for x_i in xx:
            self.s = np.ones(self.receptive_field.n_neurons)
            self.prev_output = np.zeros(self.receptive_field.n_neurons)
            spikes = np.vstack([self.step(np.array([x_i])) for _ in range(10000 // self.time_scale)])
            Fi = np.mean([np.sum(chunk,0) / chunk.shape[0] for chunk in np.split(spikes, 50)], 0)
            firing_rates = np.vstack((firing_rates, Fi))
        plt.plot(xx, firing_rates)
        plt.ylabel('Firing Rate')
        plt.xlabel('Stimulus')
        self.reset()
        plt.show()

    def plot(self, stimuli):
        self.s = np.ones(self.receptive_field.n_neurons)
        self.prev_output = np.zeros(self.receptive_field.n_neurons)
        super().plot(stimuli)
        self.reset()

@encoding_registry
class RankOrderCoding(Encoder):
    """ Rank Order Coding (ROC) scheme. It is a temporal encoding in which the 
    stimulus is firstly mapped to spike delays, fixing some ordering, using 
    receptive fields. Then, a matrix of spikes of shape [time_scale x num_neurons] 
    is created with ony one spike per neuron at the corresponding delay. Note 
    that this encoding imposes that time_scale is greater that 1. 
    """
    def __init__(self, *args, **kwargs):
        super(RankOrderCoding, self).__init__(*args,) #**kwargs)
        if self.time_scale < self.num_neurons:
            raise Exception(logging.error('ROC encoding can only be used with time scales '\
                    'greater than the number of neurons.'))
        
    def step(self, stimuli):
        stimuli = [stimuli] if type(stimuli) not in [list, np.ndarray] else stimuli
        delays = np.hstack([self.receptive_field(stim).astype(int) for stim in stimuli])
        spikes = np.eye(self.time_scale)[delays-1].T
        return spikes
    
    def stim_rate_curve(self):
       raise Exception(logging.error('Encoding Firing Rate - Stimulus is not '\
            'available in Rank Order Coding.'))
    
    def plot(self, stimuli):
        stimuli = [stimuli] if type(stimuli) not in [list, np.ndarray] else stimuli
        f, axes = plt.subplots(1, 2, figsize=(15, 7))
        xx = np.linspace(self.receptive_field.min_stim, self.receptive_field.max_stim, 1000)
        yy = [self.receptive_field(x) for x in xx]
        lines = axes[0].plot(xx, yy)
        for stim in stimuli:
            axes[0].vlines(stim, self.receptive_field.min_val, self.receptive_field.max_val)
            for delay in self.receptive_field(stimuli):
                axes[0].plot(stim, delay, 'ko')
        
        axes[0].legend(lines, ['Neuron ' + str(i) + ' with mu=' + str(round(mu, 2))\
                    for i, mu in enumerate(self.receptive_field.centers)])
        axes[0].set_xlabel('Stimulus')
        axes[0].set_ylabel('Time to First Spike')
        # Plot spike
        spikes = self.step(stimuli).T
        axes[1].eventplot([np.where(v)[0]  for v in spikes], lineoffsets=1, linelengths=.5,\
                            linewidths=3, color=[l.get_color() for l in lines])
        axes[1].set_ylabel('Neuron')
        axes[1].set_xlabel('Time')
        axes[1].set_xlim(-1, self.time_scale)
        axes[1].set_yticks(range(5))
        plt.show()


#!TODO
# class PopulationVectorEncoding(Encoder):
#     def __init__(self, *args, **kwargs):
#         super(PopulationVectorEncoding, self).__init__(*args,)
#         #! ojo depender de otra var y no mu_vec
#         from scipy.linalg import circulant
#         self.Phi = circulant(np.r_[[1, .5], np.zeros(8-3), [0.5]])
#         self.s = np.ones_like(self.mu_vec) # survive prob
#         self.prev_output = 0 # previous output sample (spike)

#     def step(self, stimuli):
#         yy = self.Phi.dot(stimuli)
#         rates = [self.receptive_field(yy, mu) for mu in self.mu_vec]
        
#         self.s += np.array([1e-3 * (-rate * s) for s, rate in zip(self.s, rates)])
#         output = np.array([np.random.choice(2, p=[prob, 1-prob]) for prob in self.s])
#         self.s[output.astype(bool)] = 1.
#         return output

#     def plot_ROC(self, stim_val):
#         f, axes = plot.subplots(1, 2, figsize=(15, 7))
#         xx = np.linspace(self.min_stim, self.max_stim, 1000)
#         lines = []
#         for mu in self.mu_vec:
#             y = np.array([self.receptive_field(v, mu) for v in xx])
#             lines.append(axes[0].plot(xx, y)[0])
#             axes[0].vlines(stim_val, self.min_val, self.max_val)
#             axes[0].plot(stim_val, self.receptive_field(stim_val, mu), 'ko')
#         axes[0].legend(lines, ['Neuron '+str(i)+' with mu='+str(round(mu,2)) for i, mu in enumerate(self.mu_vec)])
#         axes[0].set_xlabel('Stimulus'); axes[0].set_ylabel('Firing Rate')
        
#         spikes = [self.step(stim_val) for _ in range(1000)]
#         spikes = np.vstack(spikes).T
#         axes[1].eventplot([np.where(v)[0]  for v in spikes], lineoffsets=1, linelengths=.5, linewidths=1)
#         axes[1].set_ylabel('Neuron'); axes[1].set_xlabel('Time')
#         axes[1].set_xlim(0, 1000)
#         axes[1].set_yticks(range(5))
#         plot.show()

#     def reset(self):
#         self.t = 0
#         self.s = np.ones_like(self.mu_vec) # survive prob
#         self.prev_output = np.zeros_like(self.mu_vec)


#     def step(self, stimuli):
#         stimuli = [stimuli] if not isinstance(stimuli, list) else stimuli
#         delays = np.hstack([self.receptive_field(stim).astype(int) for stim in stimuli])
#         spikes = np.eye(self.time_scale)[delays-1].T
#         return spikes