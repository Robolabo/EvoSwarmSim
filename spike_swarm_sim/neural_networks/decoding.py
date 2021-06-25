import logging
import numpy as np
from spike_swarm_sim.register import decoding_registry, decoders
from spike_swarm_sim.utils import softmax, sigmoid, tanh
from spike_swarm_sim.globals import global_states
from spike_swarm_sim.algorithms.interfaces import GET, SET, LEN, INIT

class DecodingWrapper:
    """ Wrapper for gathering all the decoders of each ANN output. 
    It contains a dict mapping output names to Decoder instances.
    =============================================================================
    - Params:
        topology [dict] : configuration dict of the overall topology.
    - Attributes:
        motor_ensembles [dict] :  dict mapping motor ensembles to num of neurons.
        decoding_config [dict] : config. dict of the decoding (extracted from 
                topology config.).
        act_ens_map [dict] : dict mapping output names to motor ensemble names.
    =============================================================================
    """
    def __init__(self, topology=None):
        self.motor_ensembles = None
        self.decoding_config = None
        self.act_ens_map = None
        self._decoders = {}
        if topology is not None:
            self.build(topology)

    def step(self, spikes):
        """ Steps all the decoders with the corresponding spikes or activities. """
        if spikes.shape[1] != sum(tuple(self.motor_ensembles.values())):
            raise Exception(logging.error('The dim. of spikes to be decoded must be the same '\
                'as the number of motor neurons.'))
        actions = {}
        current_idx = 0
        for name, decoder in self._decoders.items():
            dec_spikes = spikes[:, current_idx : current_idx + self.motor_ensembles[self.act_ens_map[name]]]
            actions.update({name : decoder.step(dec_spikes)})
            current_idx += self.motor_ensembles[self.act_ens_map[name]]
        return actions        

    def build(self, topology):
        """ Builds the decoders using the topology config. dict. """
        self.motor_ensembles = {out['ensemble'] : topology['ensembles'][out['ensemble']]['n']\
                for out in topology['outputs'].values()}
        self.decoding_config = topology['decoding'].copy()
        self.act_ens_map = {out_name : out['ensemble']  for out_name, out in topology['outputs'].items()}
        for decoder_name, decoder in topology['decoding'].items():
            self._decoders.update({decoder_name : decoders[decoder['scheme']]({
                self.act_ens_map[decoder_name] : self.motor_ensembles[self.act_ens_map[decoder_name]]},\
                **decoder['params'])})

    @property
    def all(self):
        """ Return all the decoders. """
        return self._decoders

    def get(self, key):
        """ Return the decoder corresponding to key. """
        if key not in self._decoders.keys():
            raise Exception(logging.error('Decoder corresponding to key {} does not exist.'\
                    .format(key)))
        return self._decoders[key]

    def reset(self):
        """ Resets all the decoders. """
        for decoder in self._decoders.values():
            decoder.reset()

    @GET('decoders:weights')
    def get_decoding_weights(self, dec_name, ann_graph, min_val=0., max_val=1., only_trainable=True):
        if dec_name == 'all':
            weights = np.hstack([decoder.w.copy() for decoder in self._decoders.values() if decoder.trainable])
            return (weights - min_val) / (max_val - min_val)
        return (self._decoders[dec_name].w.copy() - min_val) / (max_val - min_val)

    @SET('decoders:weights')
    def set_decoding_weights(self, dec_name, ann_graph, data, min_val=0., max_val=1.):
        data = min_val + data * (max_val - min_val)
        if dec_name == 'all':
            pointer = 0
            for decoder in self._decoders.values():
                if decoder.trainable:
                    decoder.w = data[pointer:pointer+decoder.w.shape[0]].copy()
                    pointer += decoder.w.shape[0]
        else:
            self._decoders[dec_name].w = data.copy()
        return ann_graph#!

    @INIT('decoders:weights')
    def init_decoding_weights(self, dec_name, ann_graph, min_val=0., max_val=1., only_trainable=True):
        weights_len = self.len_decoding_weights(dec_name, ann_graph)
        random_weights = np.random.random(size=weights_len)
        return self.set_decoding_weights(dec_name, ann_graph, random_weights, min_val=min_val, max_val=max_val)

    @LEN('decoders:weights')
    def len_decoding_weights(self, conn_name, ann_graph, only_trainable=True):
        return len(self.get_decoding_weights(conn_name, ann_graph, only_trainable=True))


class Decoder:
    """ Base class for decoding ANN outputs (spikes, activities, ...) 
    into actions.
    ===================================================================
    - Params :
        out_ensembles [dict] : dict mapping output/motor ensembles to 
                number of neurons.
        trainable [bool] : whether there are trainable/optimizable 
                variables in the decoder. # TODO check
        is_cat [bool] : whether the decoded actions are categorical or 
                numerical.
    ===================================================================
    """
    def __init__(self, out_ensembles, trainable=False, is_cat=True):
        self.out_ensembles = out_ensembles
        self.trainable = trainable
        self.is_categorical = is_cat

    def reset(self):
        pass

@decoding_registry
class IdentityDecoding(Decoder):
    """ Identity of dummy decoding of activities (not supported for spikes). 
    It outputs the same value as the input. If time_scale > 1 then it returns the 
    input value at the last time step.
    """
    def __init__(self, *args, **kwargs):
        super(IdentityDecoding, self).__init__(*args, **kwargs, trainable=False)

    def step(self, spikes):
        #* choose last state for action generation
        return spikes[-1].tolist()


@decoding_registry
class ThresholdDecoding(Decoder):
    """ Decodes activities into binary actions by means of applying a 
    heaviside function.
    ====================================================================
    - Args:
        threshold [float]
    """
    def __init__(self, *args, stochastic=False, **kwargs):
        super(ThresholdDecoding, self).__init__(*args, **kwargs, trainable=False)
        self.threshold = 0.5
        self.stochastic = stochastic

    def step(self, spikes):
        spikes_last = spikes[-1].flatten() #* choose last state for action generation
        if len(spikes_last) == 1:
            if self.stochastic:
                return np.random.choice(2, p=(1-spikes_last[0], spikes_last[0]))
            else:
                return int(spikes_last >= self.threshold)
        else:
            return (spikes_last >= self.threshold).astype(int).tolist()


@decoding_registry
class ArgmaxDecoding(Decoder):
    """ #TODO
    """
    def __init__(self, *args, **kwargs):
        kwargs['trainable'] = False
        super(ArgmaxDecoding, self).__init__(*args, **kwargs)

    def step(self, s):
        pass

@decoding_registry
class SoftmaxDecoding(Decoder):
    """ Decodes activities into categorical actions by transforming activities 
    into probabilities with softmax and sampling the action with a categorical 
    dist.
    """
    def __init__(self, *args, **kwargs):
        kwargs['trainable'] = False
        super(SoftmaxDecoding, self).__init__(*args, **kwargs)

    def step(self, activities):
        activities = activities[-1] #* choose last state for action generation
        action = np.random.choice(len(activities), p=softmax(5 * activities))
        return action
    
@decoding_registry
class FirstToSpike(Decoder):
    """ Temporal decoding process in which selected action in a time 
        window is defined by the output neuron that spiked first.
        It is used when the action is categorical and low dimensional.
    """
    def __init__(self, *args, **kwargs):
        kwargs['trainable'] = False
        super(FirstToSpike, self).__init__(*args, **kwargs)
        
    def step(self, spikes):
        action = np.argmin([(spikes.shape[0], np.argmax(v))[v.sum() > 0] for v in spikes.T])
        return action

@decoding_registry
class RankOrderDecoding(Decoder):
    # TODO: Not tested. May produce errors
    """ Temporal decoding process in which the selected action within a time window is defined 
        based on a linear combination of the order of neuron spikes.
    """
    def __init__(self, *args, **kwargs,):
        kwargs['trainable'] = True
        super(RankOrderDecoding, self).__init__(*args, **kwargs)
        self.w = np.random.random(4)
        
    def step(self, spikes):
        order = np.argsort([(spikes.shape[0], np.argmax(v))[v.sum() > 0] for v in spikes.T])
        action = (spikes.sum(0) * self.w).dot(.25**order)
        return action
    
@decoding_registry
class LinearPopulationDecoding(Decoder):
    """ Decodes spikes as a linear transformation of the computed activities of 
    each motor neuron. More specifically, it is computed as follows:
        -- actions = sigmoid(3 * W.dot(u(t)) --
    where W is a weight matrix and u(t) is the vector of activities/rates computed 
    as the filtering of the spike trains.
    The weights are normally optimized and can be addressed by the evolutionary alg. 
    with the query "decoders:weights:all".
    ====================================================================================
    - Params:
        num_outputs [int] : number of output actions.
        tau_decay [float] : decaying time constant of the neuron's activity after spike. 
        tau_rise [float] : rise time constant of the neuron's activity after spike. 
    - Attributes:
        dt [float] : Euler step to compute the activities.
        activities [np.ndarray] : current activities of the neurons of the ensemble.
        decoded_activities [np.ndarray] : decoded activities.
        x [np.ndarray] : auxiliary variable to filter spikes.
        w [np.ndarray] :  flattened or vectorized decoding weight matrix.
        action_recodings [np.ndarray] : Data recodings of the decoded activities.
                # TODO: Provisional, pasar a monitor.
    ====================================================================================
    """
    def __init__(self, *args, num_outputs=1, tau_decay=30.,
                    tau_rise=5., non_linearity='sigmoid',
                    rest_value=0.5, **kwargs):
        kwargs['trainable'] = True
        super(LinearPopulationDecoding, self).__init__(*args, **kwargs)
        self.dt = 1.
        self.num_outputs = num_outputs
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay
        self.rest_value = rest_value
        self.non_linearity = {
            'tanh' : tanh,
            'sigmoid' : sigmoid,
        }.get(non_linearity, 'sigmoid')

        # Dynamics
        self.activities = np.zeros(tuple(self.out_ensembles.values())[0])
        self.x = np.zeros_like(self.activities)
        self.w = np.ones(self.num_outputs * self.activities.shape[0]) # not important, set in initialize 
        self.action_recording = np.empty((0, num_outputs)) if global_states.DEBUG else None

    def step(self, spikes_batch):
        spikes_batch = spikes_batch.astype(bool)
        for spikes in spikes_batch:
            self.x[spikes] = 1
            self.x[~spikes] += self.dt * (-self.x[~spikes] / self.tau_rise)
            self.activities += self.dt * (.5 * self.x * (1 - self.activities)\
                            - self.activities / self.tau_decay)
        actions = self.decoded_activities if not self.is_categorical \
                else (self.decoded_activities >= .5).astype(int)
        if global_states.DEBUG:
            self.action_recording = np.vstack((self.action_recording, actions))
        return actions

    @property
    def decoded_activities(self):
        W = self.w.reshape(self.num_outputs, self.activities.shape[0]).T
        # output = sigmoid(3 * (W.T.dot(self.activities) - 1))
        output = self.non_linearity(3 * (W.T.dot(self.activities)))
        # output += (self.rest_value - 0.5)
        return output

    def reset(self):
        self.x = np.zeros_like(self.x)
        self.activities = np.zeros_like(self.x)
