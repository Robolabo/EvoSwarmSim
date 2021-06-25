from abc import ABC, abstractmethod, abstractproperty
import matplotlib.pyplot as plot
import numpy as np
from spike_swarm_sim.algorithms.interfaces import GET, SET, LEN, INIT
from spike_swarm_sim.register import neuron_model_registry
from spike_swarm_sim.utils import sigmoid, tanh, increase_time

class BaseNeuronModel(ABC):
    """ Base abstract class for neuron models."""
    def __init__(self, dt):
        self.dt = dt
        self.bias = np.empty(0)
        self._volt = np.empty(0)

    @abstractmethod
    def step(self, Isyn):
        pass
    
    def __len__(self):
        return len(self._volt)

    # @abstractmethod
    # def add(self, Isyn):
    #     pass

    def build(self, **kwargs):
        for var, val in kwargs.items():
            self.__dict__[var] = np.array(val) if isinstance(val, list) else val
            if isinstance(self.__dict__[var], float) or isinstance(self.__dict__[var], int):
                self.__dict__[var] = np.repeat(self.__dict__[var], len(self))
        self.reset()

    @abstractmethod
    def reset(self):
        pass
    
    @property
    def voltages(self):
        return self._volt.copy()

class SpikingNeuronModel(BaseNeuronModel):
    def __init__(self, *args, **kwargs):
        super(SpikingNeuronModel, self).__init__(*args, **kwargs)
        self._theta = np.empty(0)
        self._recov = np.empty(0)
    
    @property
    def recovery(self):
        return self._recov.copy()
    
    @property
    def theta(self):
        return self._theta.copy()

class NonSpikingNeuronModel(BaseNeuronModel):
    def __init__(self, *args, **kwargs):
        super(NonSpikingNeuronModel, self).__init__(*args, **kwargs)
        self.bias = np.empty(0)
        
    @GET('neurons:bias')
    def get_bias(self, neuron_name, ann_graph, min_val=0, max_val=1):
        bias_vals = np.array({
            'all' : [neuron['bias'] for neuron in ann_graph['neurons'].values()],
            'hidden' : [neuron['bias'] for neuron in ann_graph['neurons'].values() if not neuron['is_motor']], 
            'motor' : [neuron['bias'] for neuron in ann_graph['neurons'].values() if neuron['is_motor']]
        }.get(neuron_name,  [neuron['bias'] for neuron in ann_graph['neurons'].values() \
                if neuron['ensemble'] == neuron_name]))
        return (bias_vals - min_val) / (max_val - min_val)

    @SET('neurons:bias')
    def set_bias(self, neuron_name, ann_graph, data, min_val=0, max_val=1):
        neuron_iterable = {
            'all' : ann_graph['neurons'].values(),
            'hidden' : filter(lambda x: not x['is_motor'], ann_graph['neurons'].values()),
            'motor' : filter(lambda x: x['is_motor'], ann_graph['neurons'].values())
        }.get(neuron_name, filter(lambda x: x['ensemble'] == neuron_name, ann_graph['neurons'].values()))
        for bias, neuron in zip(data, neuron_iterable):
            neuron['bias'] = bias * (max_val - min_val) + min_val
            self.bias[neuron['idx']] = neuron['bias']
        return ann_graph
    
    @LEN('neurons:bias')
    def len_bias(self, neuron_name, ann_graph):
        return len(self.get_bias(neuron_name, ann_graph))

    @INIT('neurons:bias')
    def init_bias(self, neuron_name, ann_graph, min_val=-1., max_val=1.):
        biases_len = self.len_bias(neuron_name, ann_graph)
        random_biases = 0.5 * np.random.randn(biases_len) * 0.3
        random_biases = np.clip(random_biases, a_min=0, a_max=1)
        return self.set_bias(neuron_name, ann_graph, random_biases, min_val=min_val, max_val=max_val)

@neuron_model_registry(name='rate_model')
class RateModel(NonSpikingNeuronModel):
    """ Class for the Rate model or non spiking model mainly used as building block 
    of CTRNNs. 
    """
    def __init__(self, *args):
        super(RateModel, self).__init__(*args)
        self.tau = np.empty(0)
        self.bias = np.empty(0)
        self.gain = np.empty(0)
        self.activation = np.empty(0)
        # self.build(tau=tau, gain=gain, bias=bias, activation=activation)
        # self.reset()
    
    def step(self, Isyn):
        self._volt += (self.dt / self.tau) * (Isyn.copy() - self._volt)
        outputs = self.gain * self._volt.copy() + self.bias
        outputs[self.activation == 'sigmoid'] = sigmoid(outputs[self.activation == 'sigmoid'])
        outputs[self.activation == 'tanh'] = tanh(outputs[self.activation == 'tanh'])
        return outputs, self._volt.copy()

    #! pasarlo a base
    def add(self, tau=1., gain=1., bias=0., activation='sigmoid'):
        self._volt = np.hstack((self._volt, 0))
        self.tau = np.hstack((self.tau, tau)) # np.insert(self.tau, index, values=tau, axis=0)
        self.bias = np.hstack((self.bias, bias))
        self.gain = np.hstack((self.gain, gain))
        self.activation = np.hstack((self.activation, activation))

    def delete(self, index):
        self._volt = np.delete(self._volt, index)
        self.tau = np.delete(self.tau, index)
        self.bias = np.delete(self.bias, index)
        self.gain = np.delete(self.gain, index)
        self.activation = np.delete(self.activation, index)
        
    def reset(self):
        self._volt = np.zeros(len(self))

    def build(self, tau=1., gain=1., bias=0., activation='sigmoid'):
        super().build(tau=tau, gain=gain, bias=bias)
        self.activation = np.array(activation) if isinstance(activation, list) else activation
        if isinstance(activation, str):
            self.activation = np.repeat(activation, len(self))
     
    @GET('neurons:tau')
    def get_tau(self, neuron_name, ann_graph, min_val=0, max_val=1):
        tau_vals = np.array({
            'all' : [neuron['tau'] for neuron in ann_graph['neurons'].values()],
            'hidden' : [neuron['tau'] for neuron in ann_graph['neurons'].values() if not neuron['is_motor']], 
            'motor' : [neuron['tau'] for neuron in ann_graph['neurons'].values() if neuron['is_motor']]
        }.get(neuron_name,  [neuron['tau'] for neuron in ann_graph['neurons'].values() \
                if neuron['ensemble'] == neuron_name]))
        if any(tau_vals == 0): import pdb; pdb.set_trace() 
        return (np.log10(0.5 * tau_vals) - min_val) / (max_val - min_val)

    @SET('neurons:tau')
    def set_tau(self, neuron_name, ann_graph, data, min_val=0, max_val=1):
        neuron_iterable = {
            'all' : ann_graph['neurons'].values(),
            'hidden' : filter(lambda x: not x['is_motor'], ann_graph['neurons'].values()),
            'motor' : filter(lambda x: x['is_motor'], ann_graph['neurons'].values())
        }.get(neuron_name, filter(lambda x: x['ensemble'] == neuron_name, ann_graph['neurons'].values()))
        for tau, neuron in zip(data, neuron_iterable):
            neuron['tau'] = 2 * 10 ** (tau * (max_val - min_val) + min_val)
            if neuron['tau'] == 0: import pdb; pdb.set_trace() 
            self.tau[neuron['idx']] = neuron['tau']
        return ann_graph

    @LEN('neurons:tau')
    def len_tau(self, neuron_name, ann_graph):
        return len(self.get_tau(neuron_name, ann_graph, min_val=-1, max_val=.5))

    @INIT('neurons:tau')
    def init_tau(self, neuron_name, ann_graph, min_val, max_val):
        tau_len = self.len_tau(neuron_name, ann_graph)
        random_taus = np.random.random(size=tau_len)
        random_taus = np.random.rayleigh(scale=0.6, size=tau_len) / 3
        random_taus = np.clip(random_taus, a_min=0, a_max=1)
        return self.set_tau(neuron_name, ann_graph, random_taus, min_val=min_val, max_val=max_val)

    @GET('neurons:gain')
    def get_gain(self, neuron_name, ann_graph, min_val=0, max_val=1):
        gain_vals = np.array({
            'all' : [neuron['gain'] for neuron in ann_graph['neurons'].values()],
            'hidden' : [neuron['gain'] for neuron in ann_graph['neurons'].values() if not neuron['is_motor']], 
            'motor' : [neuron['gain'] for neuron in ann_graph['neurons'].values() if neuron['is_motor']]
        }.get(neuron_name,  [neuron['gain'] for neuron in ann_graph['neurons'].values() \
                if neuron['ensemble'] == neuron_name]))
        return (gain_vals - min_val) / (max_val - min_val)
    
    @SET('neurons:gain')
    def set_gain(self, neuron_name, ann_graph, data, min_val=0, max_val=1):
        neuron_iterable = {
            'all' : ann_graph['neurons'].values(),
            'hidden' : filter(lambda x: not x['is_motor'], ann_graph['neurons'].values()),
            'motor' : filter(lambda x: x['is_motor'], ann_graph['neurons'].values())
        }.get(neuron_name, filter(lambda x: x['ensemble'] == neuron_name, ann_graph['neurons'].values()))
        for gain, neuron in zip(data, neuron_iterable):
            neuron['gain'] = gain * (max_val - min_val) + min_val
            self.gain[neuron['idx']] = neuron['gain']
        return ann_graph

    @LEN('neurons:gain')
    def len_gain(self, neuron_name, ann_graph):
        return len(self.get_gain(neuron_name, ann_graph))

    @INIT('neurons:gain')
    def init_gain(self, neuron_name, ann_graph, min_val, max_val):
        gain_len = self.len_gain(neuron_name, ann_graph)
        random_gains = np.random.random(size=gain_len)
        return self.set_gain(neuron_name, ann_graph, random_gains, min_val=min_val, max_val=max_val)

@neuron_model_registry(name='adex')
class AdExModel(SpikingNeuronModel):
    """ Class for the Adaptive Exponenitial LIF spiking neuron model. """
    def __init__(self, *args):
        super(AdExModel, self).__init__(*args)
        self.tau_w = np.empty(0)
        self.tau_m = np.empty(0)
        self.V_rest = np.empty(0)
        self.V_reset = np.empty(0)
        self.A = np.empty(0)
        self.B = np.empty(0)
        self.theta_rest = np.empty(0)
        self.time_refrac = 10
        self.R = 1.
        self.refractoriness = None
        self.t = 0
        #* --- Build Params ---
        # self.build(tau_m=tau_m, tau_w=tau_w, V_rest=V_rest,
        #             V_reset=V_reset, A=A, B=B, theta_rest=theta_rest)
        # self.reset()
        

    @increase_time
    def step(self, Isyn):
        # Isyn += 30 * (np.cos(2 * np.pi * 10 * 1e-3 * self.t) + np.random.randn(Isyn.shape[0]))
        Isyn += 20 * np.random.randn(Isyn.shape[0])
        self._volt += (self.dt / self.tau_m) * (self.V_rest - self._volt\
                    + 2 * np.exp((self._volt - (self._theta)) / 2) - self._recov + Isyn)
        self._volt = np.clip(self._volt, a_min=None, a_max=30.)
        spikes = (self._volt >= 30.).astype(int)
        out_voltage = self._volt.copy()
        self._volt[self._volt >= 30.] = self.V_reset[self._volt >= 30.]
        self._recov += (self.dt / self.tau_w) * (self.A * (self._volt - self.V_rest)\
                     - self._recov + self.B * self.tau_w * spikes)
        self._theta += self.dt * ((1 - spikes) * (self.theta_rest - self._theta) / 50. + spikes * 20.)
        # self.refractoriness += self.time_refrac * spikes
        return spikes, out_voltage

    #! pasarlo a base
    def add(self, tau_m=10., tau_w=70., V_rest=-70.,
                V_reset=-55., A=0., B=5., theta_rest=-45.,):
        self._volt = np.hstack((self._volt, 0))
        self.tau_m = np.hstack((self.tau_m, tau_m)) # np.insert(self.tau, index, values=tau, axis=0)
        self.tau_w = np.hstack((self.tau_w, tau_w))
        self.V_rest = np.hstack((self.V_rest, V_rest))
        self.V_reset = np.hstack((self.V_reset, V_reset))
        self.A = np.hstack((self.A, A))
        self.B = np.hstack((self.B, B))
        self.theta_rest = np.hstack((self.theta_rest, theta_rest))

    def delete(self, index):
        self._volt = np.delete(self._volt, index)
        self.tau_m = np.delete(self.tau_m, index)
        self.tau_w = np.delete(self.tau_w, index)
        self.V_rest = np.delete(self.V_rest, index)
        self.V_reset = np.delete(self.V_reset, index)
        self.A = np.delete(self.A, index)
        self.B = np.delete(self.B, index)
        self.theta_rest = np.delete(self.theta_rest, index)

    def reset(self):
        self.t = 0
        self._volt = self.V_rest * np.ones(len(self))
        self._recov = self.A * (self._volt - self.V_rest)
        self.refractoriness = np.zeros_like(self._volt)
        self._theta = self.theta_rest * np.ones(len(self))


@neuron_model_registry(name='izhikevich')
class IzhikevichModel(SpikingNeuronModel):
    """ Class for the Izhikevich spiking neuron model. """
    def __init__(self, *args,
                A=0.02, B=0.2, C=-65., D=8.):
        super(IzhikevichModel, self).__init__(*args)
        # define vars, they are initialized in reset
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.build(A=A, B=B, C=C, D=D)
        self.reset()
        
    def step(self, Isyn):
        Isyn *= 0.1
        self._recov[self._volt >= 30.] += self.D[self._volt >= 30.].copy()
        self._volt[self._volt >= 30.] = self.C[self._volt >= 30.].copy()
        self._volt += self.dt * (.04 * (self._volt ** 2) \
                    + 5 * self._volt + 140 - self._recov + Isyn)
        self._volt = np.clip(self._volt, a_max=30., a_min=-85.)
        self._recov += self.dt * (self.A * (self.B * self._volt - self._recov))
        spikes = (self._volt.copy() >= 30).astype(int)
        return spikes, self._volt.copy()

    @property
    def theta(self):
        """ There is no theta in this neuron model. Zero array returned. """
        return np.zeros_like(self._volt)

    def reset(self):
        # Initialize at stable fixed point
        self._volt = self.C.copy()
        self._recov = (self.B * self._volt).copy()



@neuron_model_registry(name='lif')
class LIFModel(SpikingNeuronModel):
    """ Class for the Leaky Integrate and Fire (LIF) spiking neuron model. """
    def __init__(self, *args):
        super(LIFModel, self).__init__(*args)

        # define vars, they are initialized in reset
        self.tau = np.empty(0)
        self.v_rest = np.empty(0)
        self.thresh = np.empty(0)
        self.time_refrac = np.empty(0)
        self.R = np.empty(0)
        self.refractoriness = None
        # self.build(tau=tau, R=R, v_rest=v_rest, \
        #             thresh=thresh, time_refrac=time_refrac)
        # self.reset()
    
    def step(self, Isyn):
        Isyn[self.refractoriness > 0] = 0.
        self._volt += (self.dt / self.tau) * (self.v_rest - self._volt + Isyn)
        spikes = (self._volt >= self.thresh).astype(int)
        out_voltage = self._volt.copy()
        self._volt[spikes.astype(bool)] = self.v_rest[spikes.astype(bool)]
        self.refractoriness[self.refractoriness > 0] -= 1
        self.refractoriness[spikes.astype(bool)] = self.time_refrac[spikes.astype(bool)]
        return spikes.copy(), out_voltage

    #! pasarlo a base
    def add(self, tau=20., R=1., v_rest=-65., thresh=-50., time_refrac=5.):
        self._volt = np.hstack((self._volt, 0))
        self.tau = np.hstack((self.tau, tau)) # np.insert(self.tau, index, values=tau, axis=0)
        self.R = np.hstack((self.R, R))
        self.v_rest = np.hstack((self.v_rest, v_rest))
        self.thresh = np.hstack((self.thresh, thresh))
        self.time_refrac = np.hstack((self.time_refrac, time_refrac))


    @property
    def recovery(self):
        """ There is no recovery var. in this neuron model. Zero array returned."""
        return np.zeros_like(self._volt)

    @property
    def theta(self):
        """ There is no theta in this neuron model. Zero array returned. """
        return np.zeros_like(self._volt)

    def reset(self):
        self._volt = self.v_rest * np.ones(len(self))
        self.refractoriness = np.zeros_like(self._volt)



@neuron_model_registry(name='exp_lif')
class ExpLIFModel(SpikingNeuronModel):
    """ Class for the Exponetial LIF spiking neuron model. """
    def __init__(self, *args, tau=20., R=1., v_rest=-65., time_refrac=10., thresh=-40.):
        super(ExpLIFModel, self).__init__(*args)
        self.tau = None
        self.v_rest = None
        self.thresh = None
        self.time_refrac = None
        self.R = None
        self.refractoriness = None
        self.build(tau=tau, R=R, v_rest=v_rest,\
                    thresh=thresh, time_refrac=time_refrac)        
        self.reset()
    
    def step(self, Isyn):
        Isyn *= 0.6
        Isyn[self.refractoriness > 0] = 0.
        self._volt += (self.dt / self.tau) * (self.v_rest - self._volt \
                    + .3 * np.exp((self._volt - self.thresh) / 1) + Isyn)
        self._volt = np.clip(self._volt, a_max=30., a_min=None)
        spikes = (self._volt >= 30.).astype(int)
        out_voltages = self._volt.copy()
        self._volt[self._volt >= 30.] = self.v_rest[self._volt >= 30.]
        self.refractoriness[self.refractoriness > 0] -= 1
        self.refractoriness[spikes.astype(bool)] = self.time_refrac[spikes.astype(bool)]
        return spikes, out_voltages

    @property
    def recovery(self):
        """ There is no recovery var. in this neuron model. Zero array returned."""
        return np.zeros_like(self._volt)

    @property
    def theta(self):
        """ There is no theta in this neuron model. Zero array returned. """
        return np.zeros_like(self._volt)

    def reset(self):
        self._volt = self.v_rest * np.ones(self.num_neurons)
        self.refractoriness = np.zeros_like(self._volt)

@neuron_model_registry(name='morris_lecar')
class MorrisLecarModel(SpikingNeuronModel):
    """ Class for the Morris Lecar spiking neuron model. """
    def __init__(self, *args, phi=0.067, g_Ca=4., g_K=8., g_L=2., V1=-1.2, V2=18.,
                    V3=12., V4=17.4, E_Ca=120., E_K=-84., E_L=-60., Cm=20.):
        super(MorrisLecarModel, self).__init__(*args)
        # Model parameters (fixed for the moment)
        self.phi = None
        self.g_Ca = None
        self.g_K = None
        self.g_L = None
        self.V1 = None
        self.V2 = None
        self.V3 = None
        self.V4 = None
        self.E_Ca = None
        self.E_K = None
        self.E_L = None
        self.Cm = None
        self.build(phi=phi, g_Ca=g_Ca, g_K=g_K, g_L=g_L, V1=V1, V2=V2,
                    V3=V3, V4=V4, E_Ca=E_Ca, E_K=E_K, E_L=E_L, Cm=Cm)
        self.reset()
       
    def step(self, Isyn):
        m_inf = .5 * (1 + np.tanh((self._volt - self.V1) / self.V2))
        n_inf = .5 * (1 + np.tanh((self._volt - self.V3) / self.V4))
        tau_n = 1 / np.cosh((self._volt - self.V3) / (2 * self.V4))
        self._volt += (self.dt / self.Cm) * (Isyn - self.g_L * (self._volt - self.E_L)\
                - self.g_K * self._recov * (self._volt - self.E_K)\
                - self.g_Ca * m_inf * (self._volt - self.E_Ca))
        self._volt = np.clip(self._volt, a_max=40., a_min=-85.)
        self._recov += self.dt * (self.phi * (n_inf - self._recov) / tau_n)
        spikes = ((self._volt.copy() >= 35.5) * self._volt.copy() / 35.5).astype(int)
        return spikes, self._volt.copy()

    def reset(self):
        # Initialize at stable fixed point
        self._volt = -50 * np.ones(self.num_neurons)
        self._recov = .5 * (1 + np.tanh((self._volt - self.V3) / self.V4))

    @property
    def theta(self):
        """ There is no theta in this neuron model. Zero array returned. """
        return np.zeros_like(self._volt)