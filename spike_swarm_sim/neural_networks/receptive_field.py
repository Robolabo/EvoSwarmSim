from abc import ABC, abstractmethod
import logging
import numpy as np
import matplotlib.pyplot as plt
from spike_swarm_sim.register import receptive_field_registry

class ReceptiveField(ABC):
    """ Base class for receptive fields (RF). It is composed by a number of 'neurons' 
    or sensing nodes that activate depending on the value of the stimuli. Normally, 
    the sensing fields of each neuron overlap at some extent. Receptive fields can 
    be mathematically thought as basis expansions of the stimulus variable. 
    The mathematical function that defines the receptive field is defined in each 
    class inheriting from this one and specified in the receptive_function method.
    ==============================================================================
    - Params:
        n_inputs [int] : dimension of the input stimuli vector.
        n_neurons [int] : number of neurons encoding each stimulus.
        trainable [bool] : whether RF parameters can be optimized or not.
        max_val [float] : maximum value output of the receptive field.
        min_val [float] : minimum value output of the receptive field.
        min_stim [float] : minimum value of the stimuli components.
        max_stim [float] : maximum value of the stimuli components.
        inverted [bool] : whether to use 1 - receptive_field(stim) or not.
    ==============================================================================
    """
    def __init__(self, n_inputs, n_neurons=1, trainable=False, max_val=1., min_val=0.,
                min_stim=0., max_stim=1., inverted=False):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.trainable = trainable
        self.min_val = min_val
        self.max_val = max_val
        self.min_stim = min_stim
        self.max_stim = max_stim
        self.inverted = inverted
        self.centers = np.linspace(min_stim, max_stim, self.n_neurons)

    @abstractmethod
    def receptive_function(self, s, center):
        """ Method that defines the mathematical function of each neuron of the 
        receptive field. For the moment only radial receptive fields are supported, 
        requiring the neuron specific center where their sensing field is maximum 
        (or minimum if inverted=True).
        ============================================================================
        - Args:
            s : the stimulus to be processed.
            center :  the center of the radial function.
        ============================================================================
        """
        raise NotImplementedError

    def __call__(self, stim):
        """ Call method of the receptive fields. It iterated over centers (or neurons) 
        and computes the corresponding response to the stimulus. It linearly transforms 
        the responses to the specified range [min_val, max_val].
        """
        normalized = np.array([self.receptive_function(stim, center)\
                for center in self.centers])
        if self.inverted:
            normalized = 1 - normalized
        return (self.max_val - self.min_val) * normalized + self.min_val 

    def plot(self, stimulus, ax=None):
        if ax is None:
            f, ax = plt.subplots()
        xx = np.linspace(self.min_stim, self.max_stim, 1000)
        yy = [self(x) for x in xx]
        lines = ax.plot(xx, yy)
        ax.vlines(stimulus, self.min_val, self.max_val)
        for val in self(stimulus):
            ax.plot(stimulus, val, 'ko')
        ax.legend(lines, ['Neuron ' + str(i) + ' with mu=' + str(round(mu, 2))\
                    for i, mu in enumerate(self.centers)])
        ax.set_xlabel('Stimulus')
        ax.set_ylabel('Firing Rate [Hz]')
        return ax

@receptive_field_registry(name='identity_receptive_field')
class IdentityReceptiveField(ReceptiveField):
    """ Identity or dummy receptive field. It merely serves as a placeholder 
    when the neural encoder doesn't employ receptive field. The number of
    RF neurons/nodes must be always 1 (otherwise it would be ignored). 
    """
    def __init__(self, *args, **kwargs):
        super(IdentityReceptiveField, self).__init__(*args, **kwargs)
        if self.n_neurons > 1:
            logging.warning('Identity receptive field does not support '\
                'multiple sensing nodes. Using n_neurons=1 instead.')

    def receptive_function(self, stim, center):
        return stim

    def __call__(self, stim):
        return stim

@receptive_field_registry(name='gaussian_receptive_field')
class GaussianReceptiveField(ReceptiveField):
    """ Gaussian or radial basis receptive field. It encodes the stimuli 
    of each RF neuron as exp(-sigma * ||stim-center||_2^2).
    ====================================================================
    - Args:
        sigma [float]: std. dev. of the gaussian receptive fields.
    """
    def __init__(self, *args, sigma=6., **kwargs):
        super(GaussianReceptiveField, self).__init__(*args, **kwargs)
        self.sigma = sigma

    def receptive_function(self, stim, center):
        return np.exp(-self.sigma * np.linalg.norm(stim - center) ** 2)

@receptive_field_registry(name='triangular_receptive_field')
class TriangularReceptiveField(ReceptiveField):
    """ Triangular receptive field. It encodes the stimuli 
    of each RF neuron as 1 - slope * |stim - center|. Only supported for
    1D stim.
    """
    def __init__(self, slope=5., *args, **kwargs):
        super(TriangularReceptiveField, self).__init__(*args, **kwargs)
        self.slope = slope

    def receptive_function(self, stim, center):
        return np.clip(1 - self.slope * np.abs(stim - center), a_min=0., a_max=1.)

@receptive_field_registry(name='conic_receptive_field')
class ConicReceptiveField(ReceptiveField):
    def __init__(self, slope=5., *args, **kwargs):
        super(ConicReceptiveField, self).__init__(*args, **kwargs)
        self.slope = slope

    def receptive_function(self, stim, center):
        return np.clip(1 - self.slope * np.linalg.norm(stim - center), a_min=0., a_max=1.)

@receptive_field_registry(name='parabolic_receptive_field')
class ParabolicReceptiveField(ReceptiveField):
    def __init__(self, slope=5., *args, **kwargs):
        super(ParabolicReceptiveField, self).__init__(*args, **kwargs)
        self.slope = slope

    def receptive_function(self, stim, center):
        return np.clip(1 - self.slope * np.linalg.norm(stim - center) ** 2, a_min=0., a_max=1.)

#TODO
# @receptive_field_registry(name='linear_receptive_field')
# class LinearReceptiveField(ReceptiveField):
#     def __init__(self,  *args, **kwargs):
#         super(LinearReceptiveField, self).__init__(*args, **kwargs, trainable=True)
#         self.weights = np.ones(self.n_inputs * self.n_neurons + self.n_neurons)
#         self.initialize()

#     def receptive_function(self, stim, center):
#         Phi = self.weights[:self.n_inputs * self.n_neurons].reshape(self.n_inputs, self.n_neurons)
#         bias = self.weights[-self.n_neurons:]
#         return (Phi.T.dot(stim) + bias).flatten()

#     def initialize(self):
#         base_rf = [[np.exp(-np.abs(x - center)) for x in range(self.n_inputs)]\
#                    for center in np.linspace(0, self.n_inputs, self.n_neurons)]
#         Phi = np.array(base_rf)
#         self.weights = Phi.flatten()