import numpy as np
from .base_actuator import Actuator
from spike_swarm_sim.register import actuator_registry
from spike_swarm_sim.utils import softmax

@actuator_registry(name='RF_transmitter')
class RF_Transmitter(Actuator):
    """ Communication transmitter actuator. It isotropically transmits a 
    frame with a message and its context. The propagation simulation is 
    implemented at the receiver side, this class only updates the transmitted 
    frame of each robot. 
    =========================================================================
    - Params:
        range [float] : maximum distance of message reception, in centimeters.
        msg_length [int] : number of components of the message.
        quantize [bool] : whether to quantize the message to a set of possible 
                symbols or not.
    """
    def __init__(self, *args, range=10, msg_length=1, **kwargs):
        super(RF_Transmitter, self).__init__(*args, **kwargs)
        self.channel = 0
        self.msg_length = msg_length
        self.range = range
        self.frame = None
        self.reset()
        
    def step(self, action):
        self.frame['msg'] = action
        self.frame['enabled'] = bool(action[0]) if isinstance(action, list) else bool(action)

    def reset(self):
        self.frame = {
            'msg' : 0,
            'enabled' : True
        }
