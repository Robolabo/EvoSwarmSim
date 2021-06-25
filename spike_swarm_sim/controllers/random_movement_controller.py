import numpy as np
from spike_swarm_sim.controllers import Controller
import pdb

class RandomMovementController(Controller):
    def __init__(self):
        super().__init__()
        self.enabled_actuators['wheel_actuator'] = True
        
    def step(self, state):
        return {'wheel_actuator' : np.random.choice([-1,1],size=2)}