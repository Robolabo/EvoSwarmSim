import numpy as np
from spike_swarm_sim.controllers import RobotController
from spike_swarm_sim.register import controller_registry

@controller_registry(name='Braitenberg2')
class Braitenberg2Controller(RobotController):
    def __init__(self, *args, **kwargs):
        super(Braitenberg2Controller, self).__init__(*args, **kwargs)

    def step(self, state, reward=0.0):
        action = np.zeros(2)
        action[0] = np.mean(state['light_sensor'][[1, 2, 3]])
        action[1] = np.mean(state['light_sensor'][[-1, -2, -3]])
        return {'wheel_actuator' : np.clip(5 * action, a_min=-1, a_max=1)}