from matplotlib import colors
import pybullet as p
from spike_swarm_sim.register import actuator_registry
from .base_actuator import Actuator

@actuator_registry(name='led_actuator')
class LedActuator(Actuator):
    """ LED actuator that turns on or off the LED depending on 
    the action. """
    def __init__(self, *args, **kwargs):
        super(LedActuator, self).__init__(*args, **kwargs)
        self.on = 0
        self.fault = False

    def step(self, action):
        self.on = action

    def reset(self):
        self.on = 0
        self.fault = False

@actuator_registry(name='led_actuator_3D')
class LedActuator3D(Actuator):
    """ LED actuator that turns on or off the LED depending on 
    the action. """
    def __init__(self, *args, color_on='blue', color_off='white', **kwargs):
        super(LedActuator3D, self).__init__(*args, **kwargs)
        self.color_on = color_on
        self.color_off = color_off
        self.color_fault = 'red'
        self.on = 0
        self.fault = False

    def step(self, action):
        self.on = action if not self.fault else 0
        #! Check color with colors.is_color_like
        color = (self.color_on if self.on else self.color_off) if not self.fault else self.color_fault
        color = colors.to_rgb(color)
        # In epuck, 3 is the led piece
        p.changeVisualShape(self.actuator_owner.id, 3, rgbaColor=list(color) + [0.6],\
            physicsClientId=self.actuator_owner.physics_client)

    def reset(self):
        self.on = 0
        self.fault = False