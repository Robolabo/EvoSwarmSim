import numpy as np
import pybullet as p
from .base_sensor import DirectionalSensor
from spike_swarm_sim.register import sensor_registry
from spike_swarm_sim.objects import Robot, Robot3D
from spike_swarm_sim.utils import compute_angle, angle_diff, issubclass_of_any, circle_distance
from .utils.propagation import ExpDecayPropagation, RSSI_Propagation

@sensor_registry(name='RF_receiver')
class RF_Receiver(DirectionalSensor):
    """ Communication Receiver mimicking RF.
    ========================================================================
    - Args:
        msg_length [int] : number of components of the message. Must be the 
                    same as in the Transmitter definition.
        max_hops [int] : maximum number of hops before frame discard. 
    ========================================================================
    """
    def __init__(self, *args, **kwargs):
        kwargs['n_sectors'] = 1
        super(RF_Receiver, self).__init__(*args,  **kwargs)
        self.propagation = RSSI_Propagation(noise_sigma=self.noise_sigma)

    def _target_filter(self, obj):
        """ Filtering of potential sender robots. """
        return issubclass_of_any(obj, [Robot, Robot3D]) and 'RF_transmitter' in obj.actuators

    def _step_direction(self, rho, phi, direction_reading, direction, obj=None, diff_vector=None):
        """ Step the sensor of a sector, receiving the frame messages and the underlying
        context. For a detailed explanation of this method see DirectionalSensor._step_direction.
        """
        condition = obj is not None\
                    and rho <= self.range\
                    and rho <= obj.actuators['RF_transmitter'].range\
                    and obj.actuators['RF_transmitter'].frame['enabled']
        #* Fill initial reading with empty frame
        if direction_reading is None:
            direction_reading = self.empty_msg
        if condition:
            signal_strength = self.propagation(rho, phi)
            #* MSG={0,1}, so it is 1 if at least 1 robot sends a 1. And if a robot is enabled, it 
            #* must send a 1 to reduce overhead.
            direction_reading['msg'] = np.array(obj.actuators['RF_transmitter'].frame['msg'])
            direction_reading['signal'].append(np.array([signal_strength]))
        return direction_reading

    def step(self, *args, **kwargs):
        """ Steps the communication receiver. With the sensed frames from all directions it 
        firstly discards those with more hops than a thresh. Then, the selection of a 
        unique frame is carried out stochastically among those frames whose sender is not the receiver. 
        If no message is sensed, the measurement is an empty frame.
        """
        #* Imposed only 1 direction
        frame = super().step(*args, **kwargs)[0]
        frame['signal'] = np.mean(frame['signal']) if len(frame['signal']) else np.random.randn() * 0.05
        frame['signal'] = np.array(frame['signal'])
        frame['msg'] = np.array(frame['msg'])
        return frame
     
    @property
    def empty_msg(self):
        """ Format of an empty message. """
        return {'signal' : [], 'msg' : 0}