import numpy as np
import pybullet as p
from .base_sensor import DirectionalSensor
from .distance_sensor import DistanceSensor3D
from spike_swarm_sim.register import sensor_registry
from spike_swarm_sim.objects import Robot, Robot3D
from spike_swarm_sim.utils import compute_angle, angle_diff, issubclass_of_any, circle_distance
from .utils.propagation import ExpDecayPropagation

@sensor_registry(name='IR_receiver2')
class IRCommunicationReceiver(DirectionalSensor):
    """ Communication Receiver mimicking IR technology.
    ========================================================================
    - Args:
        msg_length [int] : number of components of the message. Must be the 
                    same as in the Transmitter definition.
        max_hops [int] : maximum number of hops before frame discard. 
    ========================================================================
    """
    def __init__(self, *args, msg_length=1, max_hops=10, selection_scheme='random', **kwargs):
        super(IRCommunicationReceiver, self).__init__(*args, **kwargs)
        self.msg_length = msg_length
        self.max_hops = max_hops
        self.selection_scheme = selection_scheme
        self.current_direction = 0
        self.propagation = ExpDecayPropagation(rho_att=0.7, phi_att=1.)
        self.current_sender = None
        self.aux_t = 0

    def _target_filter(self, obj):
        """ Filtering of potential sender robots. """
        return issubclass_of_any(obj, [Robot, Robot3D]) and 'wireless_transmitter' in obj.actuators

    def _step_direction(self, rho, phi, direction_reading, direction, obj=None, diff_vector=None):
        """ Step the sensor of a sector, receiving the frame messages and the underlying
        context. For a detailed explanation of this method see DirectionalSensor._step_direction.
        """
        condition = obj is not None\
                    and rho <= self.range\
                    and phi <= self.aperture\
                    and rho <= obj.actuators['wireless_transmitter'].range\
                    and obj.actuators['wireless_transmitter'].frame['enabled']
        
        #* Fill initial reading with empty frame
        if direction_reading is None:
            direction_reading = self.empty_msg
        if condition:
            signal_strength = self.propagation(rho, phi)
            if signal_strength > direction_reading['signal']:
                my_pos = self.get_position(self.sensors_idx[direction]) + np.r_[0, 0, 0.1] #+ np.r_[0, 0, 0.017]
                tar_pos = obj.position + np.r_[0, 0, 0.07] # my_pos[2]]
                ray_res = p.rayTest(my_pos, tar_pos, physicsClientId=self.sensor_owner.physics_client)[0][0]
                if ray_res == obj.id:
                    sending_direction = 0 #!np.argmin([angle_diff(sdir, compute_angle(diff_vector) + np.pi) for sdir in self.directions(obj.orientation)])
                    sending_angle = self.directions(0.)[sending_direction]
                    receiving_angle = self.directions(0.)[direction]
                    direction_reading['sending_direction'] = np.r_[np.cos(sending_angle), np.sin(sending_angle)].round(2)
                    direction_reading['receiving_direction'] = np.r_[np.cos(receiving_angle), np.sin(receiving_angle)].round(2)
                    direction_reading['receiving_direction'][np.abs(direction_reading['receiving_direction']) < 1e-5] = 0.0
                    # msg = .actuators['wireless_transmitter'].msg[send_dir] #! if directional transmission
                    direction_reading['msg'] = np.array(obj.actuators['wireless_transmitter'].frame['msg'])  #! if isotropic
                    direction_reading['signal'] = np.array([signal_strength])
                    direction_reading['priority'] = np.array([obj.actuators['wireless_transmitter'].frame['priority']])
                    direction_reading['destination'] = np.array([obj.actuators['wireless_transmitter'].frame['destination']])
                    direction_reading['sender'] = np.array([obj.id]) if obj.actuators['wireless_transmitter'].frame['state'] \
                                                else obj.actuators['wireless_transmitter'].frame['sender']
                    direction_reading['n_hops'] = obj.actuators['wireless_transmitter'].frame['n_hops']
                    if direction_reading['n_hops'] > 1:
                        direction_reading['sending_direction'] = obj.actuators['wireless_transmitter'].frame['sending_direction']
        return direction_reading

    def __random_selection(self, frames):
        # Discard very old frames (max 10 hops)
        frames = [frame for frame in frames if frame['n_hops'] < self.max_hops and frame['sender'] != -1]
        if len(frames) == 0:
            frames = [self.empty_msg]
        #* Select only a direction
        signal_strengths = np.hstack([frame['signal'] for frame in frames])
        senders = np.hstack([frame['sender'].item() for frame in frames])
        selected_direction = np.argmax(signal_strengths)
        if any(np.logical_and(senders != self.sensor_owner.id, senders != -1)):
            elements = np.where(np.logical_and(senders != self.sensor_owner.id, senders != -1))[0]
            selected_direction = np.random.choice(elements,)
            frames[selected_direction]['am_i_sender'] = np.array([0])
        else:
            selected_direction = 0
            frames = [self.empty_msg]
            frames[selected_direction]['am_i_sender'] = np.array([0])
            frames[selected_direction]['am_i_targeted'] = np.array([0])
        return frames[selected_direction]

    def __random_selection2(self, frames):
        if self.current_sender is not None and self.aux_t < 50:
            senders = np.hstack([frame['sender'].item() for frame in frames])
            if self.current_sender in senders:
                frame = [fr for fr in frames if fr['sender'] == self.current_sender][0]
                self.aux_t += 1
            else:
                frame = self.__random_selection(frames)
                self.current_sender = frame['sender'].item()
                self.aux_t = 0
        else: #* Chose new sender randomly.
            frame = self.__random_selection(frames)
            self.current_sender = frame['sender'].item()
            self.aux_t = 0
        return frame
        # # Discard very old frames (max 10 hops)
        # frames = [frame for frame in frames if frame['n_hops'] < self.max_hops and frame['sender'] != -1]
        # if len(frames) == 0:
        #     frames = [self.empty_msg]
        # #* Select only a direction
        # signal_strengths = np.hstack([frame['signal'] for frame in frames])
        # senders = np.hstack([frame['sender'].item() for frame in frames])
        # selected_direction = np.argmax(signal_strengths)
        # if any(np.logical_and(senders != self.sensor_owner.id, senders != -1)):
        #     import pdb; pdb.set_trace()
        #     elements = np.where(np.logical_and(senders != self.sensor_owner.id, senders != -1))[0]
        #     selected_direction = np.random.choice(elements,)
        #     frames[selected_direction]['am_i_sender'] = np.array([0])
        # else:
        #     selected_direction = 0
        #     frames = [self.empty_msg]
        #     frames[selected_direction]['am_i_sender'] = np.array([0])
        #     frames[selected_direction]['am_i_targeted'] = np.array([0])
        # return frames[selected_direction]


    def __cyclic_selection(self, frames):
        #* --- MESSAGE SELECTION Cyclic --- *#
        selected_frame = frames[self.current_direction].copy()
        self.current_direction = (self.current_direction + 1) % self.n_sectors
        return selected_frame

    def step(self, *args, **kwargs):
        """ Steps the communication receiver. With the sensed frames from all directions it 
        firstly discards those with more hops than a thresh. Then, the selection of a 
        unique frame is carried out stochastically among those frames whose sender is not the receiver.
        If no message is sensed, the measurement is an empty frame.
        """
        frames = super().step(*args, **kwargs)
        if len(np.where(np.array(frames) == 0.0)[0]):
            frames = [self.empty_msg for _ in range(len(frames))]
        selected_frame = {
            'cyclic' : self.__cyclic_selection(frames),
            'random' : self.__random_selection(frames),
            'random_2' : self.__random_selection2(frames),
        }[self.selection_scheme]
        return selected_frame


    def reset(self):
        super().reset()
        self.current_sender = None
        self.aux_t = 0
        self.current_direction = 0
    
    def remove_duplicates(self, frames):
        """#TODO: Remove received duplicate frames."""
        raise NotImplementedError
     
    @property
    def empty_msg(self):
        """ Format of an empty message. """
        return {'signal' : np.array([0.0]), 'msg' : np.zeros(self.msg_length), \
                'sending_direction' : np.zeros(2), 'receiving_direction' : np.zeros(2),\
                'priority' : np.zeros(1), 'destination' : np.array([-1]), \
                'sender' : -1 * np.ones(1), 'n_hops' : 1}

@sensor_registry(name='IR_receiver')
class BufferedIRCommRX(IRCommunicationReceiver):
    def __init__(self,  *args, **kwargs):
        super(BufferedIRCommRX, self).__init__(*args, **kwargs)
        self.prev_msg = np.zeros(self.n_sectors)

    def step(self, *args, **kwargs):
        curr_dir = self.current_direction
        frame = super().step(*args, **kwargs)
        curr_msg = self.prev_msg.copy()
        curr_msg[curr_dir] = frame['msg'][0]
        frame['msg'] = curr_msg.copy()
        self.prev_msg = curr_msg
        if self.noise_sigma > 0:
            frame['msg'] += np.random.randn(len(frame['msg'])) * self.noise_sigma
            frame['signal'] += np.random.randn() * self.noise_sigma
        return frame

    def reset(self):
        super().reset()
        self.prev_msg = np.zeros(self.n_sectors)

# @sensor_registry(name='IR_receiver2')
# class IRCommunicationReceiver3D(IRCommunicationReceiver):
#     def _step_direction(self, rho, phi, direction_reading, direction, obj=None, diff_vector=None):
#         """ Step the sensor of a sector, receiving the frame messages and the underlying
#         context. For a detailed explanation of this method see DirectionalSensor._step_direction.
#         """
#         condition = obj is not None\
#                     and rho <= self.range\
#                     and rho <= obj.actuators['wireless_transmitter'].range\
#                     and obj.actuators['wireless_transmitter'].frame['enabled']
#         #* Fill initial reading with empty frame
#         if direction_reading is None:
#             direction_reading = self.empty_msg
#         if condition:
#             misalignments = [np.pi - circle_distance(angle, self.directions(self.sensor_owner.orientation[-1])[direction])\
#                 for angle in self.directions(obj.orientation[-1])]
#             tx_sensor = np.argmin(misalignments)
#             phi = misalignments[tx_sensor]
#             signal_strength = self.propagation(rho, phi) #! we use this rho for the moment
#             if signal_strength > direction_reading['signal']:
#                 my_pos = self.get_position(self.sensors_idx[direction]) + np.r_[0, 0, 0.02]
#                 #! Cambiar esto
#                 tar_pos = obj.sensors['IR_receiver'].get_position(self.sensors_idx[tx_sensor]) + np.r_[0, 0, 0.02]
#                 ray_res = p.rayTest(my_pos, tar_pos, physicsClientId=self.sensor_owner.physics_client)[0][0]
#                 # print(obj.id, ray_res, direction, tx_sensor)
#                 if ray_res == obj.id or ray_res == -1: # No beam collisions 
#                     sending_angle = self.directions(0.)[tx_sensor]
#                     receiving_angle = self.directions(0.)[direction]
#                     direction_reading['sending_direction'] = np.r_[np.cos(sending_angle), np.sin(sending_angle)]
#                     direction_reading['receiving_direction'] = np.r_[np.cos(receiving_angle), np.sin(receiving_angle)]
#                     direction_reading['receiving_direction'][np.abs(direction_reading['receiving_direction']) < 1e-5] = 0.0
#                     direction_reading['msg'] = np.array(obj.actuators['wireless_transmitter'].frame['msg'])
#                     direction_reading['signal'] = np.array([signal_strength])
#                     direction_reading['priority'] = np.array([obj.actuators['wireless_transmitter'].frame['priority']])
#                     direction_reading['destination'] = np.array([obj.actuators['wireless_transmitter'].frame['destination']])
#                     direction_reading['sender'] = np.array([obj.id]) if obj.actuators['wireless_transmitter'].frame['state'] \
#                                                 else obj.actuators['wireless_transmitter'].frame['sender']
#                     direction_reading['n_hops'] = obj.actuators['wireless_transmitter'].frame['n_hops']
#                     if direction_reading['n_hops'] > 1:
#                         direction_reading['sending_direction'] = obj.actuators['wireless_transmitter'].frame['sending_direction']
#         return direction_reading

#     def reset(self):
#         joints = np.array([p.getJointInfo(self.sensor_owner.id, i, physicsClientId=self.sensor_owner.physics_client)[:2]\
#                 for i in range(p.getNumJoints(self.sensor_owner.id, physicsClientId=self.sensor_owner.physics_client))])
#         self.sensors_idx = {i : np.where(np.array(joints) == bytes('base_to_IR'+str(i), 'utf-8'))[0][0]\
#                 for i in range(self.n_sectors)}

#     def get_position(self, idx):
#         return np.array(p.getLinkState(self.sensor_owner.id, idx,\
#                 physicsClientId=self.sensor_owner.physics_client)[0])
