import logging
import pybullet as p
import numpy as np
from .base_actuator import Actuator
from spike_swarm_sim.register import actuator_registry

@actuator_registry(name='joint_velocity_actuator')
class JointVelocityActuator(Actuator):
    """ Robot wheel actuator using a differential drive system. 
    """
    def __init__(self, *args, joint_ids=[0], inverse_mirrored=None, max_velocity=10., **kwargs):
        super(JointVelocityActuator, self).__init__(*args, **kwargs)
        self.joint_ids = joint_ids
        self.max_velocity = max_velocity
        self.inverse_mirrored = inverse_mirrored

    def step(self, action):
        if len(action) != len(self.joint_ids):
            raise Exception(logging.error('Size of the action in Joint Actuator differs from '\
            	'the number of controllable joints.'))
        for ac, joint in zip(action, self.joint_ids):
            p.setJointMotorControl2(self.actuator_owner.id, joint, targetVelocity=ac * self.max_velocity,\
                controlMode=p.VELOCITY_CONTROL, physicsClientId=self.actuator_owner.physics_client, velocityGain=1.1)
        if self.inverse_mirrored is not None and len(action) == 1: #! mejorar
            p.setJointMotorControl2(self.actuator_owner.id, self.inverse_mirrored,\
                targetVelocity=-action[0] * self.max_velocity, controlMode=p.VELOCITY_CONTROL,\
                physicsClientId=self.actuator_owner.physics_client, velocityGain=1.1)

    def reset(self,):
        for joint in self.joint_ids:
            p.setJointMotorControl2(self.actuator_owner.id, joint, targetVelocity=0, velocityGain=0,\
                controlMode=p.VELOCITY_CONTROL, physicsClientId=self.actuator_owner.physics_client)

@actuator_registry(name='joint_position_actuator')
class JointPositionActuator(Actuator):
    """ Robot wheel actuator using a differential drive system. 
    """
    def __init__(self, *args, joint_ids=[0], **kwargs):
        super(JointPositionActuator, self).__init__(*args, **kwargs)
        self.joint_ids = joint_ids
        self.max_velocity = 1.
         
    def step(self, action):
        if len(action) != len(self.joint_ids):
            raise Exception(logging.error('Size of the action in Joint Actuator differs from '\
            	'the number of controllable joints.'))

        for ac, joint in zip(action, self.joint_ids):
            p.setJointMotorControl2(self.actuator_owner.id, joint, targetPosition=ac * np.pi,\
                controlMode=p.POSITION_CONTROL, physicsClientId=self.actuator_owner.physics_client,\
                positionGain=1.1, velocityGain=1.1, maxVelocity=self.max_velocity)
    
    # def reset(self,):
    #     for joint in self.joint_ids:
    #         p.setJointMotorControl2(self.actuator_owner.id, joint, targetVelocity=0, velocityGain=0,\
    #             controlMode=p.VELOCITY_CONTROL, physicsClientId=self.actuator_owner.physics_client)
