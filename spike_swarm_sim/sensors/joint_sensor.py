import numpy as np
import pybullet as p
from spike_swarm_sim.register import sensor_registry
from spike_swarm_sim.utils import compute_angle
from spike_swarm_sim.sensors import Sensor

@sensor_registry(name='joint_position_sensor')
class JointPositionSensor(Sensor):
    """
    """
    def __init__(self, *args, joints=[0], **kwargs):
        super(JointPositionSensor, self).__init__(*args, **kwargs)
        self.joints = joints
        
    def step(self, neighborhood):
        joint_vals = [p.getJointState(self.sensor_owner.id, joint, \
            physicsClientId=self.sensor_owner.physics_client)[0] for joint in self.joints]
        return np.array(joint_vals)

@sensor_registry(name='joint_velocity_sensor')
class JointVelocitySensor(Sensor):
    """
    """
    def __init__(self, *args, joints=[0], **kwargs):
        super(JointVelocitySensor, self).__init__(*args, **kwargs)
        self.joints = joints
        
    def step(self, neighborhood):
        joint_vals = [p.getJointState(self.sensor_owner.id, joint, \
            physicsClientId=self.sensor_owner.physics_client)[1] for joint in self.joints]
        return np.array(joint_vals)