import numpy as np
import pybullet as p
from spike_swarm_sim.register import sensor_registry
from spike_swarm_sim.utils import compute_angle
from spike_swarm_sim.sensors import Sensor, DirectionalSensor
from .utils.propagation import ExpDecayPropagation


@sensor_registry(name='color_sensor')
class ColorSensor(DirectionalSensor):
    """
    """
    def __init__(self, *args, **kwargs):
        super(ColorSensor, self).__init__(*args, **kwargs)
        # self.color = color
        self.aperture = 1.5 * np.pi / self.n_sectors
        self.propagation = ExpDecayPropagation(rho_att=0.2, phi_att=1)
    
    def _target_filter(self, obj):
        """ Filtering of potential target WorldObjects.
        #TODO Support for more luminous objects.
        """
        return type(obj).__name__ in ['Cube'] and not obj.is_grasped # List because may be extended to other objects.

    def _step_direction(self, rho, phi, direction_reading, *args, **kwargs):
        """ Step the sensor of a sector. For a detailed explanation of
        this method see DirectionalSensor._step_direction.
        """
        condition = kwargs['obj'] is not None\
                    and rho <= self.range\
                    and phi <= self.aperture #<= 3*np.pi/self.n_sectors
        if direction_reading is None:
            direction_reading = 0.
        # import pdb; pdb.set_trace()
        if condition and direction_reading == 0.0:
            my_pos = self.get_position(self.sensors_idx[args[0]]) + np.r_[0, 0, 0.1] #+ np.r_[0, 0, 0.017]
            tar_post = kwargs['obj'].position + np.r_[0, 0, 0.07] # my_pos[2]]
            ray_res = p.rayTest(my_pos, tar_post, physicsClientId=self.sensor_owner.physics_client)[0][0]
            # signal_strength = self.propagation(rho, phi)
            if ray_res == kwargs['obj'].id:
                direction_reading = 1.
        return direction_reading
    
    # def reset(self):
    #     joints = np.array([p.getJointInfo(self.sensor_owner.id, i, physicsClientId=self.sensor_owner.physics_client)[:2]\
    #         for i in range(p.getNumJoints(self.sensor_owner.id, physicsClientId=self.sensor_owner.physics_client))])
    #     self.sensors_idx = {i : np.where(np.array(joints) == bytes('base_to_IR'+str(i), 'utf-8'))[0][0]\
    #             for i in range(self.n_sectors)}

    # def get_position(self, idx):
    #     return np.array(p.getLinkState(self.sensor_owner.id, idx, physicsClientId=self.sensor_owner.physics_client)[0])


@sensor_registry(name='object_grasped_sensor')
class ObjectGraspedSensor(Sensor):
    def __init__(self, *args, **kwargs):
        super(ObjectGraspedSensor, self).__init__(*args, **kwargs)
    
    def step(self, neighborhood):
        reading = int(self.sensor_owner.actuators['grasp_actuator'].cube_grasped is not None)
        return np.array([reading])