import numpy as np
import pybullet as p
from spike_swarm_sim.register import sensor_registry
from spike_swarm_sim.utils import compute_angle
from spike_swarm_sim.sensors import DirectionalSensor
from .utils.propagation import ExpDecayPropagation

@sensor_registry(name='light_sensor')
class LightSensor(DirectionalSensor):
    """ Directional ambient light sensor that enables the sensing 
    of the light intensity resulting from the emission of luminous 
    WorldObjects (e.g. LightSource).
    """
    def __init__(self, *args, color='red', **kwargs):
        super(LightSensor, self).__init__(*args, **kwargs)
        self.color = color
        self.aperture = 3 * np.pi / self.n_sectors
        self.propagation = ExpDecayPropagation(rho_att=0.2, phi_att=1)
        

    def _step_direction(self, rho, phi, direction_reading, *args, **kwargs):
        """ Step the sensor of a sector. For a detailed explanation of 
        this method see DirectionalSensor._step_direction.
        """
        condition = kwargs['obj'] is not None\
                    and rho <= kwargs['obj'].range\
                    and kwargs['obj'].color == self.color
                    #and phi <= self.aperture #<= 3*np.pi/self.n_sectors
        if direction_reading is None:
            direction_reading = 0.0
        if condition:
            signal_strength = self.propagation(rho, phi)
            direction_reading += signal_strength
            direction_reading = np.clip(direction_reading, a_min=0, a_max=1)
        return direction_reading

    def _target_filter(self, obj):
        """ Filtering of potential target WorldObjects. 
        #TODO Support for more luminous objects.
        """
        return type(obj).__name__ == 'LightSource3D'


@sensor_registry(name='light_sensor3D')
class LightSensor3D(DirectionalSensor):
    """ Directional ambient light sensor that enables the sensing 
    of the light intensity resulting from the emission of luminous 
    WorldObjects (e.g. LightSource). This sensor is sensitive to 
    light beams of any color.
    """
    def __init__(self, *args, color='red', **kwargs):
        super(LightSensor3D, self).__init__(*args, **kwargs)
        self.color = color
        self.aperture = 3 * np.pi / self.n_sectors
        self.propagation = ExpDecayPropagation(rho_att=0.1, phi_att=1)
        # self.propagation = ExpDecayPropagation(rho_att=0.15, phi_att=1)
    
    def _target_filter(self, obj):
        """ Filtering of potential target WorldObjects. 
        #TODO Support for more luminous objects.
        """
        return type(obj).__name__ == 'LightSource3D'

    def _step_direction(self, rho, phi, direction_reading, *args, **kwargs):
        """ Step the sensor of a sector. For a detailed explanation of 
        this method see DirectionalSensor._step_direction.
        """
        condition = kwargs['obj'] is not None\
                    and rho <= kwargs['obj'].range\
                    # and kwargs['obj'].color == self.color
                    #and phi <= self.aperture #<= 3*np.pi/self.n_sectors
        if direction_reading is None:
            direction_reading = np.random.randn() * self.noise_sigma if self.noise_sigma > 0 else 0.
        # import pdb; pdb.set_trace()
        if condition:
            my_pos = self.get_position(self.sensors_idx[args[0]]) + np.r_[0,0,0.02]
            tar_post = kwargs['obj'].position
            ray_res = p.rayTest(my_pos, tar_post, physicsClientId=self.sensor_owner.physics_client)
            signal_strength = self.propagation(rho, phi)
            if ray_res[0][0] == -1:
                direction_reading += signal_strength
                if self.noise_sigma > 0:
                    direction_reading += np.random.randn() * self.noise_sigma
                direction_reading = np.clip(direction_reading, a_min=0, a_max=1)
        return direction_reading 
    
    def reset(self):
        joints = np.array([p.getJointInfo(self.sensor_owner.id, i, physicsClientId=self.sensor_owner.physics_client)[:2]\
            for i in range(p.getNumJoints(self.sensor_owner.id, physicsClientId=self.sensor_owner.physics_client))])
        self.sensors_idx = {i : np.where(np.array(joints) == bytes('base_to_IR'+str(i), 'utf-8'))[0][0]\
                for i in range(self.n_sectors)}

    def get_position(self, idx):
        return np.array(p.getLinkState(self.sensor_owner.id, idx, physicsClientId=self.sensor_owner.physics_client)[0])


class ColoredLightSensor(LightSensor3D):
    def __init__(self, color, *args, **kwargs):
        super(ColoredLightSensor, self).__init__(*args, **kwargs)
        self.color = color

    def _target_filter(self, obj):
        """ Filtering of potential target WorldObjects. """
        return super()._target_filter(obj) and obj.color == self.color

@sensor_registry(name='blue_light_sensor')
class BlueLightSensor(ColoredLightSensor):
    def __init__(self, *args, **kwargs):
        super(BlueLightSensor, self).__init__('blue', *args, **kwargs)

@sensor_registry(name='yellow_light_sensor')
class YellowLightSensor(ColoredLightSensor):
    def __init__(self, *args, **kwargs):
        super(YellowLightSensor, self).__init__('yellow', *args, **kwargs)
    
@sensor_registry(name='red_light_sensor')
class RedLightSensor(ColoredLightSensor):
    def __init__(self, *args, **kwargs):
        super(RedLightSensor, self).__init__('red', *args, **kwargs)

@sensor_registry(name='green_light_sensor')
class GreenLightSensor(ColoredLightSensor):
    def __init__(self, *args, **kwargs):
        super(GreenLightSensor, self).__init__('green', *args, **kwargs)
