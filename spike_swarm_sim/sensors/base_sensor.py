import numpy as np
import numpy.linalg as LA
import pybullet as p
from spike_swarm_sim.objects import WorldObject2D, WorldObject3D, LightSource3D, Robot3D, Wall
from spike_swarm_sim.utils import compute_angle, angle_diff, toroidal_difference

class Sensor:
    """ Base class of sensors.
    ======================================================================
    - Args:
        sensor_owner [Robot] : instance of the agent executing the sensor.
        range [float] : range of coverage of the sensor.
        noise_sigma [float] : white noise std. dev.  
    ======================================================================
    """
    def __init__(self, sensor_owner, range=100, noise_sigma=0.0):
        self.sensor_owner = sensor_owner
        self.noise_sigma = noise_sigma
        self.range = range
        self.reading = None
        # self.sensor_idx = {}#!

    def step(self, neighborhood):
        raise NotImplementedError

class DirectionalSensor(Sensor):
    """
    Base class for directional sensors.
    ==========================================
    - Args:
        - n_sectors [int] : number of sectors.
    ===========================================
    """
    def __init__(self, *args, n_sectors=8, **kwargs):
        super(DirectionalSensor, self).__init__(*args, **kwargs)
        self.n_sectors = n_sectors
        self.aperture = np.pi / self.n_sectors
        self.sensors_idx = None

    def _target_filter(self, obj):
        """
        Method devoted to filtering the world objects that should be targeted for a particular sensor.
        For example: a light sensor will filter out only luminous objects.
        This method must be overwritten in each directional sensor that inherits from DirectionalSensor 
        in order to particularize its functioning.
        ===============================================================================================
        - Args:
            obj [WorldObject] -> Potential world object to be sensed.
        - Returns:
            Boolean response revealing whether the obj should be explored by the sensor or not.
        ===============================================================================================
        """
        raise NotImplementedError

    def _step_direction(self, rho, phi, direction_reading, direction, obj=None, diff_vector=None):
        """
        Method that specifies the particular behavior of a directional sensor in each sensing direction.
        It must return the reading of the current direction.
        This method must be overwritten in each directional sensor that inherits from DirectionalSensor 
        in order to particularize its functioning.
        ==================================================================================================
        - Args:
            rho [float] -> distance between the object sensing and the object (obj) sensed.
            phi [float] -> angle between the direction of the sensor and the line passing through 
                    both sensing and sensed object positions.
            direction_reading [float or dict] -> Current reading in the featured direction 
                    to be potentially overwritten.
            direction [int] -> integer refering to the current sensing direction between 0 and n_sectors - 1.
            obj [WorldObject] -> Optionally, the object that is being sensed can be used.
            diff_vector [np.ndarray] -> Optionally, the vector resulting from the difference 
                    of the between object positions can be used. However, most of the times, 
                    rho and phi are sufficient. Notice that rho=|diff_vector|.
        - Returns:
            Reading of the sensor in the current direction.
        =================================================================================================
        """
        raise NotImplementedError

    def directions(self, theta):
        """
        Returns the vector of sensing directions of the sectors based on the robot orientation.
        - Args:
            theta [float] -> orientation of the robots using the sensor.
        - Returns:
            Numpy Array with the absolute directions of each sensor (starting from theta).
        """
        return np.array([theta + i * (2 * np.pi / self.n_sectors) for i in range(self.n_sectors)])

    def reset(self):
        self.reading = None
        joints = np.array([p.getJointInfo(self.sensor_owner.id, i, physicsClientId=self.sensor_owner.physics_client)[:2]\
            for i in range(p.getNumJoints(self.sensor_owner.id, physicsClientId=self.sensor_owner.physics_client))])
        self.sensors_idx = {i : np.where(np.array(joints) == bytes('base_to_IR'+str(i), 'utf-8'))[0][0]\
                for i in range(self.n_sectors)}

    def get_position(self, idx):
        return np.array(p.getLinkState(self.sensor_owner.id, idx, physicsClientId=self.sensor_owner.physics_client)[0])
    

    def step(self, neighborhood):
        """
        Main method for steping the sensor and capturing nearby environment events.
        It senses the environment independently in each of the sensing sectors. The 
        particular behavior of each sensor inheriting this class should be specified in 
        _step_direction method.
        ================================================================================
        - Args: 
            neighborhood -> List of neighboring WorldObjects.

        - Returns:
            Numpy array with the measurement in each direction. In exceptional cases 
            it may return a list of python dictionaries (see CommunicationReceiver).
        ================================================================================
        """
        readings = [self._step_direction(0, 0, None, 0, obj=None) for _ in range(self.n_sectors)]
        orientation = self.sensor_owner.orientation
        featured_objects = [obj for obj in neighborhood \
                            if self._target_filter(obj) and obj.id != self.sensor_owner.id]
        #! Improve
        for obj in featured_objects:
            if type(obj).__name__ == 'Wall':
                closest_points = p.getClosestPoints(self.sensor_owner.id, obj.id, 200,\
                        linkIndexA=-1, linkIndexB=-1, physicsClientId=self.sensor_owner.physics_client)
                v = np.array(closest_points[0][6]) - self.sensor_owner.position
            else:
                v = obj.position - self.sensor_owner.position #!OJO
            orientation = self.sensor_owner.orientation[-1]
            rho = LA.norm(v)
            if rho >= self.range:
                continue

            # Angle difference between sensor directions and ang(v)
            # import pdb; pdb.set_trace()
            phi_values = np.array([angle_diff(compute_angle(v[:2]), theta) for theta in self.directions(orientation)])
            featured_sensors = np.where(phi_values <= self.aperture)[0]
            phi_values = phi_values[featured_sensors]
            for k, phi in zip(featured_sensors, phi_values):
                readings[k] = self._step_direction(rho, phi, readings[k], k, obj=obj, diff_vector=v)
        return np.array(readings) if not isinstance(readings[0], dict) else readings

    # def step2(self, neighborhood):
    #     """
    #     Main method for steping the sensor and capturing nearby environment events.
    #     It senses the environment independently in each of the sensing sectors. The 
    #     particular behavior of each sensor inheriting this class should be specified in 
    #     _step_direction method.
    #     ================================================================================
    #     - Args: 
    #         neighborhood -> List of neighboring WorldObjects.

    #     - Returns:
    #         Numpy array with the measurement in each direction. In exceptional cases 
    #         it may return a list of python dictionaries (see CommunicationReceiver).
    #     ================================================================================
    #     """
    #     readings = [self._step_direction(0, 0, None, 0, obj=None) for _ in range(self.n_sectors)]
    #     orientation = self.sensor_owner.orientation[-1]
    #     featured_objects = [obj for obj in neighborhood \
    #                         if self._target_filter(obj) and obj.id != self.sensor_owner.id]
    #     for obj in featured_objects:
    #         for k, direction in enumerate(self.directions(orientation)):
    #             robot_sensor = self.sensors_idx[k]
    #             if issubclass(type(obj), WorldObject3D):
    #                 if obj.tangible:
    #                     closest_points = p.getClosestPoints(self.sensor_owner.id, obj.id, 200,\
    #                             linkIndexA=robot_sensor, linkIndexB=-1, physicsClientId=self.sensor_owner.physics_client)
    #                     v = np.array(closest_points[0][6]) - self.sensor_owner.position
    #                 else:
    #                     v = obj.position - self.sensor_owner.position #!OJO: No pilla bien la altura de los objetos del URDF.
    #                 orientation = self.sensor_owner.orientation[-1]
    #             else:
    #                 v = toroidal_difference(obj.position, self.sensor_owner.position)
    #             rho = LA.norm(v)
    #             phi = angle_diff(compute_angle(v[:2]), direction)
    #             # import pdb; pdb.set_trace()
    #             if rho < self.range and phi < self.aperture:
    #                 readings[k] = self._step_direction(rho, phi, readings[k], k, obj=obj, diff_vector=v)
    #     reading = np.array(readings) if not isinstance(readings[0], dict) else readings
    #     self.reading = reading.copy()
    #     return reading
