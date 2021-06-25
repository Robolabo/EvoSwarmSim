import numpy as np
from spike_swarm_sim.register import sensor_registry
from spike_swarm_sim.sensors import DirectionalSensor, Sensor

@sensor_registry(name='ground_sensor')
class GroundSensor(Sensor):
    """ Sensor that detects if there is a ground area underneath the robot (binary reading). 
    Additionally, it only detects ground areas of a particular color.
    """
    def __init__(self, *args, **kwargs):
        super(GroundSensor, self).__init__(*args, **kwargs)

    def step(self, neighborhood):
        for ground_area in filter(lambda x: type(x).__name__ == 'GroundArea', neighborhood):
            if np.linalg.norm(self.sensor_owner.position[:2] - ground_area.position[:2]) <= ground_area.radius:
                return np.array([1.0])
        return np.array([0.0])

# @sensor_registry(name='grey_ground_sensor')
# class GreyGroundSensor(Sensor):
#     """ Sensor that detects if there is a ground area underneath the robot (binary reading).
#     Additionally, it only detects ground areas of a particular color.
#     """
#     def __init__(self, *args, color='grey', **kwargs):
#         super(GroundSensor, self).__init__(*args, **kwargs)

#     def step(self, neighborhood):
#         for ground_area in filter(lambda x: type(x).__name__ == 'GroundArea', neighborhood):
#             if np.linalg.norm(self.sensor_owner.position[:2] - ground_area.position[:2]) >= ground_area.radius:
#                 return 1.0
#         return 0.0