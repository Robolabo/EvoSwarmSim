import numpy as np
import numpy.linalg as LA
from spike_swarm_sim.register import sensor_registry
from spike_swarm_sim.sensors import Sensor

""" ALSO INCLUDES NEIGHBORHOOD ORIENTATIONS """
@sensor_registry(name='neighborhood_pos_sensor')
class NeighborhoodPositionSensor(Sensor):
    def __init__(self, *args, **kwargs):
        super(NeighborhoodPositionSensor, self).__init__(*args, **kwargs)
        self.range = 5000

    def step(self, neighborhood):
        neighborhood_pos = []
        for obj in neighborhood: 
            if self.sensor_owner.id != obj.id and obj.controllable:
                dist = LA.norm(obj.position - self.sensor_owner.position)
                neighborhood_pos.append(np.hstack((obj.position, obj.orientation)))
        return np.array(neighborhood_pos)
