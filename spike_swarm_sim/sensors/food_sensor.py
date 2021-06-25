import numpy as np
from spike_swarm_sim.register import sensor_registry
from spike_swarm_sim.sensors import DirectionalSensor, Sensor

@sensor_registry(name='food_sensor')
class FoodSensor(Sensor):
    """ Food sensor that detects whether the robot is 
    holding/transporting a food piece or not. """
    def __init__(self, *args, **kwargs):
        super(FoodSensor, self).__init__(*args, **kwargs)

    def step(self, neighborhood):
       return np.array([int(self.sensor_owner.food)])

@sensor_registry(name='food_area_sensor')
class FoodAreaSensor(DirectionalSensor):
    """ Food area sensor that detects whether there is a food 
    area underneath the robot or not. """
    def __init__(self, *args, **kwargs):
        super(FoodAreaSensor, self).__init__(n_sectors=1, *args, **kwargs)
    
    def _step_direction(self, rho, phi, direction_reading, *args, **kwargs):
        condition = (rho <= kwargs['obj'].range)
        if direction_reading is None:
            direction_reading = 0.
        if condition:
            direction_reading = 1.
        return direction_reading

    def _target_filter(self, obj):
        return type(obj).__name__ == 'FoodArea'

@sensor_registry(name='nest_sensor')
class NestSensor(DirectionalSensor):
    """ Nest area sensor that detects whether there is a nest 
    area underneath the robot or not. """
    def __init__(self, *args, **kwargs):
        super(NestSensor, self).__init__(n_sectors=1, *args, **kwargs)

    def _step_direction(self, rho, phi, direction_reading, *args, **kwargs):
        condition = (rho <= kwargs['obj'].range)
        if direction_reading is None:
            direction_reading = 0.
        if condition:
            direction_reading = 1.
        return direction_reading

    def _target_filter(self, obj):
        return type(obj).__name__ == 'Nest'