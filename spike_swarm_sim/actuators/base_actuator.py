
class Actuator:
    """ Base class of actuator.
    ======================================================================
    - Args:
        sensor_owner [Robot] : instance of the agent executing the sensor.
        range [float] : range of coverage of the sensor.
        noise_sigma [float] : white noise std. dev.  
    ======================================================================
    """
    def __init__(self, actuator_owner):
        self.actuator_owner = actuator_owner


    def step(self, action):
        raise NotImplementedError


class HighLevelActuator(Actuator):
    def __init__(self, *args, **kwargs):
        super(HighLevelActuator, self).__init__(*args, **kwargs)


    def step(self, action, neighborhood):
        raise NotImplementedError