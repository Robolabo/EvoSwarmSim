import os
import numpy as np
import pymunk
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from pygame.color import THECOLORS
# from shapely.geometry import Point
from spike_swarm_sim.objects import WorldObject2D
from spike_swarm_sim.register import sensors, actuators, world_object_registry

@world_object_registry(name='robot')
class Robot(WorldObject2D):
    """
    Base class for the robot world object.
    """
    def __init__(self, position, orientation, *args, **kwargs):
        super(Robot, self).__init__(position, orientation, static=False, luminous=False,\
                        tangible=True, *args, **kwargs)
        self.radius = .11
        self._food = False

        #* Initialize sensors and actuators according to controller requirements
        self.sensors = {k : s(self, **self.controller.enabled_sensors[k])\
                            for k, s in sensors.items()\
                            if k in self.controller.enabled_sensors.keys()}
        self.actuators = {k : k == 'wheel_actuator' and a(self, robot_radius=self.radius, **self.controller.enabled_actuators[k])\
                            or a(self, **self.controller.enabled_actuators[k])\
                            for k, a in actuators.items()\
                            if k in self.controller.enabled_actuators.keys()}

        #* Storage for actions selected by the controllers to be fed to actuators
        self.planned_actions = {k : [None] for k in actuators.keys()}
        self.reset()

    def add_physics(self, engine):
        super().add_physics(engine)
        mass = 1
        inertia = pymunk.moment_for_circle(mass, 0, self.radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = tuple(self.init_position * 100 + 500)
        body.orientation = self.init_orientation
        shape = pymunk.Circle(body, self.radius, (0, 0))
        # shape_ori = pymunk.Segment(body, (0, 0), \
        #     (self.radius * np.cos(body.orientation), self.radius * np.sin(body.orientation)), 2)
        shape.friction = 1
        shape.color = THECOLORS['green']
        # shape_ori.color = THECOLORS['black']
        engine.add(self.id, [body], [shape,])


    def step(self, neighborhood, reward=None, perturbations=None):
        """
        Firstly steps all the sensors in order to perceive the environment.
        Secondly, the robot executes its controller in order to compute the
        actions based on the sensory information.
        Lastly, the actions are stored as planned actions to be eventually executed.
        =====================
        - Args:
            neighborhood [list] -> list filled with the neighboring world objects.
            reward [float] -> reward to be fed to the controller update rules, if any.
            perturbations [list of PostProcessingPerturbation or None] -> perturbation to apply 
                        to the stimuli before controller step.
        - Returns:
            State and action tuple of the current timestep. Both of them are expressed as 
            a dict with the sensor/actuator name and the corresponding stimuli/action.
        =====================
        """
        #* Sense environment surroundings.
        state = self.perceive(neighborhood)
        #* Apply perturbations to stimuli 
        if perturbations is not None:
            for pert in perturbations:
                state = pert(state, self)
        #* Obtain actions using controller.
        actions = self.controller.step(state, reward=reward)
        #* Plan actions for future execution
        self.plan_actions(actions)

        # #* Render robot LED
        # self.update_colors(state, actions)
        # print(self.orientation)
        return state, actions

    def update_colors(self, state, action):
        colors = ['black', 'red', 'yellow', 'blue']
        if 'wireless_transmitter' in self.actuators.keys():
            for k, msg in enumerate(action['wireless_transmitter']['msg']):
                symbol = np.argmin([np.abs(sym - msg) for sym in [0, 0.33, 0.66, 1]])
                if k == 0:
                    self.colorA = colors[symbol]
                if k == 1:
                    self.colorB = colors[symbol]
    
        if 'led_actuator' in self.actuators.keys():
            self.color2 = ('green', 'white', 'red')[action['led_actuator']] #[actions['wireless_transmitter']['state']]#
      
    def plan_actions(self, actions):
        for actuator, action in actions.items():
            self.planned_actions[actuator] = (actuator == 'wheel_actuator')\
                and [action, self.position, self.orientation]  or [action]

    def actuate(self):
        """
        Executes the previously planned actions in order to be processed in the world.
        =====================
        - Args: None
        - Returns: None
        =====================
        """
        for actuator_name, actuator in self.actuators.items():
            # if actuator_name not in self.planned_actions:
            #     raise Exception('Error: Actuator does not have corresponding planned action.')
            actuator.step(*iter(self.planned_actions[actuator_name]))

    def perceive(self, neighborhood):
        """
        Computes the observed stimuli by steping each of the active sensors.
        =====================
        - Args:
            neighborhood [list] -> list filled with the neighboring world objects.
        -Returns:
            A dict with each sensor name as key and the sensor readings as value.
        =====================
        """
        return {sensor_name : sensor.step(neighborhood)\
                for sensor_name, sensor in self.sensors.items()}

    def reset(self):
        """
        Resets the robot dynamics, sensors, actuators and controller. Position and orientation 
        can be randomly initialized or fixed. In the former case a seed can be specified.
        =====================
        - Args:
            seed [int] -> seed for random intialization.
        - Returns: None
        =====================
        """
        self.delta_pos = np.zeros(2)
        self.delta_theta = 0.0
        self._food = False
        #* Reset Controller
        if self.controller is not None:
            self.controller.reset()
        #* Reset Actuators
        for actuator in self.actuators.values():
            if hasattr(actuator, 'reset'):
                actuator.reset()
        #* Reset Sensors
        for sensor in self.sensors.values():
            if hasattr(sensor, 'reset'):
                sensor.reset()

    @property
    def food(self):
        """Getter for the food attribute. It is a boolean attribute active if the robot stores food.
        """
        return self._food

    @food.setter
    def food(self, hasfood):
        """Setter for the food attribute. It is a boolean attribute active if the robot stores food.
        """
        self._food = hasfood