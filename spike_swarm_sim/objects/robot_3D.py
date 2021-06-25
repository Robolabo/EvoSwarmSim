import numpy as np
from shapely.geometry import Point
from spike_swarm_sim.objects import WorldObject3D
from spike_swarm_sim.actuators.base_actuator import HighLevelActuator
from spike_swarm_sim.register import sensors, actuators, world_object_registry


@world_object_registry(name='robot')
class Robot3D(WorldObject3D):
    """
    Base class for the robot world object.
    """
    def __init__(self, position, orientation,  *args, urdf_file='epuck', **kwargs):
        super(Robot3D, self).__init__(urdf_file, position, orientation,\
                        static=False, luminous=False, tangible=True, \
                        *args, **kwargs)
        self._food = False

        #* Initialize sensors and actuators according to controller requirements
        self.sensors = {k : s(self, **self.controller.enabled_sensors[k])\
                            for k, s in sensors.items()\
                            if k in self.controller.enabled_sensors.keys()}
        self.actuators = {k : a(self, **self.controller.enabled_actuators[k])\
                            for k, a in actuators.items()\
                            if k in self.controller.enabled_actuators.keys()}

        #* Storage for actions selected by the controllers to be fed to actuators
        self.planned_actions = {k : [None] for k in actuators.keys()}

        #* Rendering colors (TO BE MOVED TO RENDER FILE IN THE FUTURE)
        self.colorA = 'black'
        self.colorB = 'black'
        self.color2 = ('skyblue3', 'green')[self.trainable]
        # self.reset()
        self.st_aux = [] #!

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

        # #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # freq = {5 : 1, 6: 0.5, 7: 1, 8: 6}[self.id]
        # tt = self.controller.neural_network.t
        # actions['wireless_transmitter']['msg'] = [np.sin(2 * np.pi * freq * tt * 1e-3)**2]
        # self.st_aux.append(state['IR_receiver']['msg'])
        
        
        # if self.id == 5 and tt > 1000:
        #     self.st_aux = np.vstack(self.st_aux)
        #     import matplotlib.pyplot as plt
        #     import matplotlib as mpl
        #     mpl.rcParams['lines.linewidth'] = 2
        #     mpl.rcParams['font.size'] = 20
        #     mpl.rc('xtick', labelsize=17)
        #     mpl.rc('ytick', labelsize=17)
        #     plt.plot(self.st_aux)
        #     plt.xlabel('Time')
        #     plt.ylabel('Message')
        #     plt.legend(['Sector 1', 'Sector 2','Sector 3','Sector 4'], ncol=4)
        #     plt.show()
        #     import pdb; pdb.set_trace()
        # #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        #* Plan actions for future execution
        self.plan_actions(actions)
        # #* Handle robot food pickup
        # if 'food_area_sensor' in state.keys() and bool(state['food_area_sensor'][0]):
        #     self.food = True
        # if 'nest_sensor' in state.keys() and bool(state['nest_sensor'][0]):
        #     self.food = False
        return state, actions

    def plan_actions(self, actions):
        for actuator, action in actions.items():
            self.planned_actions[actuator] = (actuator == 'wheel_actuator')\
                    and [action, self.position, self.orientation]  or [action]

    def actuate(self, neighborhood):
        """
        Executes the previously planned actions in order to be processed in the world.
        =====================
        - Args: None
        - Returns: None
        =====================
        """
        for actuator_name, actuator in self.actuators.items():
            if issubclass(type(actuator), HighLevelActuator):
                actuator.step(*iter(self.planned_actions[actuator_name]), neighborhood)
            else:
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

    def reset(self, seed=None):
        """
        Resets the robot dynamics, sensors, actuators and controller. Position and orientation 
        can be randomly initialized or fixed. In the former case a seed can be specified.
        =====================
        - Args:
            seed [int] -> seed for random intialization.
        - Returns: None
        =====================
        """
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




@world_object_registry(name='minitaur')
class Minitaur(Robot3D):
    def __init__(self, *args, **kwargs):
        super(Minitaur, self).__init__(*args, urdf_file='quadruped/minitaur', z_offset=0.5,**kwargs)

@world_object_registry(name='epuck')
class Epuck3D(Robot3D):
    def __init__(self, *args, **kwargs):
        super(Epuck3D, self).__init__(*args, urdf_file='epuck', **kwargs)