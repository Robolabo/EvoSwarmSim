import numpy as np
from spike_swarm_sim.controllers import Controller
from spike_swarm_sim.register import controller_registry
from spike_swarm_sim.utils import compute_angle, normalize, toroidal_difference, increase_time

@controller_registry(name='light_orbit_controller')
class LightOrbitController(Controller):
    """ Class for the control of light sources as orbits
    around a central position.
    There is a small probability of rotation sense inversion. 
    """
    def __init__(self):
        self.t = 1
        self.dir = 1

    @increase_time
    def step(self, pos):
        # import pdb; pdb.set_trace()
        new_pos = pos[:2].copy()
        # self.dir = np.random.choice([self.dir, -self.dir], p=[0.99, 0.01])
        current_angle = compute_angle(new_pos - np.array([0, 0]))
        new_angle = current_angle + self.dir * 0.005 #0.012 #0.01
        new_rad = min(np.linalg.norm(new_pos - np.array([0, 0])) + 0.004, 2.5)
        new_pos = new_rad * np.r_[np.cos(new_angle), np.sin(new_angle)] + np.array([0, 0])
        
        if len(pos) == 3:
            new_pos = np.r_[new_pos, pos[-1].copy()]
        return new_pos

    def reset(self):
        self.t = 1
        self.dir = 1

@controller_registry(name='light_rnd_pos_controller')
class LightRndPositionController(Controller):
    """ Class for the control of light sources as straight 
    trajectories to randomly sampled goal locations. Every 
    200 time steps the goal location is resampled.
    """
    def __init__(self):
        self.t = 1
        self.tar_pos = np.random.uniform(-2, 2, size=2)

    @increase_time
    def step(self, pos):
        new_pos = pos[:2].copy()
        if self.t % 100 == 0:
            self.tar_pos = np.random.uniform(-5, 5, size=2)
        new_pos = new_pos + 0.012 * normalize(self.tar_pos - new_pos)
        if len(pos) == 3:
            new_pos = np.r_[new_pos, pos[-1]]
        return new_pos

    def reset(self):
        self.t = 1
        self.tar_pos = np.random.uniform(-4, 4, size=2)

# @controller_registry(name='prey_controller')
# class PreyController(RobotController):
#     """ Class for the control of light sources mimicking prey escape. 
#     The controller deterministically computes the escape direction (steering) 
#     based on the known positions of the predators. If a predator is at a distance 
#     lower than 30 (3cm), then the prey is hunted and stops its motion.
#     """
#     def __init__(self,  cc, *args, **kwargs):
#         super(PreyController, self).__init__(*args, **kwargs)
#         self.t = 1
#         self.direction = np.r_[np.cos(np.pi/4), np.sin(np.pi/4)]
#         self.hunted = 0

#     @increase_time
#     def step(self, state):
#         import pdb; pdb.set_trace()
#         # new_pos = my_pos[:2].copy()
#         # if len(robot_positions) == 0:
#         #     return my_pos
#         # # robot_light_vecs = np.stack([toroidal_difference(robot_pos[:2], my_pos[:2]) for robot_pos in robot_positions])
#         # robot_light_vecs = np.stack([robot_pos[:2] - my_pos[:2] for robot_pos in robot_positions])
#         # distances = np.array([np.linalg.norm(v) for v in robot_light_vecs])
#         # near_robots = [d < 1 for d in distances]
#         # if any(near_robots):
#         #     weights = (5 - distances[near_robots]) / 5
#         #     weights /= sum(weights)
#         #     self.direction = -normalize(np.dot(weights, robot_light_vecs[near_robots]))
#         # if not self.hunted:
#         #     new_pos += 0.01 * self.direction
#         #     self.hunted = any([np.linalg.norm(v) < 0.2 for v in robot_light_vecs])
        
#         # if len(my_pos) == 3:
#         #     new_pos = np.r_[new_pos, my_pos[-1].copy()]

#         # #! luz no puede pasar de la pared.


#         # return new_pos

#     def reset(self):
#         self.t = 1
#         self.hunted = 0
