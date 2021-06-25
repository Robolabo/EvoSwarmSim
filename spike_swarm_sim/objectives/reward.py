import numpy as np
import numpy.linalg as LA
from spike_swarm_sim.utils import angle_mean, angle_diff, increase_time
from spike_swarm_sim.register import reward_registry


class AlignmentReward:
    def __init__(self):
        self.required_info = ("robot_positions", "robot_orientations",)
    
    def __call__(self, actions, states, info=None):
        thetas = info['robot_orientations']
        angle_errs = np.mean([angle_diff(th1, th2)\
                            for j, th1 in enumerate(thetas)\
                            for i, th2 in enumerate(thetas) if i != j])
        rA = 1 - (angle_errs / np.pi) ** 0.7
        rB = 1 - np.mean([np.abs(ac['wheel_actuator'][0]) for ac in actions])
        return 0.7 * rA + 0.3 * rB


@reward_registry(name='goto_light')
class GoToLightReward:
    def __init__(self):
        self.required_info = ("generation", "robot_positions", "light_positions")

    def __call__(self, actions, states, info=None):
        # positions = info['robot_positions']
        # light_pos = info['light_positions']
        # distances = [LA.norm(robot_pos - light_pos) for robot_pos in positions]

        # rew = np.mean([np.clip(1 - (dist / 100), a_min=0, a_max=1) for dist in distances])
        # return rew
        
        rew_obst = -1. if np.max(states['distance_sensor3D']) > 0.4 else 0.0
        rew_ls = 1. if np.max(states['light_sensor3D']) > 0.4 else 0.0
        return  rew_obst + rew_ls



@reward_registry(name='many_lights')
class GoToLightReward:
    def __init__(self):
        pass
    def __call__(self, actions, states, info=None):
        rewards = np.zeros(len(actions))
        robots = [obj for obj in info.values() if type(obj).__name__ == 'Robot3D']
        lights = [obj for obj in info.values() if type(obj).__name__ == 'Light_Source']
        # Assume only one of each
        green_ls_pos = [ls.position for ls in lights if ls.color == 'green'][0]
        red_ls_pos = [ls.position for ls in lights if ls.color == 'red'][0]
        yellow_ls_pos = [ls.position for ls in lights if ls.color == 'yellow'][0]
        for i, robot in enumerate(robots):
            mask_ls = [LA.norm(robot.position[:2] - ls.pos[:2]) < 1 for ls in [green_ls_pos, red_ls_pos, yellow_ls_pos]]
            nearest_robot = np.min([LA.norm(robot.position - robotB.position) for j, robotB in enumerate(robots)])
            
        # return  rew_obst + rew_ls


@reward_registry(name='transport_cube')
class TransportCubeReward:
    def __init__(self):
        self.t = 0
        self.required_info = ("generation", "robot_positions", "light_positions")
        self.prev_cubes_pos = None

    @increase_time
    def __call__(self, actions, states, info=None):
        rewards = np.zeros(len(actions))
        robots = [obj for obj in info.values() if type(obj).__name__ == 'Robot3D']
        robots_pos = [robot.position for robot in robots]
        cubes_pos = np.array([obj.position for obj in info.values() if type(obj).__name__ == 'Cube'])
        if self.prev_cubes_pos is None:
            self.prev_cubes_pos = cubes_pos.copy()
            return rewards
        delta_pos_cubes = cubes_pos[:, :2] - self.prev_cubes_pos[:, :2]
        delta_pos_cubes[np.abs(delta_pos_cubes) < 1e-2] = 0
        ground_area_pos = np.array([obj.position for obj in info.values() if type(obj).__name__ == 'GroundArea'])
        ground_area_rad = np.array([obj.radius for obj in info.values() if type(obj).__name__ == 'GroundArea'])
        mask_ground_area = LA.norm(cubes_pos - ground_area_pos, axis=1) < ground_area_rad
        mask_prev_ground_area = LA.norm(self.prev_cubes_pos - ground_area_pos, axis=1) < ground_area_rad
        # Reward when cube enters ground area and penalize when it exits ground area.
        rewards += np.sum(mask_ground_area & ~mask_prev_ground_area)
        rewards -= np.sum(~mask_ground_area & mask_prev_ground_area)
        mask_delta_cube = LA.norm(delta_pos_cubes, axis=1) > 0
        
        mask_direction_moved = LA.norm(cubes_pos - ground_area_pos, axis=1) < LA.norm(self.prev_cubes_pos - ground_area_pos, axis=1)
        mask_direction_moved = np.where(~mask_direction_moved, -1, mask_direction_moved).astype(float)
        rewards += np.sum(LA.norm(delta_pos_cubes, axis=1) * mask_delta_cube * mask_direction_moved)
        # if any(rewards != 0): import pdb; pdb.set_trace()
        self.prev_cubes_pos = cubes_pos.copy()
        return rewards

    def reset(self):
        self.t = 0
        self.prev_cubes_pos = None