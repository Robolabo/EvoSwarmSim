import numpy as np
import numpy.linalg as LA
from .base_actuator import HighLevelActuator
from spike_swarm_sim.register import actuator_registry

@actuator_registry(name='grasp_actuator')
class GraspActuator(HighLevelActuator):
    def __init__(self, *args, **kwargs):
        super(GraspActuator, self).__init__(*args, **kwargs)
        self.grasp_range = 0.5
        self.grasp_cooldown = 0
        self.cube_grasped = None

    def __grasp(self, cube):
        new_pos = self.actuator_owner.position.copy()
        new_pos[-1] = 0.23
        cube.position = new_pos
        cube.is_grasped = True
        self.cube_grasped = cube
        # self.grasp_cooldown = 10


    def __drop(self):
        robot_pos = self.actuator_owner.position.copy()
        robot_ori = self.actuator_owner.orientation.copy()[-1]
        drop_sector = np.argmin(self.actuator_owner.sensors['distance_sensor3D'].reading)
        drop_ori = self.actuator_owner.sensors['distance_sensor3D'].directions(robot_ori)[drop_sector]
        new_pos = robot_pos + 0.3 * np.r_[np.cos(drop_ori), np.sin(drop_ori), 0.]
        self.cube_grasped.is_grasped = False
        self.cube_grasped.position = new_pos
        self.cube_grasped = None
        self.grasp_cooldown = 50

    def step(self, action, neighborhood):
        if self.cube_grasped is not None:
            #* Avoid cube from falling from robot.
            if LA.norm(self.cube_grasped.position - self.actuator_owner.position) > 0.3:
                self.__grasp(self.cube_grasped)
        #* Action is a\in{0,1,2}. a=0 means do nothing, a=1 means grasp and a=2 means drop.
        if action == 0 or 0 > action > 2:
            return
        if action == 1: #* Grasp    
            if self.cube_grasped is not None or self.grasp_cooldown > 0:
                self.grasp_cooldown -= 1
                return
            cubes = [obj for obj in neighborhood.values() if type(obj).__name__ == 'Cube']
            if len(cubes) == 0:
                return
            cube_distances = np.array([LA.norm(cube.position - self.actuator_owner.position) for cube in cubes])
            if not any(cube_distances < 0.5):
                return
            tar_cube = cubes[np.argmin(cube_distances)]
            #* Grasp cube if no cube already grasped or the cube has not been grasped by other robots.
            if not tar_cube.is_grasped:
                self.__grasp(tar_cube)
        elif action == 2: #*Drop
            if self.cube_grasped is None:
                return
            self.__drop()

    def reset(self):
        self.grasp_cooldown = 0
        self.cube_grasped = None