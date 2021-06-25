import numpy as np            
from matplotlib import colors
import matplotlib.pyplot as plot
import pybullet as p
from spike_swarm_sim.objects import WorldObject2D, WorldObject3D
from spike_swarm_sim.register import world_object_registry



@world_object_registry(name='ball')
class Ball(WorldObject3D):
    def __init__(self, position, orientation, *args, color='red', range=1., **kwargs):
        super(Ball, self).__init__('ball', position, orientation,\
                        *args, **kwargs)
        self.color = color

    def add_physics(self, physics_client):
        super().add_physics(physics_client)
        color = list(colors.to_rgb(self.color)) + [1.]
        # import pdb; pdb.set_trace()
        p.changeVisualShape(self.id, -1, rgbaColor=color, physicsClientId=physics_client)
    
    def step(self, world_dict):
        pass

    def reset(self, seed=None):
        pass


@world_object_registry(name='cube')
class Cube(WorldObject3D):
    def __init__(self, position, orientation, *args, color='blue', mass=1., side_len=0.3, **kwargs):
        position = list(position)
        position[-1] = side_len / 2 - 0.1
        super(Cube, self).__init__('cube', position, orientation,\
                        *args, **kwargs)
        self.color = color
        self.mass = mass
        self.side_len = side_len
        self.is_grasped = False # Whether a robot is grasping the cube or not.

    def add_physics(self, physics_client):
        super().add_physics(physics_client, scaling=self.side_len)
        color = list(colors.to_rgb(self.color)) + [1.]
        # import pdb; pdb.set_trace()
        p.changeVisualShape(self.id, -1, rgbaColor=color, physicsClientId=physics_client)
        p.changeDynamics(self.id, -1, mass=self.mass, physicsClientId=physics_client)
    
    def step(self, world_dict):
        pass

    def reset(self, seed=None):
        self.is_grasped = False

@world_object_registry(name='ground_area')
class GroundArea(WorldObject3D):
    def __init__(self, position, orientation, *args, color='red', radius=1., **kwargs):
        super(GroundArea, self).__init__('ground_area', position, orientation,\
                        *args, tangible=False, **kwargs)
        self.color = color
        self.radius = radius

    def add_physics(self, physics_client):
        super().add_physics(physics_client, scaling=self.radius)
        color = list(colors.to_rgb(self.color)) + [1.]
        p.changeVisualShape(self.id, -1, rgbaColor=color, physicsClientId=physics_client)
        
    def reset(self, seed=None):
        pass