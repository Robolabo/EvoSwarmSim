import time
import os
import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import pygame
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pymunk
import pymunk.pygame_util
# from pygame.color import THECOLORS
# from pymunk.vec2d import Vec2d
from spike_swarm_sim.globals import global_states

#! Inherit from Bullet??
class Engine3D:
    def __init__(self):
        self.connected = False
        self.render = global_states.RENDER
        self.engine = None
        self.gui_params = {}

    def step_physics(self):
        p.stepSimulation()

    def step_render(self):
        # if self.physics_client.readUserDebugParameter(self.gui_params['robot_focus']) == 1:
        #     self.physics_client.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=30,\
        #         cameraTargetPosition=self.robots['robotA_0'].position, cameraPitch=-70)#-60,)
        time.sleep(1/240.) # Fast mode
        # time.sleep(1/10) # Slow mode

    def connect(self, objects):
        self.engine = bc.BulletClient(connection_mode=p.GUI if self.render else p.DIRECT)
        self.engine.resetSimulation(physicsClientId=self.engine._client)
        self.engine.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.engine.setGravity(0, 0, -9.8)
        self.engine.setTimeStep(1/50.)
        # self.engine.setPhysicsEngineParameter(numSolverIterations=10)
        # self.engine.setPhysicsEngineParameter(fixedTimeStep=1000)
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.engine._client)
        # self.engine.changeDynamics(planeId, linkIndex=-1, lateralFriction=0.9)
        self.add_objects(objects)
        self.connected = True
        # self.gui_params = {}
        if self.render:
            self.gui_params['light_coverage'] = self.engine.addUserDebugParameter("Show lights' coverage", 1, -1, -1)
            # self.gui_params['robot_focus'] = self.physics_client.addUserDebugParameter('Robot focus', 1, -1, 1)
            self.engine.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=30,\
                    cameraPitch=-90, cameraTargetPosition=[0, 0, 0])

    def disconnect(self):
        # self.engine.resetSimulation(physicsClientId=self.engine._client)
        self.engine.disconnect()
        self.connected = False

    def add_objects(self, objects):
        for obj in objects:
            obj.add_physics(self.engine._client)

    def ray_cast(self, origin, destination):
        ray_res = p.rayTest(origin, destination, physicsClientId=self.engine._client)
        return ray_res[0][0]

    #! USELESS?
    def initialize_render(self):
        self.gui_params['light_coverage'] = self.engine.addUserDebugParameter("Show lights' coverage", 1, -1, 1)
        self.engine.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=30,\
                    cameraPitch=-60, cameraTargetPosition=[0, 0, 0])



class Engine2D:
    def __init__(self, height=1000, width=1000):
        self.height = height
        self.width = width
        self.world_delay = 1 #! TO BE REMOVED
        self.render = global_states.RENDER
        self.screen = None
        self.engine = None
        self.draw_options = None
        self.objects = {}

    def connect(self, objects):
        self.engine = pymunk.Space()
        self.engine.gravity = (0.0, 0.0)
        self.add_objects(objects)
        self.connected = True
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.height, self.width))
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.clock = pygame.time.Clock()

    def add(self, identifier, bodies, shapes):
        self.objects[identifier] = {'bodies' : bodies, 'shapes' : shapes}
        self.engine.add(*bodies, *shapes)

    def get_body_position(self, identifier, body_id):
        return self.objects[identifier]['bodies'][body_id].position

    def reset_body_position(self, identifier, body_id, position):
        self.objects[identifier]['bodies'][body_id].position = position

    def get_body_orientation(self, identifier, body_id):
        return self.objects[identifier]['bodies'][body_id].angle

    def reset_body_orientation(self, identifier, body_id, orientation):
        self.objects[identifier]['bodies'][body_id].angle = orientation

    def disconnect(self):
        pass

    def initialize_render(self):
        pass

    def step_physics(self):
        self.engine.step(1 / 60.0)

    def step_render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                import sys; sys.exit(0)
        pygame.event.get()
        self.screen.fill((255, 255, 255))
        self.engine.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(30)

    def add_objects(self, objects):
        for obj in objects:
            obj.add_physics(self)
    
    def ray_cast(self, origin, destination):
        if len(origin) == 3:
            origin = origin[:2]
        if len(destination) == 3:
            destination = destination[:2]
        ray_res = self.engine.segment_query_first(origin, destination, 1, pymunk.ShapeFilter())
        if ray_res is not None:
            # Find ID of body
            ray_res.shape.body
        #* 
        return ray_res[0][0]
