import numpy as np            
from matplotlib import colors
import matplotlib.pyplot as plot
import pybullet as p
from spike_swarm_sim.objects import WorldObject2D, WorldObject3D
from spike_swarm_sim.register import world_object_registry

@world_object_registry(name='light_source')
class LightSource3D(WorldObject3D):
    def __init__(self, position, orientation, *args, color='red', range=1., **kwargs):
        super(LightSource3D, self).__init__('light', position, orientation, z_offset=0.8,\
                        static=False, luminous=True, tangible=False, *args, **kwargs)
        self.range = range
        self.color = color
        self.shadow_id = None
        self.reset()

    def add_physics(self, physics_client):
        super().add_physics(physics_client)
        color = list(colors.to_rgb(self.color)) + [0.8]
        p.changeVisualShape(self.id, -1, rgbaColor=color, physicsClientId=physics_client)
    
    # def set_color(self):

    def step(self, world_dict):
        if self.controllable:
            if type(self.controller).__name__ == 'PreyController':
                robot_pos = [robot.position for robot in world_dict]
                self.position = self.controller.step(self.position, robot_pos)
            else:
                self.position = self.controller.step(self.position)
        return (0, 0)

    def reset(self, seed=None):
        self.shadow_id = None
        if self.controller is not None:
            self.controller.reset()

    def show_coverage(self):
        if self.shadow_id is None:
            self.shadow_id = p.loadURDF('spike_swarm_sim/objects/urdf/shadow.urdf', self.position,\
                p.getQuaternionFromEuler(self.orientation), physicsClientId=self.physics_client, globalScaling=self.range)
            color = list(colors.to_rgb(self.color)) + [0.3]
            # import pdb; pdb.set_trace()
            # p.resetBasePositionAndOrientation(self.shadow_id, p.getBasePositionAndOrientation(\
            #     self.id, physicsClientId=self.physics_client)[0], [0,0,0,0], physicsClientId=self.physics_client)
            p.changeVisualShape(self.shadow_id, -1, rgbaColor=color, physicsClientId=self.physics_client)
        # else:
        #     p.resetBasePositionAndOrientation(self.shadow_id, p.getBasePositionAndOrientation(\
        #         self.id, physicsClientId=self.physics_client)[0], [0,0,0,0], physicsClientId=self.physics_client)

    def hide_coverage(self):
        if self.shadow_id is not None:
            p.removeBody(self.shadow_id, physicsClientId=self.physics_client)
            self.shadow_id = None

class IsotropicEmitter(WorldObject2D):
    def __init__(self, *args, color='red', range=150, **kwargs):
        super(IsotropicEmitter, self).__init__(*args, **kwargs)
        self.range = range
        self.color = color
        self.reset()
    
    def step(self, world_dict):
        if self.controllable:
            if type(self.controller).__name__ == 'PreyController':
                robot_pos = [robot.pos for robot in world_dict]
                self.pos = self.controller.step(self.pos, robot_pos)
            else:
                self.pos = self.controller.step(self.pos)
        return (0, 0)

    def reset(self):
        if self.controller is not None:
            self.controller.reset()

    def initialize_render(self, canvas):
        x, y = tuple(self.pos)
        coverage_shadow_id = canvas.create_oval(x-10, y-10, x + 10, y + 10, fill=self.color)
        render_id = canvas.create_oval(x-10, y-10, x + 10, y + 10, fill=None)
        self.render_dict = {
            'body' : render_id,
            'shadow' : coverage_shadow_id,
        }
        return canvas 

    def render(self, canvas):
        x, y = tuple(self.pos)
        canvas.coords(self.render_dict['body'],
                        x-10, y-10,
                        x + 10, y + 10)
        canvas.itemconfig(self.render_dict['body'], fill=self.color)
        # #coverage shadow
        canvas.coords(self.render_dict['shadow'],
                x-10-self.range, y-10-self.range,
                x + 10+self.range, y + 10+self.range)
        canvas.itemconfig(self.render_dict['shadow'], fill=self.color)
        canvas.tag_lower(self.render_dict['body'])
        canvas.tag_lower(self.render_dict['shadow'])
        return canvas

@world_object_registry(name='light_source')
class LightSource(IsotropicEmitter):
    def __init__(self, *args, **kwargs):
        super(LightSource, self).__init__(*args, **kwargs)

@world_object_registry(name='food_area')
class FoodArea(IsotropicEmitter):
    def __init__(self, *args, **kwargs):
        super(FoodArea, self).__init__(*args, color='green', **kwargs)

@world_object_registry(name='nest')
class Nest(IsotropicEmitter):
    def __init__(self, *args, **kwargs):
        super(Nest, self).__init__(*args, color='black', **kwargs)
        self._food_items = 0
    
    def reset(self):
        super().reset()
        self._food_items = 0

    @property
    def food_items(self):
        return self._food_items
    
    def increase_food(self):
        self._food_items += 1
    
    def decrease_food(self):
        if self._food_items > 0:
            self._food_items = 1
    
    def initialize_render(self, canvas):
        x, y = tuple(self.pos)
        coverage_shadow_id = canvas.create_oval(x-10, y-10, x + 10, y + 10, fill=self.color)
        render_id = canvas.create_oval(x-10, y-10, x + 10, y + 10, fill=None)
        self.render_dict = {
            'body' : render_id,
            'shadow' : coverage_shadow_id,
        }
        return canvas 

    def render(self, canvas):
        x, y = tuple(self.pos)
        canvas.coords(self.render_dict['body'],
                        x-10, y-10,
                        x + 10, y + 10)
        canvas.itemconfig(self.render_dict['body'], fill=self.color)
        #coverage shadow
        canvas.coords(self.render_dict['shadow'],
                x-10-self.range, y-10-self.range,
                x + 10+self.range, y + 10+self.range)
        canvas.itemconfig(self.render_dict['shadow'], fill=self.color)
        canvas.tag_lower(self.render_dict['body'])
        canvas.tag_lower(self.render_dict['shadow'])
        return canvas