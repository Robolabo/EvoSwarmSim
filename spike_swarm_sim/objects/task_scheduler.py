import numpy as np
import pybullet as p
from spike_swarm_sim.objects.world_object import WorldObject
from spike_swarm_sim.register import world_object_registry
from spike_swarm_sim.utils import increase_time
from spike_swarm_sim.globals import global_states

@world_object_registry(name='task_scheduler')
class TaskScheduler(WorldObject):
    def __init__(self, *args, total_timesteps=1000, num_tasks=2, num_slots=2, replacement=False, **kwargs):
        super(TaskScheduler, self).__init__(*args, tangible=False, **kwargs)
        self.total_timesteps = total_timesteps
        self.num_tasks = num_tasks
        self.num_slots = num_slots
        self.replacement = replacement
        
        # self.min_slot_duration = int(total_timesteps
        # self.max_slot_duration = 
        self.task_order = None
        # self.task_switch = None
        self.t = 0
        self.prev_tasks = []

    def step(self, neighborhood):
        if global_states.RENDER:
            task_labels = ['Task A', 'Task B'] #['Red Light Pursuit', 'Cube Transportation']
            if self.t == 0:
                self.label_id = p.addUserDebugText(task_labels[self.current_task], (0,0,3), 
                                textColorRGB=(0,0,0), textSize=2, )
            else:
                self.label_id = p.addUserDebugText(task_labels[self.current_task], (-0.5,0,3), 
                                textColorRGB=(0,0,0), textSize=2, replaceItemUniqueId=self.label_id)
        # print(self.t, ('RED', 'YELLOW')[self.current_task])
        self.t += 1

    def controllable(self):
        return True

    @property
    def current_task(self):
        assert self.task_order is not None
        return self.task_order[self.t // (self.total_timesteps // self.num_slots + 1)]

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.t = 0
        # if seed is not None:
        #     self.task_order = np.array([1 if seed % 2 != 0 else 0]) #np.random.choice(self.num_tasks, size=self.num_slots, replace=False)
        #     print(seed, self.task_order)
        # else:
        #     self.task_order = np.random.choice(self.num_tasks, size=self.num_slots, replace=False)
        self.task_order = np.random.choice(self.num_tasks, size=self.num_slots, replace=self.replacement)
        if seed is not None:
            np.random.seed()

    def add_physics(self, engine):
        pass

    def position(self):
        pass
    
    def orientation(self):
        pass
    
