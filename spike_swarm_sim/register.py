from functools import wraps
from spike_swarm_sim.objects.world_object import WorldObject3D, WorldObject, WorldObject2D

# Set of registers for easing class and function automatic discovery
world_objects = {'2D' : {}, '3D' : {}}
sensors = {}
actuators = {}
neuron_models = {}
synapse_models = {}
controllers = {}
encoders = {}
decoders = {}
evo_operators = {}
fitness_functions = {}
algorithms = {}
initializers = {}
env_perturbations = {}
receptive_fields = {}
learning_rules = {}
rewards = {}

def world_object_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        # if name == 'task_scheduler': import pdb; pdb.set_trace()
        engine = '3D' if issubclass(cls, WorldObject3D) else '2D'
        world_objects[engine][name] = cls
        if all([not issubclass(cls, WorldObject3D), 
                not issubclass(cls, WorldObject2D), 
                issubclass(cls, WorldObject)]):
            world_objects['3D'][name] = cls # Add to both 
        return cls
    return wrapper

def fitness_func_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        fitness_functions[name] = cls
        return cls 
    return wrapper 


# def sensor_registry(*args, **kwargs): 
#     def wrapper(cls):
#         name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
#         sensors[name] = cls
#         @wraps(cls)
#         def _wrapper(*args, **kwargs):      
#             return cls
#         return _wrapper
#     return wrapper 

def sensor_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        sensors[name] = cls
        return cls 
    return wrapper 

def actuator_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        actuators[name] = cls
        return cls 
    return wrapper 


def controller_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        controllers[name] = cls
        return cls 
    return wrapper 

def neuron_model_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        neuron_models[name] = cls
        return cls 
    return wrapper

def synapse_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        synapse_models[name] = cls
        return cls
    return wrapper

def algorithm_registry(*args, **kwargs):
    def wrapper(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        algorithms[name] = cls
        return cls 
    return wrapper
 
def encoding_registry(cls):
    encoders[cls.__name__] = cls
    return cls

def decoding_registry(cls):
    decoders[cls.__name__] = cls
    return cls

def evo_operator_registry(*args, **kwargs):
    def decorator(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        evo_operators[name] = cls
        return cls
    return decorator

def initializer_registry(*args, **kwargs):
    def decorator(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        initializers[name] = cls
        return cls
    return decorator

def env_perturbation_registry(*args, **kwargs):
    def decorator(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        env_perturbations[name] = cls
        return cls
    return decorator

def receptive_field_registry(*args, **kwargs):
    def decorator(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        receptive_fields[name] = cls
        return cls
    return decorator

def learning_rule_registry(*args, **kwargs):
    def decorator(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        learning_rules[name] = cls
        return cls
    return decorator

def reward_registry(*args, **kwargs):
    def decorator(cls):
        name = (cls.__name__, kwargs['name'])['name' in kwargs.keys()]
        rewards[name] = cls
        return cls
    return decorator