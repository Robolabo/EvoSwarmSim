import time
import logging 
from functools import wraps
from collections import deque
import numpy as np
from spike_swarm_sim.globals import global_states

def time_elapsed(func):
    """ Computes the amount of time that a function elapses.
    Only works in DEBUG mode.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        logging.info('Function {} elapsed {}.'.format(func.__qualname__, round(time.time() - t0, 4)))
        return result
    return wrapper


def mov_average_timeit(func):
    """ Computes the mean time that a function elapses using 50 buffered samples.
    It logs the mean every 50 executions and only works in DEBUG mode.  
    """
    times_buffer = deque([])
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        if global_states.DEBUG:
            if len(times_buffer) == 50:
                times_buffer.appendleft(time.time() - t0)
                mean_time = np.mean(times_buffer)
                std_time = np.std(times_buffer)
                logging.debug('Function {} mean elapsed time is {} (50 samples).'\
                    .format(func.__qualname__, round(mean_time, 4)))
                times_buffer.clear()
            else:
                times_buffer.appendleft(time.time() - t0)
        return result
    return wrapper

def increase_time(func):
    """ Decorator that increases a time or counter variable of a class.  
    The variable name must be called t.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        args[0].t += 1
        return result
    return wrapper