import logging
import sys
import os
from colorama import Fore, Style
from pathlib import Path


# from .world import  World2D
from .world_3D import World3D, World2D, MultiWorldWrapper
from .physics_engine import Engine2D, Engine3D

from .env_perturbations import *
from .algorithms import *
from .objects import *
from .actuators import *
from .sensors import *
from .controllers import *
from .objectives import *
from .neural_networks import *
from .config_parser import *
from .register import *
from .globals import Globals, global_states

ROOT_DIR = Path(__file__).parents[0].parents[0]
sys.path.append(os.path.abspath(os.path.join('..', 'config')))

formatter = logging.Formatter('%(asctime)s %(name)s [%(levelname)s]: %(message)s',\
            datefmt='%Y-%m-%d %H:%M:%S')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

log_color = {
   logging.WARNING : Fore.YELLOW,#'\x1b[39;1m',
   logging.INFO : Fore.BLUE,
   logging.DEBUG : Fore.GREEN,#'\x1b[32;1m',
   logging.ERROR : Fore.RED,
   logging.CRITICAL : Fore.RED,
}

def apply_color_log(func):
    def wrapper(*args, **kwargs):      
        args[0].levelname = f"{{}}".format(log_color[args[0].levelno]) + "{0}".format(args[0].levelname)\
                        + f'{Style.RESET_ALL}'
        return func(*args)
    return wrapper
console_handler.emit = apply_color_log(console_handler.emit)
logging.getLogger().addHandler(console_handler)