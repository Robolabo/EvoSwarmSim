import click
import logging
from mpi4py import MPI
from spike_swarm_sim import World2D, World3D, MultiWorldWrapper
from spike_swarm_sim.algorithms.evolutionary import GeneticAlgorithm, CMA_ES, xNES
from spike_swarm_sim.register import fitness_functions
from spike_swarm_sim.config_parser import json_parser
from spike_swarm_sim.register import algorithms
from spike_swarm_sim.globals import global_states

@click.command()
@click.option('-R', '--render', default=False, is_flag=True, help='Execute in render mode.')
@click.option('-d', '--debug', default=False, is_flag=True,  help='Execute in debug mode.')
@click.option('-r', '--resume', default=False, is_flag=True,\
        help='Resume optimization stored in the checkpoint settled in the JSON config.')
@click.option('-e', '--eval', default=False, is_flag=True, \
        help='Execute in eval mode. No optimization will be carried out.')
@click.option('-v', '--verbose', default=False, is_flag=True,\
        help='Execute in verbose mode (info msgs enabled).')
@click.option('-n', '--ncpu', default=1, help='Number of CPU cores.')
@click.option('-f', '--cfg', default='default', help='Name of the JSON config. file.')
def main(render, resume, cfg, debug, eval, verbose, ncpu):
    global_states.set_states(render=render, eval=eval, debug=debug, info=verbose)
    cfg_dict = json_parser(cfg)
    if debug:
        # logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().level = logging.DEBUG
        logging.getLogger().debug('Executing in DEBUG mode.')
    elif verbose:
        logging.getLogger().level = logging.INFO
        logging.getLogger().info('Executing in VERBOSE mode.')
    if ncpu > 1 or MPI.COMM_WORLD.Get_size() > 1:
        world = MultiWorldWrapper(max(ncpu, MPI.COMM_WORLD.Get_size()), 
                    height=cfg_dict['world']["height"], width=cfg_dict['world']["width"],\
                    world_delay=cfg_dict['world']["world_delay"])
    else:
        world_cls = {'2D' : World2D, '3D' : World3D}[cfg_dict['world']['engine']]
        world = world_cls(height=cfg_dict['world']["height"], width=cfg_dict['world']["width"],\
             world_delay=cfg_dict['world']["world_delay"])
        # world = World3D(height=cfg_dict['world']["height"], width=cfg_dict['world']["width"],\
        #                 world_delay=cfg_dict['world']["world_delay"],)
    world.build_from_dict(cfg_dict['world'], ann_topology=cfg_dict['topology'])

    if cfg_dict['algorithm'] is not None and len(cfg_dict['algorithm']):
        ga_config = cfg_dict['algorithm']
        fitness = fitness_functions[ga_config['fitness_function']]()
        algorithm_cls = algorithms[cfg_dict['algorithm']['name']]
        opt_alg = algorithm_cls(cfg_dict['algorithm']['populations'], world,\
                    population_size=ga_config['population_size'], n_generations=ga_config['generations'],\
                    eval_steps=ga_config['evaluation_steps'], num_evaluations=ga_config['num_evaluations'],\
                    n_processes=ncpu, resume=resume, fitness_fn=fitness, checkpoint_name=cfg_dict["checkpoint_file"])
        #* Run GA
        if not eval:
            opt_alg.run()
        else:
            #* Evaluate after evolution
            opt_alg.evaluate()
    # else:
    #     world.reset()
    #     # world.draw_node_coords()
    #     while(True):
    #         state, action = world.step()

if __name__ == "__main__":
    main()