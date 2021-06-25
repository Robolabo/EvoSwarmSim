from functools import reduce 
import numpy as np
from spike_swarm_sim.register import env_perturbation_registry
from spike_swarm_sim.utils import increase_time, angle_diff

class EnvironmentalPerturbation:
    def __init__(self, total_robots, affected_robots='all'):
        self.random_selection = affected_robots is not 'all' and not isinstance(affected_robots, list)
        self.total_robots = total_robots
        self.affected_robots = total_robots if affected_robots == 'all' else affected_robots
        if affected_robots == 'all':
            self.affected_robots = np.arange(total_robots)
        else:
            if isinstance(affected_robots, list):
                self.affected_robots = np.array(affected_robots)
            else:
                self.affected_robots = np.random.choice(range(total_robots), size=affected_robots, replace=False)
        self.t = 0
    
    def reset(self):
        self.t = 0
        if self.random_selection:
            self.affected_robots = np.random.choice(range(self.total_robots),\
                size=self.affected_robots.shape[0], replace=False)

class PostProcessingPerturbation(EnvironmentalPerturbation):
    def __init__(self, *args, **kwargs):
        super(PostProcessingPerturbation, self).__init__(*args, **kwargs)
        self.postprocessing = True

    def __call__(self, states, actions, robots):
        raise NotImplementedError

class PreProcessingPerturbation(EnvironmentalPerturbation):
    def __init__(self, *args, **kwargs):
        super(PreProcessingPerturbation, self).__init__(*args, **kwargs)
        self.postprocessing = False

    def __call__(self, state, robot):
        raise NotImplementedError

@env_perturbation_registry(name='leader_failure')
class LeaderFailure(PostProcessingPerturbation):
    """ Constrain Specific to leader selection problem. 
    It produces the fault of the leader after a certain amount 
    of time of consecutive leadership.
    """
    def __init__(self, *args, time_to_failure=50, **kwargs):
        super(LeaderFailure, self).__init__(*args, **kwargs)
        self.time_to_failure = time_to_failure
        self.blacklist = []
        self.leaders_consec = None
    
    @increase_time
    def __call__(self, states, actions, robots):
        for idx in self.blacklist:
            #! actions[idx]['led_actuator'] = 0
            actions[idx]['led_actuator_3D'] = 0
            # Impose relay mode 
            actions[idx]['wireless_transmitter']['msg'] = states[idx]['IR_receiver']['msg']
            tuple(robots.values())[idx].actuators['led_actuator_3D'].fault = True
            #! tuple(robots.values())[idx].update_colors(states[idx], actions[idx])
            #! tuple(robots.values())[idx].color2 = 'red'
        leds = [ac['led_actuator_3D'] for ac in actions]#!
        if np.sum(leds) == 1:
            leader = np.argmax(leds)
            self.leaders_consec[leader] += 1
            self.leaders_consec[(1 - np.array(leds)).astype(bool)] = 0
            if self.leaders_consec[leader] > 50:
                print(leader, ' in blacklist')
                self.blacklist.append(leader)
                self.leaders_consec[leader] = 0
        else:
            self.leaders_consec *= 0 # reset
        return (states, actions)

    def reset(self):
        super(LeaderFailure, self).reset()
        self.blacklist = []
        self.leaders_consec = np.zeros(self.total_robots)


@env_perturbation_registry(name='uncontrollable_rotation')
class UncontrollableRotation(PostProcessingPerturbation):
    def __init__(self, *args,  **kwargs):
        super(UncontrollableRotation, self).__init__(*args, **kwargs)

    @increase_time
    def __call__(self, states, actions, robots):
        if self.t > 100:
            for i in self.affected_robots:
                tuple(robots.values())[i].color2 = 'red'
                tuple(robots.values())[i].planned_actions['wheel_actuator'][0] = np.array([.3, -.3])
                if angle_diff(tuple(robots.values())[i].orientation, np.radians([270, 90, 15, 180][self.t // 600])) < .15:
                    actions[i]['wheel_actuator'] = np.array([0., 0.])
                    tuple(robots.values())[i].planned_actions['wheel_actuator'][0] = np.array([.0, 0.])
        return (states, actions)

@env_perturbation_registry(name='stimuli_inhibition')
class StimuliInhibition(PreProcessingPerturbation):
    def __init__(self, *args, stimuli='light_sensor', replace_value=0, **kwargs):
        super(StimuliInhibition, self).__init__(*args, **kwargs)
        self.replace_value = replace_value
        self.stimuli = stimuli.split(':')

    @increase_time
    def __call__(self, state, robot):
        stim_val = reduce(lambda x, y: x[y], self.stimuli, state)
        noise_sigma = robot.sensors[self.stimuli[0]].noise_sigma if hasattr(robot.sensors[self.stimuli[0]], 'noise_sigma') else 0.
        if len(self.stimuli) > 1:
            state[self.stimuli[0]][self.stimuli[1]] = np.random.randn(len(stim_val)) * noise_sigma\
                    if noise_sigma > 0 else np.zeros_like(stim_val)
            state[self.stimuli[0]][self.stimuli[1]] += self.replace_value  
        else:
            state[self.stimuli[0]] = np.random.randn(len(stim_val)) * noise_sigma\
                    if noise_sigma > 0 else np.zeros_like(stim_val)
            state[self.stimuli[0]] += self.replace_value       
        # state.update(reduce(lambda x, y: {y : x}, self.stimuli[::-1], np.zeros_like(stim_val)))
        return state