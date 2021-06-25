from collections import deque
import numpy as np
from spike_swarm_sim.algorithms.interfaces import GET, SET, LEN, INIT
from spike_swarm_sim.register import learning_rule_registry

def append_and_pop(queue, new_elem):
    queue.append(new_elem)
    queue.popleft()
    return queue

@learning_rule_registry(name='generalized_hebbian')
class GeneralizedHebbian:
    def __init__(self):
        self.modulated = False #!
        self.learning_rate = 5e-3
        self.A = 1.0
        self.B = 0.0
        self.C = 0.0
        self.D = 0.0
        self.gamma = 0.9
        self.timesteps_update = 50 # Gather XX rewards before updating for computing value func.
        self.reward_queue = deque([])
        self.activities_queue = deque([])
        self.inputs_queue = deque([])
        self.t = 0

    def __step(self, inputs, activities, reward=None):
        act_inpt_cat = np.r_[inputs, activities]
        weight_update = self.learning_rate * (
                        self.A * np.outer(activities, act_inpt_cat)\
                        + self.B * np.outer(activities, np.ones_like(act_inpt_cat))\
                        + self.C * np.outer(np.ones_like(activities), act_inpt_cat)\
                        + self.D) 
        return weight_update * reward if reward is not None else weight_update
    
    def step(self, inputs, activities, reward=None):
        if not self.modulated:
            return self.__step(inputs, activities, reward=1.)
        if self.t < self.timesteps_update:
            self.activities_queue.append(activities)
            self.inputs_queue.append(inputs)
            if reward is not None:
                self.reward_queue.append(reward)
            self.t += 1
            return 0.0
        weight_update = 0.0
        if self.timesteps_update > 1:
            self.activities_queue = append_and_pop(self.activities_queue, activities)
            self.inputs_queue = append_and_pop(self.inputs_queue, inputs)
            if reward is not None:
                self.reward_queue = append_and_pop(self.reward_queue, reward)
            value_fn = np.sum([rew * self.gamma ** k for k, rew in enumerate(self.reward_queue)])
            weight_update = self.__step(self.inputs_queue[0], self.activities_queue[0], reward=value_fn)
        else:
            weight_update = self.__step(inputs, activities, reward=reward)
        self.t += 1
        return weight_update

    def build(self, ann_graph):
        mask = np.full((len(ann_graph['neurons']), len(ann_graph['inputs']) + len(ann_graph['neurons'])), False)
        trainable_mask = mask.copy()
        A = mask.copy().astype(float)
        B = A.copy()
        C = A.copy()
        D = A.copy()
        for name, node in ann_graph['neurons'].items():
            in_connections = [syn for syn in ann_graph['synapses'].values() if syn['post'] == name]
            for syn in in_connections:
                if syn['enabled']:
                    pre_idx = ann_graph['inputs'][syn['pre']]['idx'] if syn['pre'] in ann_graph['inputs']\
                                else ann_graph['neurons'][syn['pre']]['idx'] + len(ann_graph['inputs'])
                    mask[node['idx'], pre_idx] = True
                    trainable_mask[node['idx'], pre_idx] = True
                    A[node['idx'], pre_idx] = syn.get('learning_rule', {}).get('A', 0.)
                    B[node['idx'], pre_idx] = syn.get('learning_rule', {}).get('B', 0.)
                    C[node['idx'], pre_idx] = syn.get('learning_rule', {}).get('C', 0.)
                    D[node['idx'], pre_idx] = syn.get('learning_rule', {}).get('D', 0.)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        # self.learning_rate = np.random.randn(*self.A.shape) * 1e-3    


    def reset(self):
        self.t = 0
        self.reward_queue = deque([])
        self.activities_queue = deque([])
        self.inputs_queue = deque([])

    #! OJO refactorizar queries!!!!
    @GET("learning_rule:params")
    def get_params(self, conn_name, ann_graph, min_val=0., max_val=1., only_trainable=True):
        #* Return scaled in [0,1]
        if conn_name == 'all':
            params = np.hstack([np.array([syn['learning_rule'][param] for syn in ann_graph['synapses'].values()\
                    if syn['trainable']]) for param in ['A', 'B', 'C', 'D']])
            return (params - min_val) / (max_val - min_val)
        #* Special queries of synapses
        conn_name = {
            'sensory' : [key for key, syn in ann_graph['synapses'].items()\
                            if syn['pre'] in ann_graph['inputs']],
            'hidden' : [key for key, syn in ann_graph['synapses'].items()\
                        if syn['pre'] in ann_graph['neurons']\
                        and not ann_graph['neurons'][syn['pre']]['is_motor']],
            'motor' : [key for key, syn in ann_graph['synapses'].items()\
                        if syn['pre'] in ann_graph['neurons']\
                        and ann_graph['neurons'][syn['pre']]['is_motor']]
        }.get(conn_name, [conn_name])
        weights = np.hstack([np.array([ann_graph['synapses'][name]['learning_rule'][param] for name in conn_name\
                    if not only_trainable or ann_graph['synapses'][name]['trainable']]) for param in ['A', 'B', 'C', 'D']])
        return (weights - min_val) / (max_val - min_val)
    

    @SET("learning_rule:params")
    def set_params(self, conn_name, ann_graph, data, min_val=0., max_val=1.,):
        """
        """
        #* rescale genotype segment to weight range
        data = min_val + data * (max_val - min_val)
        if conn_name == 'all':
            for i, param in enumerate(['A', 'B', 'C', 'D']):
                param_data = data[i * len(data) // 4 : (i + 1) * len(data) // 4]
                for val, syn in zip(param_data, filter(lambda x: x['trainable'], ann_graph['synapses'].values())):
                    syn['learning_rule'][param] = val
            return ann_graph
        else:
            raise NotImplementedError #!!!
   
    @INIT("learning_rule:params")
    def init_params(self, conn_name, ann_graph, min_val=0., max_val=1., only_trainable=True):
        """
        """
        params_len = self.len_params(conn_name, ann_graph)
        random_params = 0.5 + np.random.randn(params_len) * 0.2
        random_params = np.clip(random_params, a_min=0, a_max=1)
        return self.set_params(conn_name, ann_graph, random_params, min_val=min_val, max_val=max_val)

    @LEN("learning_rule:params")
    def len_params(self, conn_name, ann_graph, only_trainable=True):
        """
        """
        return self.get_params(conn_name, ann_graph, only_trainable=True).shape[0]




class BufferedHebb(GeneralizedHebbian):
    def __init__(self, *args, **kwargs):
        super(BufferedHebb, self).__init__(*args, **kwargs)
        self.buffer_len = 30
        self.gamma = 0.9
        self.buffer = {'in' : deque([]), 'activ': deque([]), 'R': deque([])}
        self.t = 0

    def step(self, inputs, activities, reward=None):
        weight_update = 0.0
        if self.t >= self.buffer_len:
            cumm_reward = np.sum([rew*self.gamma**k for k, rew in enumerate(self.buffer['R'])])
            weight_update = super().step(self.buffer['in'][0].copy(), self.buffer['activ'][0].copy(), reward=cumm_reward)
        self.buffer['in'].append(inputs)
        self.buffer['activ'].append(activities)
        self.buffer['R'].append(reward)
        if self.t >= self.buffer_len:
            self.buffer['in'].popleft()
            self.buffer['activ'].popleft()
            self.buffer['R'].popleft()
        self.t += 1
        return weight_update

    def reset(self):
        self.t = 0
        self.buffer = {'in' : deque([]), 'activ': deque([]), 'R': deque([])}

# class HebbA2C(GeneralizedABCDHebbian):
#     def __init__(self, *args, **kwargs):
#         super(HebbA2C, self).__init__(*args, **kwargs)
#         self.critic_lr = 1e-3
#         self.gamma = 0.9
#         self.prev_state = None
#         self.prev_action = None
#         self.critic_weights = None

    
#     def step(self, outputs, state, actions, reward=None):
#         #* Compute value estimation
#         st_ac_vec = np.r_[state, actions]
#         estim_value = self.critic_weights.dot(st_ac_vec)
#         self.super().step(reward=estim_value)

#         # #* Update weights of critic
#         # td_err = reward + self.gamma * estim_value
#         # self.critic_weights += self.critic_lr * 

#     def reset(self):
#         pass
