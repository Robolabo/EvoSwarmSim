import numpy as np
import numpy.linalg as LA
from spike_swarm_sim.register import initializer_registry
from spike_swarm_sim.utils import tanh, compute_angle, isinstance_of_any


class InitializerHandler:
    def __init__(self, initializer, engine='3D', variable='positions'):
        self.initializer = initializer
        self.engine = engine
        self.variable = variable

    def __call__(self, *args, **kwargs):
        return {
        '2D' : {'orientations' : map(lambda x: x[0], self.initializer())},
        '3D' : {'positions' : map(lambda x: (x[0], x[1], 0), self.initializer()),
                'orientations' : map(lambda x: (0., 0., x[0]), self.initializer())}
        }.get(self.engine, {}).get(self.variable, self.initializer())

@initializer_registry(name='fixed')
class FixedInitializer:
    def __init__(self, num_points, fixed_values=None):
        self.num_points = num_points
        self.fixed_values = fixed_values
        if not isinstance_of_any(fixed_values[0], [list, np.ndarray]):
            self.fixed_values = [[val] for val in self.fixed_values]
        assert num_points == len(self.fixed_values)

    def __call__(self):
        return [np.array(val) for val in self.fixed_values]


@initializer_registry(name='fixed_random')
class FixedRandomInitializer:
    def __init__(self, num_points, possible_values=None, replacement=True):
        self.num_points = num_points
        self.replacement = replacement
        self.possible_values = possible_values
        if not isinstance_of_any(possible_values[0], [list, np.ndarray]):
            self.possible_values = [[val] for val in self.possible_values]
        assert self.replacement or len(self.possible_values) < self.num_points

    def __call__(self):
        sel_indices = np.random.choice(len(self.possible_values), size=self.num_points, replace=self.replacement)
        return [self.possible_values[idx] for idx in sel_indices]

@initializer_registry(name='random_uniform')
class RandomUniformInitializer:
    def __init__(self, num_points, low=[-2, -2], high=[2, 2], size=2):
        self.num_points = num_points
        self.low = low
        self.high = high
        self.size = size
        self.check_overlapping = size > 1

    def __call__(self):
        res = []
        if self.check_overlapping:
            while len(res) < self.num_points:
                # new_sample = np.random.uniform(low=self.low, high=self.high, size=self.size)
                if isinstance(self.low, int):
                    new_sample = np.random.uniform(low=self.low, high=self.high, size=self.size)
                else:
                    new_sample_x = np.random.uniform(low=self.low[0], high=self.high[0])
                    new_sample_y = np.random.uniform(low=self.low[1], high=self.high[1])
                    new_sample = np.r_[new_sample_x, new_sample_y]
                if len(res) == 0 or all(LA.norm(new_sample - pp) > 0.7 for pp in res):
                    res.append(new_sample)
        else:
            res = [np.random.uniform(low=self.low, high=self.high, size=self.size)\
                    for _ in range(self.num_points)]
        return res
        
@initializer_registry(name='random_circle')
class RandomCircleInitializer:
    def __init__(self, num_points, center=[0, 0], radius=1.):
        self.num_points = num_points
        self.center = np.array(center)
        self.radius = radius

    def __call__(self):
        res = []
        while len(res) < self.num_points:
            rnd_mod = np.random.uniform(1e-3, self.radius)
            rnd_phase = np.random.uniform(0, 2 * np.pi)
            new_sample = rnd_mod * np.r_[np.cos(rnd_phase), np.sin(rnd_phase)] + self.center
            if len(res) == 0 or all(LA.norm(new_sample - pp) > 0.6 for pp in res):
                res.append(new_sample)
        return res



@initializer_registry(name='random_circumference')
class RandomCircumference:
    """
    Class for randomly initializing objects embedded in a circumference.
    It uniformly samples a random angle and computes the cartesian position within 
    a circumference of a given radius and center.
    =================================
    - Args:
    =================================
    """
    def __init__(self, num_points, radius=130, center=[500, 500]):
        self.num_points = num_points
        self.radius = radius
        self.center = np.array(center)
    
    def __call__(self):
        theta_rnd = np.random.uniform(low=0, high=2*np.pi, size=self.num_points)
        # import pdb; pdb.set_trace()
        return self.radius * np.stack([np.cos(theta_rnd), np.sin(theta_rnd)]).T + self.center #+ np.random.randn(2) * 0



@initializer_registry(name='random_graph')
class RandomGraphInitializer:
    def __init__(self, num_points, max_rad=300, initial_pos=(500, 500)):
        self.num_points = num_points
        self.initial_pos = initial_pos
        self.max_rad = max_rad

    # def __call__(self):
    #     points = [np.array(self.initial_pos).astype(float)]
    #     for _ in range(self.num_points-1):
    #         new_pos = points[-1]
    #         while not all([np.linalg.norm(new_pos - pos) > 50 for pos in points])\
    #             or not np.min([np.linalg.norm(new_pos - pos) for pos in points]) < 100:
    #             # prob of going +1 direction in x dim
    #             px = np.exp(-.75 * ((points[-1][0] - 500) / 100) ** 2)
    #             px = px if points[-1][0] >= 500 else 1 - px
    #             # prob of going +1 direction in y dim
    #             py = np.exp(-.75 * ((points[-1][1] - 500) / 100) ** 2)
    #             py = py if points[-1][1] >= 500 else 1 - py
    #             # py = (1.-.5*np.exp(-.75 * (points[-1][1] / 100 - 5) ** 2),\
    #             #     .5 * np.exp(-.75 * (points[-1][1] / 100 - 5) ** 2))[points[-1][1] >= 500]
    #             # sampled unitary direction
    #             new_dir = np.array([np.random.choice([-1, 1], p=(1-p, p)) for p in [px, py]])
    #             new_pos = points[-1] + np.random.uniform(30, 100, size=2) * new_dir
    #         points.append(new_pos)
    #     return points
    #
    def __call__(self):
        # self.initial_pos = [700, 100]
        # self.num_points = 2
        points = [np.array(self.initial_pos).astype(float)]
        # prob_decay = [0.1, 50] #.75
        R_max = self.max_rad
        for _ in range(self.num_points-1):
            new_pos = points[-1]
            # while any([np.linalg.norm(new_pos - pos) < 40 for pos in points]) or np.min([np.linalg.norm(new_pos - pos) for pos in points]) > 99:
            while any([np.linalg.norm(new_pos - pos) < 0.4 for pos in points]) or np.min([np.linalg.norm(new_pos - pos) for pos in points]) > 1:
                delta_X = points[-1] #- 500
                mu = tanh(-(delta_X / R_max) ** 3)
                sigma_x = np.sin(compute_angle(delta_X / R_max)) ** 2 if np.linalg.norm(delta_X) > R_max/2 else 1
                sigma_y = np.sin(compute_angle(delta_X / R_max) + np.pi / 2) ** 2 if np.linalg.norm(delta_X) > R_max/2 else 1
                rho = 0.5 * np.sin(2 * compute_angle(delta_X / R_max)) ** 3
                cov_mat = np.array([[sigma_x, rho * sigma_x * sigma_y], [rho * sigma_x * sigma_y, sigma_y]])
                # new_pos = 80 * np.random.multivariate_normal(mu, cov_mat, size=1).flatten() + points[-1]
                new_pos = 0.8 * np.random.multivariate_normal(mu, cov_mat, size=1).flatten() + points[-1]
            points.append(new_pos)
        # for p in points:
        #     p[1] = 1000 - p[1]
        return points
     


@initializer_registry(name='grid_graph')
class GridInitializer:
    def __init__(self, num_points, center):
        pass
    
    def __call__(self):
        pass
