
import numpy as np
import networkx as nx
from .activations import tanh
from .alg_utils import compute_angle

def convert2graph(points, max_dist=75):
    import matplotlib.pyplot as plt
    G = nx.Graph()
    G.add_nodes_from(range(points.shape[0]))
    for i, p in enumerate(points):
        for j, q in enumerate(points): 
            if j != i and np.linalg.norm(p - q) < max_dist:
                G.add_edge(i,j)
    return G

def disjoint_subgraphs(G):
    subgraph_nodes = tuple(nx.connected_components(G))
    return [[i for i in nodes] for nodes in subgraph_nodes]

# def random_spatial_graph(num_points, initial_pos=(500, 500)):
#     points = [np.array(initial_pos)]
#     prob_decay = [0.1, 50] #.75
#     for _ in range(num_points-1):
#         new_pos = points[-1]
#         # while not all([np.linalg.norm(new_pos - pos) > 20 for pos in points])\
#         #     or not np.min([np.linalg.norm(new_pos - pos) for pos in points]) < 100:
#         while not new_pos == points[-1]:
#             # prob of going +1 direction in x dim
#             px = 0.5*np.exp(-prob_decay[0] * ((points[-1][0] - 500) / 100) ** 2)
#             px = px if points[-1][0] >= 500 else 1 - px
#             # prob of going +1 direction in y dim
#             py = 0.5*np.exp(-prob_decay[1] * ((points[-1][1] - 500) / 100) ** 2)
#             py = py if points[-1][1] >= 500 else 1 - py
#             # py = (1.-.5*np.exp(-.75 * (points[-1][1] / 100 - 5) ** 2),\
#             #     .5 * np.exp(-.75 * (points[-1][1] / 100 - 5) ** 2))[points[-1][1] >= 500]
#             # sampled unitary direction
#             new_dir = np.array([np.random.choice([-1, 1], p=(1-p, p)) for p in [px, py]])
#             new_pos = points[-1] + np.random.uniform(20, 80, size=2) * new_dir
#         points.append(new_pos)
#     return points 


def random_spatial_graph(num_points, initial_pos=(500, 500)):
    points = [np.array(initial_pos)]
    prob_decay = [0.1, 50] #.75
    R_max = 300
    import pdb; pdb.set_trace()
    for _ in range(num_points-1):
        new_pos = points[-1]
        # while not all([np.linalg.norm(new_pos - pos) > 20 for pos in points])\
        #     or not np.min([np.linalg.norm(new_pos - pos) for pos in points]) < 100:
        delta_X = np.abs(points[-1] - 500)
        mu = tanh(-2*(delta_X / R_max) ** 3)
        sigma_x = 0.1*(delta_X[1] / R_max + 1e-3)
        sigma_y = delta_X[0] / R_max + 1e-3
        rho = np.sin(2 * compute_angle(points[-1]))
        cov_mat = np.array([[sigma_x, rho*sigma_x*sigma_y], [rho*sigma_x*sigma_y, sigma_y]])
        new_pos = 50 * np.random.multivariate_normal(mu, cov_mat, size=2)
        import pdb; pdb.set_trace()
        points.append(new_pos)
    return points 

def grid_spatial_graph(num_points, initial_pos=(500, 500)):
    points = []
    for i in np.arange(initial_pos[0]-2*45, initial_pos[0]+45, 45):
        for j in np.arange(initial_pos[1]-2*45, initial_pos[1]+45, 45):
            points.append(np.array([i,j])+np.random.randn(2)*3)
    return points