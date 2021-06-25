import numpy as np

def geom_mean(v):
    """Geometrical mean of the elements of a numpy array."""
    return np.prod(v) ** (1/v.shape[0])

def angle_mean(angles):
    """ Compute average angle of a vector of angles in radians.""" 
    return np.angle(np.sum([np.exp(1j * ang) for ang in angles]))


def compute_angle(u, v=None):
    '''
    Computes the angle between vectors u and v.
    If v is None, computes the angle of u wrt y=0
    '''
    # if v is None: v = np.array([1, 0])
    if v is None:
        if all(u == 0):
            return 0
        ang = np.arccos(u[0] / np.linalg.norm(u))
        return ang if u[1] >= 0 else -ang
    else:
        if u.sum() == 0 or v.sum() == 0 or u == v:
            return 0
        cos_theta = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
        theta = np.arccos(np.clip(cos_theta, a_min=-1, a_max=1))
        # if(u[0]*v[1] - u[1]*v[0] < 0): theta *= -1
        return theta


def angle_diff(x, y):
    """ Compute the difference between two angles in radians."""
    # abs_diff = np.abs(x - y)
    # return min(abs_diff, 2 * np.pi - abs_diff)
    return min((x - y) % (2 * np.pi), (y - x) % (2 * np.pi))

def normalize(v):
    """ Normalize a numpy array.""" 
    return v / np.linalg.norm(v)

def eigendecomposition(C):
    """ Eigendecomposition of matrix C. """ 
    eigenvals, B = np.linalg.eig(C) 
    D = np.diag(eigenvals)
    return B, D, B.T


def toroidal_difference(v, u):
    v_diff = v - u
    abs_diff = np.abs(v_diff)
    res = v_diff.copy()
    res[abs_diff > 500] = 1000 - abs_diff[abs_diff > 500]
    res[abs_diff > 500] *= -np.sign(v_diff[abs_diff > 500])
    # (-1, 1)[v_diff[abs_diff > 500] > 0] # Correct sign
    # import pdb; pdb.set_trace()
    return res

#! PASAR A DISTANCES
def circle_distance(alpha, beta):
    #! Check that |alpha - beta| <= 2pi
    return min(np.abs(alpha - beta), 2*np.pi - np.abs(alpha - beta))