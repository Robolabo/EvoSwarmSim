import alphashape
import numpy as np

def get_alphashape(points, alpha=0.1):
    alpha_shape = alphashape.alphashape(points, alpha=alpha)
    border = []
    if type(alpha_shape).__name__ in ['Point', 'LineString']:
        border = points.copy()
    elif type(alpha_shape).__name__ == 'Polygon':
        border = np.array(alpha_shape.exterior.coords)
    elif type(alpha_shape).__name__ == 'MultiPolygon':
        border = np.array([point for polygon in alpha_shape for point in polygon.exterior.coords[:-1]])
    # pdb.set_trace()
    border_indices = [np.where(np.all(points == v, axis=1))[0][0] for v in border]
    return (border_indices, border)