import h5py
import numpy as np
import json
from scipy.spatial.distance import cdist

def get_distributed_inds(points, threshold):
    """
    This function loads in the data points and distribute the data points more equally according to the coordinates of
    the points. It returns a list with the indices of the nodes that need to be kept.

    :param points: an mxn np.array of points with number m and dimension n.
    :param threshold: the distance threshold between the points, in the same unit as the points.

    Contributed by Robin Laven
    """
    dist_matrix = cdist(points, points)
    available_mask = np.ones(points.shape[0], dtype=bool)
    output_points = []
    while np.any(available_mask):
        point_id = np.argmax(available_mask)
        available_mask[point_id] = False
        output_points.append(point_id)

        dists = np.where(dist_matrix[point_id,:] < threshold)
        for d in dists:
            available_mask[d] = False
    return np.array(output_points).astype(int)

def export_h5_dataset(export_fname, label, data):

    data_file = h5py.File(export_fname, 'a')
    if isinstance(data, str):
        data_file.create_dataset(label, data=np.string_(data))
    else:
        data_file.create_dataset(label, data=data)
    data_file.close()

def import_h5_dataset(import_fname, label):
    data_file = h5py.File(import_fname, 'r')
    data = data_file[label][...]
    if data.dtype.char == 'S':
        data = str(np.char.decode(data))
    data_file.close()

    return data

def export_json(array):
    def default(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        raise TypeError('Unknown type:', type(obj))

    return json.dumps(array, default=default)