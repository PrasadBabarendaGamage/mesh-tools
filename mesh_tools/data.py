import h5py
import numpy as np
import json

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