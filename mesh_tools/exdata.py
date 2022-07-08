import os
import numpy as np

def export_datapoints_csv(data, filename):
    # Shape of data should be a [num_datapoints,dim] numpy array.
    field_id = filename + '.csv'
    np.savetxt(field_id, data, delimiter=",")

def export_datapoints_ipdata(data, label, filename):
    # Shape of data should be a [num_datapoints,dim] numpy array.
    field_id = open(filename + '.ipdata', 'w')
    field_id.write('{0}\n'.format(label))
    for point_idx, point in enumerate(range(1, data.shape[0] + 1)):
        if ~np.isnan(data[point_idx, :]).any():
            field_id.write('{0} '.format(point_idx+1))
            for value_idx in range(data.shape[1]):
                field_id.write(' {0:.12E} '.format(data[point_idx, value_idx]))
            for value_idx in range(data.shape[1]):
                field_id.write('1.0 ')
            field_id.write('\n')
    field_id.close()

def export_datapoints_exdata(data, label, filename):
    # Shape of data should be a [num_datapoints,dim] numpy array.

    output_folder = os.path.dirname(filename)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    field_id = open(filename + '.exdata', 'w')
    field_id.write(' Group name: {0}\n'.format(label))
    field_id.write(' #Fields=1\n')
    field_id.write(
        ' 1) coordinates, coordinate, rectangular cartesian, #Components=3\n')
    field_id.write('  x.  Value index=1, #Derivatives=0, #Versions=1\n')
    field_id.write('  y.  Value index=2, #Derivatives=0, #Versions=1\n')
    field_id.write('  z.  Value index=3, #Derivatives=0, #Versions=1\n')

    for point_idx, point in enumerate(range(1, data.shape[0] + 1)):
        if ~np.isnan(data[point_idx, :]).any():
            field_id.write(' Node: {0}\n'.format(point))
            for value_idx in range(data.shape[1]):
                field_id.write(' {0:.12E}\n'.format(data[point_idx, value_idx]))
    field_id.close()


def exportDatapointsErrorExdata(data, error, label, filename):
    # Shape of data should be a [num_datapoints,dim] numpy array.
    field_id = open(filename + '.exdata', 'w')
    field_id.write(' Group name: {0}\n'.format(label))
    field_id.write(' #Fields=3\n')
    field_id.write(
        ' 1) coordinates, coordinate, rectangular cartesian, #Components=3\n')
    field_id.write('   x.  Value index= 1, #Derivatives=0\n')
    field_id.write('   y.  Value index= 2, #Derivatives=0\n')
    field_id.write('   z.  Value index= 3, #Derivatives=0\n')
    field_id.write(
        ' 2) error, field, rectangular cartesian, #Components=3\n')
    field_id.write('   x.  Value index= 4, #Derivatives=0\n')
    field_id.write('   y.  Value index= 5, #Derivatives=0\n')
    field_id.write('   z.  Value index= 6, #Derivatives=0\n')
    field_id.write(
        ' 3) scale, field, rectangular cartesian, #Components=3\n')
    field_id.write('   x.  Value index= 7, #Derivatives=0\n')
    field_id.write('   y.  Value index= 8, #Derivatives=0\n')
    field_id.write('   z.  Value index= 9, #Derivatives=0\n')

    for point_idx, point in enumerate(range(1, data.shape[0] + 1)):
        if ~np.isnan(data[point_idx, :]).any():
            field_id.write(' Node: {0}\n'.format(point))
            for value_idx in range(data.shape[1]):
                field_id.write(
                    ' {0:.12E}\n'.format(data[point_idx, value_idx]))
            for value_idx in range(data.shape[1]):
                field_id.write(
                    ' {0:.12E}\n'.format(error[point_idx, value_idx]))
            # Scale field is absolute of the error field to ensure vector
            # direction does not change.
            for value_idx in range(data.shape[1]):
                field_id.write(
                    ' {0:.12E}\n'.format(abs(error[point_idx, value_idx])))
    field_id.close()

def export_data_points_exdata(
        positions, label, filename, data=None, component_labels=None):
    # Shape of data should be a [num_datapoints,dim] numpy array.
    num_fields = 1
    num_points = positions.shape[0]
    num_geometric_components = positions.shape[1]
    if data is not None:
        num_fields = 2
        num_data_components = data.shape[1]
        if component_labels is None:
            # Assign component labels
            component_labels = list(range(1, num_data_components + 1))

    # Write header
    field_id = open(filename + '.exdata', 'w')
    field_id.write(' Group name: {0}\n'.format(label))
    field_id.write(' #Fields={0}\n'.format(num_fields))
    field_id.write(
        ' 1) coordinates, coordinate, rectangular cartesian, #Components=3\n')
    field_id.write('   x.  Value index= 1, #Derivatives=0\n')
    field_id.write('   y.  Value index= 2, #Derivatives=0\n')
    field_id.write('   z.  Value index= 3, #Derivatives=0\n')
    if data is not None:
        field_id.write(
            ' 2) error, field, rectangular cartesian, #Components={0}\n'.format(
                num_data_components))
        for data_component_idx, component_label in enumerate(component_labels):
            field_id.write('   {0}.  Value index= {1}, #Derivatives=0\n'.format(
                component_label, data_component_idx + 4))

    # Write field values
    for point_idx, point in enumerate(range(1, num_points + 1)):
        if ~np.isnan(positions[point_idx, :]).any():
            field_id.write(' Node: {0}\n'.format(point))
            # Write geometric components
            for value_idx in range(num_geometric_components):
                field_id.write(
                    ' {0:.12E}\n'.format(positions[point_idx, value_idx]))
            # Write data components
            if data is not None:
                for value_idx in range(num_data_components):
                    field_id.write(
                        ' {0:.12E}\n'.format(data[point_idx, value_idx]))
    field_id.close()
