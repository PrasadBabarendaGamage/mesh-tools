import mesh_tools
import numpy as np
import os

if __name__ == '__main__':
    input_mesh = '../../meshes/exfile/trilinear_cube.exfile'
    output_directory = "../../output/"
    output_filename = "test.vtk"
    dimension = 3
    xi_locations = np.array([[0.5, 0.5, 0.5]])
    elements = [1]

    mesh = mesh_tools.Zinc_mesh(input_mesh, dimension)
    mesh.evaluate(xi_locations, elements)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    mesh.export_vtk(f"{output_directory}{output_filename}")
    mesh.find_standard_elements()
