"""
This example demonstrates how to convert a morphic mesh to meshio which can
then be used to export meshes in a range of different formats.
"""

import os
import morphic
import mesh_tools

if __name__ == "__main__":
    morphic_mesh_path = './morphic_mesh/surface.mesh'
    results_folder = './output/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Read morphic mesh
    morphic_mesh = morphic.Mesh(morphic_mesh_path)

    # Convert morphic mesh to meshio
    meshio_mesh = mesh_tools.morphic_to_meshio(morphic_mesh)

    # Export to other mesh formats.
    meshio_mesh.write(os.path.join(results_folder, "mesh.vtk"))
    meshio_mesh.write(os.path.join(results_folder, "mesh.inp"))

