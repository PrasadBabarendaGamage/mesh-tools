"""
This example demonstrates how to convert an abaqus .inp file to morphic using
a truncated octahedron mesh.
"""

import os
import numpy as np
import mesh_tools

if __name__ == "__main__":

    abaqus_mesh_path = './abaqus_mesh/'
    abaqus_mesh_filename = 'mesh.inp'
    processed_mesh_path = './output/'
    processed_mesh_expected_output_path = './expected_output/'
    if not os.path.exists(processed_mesh_path):
        os.makedirs(processed_mesh_path)

    # Read abaqus mesh as a morphic mesh
    morphic_mesh = mesh_tools.abaqus_to_morphic(
        abaqus_mesh_path, abaqus_mesh_filename, debug=True)

    surface_mesh = mesh_tools.renumber_mesh(morphic_mesh)

    # Save mesh as a morphic mesh
    surface_mesh.save(os.path.join(processed_mesh_path, 'surface.mesh'))

    # Save mesh in ex format for visualisation in cmgui. To visualise the mesh,
    # the surfaceCMISS.com file needs using cm to generate the exnode and
    # exelem files used by cmgui
    mesh_tools.morphic_to_openfemlite(
        surface_mesh, export_dir=processed_mesh_path,
        export_name='surface', mesh_type='surface')
    # Run cm to export mesh in exfile
    cwd = os.getcwd()
    os.chdir(processed_mesh_path)
    os.system('export LD_LIBRARY_PATH="/home/psam012/usr/libXp/lib'
              ':$LD_LIBRARY_PATH"; /home/psam012/usr/cm/cm ' +
              'surfaceCMISS.com')
    os.chdir(cwd)