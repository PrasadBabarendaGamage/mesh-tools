"""
This example demonstrates how to convert an abaqus .inp file to morphic using
a truncated octahedron mesh.
"""

import os
import numpy as np
import mesh_tools

if __name__ == "__main__":

    initial_mesh_path = './initial_mesh/'
    initial_mesh_filename = 'truncated_octahedron_3d_elements.inp'
    processed_mesh_path = './output/'
    processed_mesh_expected_output_path = './expected_output/'
    if not os.path.exists(processed_mesh_path):
        os.makedirs(processed_mesh_path)

    # Read abaqus mesh as a morphic mesh
    lin_vol_mesh_ref1 = mesh_tools.abaqus_to_morphic(
        initial_mesh_path, initial_mesh_filename, debug=True)

    elem_nodes = np.zeros((len(lin_vol_mesh_ref1.get_element_ids()), 8))
    for idx, element in enumerate(lin_vol_mesh_ref1.elements):
        elem_nodes[idx, :] = element.node_ids

    # Save nodes and elements for further processing in matlab
    filename = 'initial_nodes.txt'
    np.savetxt(os.path.join(processed_mesh_path, filename),
               lin_vol_mesh_ref1.get_nodes())

    filename = 'initial_elements.txt'
    np.savetxt(os.path.join(processed_mesh_path, filename), elem_nodes)

    # Rearrange element nodes to obtain a consistent set of outward pointing
    # surface normals
    # os.system('matlab -nodesktop -nosplash -r post_process_mesh')
    node_file = 'final_nodes.txt'
    element_file = 'final_elements.txt'

    # Note that node 8 is at the center of the truncated octahedron and is
    # therefore not part of the surface mesh and needs to be excluded.
    surface_mesh = mesh_tools.txt_to_morphic(
        processed_mesh_path, node_file, element_file, nodes_to_exclude=[8])
    # Renumber mesh sequentially
    surface_mesh = mesh_tools.renumber_mesh(surface_mesh)

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