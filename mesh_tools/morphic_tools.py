import os
import numpy as np
import mesh_tools
from morphic.utils import export_json
import morphic


def load_morphic_mesh(path, label):
    """Loads a morphic mesh and sets the specified label.
    """
    if os.path.exists(path):
        mesh = morphic.Mesh(path)
        mesh.label = label
    else:
        message = 'Morphic mesh not found'
        print(message)
    return mesh

def generate_points_morphic_face(
        mesh, face, value, num_points=[4, 4], element_ids=[], dim=3):
    """
    Generate a grid of points on faces of selected morphic mesh elements

    Keyword arguments:
    mesh -- mesh to evaluate points in
    face -- face to evaluate points on at the specified xi value
    dim -- the number of xi directions
    """

    xi = mesh_tools.generate_xi_on_face(
        face, value, num_points=num_points, dim=dim)

    if not element_ids:
        # if elem group is empty evaluate all elements
        element_ids = mesh.elements.ids

    num_ne = len(element_ids)
    ne_num_points = np.prod(num_points)
    total_num_points = num_ne * ne_num_points
    points = np.zeros((num_ne, ne_num_points, dim))
    all_xi = np.zeros((num_ne, ne_num_points, dim))
    all_ne = np.zeros((num_ne, ne_num_points))

    for idx, element_id in enumerate(element_ids):
        element = mesh.elements[element_id]
        points[idx, :, :] = element.evaluate(xi)
        all_xi[idx, :, :] = xi
        all_ne[idx, :] = element_id

    points = np.reshape(points, (total_num_points, dim))
    all_xi = np.reshape(all_xi, (total_num_points, dim))
    all_ne = np.reshape(all_ne, (total_num_points))

    return points, all_xi, all_ne

def generate_points_morphic_elements(
        mesh, num_points=[4, 4, 4], element_ids=[], dim=3):
    """
    Generate a grid of points within selected morphic mesh elements

    Keyword arguments:
    mesh -- mesh to evaluate points in
    dim -- the number of xi directions
    """

    xi = mesh_tools.generate_xi_grid_fem(num_points=num_points, dim=3)

    if not element_ids:
        # if elem group is empty evaluate all elements
        element_ids = mesh.elements.ids

    num_ne = len(element_ids)
    ne_num_points = np.prod(num_points)
    total_num_points = num_ne * ne_num_points
    points = np.zeros((num_ne, ne_num_points, dim))
    all_xi = np.zeros((num_ne, ne_num_points, dim))
    all_ne = np.zeros((num_ne, ne_num_points))

    for idx, element_id in enumerate(element_ids):
        element = mesh.elements[element_id]
        points[idx, :, :] = element.evaluate(xi)
        all_xi[idx, :, :] = xi
        all_ne[idx, :] = element_id

    points = np.reshape(points, (total_num_points, dim))
    all_xi = np.reshape(all_xi, (total_num_points, dim))
    all_ne = np.reshape(all_ne, (total_num_points))

    return points, all_xi, all_ne


def add_fig(viewer, label=''):
    if viewer is not None:
        fig = viewer.Figure(label)
    else:
        fig = None
    return fig

def visualise_mesh(
        mesh, fig, visualise=False, face_colours=(1, 0, 0), pt_size=5,
        label=None,
        text=False, element_ids=False, text_elements=None, opacity=0.5,
        line_opacity=1., elements=None, nodes=None, node_text=False, text_size=3,
        node_size=5, node_colours=(1, 0, 0), elements_to_display_nodes=None):
    if fig is not None:
        if label is None:
            label = mesh.label
        Xnid = mesh.get_node_ids(group='_default')

        if nodes == 'all':
            nodes = Xnid[1]  # [:-5]

        if visualise:
            # View breast surface mesh
            Xs, Ts = mesh.get_surfaces(res=16, elements=elements)
            if Xs.shape[0] == 0:
                Xs, Ts = mesh.get_faces(res=16, elements=elements)
            if elements is None:
                Xl = mesh.get_lines(res=32, internal_lines=False)
            else:
                Xl = mesh.get_lines(res=32, elements=elements,
                                    internal_lines=False)
            # import ipdb; ipdb.set_trace()
            fig.plot_surfaces('{0}_Faces'.format(label), Xs, Ts,
                              color=face_colours,
                              opacity=opacity)
            # fig.plot_points('{0}_Nodes'.format(label), Xn, color=(1,0,1), size=pt_size)
            fig.plot_lines('{0}_Lines'.format(label), Xl, color=(1, 1, 0),
                           size=5, opacity=line_opacity)
            if text_elements is None:
                if text:
                    fig.plot_text('{0}_Text'.format(label), Xnid[0], Xnid[1],
                                  size=text_size)
            if elements_to_display_nodes is not None:
                for element_id in elements_to_display_nodes:
                    element = mesh.elements[element_id]
                    # import ipdb; ipdb.set_trace()
                    eXnid = mesh.get_node_ids(element.node_ids)
                    fig.plot_text(
                        '{0}_text_element{1}'.format(label, element.id),
                        eXnid[0], eXnid[1], size=pt_size, color=(1, 0, 0))
                    fig.plot_points(
                        '{0}_Nodes_element{1}'.format(label, element.id),
                        eXnid[0], color=(1, 0, 1), size=pt_size / 2)
            if element_ids:
                fig.plot_element_ids('{0}_Xecid'.format(label), mesh, size=1,
                                     color=(1, 1, 1))
            if nodes is not None:
                # import ipdb; ipdb.set_trace()
                fig.plot_points(
                    '{0}_Points'.format(label), mesh.get_nodes(nodes),
                    color=node_colours, size=node_size)
                if node_text:
                    fig.plot_text(
                        '{0}_Text'.format(label), mesh.get_nodes(nodes), nodes,
                        size=3)


def mesh_element_centroids(mesh, element_nums='all'):
    """
    Returns the centroids of mesh elements.

    These can be used as coordinates for plotting element numbers.
    Centroids are calculated at the centre of each element.

    Keyword arguments:
    mesh -- morphic mesh
    element_nums -- element numbers to calculate centroids for.
    """

    if element_nums == 'all':
        element_nums = mesh.get_element_ids()

    element_centroids = []

    for element_idx, element in enumerate(mesh.elements):
        if element_nums == 'all' or element.id in element_nums:
            element_centroids.append(
                element.evaluate([0.5]*len(element.basis)).tolist())

    return element_nums, element_centroids

def mesh_subset(mesh, element_nums):
    """
    Returns a subset of elements.

    Keyword arguments:
    mesh -- morphic mesh to renumber
    element_nums -- node offset of the renumbered mesh
    """

    mesh_subset = morphic.Mesh()
    element_nodes = []

    for element_idx, element in enumerate(mesh.elements):
        if element.id in element_nums:
            element_nodes.append(element.node_ids)
    element_nodes = np.unique(np.array(element_nodes).flatten())

    for n_id in element_nodes:
        mesh_subset.add_stdnode(
            n_id, mesh.get_nodes(n_id, group=b'_default')[0], group='_default')

    for element_idx, e_id in enumerate(element_nums):
        element = mesh.elements[e_id]
        mesh_subset.add_element(
            element.id, element.basis, element.node_ids)

    mesh_subset.generate(True)
    mesh_subset.label = mesh.label

    return mesh_subset

def renumber_mesh(
        mesh, node_offset=0, element_offset=0, label='', debug=False):
    """
    Renumbers a mesh sequentially

    Keyword arguments:
    mesh -- morphic mesh to renumber
    node_offset -- node offset of the renumbered mesh
    element_offset -- element offset of the renumbered mesh
    label -- label of the renumbered mesh
    debug -- print debug output
    """
    # Create mesh
    import morphic
    renumbered_mesh = morphic.Mesh()

    node_ids = mesh.get_node_ids(group='_default')[1]
    mapping = range(len(node_ids))
    renumbered_node_ids = range(1, len(node_ids) + 1)

    # Renumber nodes
    for node_idx, morphic_node in enumerate(mesh.nodes):
        renumbered_node_id = renumbered_node_ids[node_idx] + node_offset
        if debug:
            print('Morhpic node {0} renumbered to {1}'.format(
                morphic_node.id, renumbered_node_id))
        renumbered_mesh.add_stdnode(
            renumbered_node_id, morphic_node.values, group='_default')

    for element_idx, element in enumerate(mesh.elements):
        # Renumber element nodes
        renumbered_element_node_ids = []
        for element_node_id in element.node_ids:
            renumbered_node_ids = node_ids.index(element_node_id) + 1
            renumbered_element_node_ids.append(
                renumbered_node_ids + node_offset)
        # Add renumbered elements to new mesh
        if debug:
            print('  Element nodes', renumbered_element_node_ids)
        renumbered_mesh.add_element(
            element_idx + 1 + element_offset, ['L1', 'L1'],
            renumbered_element_node_ids)

    renumbered_mesh.generate(True)
    renumbered_mesh.label = label

    # Copy metadata
    renumbered_mesh.metadata = mesh.metadata

    try:
        # Copy node groups
        for key, morphic_nodes in mesh.nodes.groups.iteritems():
            ids = [node_ids.index(node.id) + 1 + node_offset for node in
                   morphic_nodes]
            for node in renumbered_mesh.nodes[ids]:
                node.add_to_group(key)

        # Copy element groups
        for key, morphic_elements in mesh.elements.groups.iteritems():
            ids = [elem.id + element_offset for elem in morphic_elements]
            for elem in renumbered_mesh.elements[ids]:
                elem.add_to_group(key)
    except:
        pass

    return renumbered_mesh

def surface_mesh(vol, face_node_idxs=None, label='', translations=None):
    """
    Generates a surface mesh from a volume mesh

    Keyword arguments:
    vol -- morphic volume mesh
    face_node_idxs -- volume mesh face nodes idxs to include in surface mesh
    label -- label of the surface mesh
    """
    if translations is None:
        translations = np.array([0, 0, 0])

    # Create mesh
    import morphic
    surf = morphic.Mesh()

    nd_labels = []
    for element_idx, element in enumerate(vol.elements):
        nd_labels.append(np.array(element.node_ids)[face_node_idxs])
    nd_labels = np.unique(np.array(nd_labels).flatten())
    for nd in nd_labels:
        surf.add_stdnode(
            nd, vol.get_nodes(nd, group=b'_default')[0] - translations,
            group='_default')

    for element_idx, element in enumerate(vol.elements):
        surf_element_nodes = np.array(element.node_ids)[face_node_idxs]
        surf.add_element(
            element.id, ['L3', 'L3'], surf_element_nodes)

    surf.generate(True)
    surf.label = label

    return surf

def export_morphic_nodes_csv(dir, filename, meshes):
    for mesh_idx, mesh in enumerate(meshes):
        path = os.path.join(dir, filename + '_{0}.csv'.format(mesh_idx))
        np.savetxt(path, mesh.get_nodes(), delimiter=',')

def export_morphic_animation(dir, filename, meshes, element_ids='all',
                             precision='%0.6f'):
    fp = open(os.path.join(dir, filename), 'w')

    fp.write('{\n')
    fp.write(' "nodes": {\n')
    for mesh_idx, mesh in enumerate(meshes):
        node_ids, nodes_strs, elements_strs = mesh.export_json_strs(
            element_ids,
            precision)

        fp.write('    "' + str(mesh_idx) + '": {\n')
        fp.write('        "nodes": {\n')
        fp.write(',\n'.join(nodes_strs))
        fp.write('         \n\t}\n')
        if mesh_idx == len(meshes) - 1:
            seperator = ''
        else:
            seperator = ','
        fp.write('\n\t}' + seperator + '\n')
    fp.write('},\n')
    fp.write('"elements": {\n')
    fp.write(',\n'.join(elements_strs))
    fp.write('\n\t},\n')
    fp.write(' "nodesIds": \n')
    fp.write('{0}\n'.format(
        export_json(node_ids)))
    fp.write('}')
    fp.close()