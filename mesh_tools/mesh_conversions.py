import numpy as np
import mesh_tools
import morphic

def exfile_to_morphic(nodeFilename, elementFilename, coordinateField,
                      dimension=2, interpolation='linear'):
    """Convert an exnode and exelem files to a morphic mesh.

    Only Linear lagrange elements supported.

    Keyword arguments:
    nodeFilename -- exnode filename
    elementFilename -- exelem filename
    coordinateField -- the field to read in
    dimension -- dimension of mesh to read in
    """

    # Create morphic mesh
    mesh = morphic.Mesh()

    # Load exfiles
    exnode = mesh_tools.Exnode(nodeFilename)
    exelem = mesh_tools.Exelem(elementFilename, dimension)

    # Add nodes
    if interpolation == 'hermite':
        derivatives = range(1,9)
    else:
        derivatives = [1]
    for node_num in exnode.nodeids:
        coordinates = []
        for component in range(1, 4):
            component_name = ["x", "y", "z"][component - 1]
            componentValues = []
            for derivative_idx, derivative in enumerate(derivatives):
                componentValues.append(exnode.node_value(coordinateField,
                                                     component_name, node_num,
                                                     derivative))
            coordinates.append(componentValues)

        mesh.add_stdnode(node_num, coordinates, group='_default')
        #print('Morphic node added', node_num, coordinates)

    if dimension == 2:
        if interpolation == 'linear':
            element_interpolation = ['L1', 'L1']
        if interpolation == 'quadratic':
            element_interpolation = ['L2', 'L2']
    elif dimension == 3:
        if interpolation == 'linear':
            element_interpolation = ['L1', 'L1', 'L1']
        if interpolation == 'quadratic':
            element_interpolation = ['L2', 'L2', 'L2']
        if interpolation == 'cubic':
            element_interpolation = ['L3', 'L3', 'L3']
        if interpolation == 'hermite':
            element_interpolation = ['H3', 'H3', 'H3']

    # Add elements
    for elem in exelem.elements:
        mesh.add_element(elem.number, element_interpolation, elem.nodes)
        #print('Morphic element added', elem.number)

    # Generate the mesh
    mesh.generate(True)

    return mesh

def json_to_morphic(json):
    """Convert an exnode and exelem files to a morphic mesh.

    Keyword arguments:
    nodeFilename -- mesh data in json format (see morphic.export() for more
        info).
    """

    # Create morphic mesh.
    mesh = morphic.Mesh()

    # Add nodes.
    for node_id, value in json['nodes'].items():
        mesh.add_stdnode(node_id, value, group='_default')

    # Add elements.
    for element_id, element in json['elements'].items():
        mesh.add_element(element_id, element['basis'], element['nodes'])

    # Generate the mesh.
    mesh.generate(True)

    return mesh

def exfile_to_OpenCMISS(nodeFilename, elementFilename, coordinateField, basis,
                        region, meshUserNumber, dimension=2,
                        interpolation='linear', pressure_basis=None,
                        use_pressure_basis=False, elements=[]):
    """Convert an exnode and exelem files to a morphic mesh.

    Only Linear lagrange elements supported.

    Keyword arguments:
    nodeFilename -- exnode filename
    elementFilename -- exelem filename
    coordinateField -- the field to read in
    dimension -- dimension of mesh to read in
    interpolation -- the interpolation of the mesh to read in
    """
    from opencmiss.iron import iron

    # Load exfiles
    exnode = mesh_tools.Exnode(nodeFilename)
    exelem = mesh_tools.Exelem(elementFilename, dimension)

    if elements == []:
        ex_elems = exelem.elements
    else:
        ex_elems = []
        elements = exelem.elements
        for elem in exelem.elements:
            if elem.number in elements:
                ex_elems.append(elem)

    totalNumberOfNodes = len(exnode.nodeids)
    totalNumberOfElements = len(ex_elems)

    # Start the creation of a manually generated mesh in the region
    mesh = iron.Mesh()
    mesh.CreateStart(meshUserNumber, region, dimension)
    mesh.NumberOfComponentsSet(1)
    mesh.NumberOfElementsSet(totalNumberOfElements)

    # Define nodes for the mesh
    nodes = iron.Nodes()
    nodes.CreateStart(region, totalNumberOfNodes)
    nodes.UserNumbersAllSet(exnode.nodeids)
    nodes.CreateFinish()

    MESH_COMPONENT1 = 1
    MESH_COMPONENT2 = 2

    if (use_pressure_basis):
        mesh.NumberOfComponentsSet(2)
    else:
        mesh.NumberOfComponentsSet(1)

    elements = iron.MeshElements()
    elements.CreateStart(mesh, MESH_COMPONENT1, basis)
    elemNums = []
    for elem in ex_elems:
        elemNums.append(elem.number)

    elements.UserNumbersAllSet(elemNums)
    for elem_idx, elem in enumerate(ex_elems):
        elements.NodesSet(elem_idx+1, elem.nodes)
    elements.CreateFinish()

    if (use_pressure_basis):
        linear_elem_node_idxs = [0, 3, 12, 15, 48, 51, 60, 63]
        pressure_elements = iron.MeshElements()
        pressure_elements.CreateStart(mesh, MESH_COMPONENT2, pressure_basis)
        pressure_elements.UserNumbersAllSet(elemNums)
        for elem_idx, elem in enumerate(ex_elems):
            pressure_elements.NodesSet(elem_idx+1, np.array(
                    elem.nodes, dtype=np.int32)[linear_elem_node_idxs])
        pressure_elements.CreateFinish()

    mesh.CreateFinish()

    coordinates, node_ids = mesh_tools.extract_exfile_coordinates(nodeFilename, coordinateField, interpolation)

    return mesh, coordinates, node_ids


def morphic_to_OpenCMISS(morphicMesh, region, basis, meshUserNumber,
                         dimension=2, interpolation='linear',
                         UsePressureBasis=False, pressureBasis=None,
                         include_derivatives=True):
    """Convert an exnode and exelem files to a morphic mesh.

    Only Linear lagrange elements supported.

    Keyword arguments:
    morphicMesh -- morphic mesh
    dimension -- dimension of mesh to read in
    include_derivatives -- whether to include derivatives when returning mesh
                           coordinates.
    """
    from opencmiss.iron import iron

    # Create mesh topology
    mesh = iron.Mesh()
    mesh.CreateStart(meshUserNumber, region, 3)
    if (UsePressureBasis):
        mesh.NumberOfComponentsSet(2)
    else:
        mesh.NumberOfComponentsSet(1)

    node_list = morphicMesh.get_node_ids()[1]
    if len(node_list) == 0:
        node_list = morphicMesh.get_node_ids(group = '_default')[1]
    element_list = morphicMesh.get_element_ids()

    mesh.NumberOfElementsSet(len(element_list))
    nodes = iron.Nodes()
    nodes.CreateStart(region, len(node_list))
    nodes.UserNumbersAllSet((np.array(node_list)).astype('int32'))
    nodes.CreateFinish()

    MESH_COMPONENT1 = 1
    MESH_COMPONENT2 = 2
    elements = iron.MeshElements()
    elements.CreateStart(mesh, MESH_COMPONENT1, basis)
    elements.UserNumbersAllSet((np.array(element_list).astype('int32')))
    global_element_idx = 0
    for element_idx, element in enumerate(morphicMesh.elements):
        global_element_idx += 1
        elements.NodesSet(global_element_idx, np.array(element.node_ids, dtype='int32'))
    elements.CreateFinish()

    if (UsePressureBasis):
        pressure_elements = iron.MeshElements()
        pressure_elements.CreateStart(mesh, MESH_COMPONENT2, pressureBasis)
        pressure_elements.AllUserNumbersSet((np.array(element_list).astype('int32')))
        for element_idx, element in enumerate(morphicMesh.elements):
            pressure_elements.NodesSet(element.id, np.array(element.node_ids, dtype='int32'))
        pressure_elements.CreateFinish()

    mesh.CreateFinish()

    # Add nodes
    if interpolation == 'linear' or interpolation == 'cubicLagrange':
        derivatives = [1]
    elif interpolation == 'hermite':
        derivatives = range(1,9)

    if include_derivatives:
        coordinates = np.zeros((len(node_list), 3,len(derivatives)))
    else:
        derivatives = [1]
        coordinates = np.zeros((len(node_list), 3))
    for node_idx,morphic_node in enumerate(morphicMesh.nodes):
        for component_idx in range(3):
            for derivative_idx, derivative in enumerate(derivatives):
                if include_derivatives:
                    coordinates[node_idx,component_idx, derivative_idx] = \
                        morphic_node.values[component_idx]
                else:
                    coordinates[node_idx,component_idx] = \
                        morphic_node.values[component_idx]

    return mesh, coordinates, node_list, element_list


def OpenCMISS_to_morphic(
        c_mesh, geometric_field, element_nums, node_nums, dimension=2,
        interpolation='linear'):
    """Convert an OpenCMISS mesh to a morphic mesh.

    Only Linear lagrange elements supported.

    Keyword arguments:
    morphicMesh -- morphic mesh
    dimension -- dimension of mesh to read in
    """

    from opencmiss.iron import iron
    mesh_nodes = iron.MeshNodes()
    mesh_elements = iron.MeshElements()
    c_mesh.NodesGet(1, mesh_nodes)
    c_mesh.ElementsGet(1, mesh_elements)
    # Create morphic mesh
    mesh = morphic.Mesh()

    # Add nodes
    if interpolation in ['linear', 'quadratic', 'cubic']:
        derivatives = [1]
    elif interpolation == 'hermite':
        derivatives = range(1,9)
    for node_num in node_nums:
        coordinates = []
        for c_idx, c in enumerate([1, 2, 3]): # Component
            componentValues = []
            for derivative_idx, derivative in enumerate(derivatives):
                componentValues.append(
                    geometric_field.ParameterSetGetNodeDP(
                            iron.FieldVariableTypes.U,
                            iron.FieldParameterSetTypes.VALUES, 1, derivative,
                            int(node_num), c))
            coordinates.append(componentValues)

        mesh.add_stdnode(node_num, coordinates, group='_default')
        #print('Morphic node added', node_num, coordinates)

    if dimension == 2:
        if interpolation == 'linear':
            element_interpolation = ['L1', 'L1']
            num_elem_nodes = 4
        if interpolation == 'quadratic':
            element_interpolation = ['L2', 'L2']
            num_elem_nodes = 16
    elif dimension == 3:
        if interpolation == 'linear':
            element_interpolation = ['L1', 'L1', 'L1']
            num_elem_nodes = 8
        if interpolation == 'quadratic':
            element_interpolation = ['L2', 'L2', 'L2']
            num_elem_nodes = 27
        if interpolation == 'cubic':
            element_interpolation = ['L3', 'L3', 'L3']
            num_elem_nodes = 64
        if interpolation == 'hermite':
            element_interpolation = ['H3', 'H3', 'H3']
            num_elem_nodes = 8

    # Add elements
    for elem in element_nums:
        elem_nodes = mesh_elements.NodesGet(int(elem), num_elem_nodes)
        mesh.add_element(elem, element_interpolation, elem_nodes)
        #print('Morphic element added', elem.number)

    # Generate the mesh
    mesh.generate(True)

    return mesh