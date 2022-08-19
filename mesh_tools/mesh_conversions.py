import os
import numpy as np
import scipy
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


def morphic_to_OpenCMISS(
        morphicMesh, region, basis, meshUserNumber, interpolation='linear',
        create_start_callback=False, create_finish_callback=False,
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

    if create_start_callback:
        create_start_callback(mesh, morphicMesh)
    else:
        mesh.NumberOfComponentsSet(1)

    node_list = morphicMesh.get_node_ids()[1]
    # If node list is empty then the nodes are likely grouped under the
    # '_default' group.
    if len(node_list) == 0:
        node_list = morphicMesh.get_node_ids(group = '_default')[1]
    element_list = morphicMesh.get_element_ids()

    mesh.NumberOfElementsSet(len(element_list))
    nodes = iron.Nodes()
    nodes.CreateStart(region, len(node_list))
    nodes.UserNumbersAllSet((np.array(node_list)).astype('int32'))
    nodes.CreateFinish()

    MESH_COMPONENT1 = 1
    elements = iron.MeshElements()
    elements.CreateStart(mesh, MESH_COMPONENT1, basis)
    elements.UserNumbersAllSet((np.array(element_list).astype('int32')))
    global_element_idx = 0
    for element_idx, element in enumerate(morphicMesh.elements):
        global_element_idx += 1
        elements.NodesSet(global_element_idx, np.array(element.node_ids, dtype='int32'))
    elements.CreateFinish()

    if create_finish_callback:
        create_finish_callback(mesh, morphicMesh, element_list)

    mesh.CreateFinish()

    # Add nodes
    if interpolation == 'linear' or \
            interpolation == 'quadraticLagrange' or \
            interpolation == 'cubicLagrange':
        derivatives = [1]
    elif interpolation == 'hermite':
        derivatives = range(1, 9)

    if include_derivatives:
        coordinates = np.zeros((len(node_list), 3, len(derivatives)))
    else:
        derivatives = [1]
        coordinates = np.zeros((len(node_list), 3))
    for node_idx, morphic_node in enumerate(morphicMesh.nodes):
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
    element_nums -- global OpenCMISS element numbers.
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
    for idx, elem in enumerate(element_nums):
        global_element_number = idx + 1
        elem_nodes = mesh_elements.NodesGet(
            int(global_element_number), num_elem_nodes)
        mesh.add_element(elem, element_interpolation, elem_nodes)
        #print('Morphic element added', elem.number)

    # Generate the mesh
    mesh.generate(True)

    return mesh

def txt_to_morphic(mesh_dir, node_file, element_file, nodes_subset=[],
                   nodes_to_exclude=[], elem_subset=[]):
    """Convert text files containing vertices (nodes) and faces (elements) to a
     morphic mesh.

    Only Linear lagrange elements supported.

    Keyword arguments:
    nodes_subset -- nodes to load (all if empty)
    elem_subset -- elements to load (all if empty)
    """

    # Create mesh
    mesh = morphic.Mesh()

    # Load node .txt file
    nodes = scipy.loadtxt(
        os.path.join(mesh_dir, node_file), dtype='float', delimiter=',')
    for node_idx, coordinates, in enumerate(nodes):
        node_num = node_idx + 1
        if node_num in nodes_subset or nodes_subset == []:
            if node_num not in nodes_to_exclude:
                mesh.add_stdnode(node_num, coordinates, group='_default')
                print('Morhpic node added', node_num, coordinates)

    # Add elements
    elements = scipy.loadtxt(
        os.path.join(mesh_dir, element_file), dtype='int', delimiter=',')
    for element_idx, element_nodes in enumerate(elements):
        element_num = element_idx + 1
        renumbering_idx = scipy.array([0, 1, 3, 2])
        element_nodes = element_nodes[renumbering_idx]
        if element_num in elem_subset or elem_subset == []:
            mesh.add_element(element_num, ['L1', 'L1'], element_nodes)
            print('Morphic element added', element_num)

    # Generate the mesh
    mesh.generate(True)

    return mesh

def morphic_to_meshio(
        morphic_mesh, triangulate=False, res=8, exterior_only=True):
    """Convert an morphic mesh to meshio format.

    Only Linear lagrange elements supported.

    Quad elements implemented, hex elements are a work in progress.

    Keyword arguments:
    morphic_mesh -- morphic mesh to convert.
    triangulate -- create triangulation from morphic mesh prior to conversion.
    res -- resolution to generate triangulation from. Only used when
        triangulate=True.
    exterior_only -- only triangulate exterior surface of mesh. Only used when
        triangulate=True.
    """

    import meshio

    if triangulate:
        # Generate triangulated surface mesh.
        points, faces = morphic_mesh.get_faces(
            res=res, exterior_only=exterior_only)
        cells = []
        for face in faces:
            cells.append(("triangle", np.array([face])))

        meshio_mesh = meshio.Mesh(points, cells)

    else:
        points = morphic_mesh.get_nodes()

        quad_element_nodes = []
        hex_element_nodes = []
        quad_element_numbers = []
        hex_element_numbers = []
        for element_idx, element in enumerate(morphic_mesh.elements):
            if element.basis == ['L1', 'L1']:
                reordering_idxs = scipy.array(
                    [1, 3, 2, 0])
                reordered_element_nodes = np.array(
                    element.node_ids)[reordering_idxs]
                quad_element_nodes.append(reordered_element_nodes)
                quad_element_numbers.append(element.id)
            elif element.basis == ['L1', 'L1', 'L1']:
                hex_element_nodes.append(element.node_ids)
                hex_element_numbers.append(element.id)

        cells = [
            ("quad", np.array(quad_element_nodes)-1),
            #("hex", hex_element_nodes),
        ]

        meshio_mesh = meshio.Mesh(
            points,
            cells,
            cell_data={"element_numbers": [np.array(quad_element_numbers)]},
        )

    return meshio_mesh


def abaqus_to_morphic(mesh_dir, filename, nodes_subset=[], elem_subset=[],
                      debug=False):
    """Convert an abaqus .inp file to a morphic mesh.

    Only Linear lagrange elements supported.

    Keyword arguments:
    nodes_subset -- nodes to load (all if empty)
    elem_subset -- elements to load (all if empty)
    """
    import meshio

    # Create morphic mesh
    mesh = morphic.Mesh()

    abaqus_mesh = meshio.read(os.path.join(mesh_dir, filename))

    for node_num, coordinates in enumerate(abaqus_mesh.points):
        if node_num in nodes_subset or nodes_subset == []:
            print('Morhpic node added', node_num)
            mesh.add_stdnode(
                node_num, coordinates, group='_default')

    for abaqus_mesh_cell in abaqus_mesh.cells:
        for element_num, element_nodes in enumerate(abaqus_mesh_cell.data):
            if element_num in elem_subset or elem_subset == []:
                print('Morphic element added', element_num)
                if abaqus_mesh_cell.type == 'hex':
                    reordering_idxs = scipy.array(
                        [0, 1, 2, 3, 4, 5, 6, 7])
                    reordered_element_nodes = element_nodes[reordering_idxs]
                    mesh.add_element(
                        element_num, ['L1', 'L1', 'L1'],
                        reordered_element_nodes)
                elif abaqus_mesh_cell.type == 'quad':
                    reordering_idxs = scipy.array(
                        [3, 0, 2, 1])
                    reordered_element_nodes = element_nodes[reordering_idxs]
                    mesh.add_element(
                        element_num, ['L1', 'L1'],
                        reordered_element_nodes)

    # Generate the mesh
    mesh.generate(True)

    return mesh

def abaqus_to_morphic_legacy(
        mesh_dir, filename, nodes_subset=[], elem_subset=[], debug=False):
    """Convert an abaqus .inp file to a morphic mesh.

    Only CAX4P surface meshes and SC8R volume meshes supported.
    Only Linear lagrange elements supported.

    Keyword arguments:
    nodes_subset -- nodes to load (all if empty)
    elem_subset -- elements to load (all if empty)
    """

    # Create mesh
    mesh = morphic.Mesh()

    # Load abaqus .inp file
    f = open(os.path.join(mesh_dir, filename), 'r')
    lines = f.readlines()
    num_lines = len(lines)

    # Add nodes
    for line_idx, line in enumerate(lines):
        if debug:
            print(line)
        if line.strip() == '*Node':
            for node_line_idx in range(line_idx + 1, num_lines + 1):
                node_line = lines[node_line_idx]
                if node_line.strip() == '*Element, type=SC8R':
                    break
                else:
                    coordinates = node_line.strip().split(',')

                    x = float(coordinates[1])
                    y = float(coordinates[2])
                    z = float(coordinates[3])

                    node_num = int(coordinates[0])
                    if node_num in nodes_subset or nodes_subset == []:
                        print('Morhpic node added', node_num)
                        mesh.add_stdnode(
                            node_num, scipy.array([x, y, z]), group='_default')
            break

    # Add elements
    for line_idx, line in enumerate(lines):
        if line.strip() in ['*Element, type=SC8R', '*ELEMENT, TYPE=CAX4P']:
            elem_type = line.strip().split('=')[1]
            for node_line_idx in range(line_idx + 1, num_lines + 1):
                print(node_line_idx)
                if node_line_idx == 1136:
                    a=1
                node_line = lines[node_line_idx]
                if node_line.strip() == '*System':
                    break
                else:
                    element_nodes = node_line.strip().split(',')[1:]  # [-1]
                    element_nodes = scipy.array(
                        [int(node) for node in element_nodes])
                    element_num = int(node_line.strip().split(',')[0])
                    # print 'Elem: ', element_num, 'Elem nodes: ', element_nodes
                    if element_num in elem_subset or elem_subset == []:
                        print('Morphic element added', element_num)
                        if elem_type == 'SC8R':
                            renumbering_idx = scipy.array(
                                [0, 1, 2, 3, 4, 5, 6, 7])
                            element_nodes = element_nodes[renumbering_idx]
                            mesh.add_element(
                                element_num, ['L1', 'L1', 'L1'], element_nodes)
                        elif elem_type == 'CAX4P':
                            renumbering_idx = scipy.array(
                                [0, 1, 2, 3])
                            element_nodes = element_nodes[renumbering_idx]
                            mesh.add_element(
                                element_num, ['L1', 'L1'], element_nodes)
            elem_type = ''
            break

    # Generate the mesh
    mesh.generate(True)

    return mesh


def ansys_to_morphic(mesh_dir, filename, nodes_subset=[], elem_subset=[],
                     debug=False):
    """Convert an ansys .in file to a morphic mesh.

    Only Linear lagrange elements supported.

    Keyword arguments:
    nodes_subset -- nodes to load (all if empty)
    elem_subset -- elements to load (all if empty)
    """

    # Create mesh
    mesh = morphic.Mesh()

    # Load ansys .in file
    f = open(os.path.join(mesh_dir, filename), 'r')
    lines = f.readlines()
    num_lines = len(lines)

    # Add nodes
    for line_idx, line in enumerate(lines):
        if line.split(' ,')[0] == 'NBLOCK':
            for node_line_idx in range(line_idx + 2, num_lines + 1):
                node_line = lines[node_line_idx]
                if node_line.split()[0] == 'N':
                    break
                else:
                    coordinates = node_line.split('       ')[-1]
                    # import ipdb; ipdb.set_trace()
                    x = float(coordinates[1:17])
                    y = float(coordinates[17:33])
                    z = float(coordinates[33:-1])

                    node_num = int(node_line.split()[0])
                    if node_num in nodes_subset or nodes_subset == []:
                        print('Morhpic node added', node_num)
                        mesh.add_stdnode(
                            node_num, scipy.array([x, y, z]), group='_default')
            break

    # Add elements
    for line_idx, line in enumerate(lines):
        if line.split(' ,')[0] == 'EBLOCK':
            for node_line_idx in range(line_idx + 2, num_lines + 1):
                node_line = lines[node_line_idx]
                # print node_line.split()
                if node_line.split() == []:
                    break
                else:
                    element_nodes = node_line.split()[11:15]  # [-1]
                    element_nodes = scipy.array(
                        [int(node) for node in element_nodes])
                    renumbering_idx = scipy.array([0, 1, 3, 2])
                    element_nodes = element_nodes[renumbering_idx]
                    element_num = int(node_line.split()[10])
                    # print 'Elem: ', element_num, 'Elem nodes: ', element_nodes
                    # import ipdb; ipdb.set_trace()
                    if element_num in elem_subset or elem_subset == []:
                        print('Morphic element added', element_num)
                        mesh.add_element(element_num, ['L1', 'L1'],
                                         element_nodes)
            break

    # Generate the mesh
    mesh.generate(True)

    return mesh

def morphic_to_openfemlite(
        mesh, export_dir=[], export_name='3DMeshIn3DSpace',
        node_offset=0, element_offset=0, mesh_type='volume',
        visualisation_scripts_only=False, debug=False):
    """Convert a morphic mesh to a an openfemlite mesh and export ip files.

    Cm and cmgui .com files will only be generated if they do not already
    exist in the export dir.

    Keyword arguments:
    mesh -- morphic mesh
    export_dir -- directory to export ip files (not exported if [])
    export_name -- name to export ip files (default '3DMeshIn3DSpace')
    visualisation_scripts_only -- only export CMGUI.com visualisation script
    """

    try:
        import fem_topology
    except:
        raise ValueError(
            'Openfemlite is required. This repo needs to be cloned from \
            https://github.com/PrasadBabarendaGamage/open-fem-lite an the \
            open-fem-lite/src/ folder added to the python path \
            (it is not a python module)')

    # User Numbers
    RegionUserNumber = 1
    BasisUserNumber = 1
    GeneratedMeshUserNumber = 1
    MeshUserNumber = 1
    MeshNumberOfComponents = 1
    MeshTotalNumberOfElements = 1
    GeometricFieldUserNumber = 1
    GeometricFieldNumberOfVariables = 1
    GeometricFieldNumberOfComponents = 3

    # Initialize and Create Regions
    WORLD_REGION = fem_topology.femInitialize()
    WORLD_REGION.RegionsCreateStart(RegionUserNumber)
    WORLD_REGION.RegionsCreateFinish(RegionUserNumber)
    REGION = WORLD_REGION.RegionsRegionGet(RegionUserNumber)

    # Create Basis
    REGION.BASES.BasesCreateStart(BasisUserNumber)
    if mesh_type == 'volume':
        REGION.BASES.BasisTypeSet(BasisUserNumber, "3DLinearLagrange")
        NumberOfXi = 3
    elif mesh_type == 'surface':
        REGION.BASES.BasisTypeSet(BasisUserNumber, "2DLinearLagrange")
        NumberOfXi = 2
    REGION.BASES.BasisNumberOfXiCoordinatesSet(BasisUserNumber, NumberOfXi)
    REGION.BASES.BasesCreateFinish(BasisUserNumber)

    TotalNumberOfNodes = len(mesh.get_node_ids()[1])
    REGION.NODES.NodesCreateStart(TotalNumberOfNodes)

    MeshNumberOfComponents = 1

    MeshTotalNumberOfElements = len(mesh.get_element_cids())
    MeshUserNumber = 1
    REGION.MESHES.MeshesCreateStart(MeshUserNumber)
    REGION.MESHES.MeshNumberOfDimensionsSet(MeshUserNumber, NumberOfXi)
    REGION.MESHES.MeshNumberOfComponentsSet(MeshUserNumber,
                                            MeshNumberOfComponents)
    MeshComponent = 1
    REGION.MESHES.MeshElementsCreateStart(MeshUserNumber, MeshComponent,
                                          BasisUserNumber)

    for element_idx, element in enumerate(mesh.elements):
        if debug:
            print('OpenfemLite element added', element.id)
        # elements.NodesSet(element_idx+offset, scipy.array(element.node_ids, dtype='int32'))
        REGION.MESHES.MeshElementsNodesSet(MeshUserNumber, MeshComponent,
                                           element.id + element_offset,
                                           element.node_ids)

    REGION.MESHES.MeshElementsCreateFinish(MeshUserNumber, MeshComponent)
    REGION.MESHES.MeshesCreateFinish(MeshUserNumber)

    # Define Geometric Fields
    REGION.FIELDS.FieldCreateStart(GeometricFieldUserNumber)
    REGION.FIELDS.FieldTypeSet(GeometricFieldUserNumber, "FieldGeometricType")
    REGION.FIELDS.FieldMeshSet(GeometricFieldUserNumber, MeshUserNumber)
    REGION.FIELDS.FieldNumberOfFieldVariablesSet(GeometricFieldUserNumber,
                                                 GeometricFieldNumberOfVariables)
    REGION.FIELDS.FieldNumberOfFieldComponentsSet(GeometricFieldUserNumber, 1,
                                                  GeometricFieldNumberOfComponents)
    REGION.FIELDS.FieldComponentLabelSet(GeometricFieldUserNumber, 1, 1, "x")
    REGION.FIELDS.FieldComponentLabelSet(GeometricFieldUserNumber, 1, 2, "y")
    REGION.FIELDS.FieldComponentLabelSet(GeometricFieldUserNumber, 1, 2, "z")
    REGION.FIELDS.FieldComponentMeshComponentSet(GeometricFieldUserNumber, 1,
                                                 1,
                                                 1)  # FieldUserNumber,FieldVariableUserNumber,FieldComponentUserNumber,MeshComponentUserNumber
    REGION.FIELDS.FieldComponentMeshComponentSet(GeometricFieldUserNumber, 1,
                                                 2, 1)
    REGION.FIELDS.FieldComponentMeshComponentSet(GeometricFieldUserNumber, 1,
                                                 3, 1)
    REGION.FIELDS.FieldCreateFinish(GeometricFieldUserNumber)

    FieldVariable = 1
    for morphic_node in mesh.nodes:
        # if morphic_node.id in nodes_subset or nodes_subset == []:
        for comp_idx in range(3):
            try:
                REGION.FIELDS.FieldParameterSetUpdateNode(
                    GeometricFieldUserNumber, FieldVariable, 1, 1,
                    morphic_node.id, comp_idx + 1,
                    morphic_node.values[comp_idx])  # version_idx,deri
            except:
                pass

    if export_dir:
        path = os.path.join(export_dir, export_name + 'CMGUI.com')
        if os.path.exists(path):
            print('WARNING: ' + path + ' already exists')
        else:
            REGION.WriteCmguiCom(GeometricFieldUserNumber, 1,
                                 os.path.join(export_dir, export_name))
        if not visualisation_scripts_only:
            REGION.WriteIpCoor(GeometricFieldUserNumber, 1,
                               os.path.join(export_dir, export_name))
            REGION.WriteIpNode(GeometricFieldUserNumber, 1,
                               os.path.join(export_dir, export_name))
            REGION.WriteIpElem(GeometricFieldUserNumber, 1,
                               os.path.join(export_dir, export_name))
            if not os.path.exists(
                    os.path.join(export_dir, export_name + '.ipbase')):
                REGION.WriteIpBase(GeometricFieldUserNumber, 1,
                                   os.path.join(export_dir, export_name))
            if not os.path.exists(
                    os.path.join(export_dir, export_name + 'CMISS.com')):
                REGION.WriteCmCom(GeometricFieldUserNumber, 1,
                                  os.path.join(export_dir, export_name))

    return REGION