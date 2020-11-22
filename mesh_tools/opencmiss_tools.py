import mesh_tools.fields as fields
import utilities
import numpy as np
from opencmiss.iron import iron

def interpolate_opencmiss_field(
        field, element_ids=[], xi=None, num_values=4, dimension=3,
        derivative_number=1, elems=None, face=None, value=0.):
    import mesh_tools.fields as fields

    if xi is None:
        if face == None:
            XiNd = fields.generate_xi_grid_fem(
                num_points=[num_values, num_values, num_values])
        else:
            XiNd = fields.generate_xi_on_face(
                face, value, num_points=num_values, dim=dimension)

        num_elem_values = XiNd.shape[0]
        num_Xe = len(element_ids)
        total_num_values = num_Xe * num_elem_values
        values = np.zeros((num_Xe, num_elem_values, dimension))
        xi = np.zeros((num_Xe, num_elem_values, dimension))
        elements = np.zeros((num_Xe, num_elem_values, 1))

        for elem_idx, element_id in enumerate(element_ids):
            for point_idx in range(num_elem_values):
                single_xi = XiNd[point_idx,:]
                values[elem_idx, point_idx, :] = field.ParameterSetInterpolateSingleXiDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES, derivative_number,
                    int(element_id), single_xi, dimension)
            xi[elem_idx, :, :] = XiNd
            elements[elem_idx, :] = element_id

        values = np.reshape(values, (total_num_values, dimension))
        xi = np.reshape(xi, (total_num_values, dimension))
        elements = np.reshape(elements, (total_num_values))
        return values, xi, elements
    else:
        num_values = xi.shape[0]
        values = np.zeros((num_values, dimension))
        for point_idx in range(xi.shape[0]):
            element_id = elems[point_idx]
            single_xi = xi[point_idx,:]
            values[point_idx, :] = field.ParameterSetInterpolateSingleXiDP(
                iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES,
                derivative_number, int(element_id), single_xi, dimension)
        return values


def interpolate_opencmiss_field_sample(
        field, element_ids=None, num_values=4, dimension=3, num_components=3,
        derivative_number=1, face=None, value=0., unique=False,
        geometric_field=None, debug=False):
    """ Interpolates and OpenCMISS field at selected points along xi directions

    Args:
        field (OpenCMISS field object): The general field to interpolate

        element_ids (list): Mesh element ids to interpolate. Defaults to all
          elements if none are specified.

        num_values (int): The number of values to interpolate at along each
          element xi direction

        dimension (int): dimension of the field being interpolated
          (e.g. 3 for 3D)

        num_components (int): number of field components to interpolate
          (e.g. 3 for 3D)
        Todo Identify number of components directly from the OpenCMISS field
             object

        derivative_number (int): The field derivative to interpolate

        face (int): The face number interpolate the field at

        value (int): The xi value of the face to select

        unique(bool): Return interpolated field values at only unique geometric
          values

        geometric_field(OpenCMISS field object): Required for unique option

        Todo replace face and value with OpenCMISS face variable which combines
             the two variables into one.

    Returns:
        values, xi, elements
    """

    if face == None:
        XiNd = fields.generate_xi_grid_fem(
            num_points=[num_values, num_values, num_values])
    else:
        XiNd = fields.generate_xi_on_face(
            face, value, num_points=num_values, dim=dimension)

    num_elem_values = XiNd.shape[0]
    num_Xe = len(element_ids)
    total_num_values = num_Xe * num_elem_values
    values = np.zeros((num_Xe, num_elem_values, num_components))
    xi = np.zeros((num_Xe, num_elem_values, dimension))
    elements = np.zeros((num_Xe, num_elem_values, 1), dtype=int)

    for elem_idx, element_id in enumerate(element_ids):
        for point_idx in range(num_elem_values):
            single_xi = XiNd[point_idx,:]
            values[elem_idx, point_idx,
            :] = field.ParameterSetInterpolateSingleXiDP(
                iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES,
                derivative_number, int(element_id), single_xi, num_components)
        xi[elem_idx, :, :] = XiNd
        elements[elem_idx, :] = element_id

    # Reshape interpolated field values into a vector
    values = np.reshape(values, (total_num_values, num_components))
    xi = np.reshape(xi, (total_num_values, dimension))
    elements = np.reshape(elements, (total_num_values))

    if unique:
        # Evaluate geometric field at xi values and select general field values
        # that only have unique coordinates
        if geometric_field is None:
            raise ValueError(
                'Geometric field is required for returning unique field values')
        geometric_values = np.zeros((total_num_values, dimension))
        for point_idx in range(total_num_values):
            geometric_values[point_idx,
            :] = geometric_field.ParameterSetInterpolateSingleXiDP(
                iron.FieldVariableTypes.U,
                iron.FieldParameterSetTypes.VALUES, derivative_number,
                int(elements[point_idx]), xi[point_idx, :], dimension)
            if debug:
                print('Point num         : ', point_idx+1)
                print('  True value      : ', geometric_values[point_idx, :])
                print('  Projected value : ', values[point_idx, :])
        # Identify unique geometric field values
        _, indices = utilities.np_1_13_unique(
            np.vstack((
                geometric_values[:, 0].round(decimals=4),
                geometric_values[:, 1].round(decimals=4),
                geometric_values[:, 2].round(decimals=4))).T,
            axis=0, return_index=True)

        # Select only unique general field values
        values = values[sorted(indices), :]
        xi = xi[sorted(indices), :]
        elements = elements[sorted(indices)]

        if debug:
            for point_idx in range(len(elements)):
                print('Point num: ', point_idx+1)
                print('  element num   : ', elements[point_idx])
                print('  xi            : ', xi[point_idx, :])
                print('  position      : ', values[point_idx, :])

    return values, xi, elements

def interpolate_opencmiss_field_xi(
        field, xi, element_ids=[], dimension=3, deriv=1,
        variable=iron.FieldVariableTypes.U):
    num_field_components = field.NumberOfComponentsGet(iron.FieldVariableTypes.U)
    xi = np.atleast_2d(xi)
    num_values = xi.shape[0]
    values = np.zeros((num_values, num_field_components))
    for point_idx in range(xi.shape[0]):
        element_id = element_ids[point_idx]
        values[point_idx, :] = field.ParameterSetInterpolateSingleXiDP(
            variable,
            iron.FieldParameterSetTypes.VALUES, deriv,
            int(element_id), xi[point_idx, :], num_field_components)
    return values


def get_field_values(field, node_nums, derivative=1, dimension=3,
                     variable=iron.FieldVariableTypes.U):
    coordinates = np.zeros((len(node_nums), dimension))
    for node_idx, node in enumerate(node_nums):
        for component_idx, component in enumerate(range(1, dimension + 1)):
            coordinates[node_idx, component_idx] = field.ParameterSetGetNodeDP(
                variable, iron.FieldParameterSetTypes.VALUES, 1, derivative,
                int(node), component)
    return coordinates


def set_field_values(field, node_nums, values, derivative=1,
                     variable=iron.FieldVariableTypes.U,
                     update_scale_factors=False):
    """
    Update the field parameters (only sets components 1 to number of
    coordinates of the input values array)
    """
    if update_scale_factors:
        field.ParameterSetUpdateStart(iron.FieldVariableTypes.U,
                                      iron.FieldParameterSetTypes.VALUES)
    for node_idx, node in enumerate(node_nums):
        for component_idx, component in enumerate(
                range(1, values.shape[1] + 1)):
            field.ParameterSetUpdateNodeDP(
                variable, iron.FieldParameterSetTypes.VALUES, 1, derivative,
                int(node), component, values[node_idx, component_idx])

    if update_scale_factors:
        field.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,
                                       iron.FieldParameterSetTypes.VALUES)

def num_nodes_get(mesh, mesh_component=1):
    nodes = iron.MeshNodes()
    mesh.NodesGet(mesh_component, nodes)
    return nodes.NumberOfNodesGet()

def num_element_get(mesh, mesh_component=1):
    elements = iron.MeshElements()
    mesh.ElementsGet(mesh_component, elements)
    num_elements = mesh.NumberOfElementsGet()
    element_nums = (np.arange(num_elements)+1).tolist()
    return num_elements, element_nums

def export_ex_mesh(filename, region):
    """
    Export fields in ex format.
    """

    fields = iron.Fields()
    fields.CreateRegion(region)
    fields.NodesExport(filename, "FORTRAN")
    fields.ElementsExport(filename, "FORTRAN")
    fields.Finalise()

def generate_opencmiss_geometry(
        interpolation=None, region_label='region', num_elements=None,
        dimensions=None, scaling_type=None, mesh_type=None):
    """ Generates an OpenCMISS geometry (mesh and geometric field)


    Args:
        interpolation (iron.BasisInterpolationSpecifications):
          Element interpolation e.g. LINEAR_LAGRANGE
        region_label (str): Region name
        num_elements (arr): Number of elements in the mesh along each element
          xi e.g. [1,1,1]
        dimensions (arr): Dimension of the geometry, e.g. [10, 10, 10] for a
          regular mesh
        scaling_type (iron.FieldScalingTypes): The type of field scaling to use
          e.g. 'NONE'
        mesh_type(iron.GeneratedMeshTypes): Type of mesh to generate. Options
          are: 'CYLINDER', 'ELLIPSOID', 'FRACTAL_TREE', 'POLAR', 'REGULAR'

    Returns:
        region, decomposition, mesh, geometric_field
    """

    # OpenCMISS user numbers
    coor_sys_user_num = 1
    region_user_num = 1
    basis_user_num = 1
    generated_mesh_user_num = 1
    mesh_user_num = 1
    decomposition_user_num = 1
    geometric_field_user_num = 1

    # Instances for setting up OpenCMISS mesh
    coordinate_system = iron.CoordinateSystem()
    region = iron.Region()
    basis = iron.Basis()
    generated_mesh = iron.GeneratedMesh()
    mesh = iron.Mesh()
    decomposition = iron.Decomposition()
    geometric_field = iron.Field()

    if interpolation is None:
        interpolation = iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE
    if dimensions is None:
        dimensions = np.array([1, 1, 1])  # Length, width, height
    if num_elements is None:
        num_elements = [1, 1, 1]
    components = [1, 2, 3]  # Geometric components
    dimension = 3  # 3D coordinates
    if scaling_type is None:
        scaling_type = iron.FieldScalingTypes.NONE
    if mesh_type is None:
        mesh_type=iron.GeneratedMeshTypes.REGULAR

    # Get the number of computational nodes and this computational node number
    numberOfComputationalNodes = iron.ComputationalNumberOfNodesGet()
    computationalNodeNumber = iron.ComputationalNodeNumberGet()

    # Create a 3D rectangular cartesian coordinate system
    coordinate_system.CreateStart(coor_sys_user_num)
    coordinate_system.DimensionSet(dimension)
    coordinate_system.CreateFinish()

    # Create a region and assign the coordinate system to the region
    region.CreateStart(region_user_num, iron.WorldRegion)
    region.LabelSet(region_label)
    region.CoordinateSystemSet(coordinate_system)
    region.CreateFinish()

    # Define basis
    basis.CreateStart(basis_user_num)
    basis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
    basis.NumberOfXiSet(dimension)
    basis.InterpolationXiSet([interpolation] * dimension)
    # Set number of Gauss points used for quadrature
    if interpolation == iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE:
        number_gauss_xi = 2
    elif interpolation == \
            iron.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE:
        number_gauss_xi = 3
    elif interpolation == \
            iron.BasisInterpolationSpecifications.CUBIC_LAGRANGE:
        number_gauss_xi = 4
    else:
        raise ValueError('Interpolation not supported')
    basis.QuadratureNumberOfGaussXiSet([number_gauss_xi] * dimension)
    basis.CreateFinish()

    # Start the creation of a generated mesh in the region
    generated_mesh.CreateStart(generated_mesh_user_num, region)
    generated_mesh.TypeSet(mesh_type)
    generated_mesh.BasisSet([basis])
    generated_mesh.ExtentSet(dimensions)
    generated_mesh.NumberOfElementsSet(num_elements)
    generated_mesh.CreateFinish(mesh_user_num, mesh)

    # Create a decomposition for the mesh
    decomposition.CreateStart(decomposition_user_num, mesh)
    decomposition.TypeSet(iron.DecompositionTypes.CALCULATED)
    decomposition.CalculateFacesSet(True)
    decomposition.NumberOfDomainsSet(numberOfComputationalNodes)
    decomposition.CreateFinish()

    # Create a field for the geometry
    geometric_field.CreateStart(geometric_field_user_num, region)
    geometric_field.MeshDecompositionSet(decomposition)
    geometric_field.TypeSet(iron.FieldTypes.GEOMETRIC)
    geometric_field.VariableLabelSet(iron.FieldVariableTypes.U, "geometry")
    for component in components:
        geometric_field.ComponentMeshComponentSet(
            iron.FieldVariableTypes.U, component, 1)
    geometric_field.ScalingTypeSet(scaling_type)
    geometric_field.CreateFinish()

    # Update the geometric field parameters from generated mesh
    generated_mesh.GeometricParametersCalculate(geometric_field)
    generated_mesh.Destroy()

    return region, decomposition, mesh, geometric_field

def generate_opencmiss_region(
        interpolation=None, region_label='region', dimension=2):
    """ Generates an OpenCMISS geometry (mesh and geometric field)


    Args:
        interpolation (iron.BasisInterpolationSpecifications):
          Element interpolation e.g. LINEAR_LAGRANGE
        region_label (str): Region name

    Returns:
        region, decomposition, mesh, geometric_field
    """

    # OpenCMISS user numbers
    coor_sys_user_num = 1
    region_user_num = 1
    basis_user_num = 1

    # Instances for setting up OpenCMISS mesh
    coordinate_system = iron.CoordinateSystem()
    region = iron.Region()
    basis = iron.Basis()

    if interpolation is None:
        interpolation = iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE

    components = [1, 2, 3]  # Geometric components

    # Get the number of computational nodes and this computational node number
    numberOfComputationalNodes = iron.ComputationalNumberOfNodesGet()
    computationalNodeNumber = iron.ComputationalNodeNumberGet()

    # Create a 3D rectangular cartesian coordinate system
    coordinate_system.CreateStart(coor_sys_user_num)
    coordinate_system.DimensionSet(3)
    coordinate_system.CreateFinish()

    # Create a region and assign the coordinate system to the region
    region.CreateStart(region_user_num, iron.WorldRegion)
    region.LabelSet(region_label)
    region.CoordinateSystemSet(coordinate_system)
    region.CreateFinish()

    # Define basis
    basis.CreateStart(basis_user_num)
    basis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
    basis.NumberOfXiSet(dimension)
    basis.InterpolationXiSet([interpolation] * dimension)
    # Set number of Gauss points used for quadrature
    if interpolation == iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE:
        number_gauss_xi = 2
    elif interpolation == \
            iron.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE:
        number_gauss_xi = 3
    elif interpolation == \
            iron.BasisInterpolationSpecifications.CUBIC_LAGRANGE:
        number_gauss_xi = 4
    else:
        raise ValueError('Interpolation not supported')
    basis.QuadratureNumberOfGaussXiSet([number_gauss_xi] * dimension)
    basis.CreateFinish()

    return region, basis
