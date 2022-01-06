"""Example demonstrating converting a Hermite mesh to a Lagrange mesh.

The Hermite mesh needs to be provided in exnode/exelem format.
The Hermite mesh needs to have unit scaling without versions. Conversion to
quadratic or cubic Lagrange is supported. The code will check if OpenCMISS-Iron
is available and if so, export the mesh in exnode/exelem format.

"""

import os
import mesh_tools
from morphic.utils import convert_hermite_lagrange
from importlib import util

def mesh_conversion(
        dimension, interpolation, exfile_coordinate_field, exnode_filename,
        exelem_filename, results_folder, results_filename):

    # Read in ex mesh as a morphic mesh.
    cubic_hermite_morphic_mesh = mesh_tools.exfile_to_morphic(
        exnode_filename, exelem_filename, exfile_coordinate_field,
        dimension=dimension, interpolation='hermite')

    # Convert mesh from cubic hermite to cubic Lagrange
    lagrange_morphic_mesh = convert_hermite_lagrange(
        cubic_hermite_morphic_mesh, tol=1e-9, interpolation=interpolation)

    # List nodes
    print("Node numbers")
    print(lagrange_morphic_mesh.get_node_ids(group='_default')[1])
    print("Node coordinates")
    print(lagrange_morphic_mesh.get_node_ids(group='_default')[0])

    # List node ids in each element
    for element in lagrange_morphic_mesh.elements:
        print("Element number")
        print(element.id)
        print("Element node numbers")
        print(element.node_ids)

    # Export mesh in ex format using OpenCMISS
    module_spec = util.find_spec("opencmiss")
    opencmiss_found = module_spec is not None
    if opencmiss_found:
        from opencmiss.iron import iron

        if interpolation == "quadraticLagrange":
            opencmiss_mesh_interpolation = \
                iron.BasisInterpolationSpecifications.QUADRATIC_LAGRANGE
        elif interpolation == "cubicLagrange":
            opencmiss_mesh_interpolation = \
                iron.BasisInterpolationSpecifications.CUBIC_LAGRANGE

        numberOfComputationalNodes = iron.ComputationalNumberOfNodesGet()
        computationalNodeNumber = iron.ComputationalNodeNumberGet()

        coordinateSystemUserNumber = 1
        regionUserNumber = 3
        basisUserNumber = 2
        meshUserNumber = 1
        decompositionUserNumber = 1
        geometricFieldUserNumber = 1

        coordinateSystem = iron.CoordinateSystem()
        coordinateSystem.CreateStart(coordinateSystemUserNumber)
        coordinateSystem.dimension = 3
        coordinateSystem.CreateFinish()

        basis = iron.Basis()
        basis.CreateStart(basisUserNumber)
        basis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
        basis.NumberOfXiSet(dimension)
        basis.InterpolationXiSet([opencmiss_mesh_interpolation]*dimension)
        basis.QuadratureNumberOfGaussXiSet([4]*dimension)
        basis.CreateFinish()

        region = iron.Region()
        region.CreateStart(regionUserNumber,iron.WorldRegion)
        region.LabelSet("Region")
        region.CoordinateSystemSet(coordinateSystem)
        region.CreateFinish()

        mesh, coordinates, node_nums, element_nums = mesh_tools.morphic_to_OpenCMISS(
            lagrange_morphic_mesh, region, basis, meshUserNumber,
            dimension=dimension, interpolation=interpolation,
            UsePressureBasis=False, pressureBasis=None)

        decomposition = iron.Decomposition()
        decomposition.CreateStart(decompositionUserNumber, mesh)
        decomposition.TypeSet(iron.DecompositionTypes.CALCULATED)
        decomposition.NumberOfDomainsSet(numberOfComputationalNodes)
        decomposition.CreateFinish()

        geometric_field = iron.Field()
        geometric_field.CreateStart(geometricFieldUserNumber, region)
        geometric_field.MeshDecompositionSet(decomposition)
        geometric_field.VariableLabelSet(iron.FieldVariableTypes.U, "Geometry")
        geometric_field.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 1, 1)
        geometric_field.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 2, 1)
        geometric_field.ComponentMeshComponentSet(iron.FieldVariableTypes.U, 3, 1)
        geometric_field.CreateFinish()

        # Update the geometric field parameters
        geometric_field.ParameterSetUpdateStart(
            iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)
        for node_idx, node in enumerate(node_nums):
            for component_idx, component in enumerate([1, 2, 3]):
                for derivative_idx, derivative in enumerate(
                        range(1, coordinates.shape[2] + 1)):
                    geometric_field.ParameterSetUpdateNodeDP(
                        iron.FieldVariableTypes.U,
                        iron.FieldParameterSetTypes.VALUES, 1, derivative, node,
                        component,
                        coordinates[node_idx, component_idx, derivative_idx])

        geometric_field.ParameterSetUpdateFinish(
            iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES)

        output_file = os.path.join(results_folder, results_filename)
        fields = iron.Fields()
        fields.CreateRegion(region)
        fields.NodesExport(output_file, "FORTRAN")
        fields.ElementsExport(output_file, "FORTRAN")
        fields.Finalise()
        
if __name__ == '__main__':
    
    dimension = 3  # Dimension of the coordinate system.
    interpolation = "quadraticLagrange"  # Interpolation of the converted mesh.

    # Input mesh information
    exfile_coordinate_field = 'coordinates'
    exnode_filename = 'test_mesh/lung_mesh.exnode'
    exelem_filename = 'test_mesh/lung_mesh.exelem'

    # Output mesh information
    results_folder='./results/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_filename = 'converted_mesh'

    mesh_conversion(
        dimension, interpolation, exfile_coordinate_field, exnode_filename,
        exelem_filename, results_folder, results_filename)