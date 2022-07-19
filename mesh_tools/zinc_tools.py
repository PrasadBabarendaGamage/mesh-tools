"""
Tools for evaluating zinc meshes (scaffolds)
"""
import numpy as np

from opencmiss.zinc.context import Context
from opencmiss.zinc.status import OK as ZINC_OK
from scaffoldmaker.utils import zinc_utils
from opencmiss.utils.zinc.finiteelement import get_element_node_identifiers

def evaluate_zinc_mesh(input_mesh, dim, xi_loc, elements):
    """
    Evaluate element xi locations form an input .exf format mesh.

    input_mesh - Location of the input exf file.
    dim - Dimension of the mesh i.e. 1, 2, or 3.
    xi_loc - The xi coordinates to evaluate.
    elements -
    """

    context = Context("Scaffold")
    region = context.getDefaultRegion()
    region.readFile(input_mesh)
    field_module = region.getFieldmodule()
    field = field_module.findFieldByName("coordinates")
    cache = field_module.createFieldcache()
    xi = xi_loc
    mesh = field_module.findMeshByDimension(dim)
    if len(xi) != dim:
        raise TypeError("Number of xi coordinates is not valid for {} dimension".format(dim))

    el_iter = mesh.createElementiterator()
    elemDict = dict()
    element = el_iter.next()
    while element.isValid():
        elemDict[int(element.getIdentifier())] = element
        element = el_iter.next()
    mesh_elements = elemDict

    for element in elements:
        cache.setMeshLocation(mesh_elements[element], xi)
        result, out_values = field.evaluateReal(cache, 3)
        if result == ZINC_OK:
            print(mesh_elements[element].getIdentifier(), out_values)
        else:
            raise('Error when evaluating element {0}, at xi {1}'.format(element, xi))


class Zinc_mesh:
    """
    Zinc mesh
    """

    def __init__(self, input_mesh, dim=3):
        """
        Load zinc mesh in .exf format mesh

        input_mesh - Location of the input exf file.
        dim - Dimension of the mesh i.e. 1, 2, or 3.
        """
        self.input_mesh = input_mesh
        self.dim = dim


    def evaluate(self, xi_loc, elements):
        """
        Evaluate element xi locations frmm an input .exf format mesh, at user defined element xi locations

        input_mesh - Location of the input exf file.
        xi_loc - The xi coordinates to evaluate.
        elements -The elements to evaluate the xi coordinates for
        """
        num_xi_locs = np.array(xi_loc).shape[0]
        coords = np.zeros((num_xi_locs, self.dim))

        context = Context("Scaffold")
        region = context.getDefaultRegion()
        result = region.readFile(self.input_mesh)
        if result != ZINC_OK:
            raise ValueError('Error loading mesh in {0}'.format(self.input_mesh))

        field_module = region.getFieldmodule()
        field = field_module.findFieldByName("coordinates")
        mesh = field_module.findMeshByDimension(self.dim)

        # Store elements in a dict
        el_iter = mesh.createElementiterator()
        elemDict = dict()
        element = el_iter.next()
        while element.isValid():
            elemDict[int(element.getIdentifier())] = element
            element = el_iter.next()
        mesh_elements = elemDict

        xi = xi_loc[0]
        if len(xi) != self.dim:
            raise TypeError("Number of xi coordinates is not valid for {} dimension".format(self.dim))

        cache = field_module.createFieldcache()
        for xi_idx, xi in enumerate(xi_loc):
            element = elements[xi_idx]
            cache.setMeshLocation(mesh_elements[element], xi.tolist())
            result, out_values = field.evaluateReal(cache, self.dim)
            if result == ZINC_OK:
                coords[xi_idx, :] = out_values
            else:
                raise ValueError('Error when evaluating element {0}, at xi {1}'.format(element, xi))

        return coords

    def export_vtk(self, filename):

        context = Context("Scaffold")
        region = context.getDefaultRegion()
        result = region.readFile(self.input_mesh)
        if result != ZINC_OK:
            raise ValueError('Error loading mesh in {0}'.format(self.input_mesh))

        from scaffoldmaker.utils.exportvtk import ExportVtk
        export = ExportVtk(region, 'test')
        export.writeFile(filename)

    def find_standard_elements(self,):
        """
        Returns the standard element numbers in mesh (ie meshes that have 8 nodes per element

        """

        context = Context("Scaffold")
        region = context.getDefaultRegion()
        result = region.readFile(self.input_mesh)
        if result != ZINC_OK:
            raise ValueError('Error loading mesh in {0}'.format(self.input_mesh))

        field_module = region.getFieldmodule()
        mesh = field_module.findMeshByDimension(self.dim)
        coordinates = field_module.findFieldByName('coordinates')
        # Store elements in a dict
        el_iter = mesh.createElementiterator()
        elemDict = dict()
        element = el_iter.next()
        elem_list = []
        while element.isValid():
            elem_list.append(int(element.getIdentifier()))
            elemDict[int(element.getIdentifier())] = element
            element = el_iter.next()
        mesh_elements = elemDict

        for elem in elem_list:
            mesh_element = mesh_elements[elem]
            eft = mesh_element.getElementfieldtemplate(coordinates, -1)  # assumes all components same
            nodeIdentifiers = get_element_node_identifiers(mesh_element, eft)
            a=1



