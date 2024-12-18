"""
Utilities for generating 2D structures
"""

from __future__ import annotations

import numpy as np

from czone.volume import Sphere, Plane, Volume
from czone.generator import Generator, AmorphousGenerator
from czone.transform import Translation, Rotation, Reflection, Inversion

from .plane_groups import PlaneGroup
from pymatgen.core import Structure

from functools import reduce
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from czone.types import BaseGenerator

"""
2D Volume primitives
"""


class Circle(Sphere):
    @Sphere.center.setter
    def center(self, center: np.ndarray):
        center = np.array(center)
        try:
            center = center.reshape((3,))  # cast to np array if not already
        except ValueError as e:
            center = center.reshape((2,))
            center = np.array([center[0], center[1], 0.0])

        if center.size != 3 or center.shape[0] != 3:
            raise ValueError("Center must be an array with 3 elements")

        if not np.isclose(center[2], 0.0):
            raise ValueError(
                f"Circle must have center lying in XY plane, but has Z coordinate of {center[2]:04f}"
            )

        self._center = center


class TwoPlane(Plane):
    @Plane.normal.setter
    def normal(self, normal: np.ndarray):
        normal = np.array(normal)  # cast to np array if not already

        try:
            normal = normal.reshape((3,))
        except ValueError as e:
            raise ValueError(f"Normal must be a 3D vector, but has shape {normal.shape}")

        if not np.isclose(normal[2], 0.0):
            raise ValueError(
                f"TwoPlane must have normal lying in XY plane, but has Z component of {normal[2]:04f}"
            )

        if np.linalg.norm(normal) > np.finfo(float).eps:
            self._normal = normal / np.linalg.norm(normal)
        else:
            raise ValueError(
                f"Input normal vector length {np.linalg.norm(normal)} is below machine precision."
            )


"""
2D Transforms
"""

class TwoTransform: # just for type checking
    pass

# TODO: Translation -> restrict Z to 0

# TODO: Rotation -> matrices of form [[a,b, 0], [c,d, 0], [0, 0, e]]
class TwoRotation(Rotation, TwoTransform):

    @Rotation.matrix.setter
    def matrix(self, val):

        val = np.array(val).reshape(3,3)
        if (not np.array_equal(val[:,-1], np.array([0,0,1]))) or \
            (not np.array_equal(val[-1, :], np.array([0,0,1]))):
            # rotation is not in XY plane
            raise ValueError
        super(TwoRotation, TwoRotation).matrix.__set__(self, val)


# TODO: Reflection -> Plane must be TwoPlane
class TwoReflection(Reflection, TwoTransform):
    @Reflection.plane.setter
    def plane(self, plane: TwoPlane):
        if isinstance(plane, TwoPlane):
            super().plane.setter(plane)
        else:
            raise TypeError()


# TODO: Inversion -> Origin must be in XZ plane
class TwoInversion(Inversion, TwoTransform):
    @Inversion.origin.setter
    def origin(self, val):
        pass


"""
2D Generators
"""


class TwoGenerator(Generator):
    def __init__(self, lattice, Z, basis, origin=None):
        if origin is None:
            origin = np.zeros((2))

        lattice_3D = np.eye(3)
        lattice_3D[:2, :2] = lattice

        basis_3D = np.zeros((len(basis), 3))
        basis_3D[:, :2] = np.array(basis)

        super().__init__(origin=np.array([origin[0], origin[1], 0]),
                         structure=Structure(lattice_3D, Z, basis_3D))

    def __repr__(self):
        return f"Two{super().__repr__()}"

    @staticmethod
    def _parse_plane_group_args(pg: PlaneGroup, a: float, b: float | None, theta: float | None):
        match pg.int_number:
            case 1 | 2:
                # monoclinic lattice
                if (b is None) or (theta is None):
                    raise ValueError(
                        f"b and theta must be both be supplied for spacegroups 1 (p1) and 2 (p2). Received b={b} and theta={theta}"
                    )

                lattice = np.array([[a], [b]])*np.array([[1,0], [np.cos(theta), np.sin(theta)]])
            case 3 | 4 | 5 | 6 | 7 | 8 | 9:
                # rectangular lattice

                if theta is None:
                    theta = 90

                if b is None:
                    b = a

                if not np.isclose(theta, 90):
                    raise ValueError(
                        f"Lattice parameter theta must be 90 degrees for point group {pg.int_symbol}, but received theta={theta:0.2f}"
                    )
                lattice = np.array([[a, 0], [0, b]])
            case 10 | 11 | 12:
                # square lattice
                if b is None:
                    b = a

                if theta is None:
                    theta = 90

                if not np.isclose(a, b):
                    raise ValueError(
                        f"Lattice parameters a and b must be equal for point group {pg.int_symbol}, but received a={a:0.4f} and b={b:0.4f}"
                    )
                lattice = a * np.eye(2)
            case 13 | 14 | 15 | 16 | 17:
                # hexagonal lattice
                if theta is None:
                    theta = 120

                if not np.isclose(theta, 120):
                    raise ValueError(
                        f"Lattice parameter theta must be 120 degrees for point group {pg.int_symbol}, but received theta={theta:0.2f}"
                    )
                
                if b  is None:
                    b = a
                
                if not np.isclose(a, b):
                    raise ValueError(
                        f"Lattice parameters a and b must be equal for point group {pg.int_symbol}, but received a={a:0.4f} and b={b:0.4f}"
                    )
                
                theta *= np.pi/180
                lattice = a*np.array([[1,0], [np.cos(theta), np.sin(theta)]])
                print(np.linalg.norm(lattice, axis=1))

        return lattice

    @classmethod
    def from_plane_group(
        cls,
        Z: list[int],
        basis: np.ndarray | list[np.ndarray],
        pg: int | str,
        a: float,
        b: float = None,
        theta: float = None,
        **kwargs,
    ):
        """
        Args:
            basis: (partial) set of basis atoms in crystal coordinates. Will be symmetrizd.
            pg: int, 1-17 or str, as below, corresponding to 2D planegroup as in IUCR tables (via bilbao)
            a: length of primary lattice vector
            b: length of secondary lattice vector (optional: inferred, if pg constrains b == a)
            theta: angle (in degrees) between a and b (optional: inferred, if pg constrains theta==90 or theta==120)

        no restrictions on lattice:
        1- p1
        2- p2

        at least rectangular (a != b, theta=90); can be centered or not:
        3- p1m1 (pm)
        4- p1g1 (pg)
        5- c1m1 (cm)
        6- p2mm
        7- p2mg
        8- p2gg
        9- c2mm

        must be square (a=b, theta=90):
        10- p4
        11- p4mm
        12- p4gm

        must be hexagonal (a=b, theta=120):
        13- p3
        14- p3m1 (origin at 3m)
        15- p31m (origin at 31m)
        16- p6
        17- p6mm
        """
        pg = PlaneGroup(pg) if isinstance(pg, str) else PlaneGroup(int_number=pg)

        lattice = TwoGenerator._parse_plane_group_args(pg, a, b, theta)
        basis = np.array(basis)
        if len(np.unique(Z)) == 1:
            symmetrized_basis = list(pg.generate_orbit(basis))
            Z = len(symmetrized_basis) * [Z[0]]
        else:
            ## partition Z into sets by atom type and do some extra checks
            partitions = [np.where(np.array(Z) == z)[0] for z in Z]
            Z_sets = [Z[p][0] for p in partitions]
            basis_sets = [basis[p] for p in partitions]

            sZ_sets = []
            sbasis_sets = []
            for local_Z, local_basis in zip(Z_sets, basis_sets):
                cur_symmetrized_basis = list(pg.generate_orbit(local_basis))
                sZ_sets.append(len(cur_symmetrized_basis) * [local_Z])
                sbasis_sets.append(cur_symmetrized_basis)

            ## check to make sure atoms do not overlap between bases
            if len(reduce(lambda x, y: x | y, sbasis_sets)) != reduce(
                lambda x, y: x + y, [len(s) for s in sbasis_sets]
            ):
                raise ValueError(
                    "Could not produce a consistent basis. Check for overlapping atoms between different element types."
                )

            Z = reduce(lambda x, y: x + y, sZ_sets)
            symmetrized_basis = np.stack([np.array(list(sb)) for sb in sbasis_sets])

        return cls(lattice, Z, symmetrized_basis, **kwargs)
    
    def transform(self, transformation: TwoTransform):
        if not isinstance(transformation, TwoTransform):
            raise TypeError("Supplied transformation must be a TwoTransform.")
        
        super().transform(transformation)



class TwoAmorphousGenerator(AmorphousGenerator):
    """
    Only really need to do two things:
    1) Reconcile any differences in units/measures of density to ensure correct N of atoms
    2) Make sure samples in gen_p_substrate (or, gen_multielemenet_random_block) are in XY plane
        e.g.
        new_coord = rng.uniform(size=(1, 3)) * dims
        new_coord[2] = 0
    3) ensure that voxel list has flat Z list/ dims[2] == 1 < min_dist * voxel_scale

    """

    pass


"""
2D Volumes
"""

# TODO: Restrict transforms to valid 2D transforms


class TwoVolume(Volume):
    def __init__(
        self,
        points: np.ndarray = None,
        alg_objects: np.ndarray = None,
        generator: BaseGenerator = None,
        priority: int = 0,
        tolerance: float = 1e-6,
        **kwargs,
    ):
        # All volumes should come with pair of planes with normals (0,0,+-z_tol), to restrict atoms to XY plane
        super().__init__(points, alg_objects, generator, priority, tolerance)
        self.add_alg_object(
            [Plane((0, 0, 1), (0, 0, 0.5)), Plane((0, 0, -1), (0, 0, -0.5))]
        )

    def get_bounding_box(self):
        bbox = super().get_bounding_box()
        N = len(bbox)
        bbox[: N // 2, -1] = 0.0
        bbox[N // 2 :, -1] = 1.0

        return bbox
