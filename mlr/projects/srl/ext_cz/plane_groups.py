import numpy as np

from ._plane_symm_data import PLANE_GROUP_DATA, GENERATORS, SYMBOL_TO_NUMBER, WYCKOFF_POS
from .array_set import NormedArraySet, ArraySet

from itertools import product


class PlaneGroup:
    def __init__(self, int_symbol: str = None, int_number: int = None):
        """
        Args:
            int_symbol: str, international symbol in Hermann-Mauguin notation, either full or abbreviated.
            int_number: int, international number of plane group
        """
        if (int_symbol is None) and (int_number is None):
            raise ValueError("Either one of int_symbol or int_number must be passed.")

        if int_number is None:
            self.int_number = SYMBOL_TO_NUMBER[int_symbol]
        else:
            self.int_number = int_number

        self.int_symbol = PLANE_GROUP_DATA[self.int_number]["full_symbol"]
        self.point_group = PLANE_GROUP_DATA[self.int_number]["point_group"]
        self._symmetry_generators = ArraySet(GENERATORS[self.int_number])
        self._symmetry_operators = PlaneGroup.generate_symmetry_ops(self.symmetry_generators)
        self.wyckoff_pos = WYCKOFF_POS[self.int_number]

    @staticmethod
    def generate_symmetry_ops(generators: list[np.ndarray]) -> ArraySet:
        """Produces full symmetry operators, from set of generators as affine matrices."""
        G_res = ArraySet(generators)
        G_new = ArraySet(generators)
        mod_mask = np.zeros_like(next(iter(G_res)), dtype=bool)
        mod_mask[:-1, -1] = True

        while len(G_new) > 0:
            G_trial = ArraySet(
                [
                    np.mod(
                        (g := x @ y), 1.0, out=g, where=mod_mask
                    )  # keep translation in unit cell
                    for x, y in product(G_new, G_res)
                ]
            )
            G_new = G_trial - G_res
            G_res |= G_trial

        return G_res

    @property
    def symmetry_generators(self) -> ArraySet:
        """Set of affine matrices which are used to generate full set of symmetry operations."""
        return self._symmetry_generators

    @property
    def symmetry_operators(self) -> ArraySet:
        """Full set of symmetry operations as affine matrices."""
        return self._symmetry_operators

    def generate_orbit(self, points: np.ndarray, tol=1e-5) -> NormedArraySet:
        """Generate the orbit of a set of points in the unit cell.
        Incoming points will be normalized to lie within the [0,1), as will outgoing points.

        Args:
            points: Mx2 np.ndarray, representing partial basis of unit cell.

        Returns:
            symmetrized_points: Nx2 np.ndarray, representing full basis of unit cell.
        """

        def _to_transformable(x):
            res = np.zeros((len(x), 3))
            res[:, :-1], res[:, -1] = np.array(list(x)), 1
            return res

        points = NormedArraySet(np.mod(points, 1.0), tol=tol)
        for o in self.symmetry_operators:
            basis = _to_transformable(points)
            points |= np.mod((o @ basis.T).T, 1)[:, :-1]

        return points


