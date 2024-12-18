from unittest import TestCase
import numpy as np

from _plane_symm_data import GENERATORS, WYCKOFF_POS
from array_set import ArraySet, NormedArraySet

from itertools import product

from plane_groups import PlaneGroup


class PlaneGroupTests(TestCase):
    def setUp(self):
        self.rng = np.random.default_rng()
        self.init_random_state = self.rng.bit_generator.state
        self.N_trials = 128

    def test_symmetry_op_generation(self):
        def reference_implementation(generators):
            # Reference implementation based on pymatgen symmetry operation routines
            # see https://github.com/materialsproject/pymatgen/blob/31f1e1fb8cc1bb517d59ea8e484965bb641c42a0/src/pymatgen/symmetry/groups.py#L415
            res = list(generators)
            new_ops = list(generators)
            while len(new_ops) > 0:
                gen_ops = []
                trial_ops = [x @ y for x, y in product(new_ops, res)]
                for op in trial_ops:
                    op[:-1, -1] = np.mod(op[:-1, -1], 1)
                    if not np.any([np.array_equal(op, g) for g in res]):
                        gen_ops.append(op)
                        res.append(op)
                new_ops = gen_ops
            return res

        for i, g in GENERATORS.items():
            ref_set = ArraySet(reference_implementation(g))
            test_set = PlaneGroup(int_number=i).symmetry_operators
            self.assertEqual(ref_set, test_set)

    def test_orbits(self):
        def get_generalized_orbit(i, p):
            orbit = NormedArraySet(p, tol=1e-5)

            data = WYCKOFF_POS[i]["(x, y)"]
            equiv_pos = data["equivalent_pos"]
            if len(equiv_pos) > 0:
                for epos in equiv_pos:
                    orbit |= np.mod(
                        np.stack([(lambda x, y: eval(epos))(i, j) for (i, j) in p]),
                        1.0,
                    )
            return orbit

        def get_high_sym_orbit(i, p):
            high_sym_points = NormedArraySet([], tol=1e-5)
            high_sym_orbit = NormedArraySet([], tol=1e-5)

            data = WYCKOFF_POS[i]
            for wp in data:
                if wp == ("(x, y)"):
                    continue
                equiv_pos = data[wp]["equivalent_pos"]
                if len(equiv_pos) > 0:
                    t_p = np.mod(np.stack([(lambda x, y: eval(wp))(i, j) for (i, j) in p]), 1.0)

                    high_sym_points |= t_p
                    high_sym_orbit |= high_sym_points
                    for epos in equiv_pos:
                        high_sym_orbit |= np.mod(
                            np.stack([(lambda x, y: eval(epos))(i, j) for (i, j) in t_p]),
                            1.0,
                        )

            return high_sym_points, high_sym_orbit

        def get_invariants(i, p):
            data = WYCKOFF_POS[i]

            invariant_pos = NormedArraySet([], tol=1e-5)
            for wp in data:
                equiv_pos = data[wp]["equivalent_pos"]
                if len(equiv_pos) == 0:
                    invariant_pos |= np.mod(
                        np.stack([(lambda x, y: eval(wp))(i, j) for (i, j) in p]),
                        1.0,
                    )
            return invariant_pos

        points = self.rng.uniform(size=(16, 2))
        for i in range(1, 18):
            pg = PlaneGroup(int_number=i)
            test_orbit = pg.generate_orbit(points)
            ref_orbit = get_generalized_orbit(i, points)
            self.assertEqual(ref_orbit, test_orbit, msg=f"Plane group {i} failed general orbit.")

            ref_invariants = get_invariants(i, points)
            if len(ref_invariants) > 0:
                test_invariants = pg.generate_orbit(np.stack(list(ref_invariants)))
                self.assertEqual(ref_invariants, test_invariants,
                                 msg=f"Plane group {i} failed invariant orbit.")

            ref_hs_points, ref_hs_orbit = get_high_sym_orbit(i, points)
            if len(ref_hs_points) > 0:
                test_hs_orbit = pg.generate_orbit(np.stack(list(ref_hs_points)))
                self.assertEqual(
                    ref_hs_orbit, test_hs_orbit, 
                    msg=f"Plane group {i} failed high symmetry orbit."
                )
