
from collections.abc import Callable
from abc import ABC
from typing import List, Tuple

import numpy as np

from .hilbert import hilbert_sort_3D_unrolled_fixed_iter
from .voxel_grid import get_voxel_grid
from ttt.timers import ScopedTimer

"""
For searching clusters with sparse graphs, need to store vector-valued elements in sparse matrix; or, sparse tensors

pydata sparse has generalized sparse tensor operations for cpu python
pytorch sparse has COO sparse tensor format

should look into the featurization of the data you want to do, and what computed quantities from the matrix you want to include

for now, store:
    - unweighted upper triangle graph (A)
    - weighted upper triangle graph (A_d)
    - x/y/z weighted upper triangle graphs (A_x, A_y, A_z) 

"""


"""
General procedure to make an atomic graph

Define r_e = radius_edge in angstroms
Define r_c = radius_cluster in angstroms

1) Shift and scale points to unit cube
2) Spatially sort with 3D Hilbert order
3) Put into voxels with side length r_e; also mapped with Hilbert order
4) For each atom i:
    a) Get list of all atoms in voxel and neighboring voxels
    b) Calculate squared distance d2
    c) For each atom j in list that is < r_e**2.0 away, create an edge weighted by d
    

Define N_max = maximum degree of separation 
To return atomic clusters with centered about atom i with radius r_c

Initialize v_x, v_y, v_z = e_i

For i in range(N_max):
    v_x = A_x @ v_x + v_x
    v_y = A_y @ v_y + v_y
    v_z = A_z @ v_z + v_z

"""


class AtomicGraph(ABC):

    def __init__(self,):

        return

class PackedBins(ABC):

    def __init__(self,):

        return

def remap_voxels(voxel_grid, new_index, new_order):

    new_grid = []
    for j in new_order:
        vn = voxel_grid[j]
        new_grid.append(set([new_index[i] for i in vn]))

    return new_grid

def batch_pairwise_dist(points):
    dist = np.linalg.norm(points-points[:,None,:],axis=2)
    # dist += np.eye(dist.shape[0])*1e5
    return np.max(dist, axis=1)

def batch_batch_pairwise_dist(batch_0, batch_1):
    dist = np.linalg.norm(batch_0-batch_1[:,None,:],axis=2)
    return np.max(dist, axis=1)

def neighbors_in_radius(idx, vidx, pos, voxel_n, voxel_s, r_c):
    nn_v = set([])
    for n in voxel_n[vidx]:
        if len(voxel_s[n]) > 0:
            c_atoms = np.concatenate([pos[j,:].reshape(1,3) for j in voxel_s[n]], axis=0)
            c_inds = np.array([j for j in voxel_s[n]])
            dists = np.linalg.norm(c_atoms-pos[idx,:].reshape(1,3),axis=1)
            nn_ind = c_inds[np.flatnonzero(dists < r_c)]
            nn_v.update(nn_ind)

    return nn_v

def test(r_e=3.0, radius=30):
    from czone.generator import AmorphousGenerator
    from czone.volume import Volume, Sphere

    seed = 23
    rng = np.random.default_rng(seed=seed)

    with ScopedTimer('Generating and sorting atoms'):
        a_gen = AmorphousGenerator()
        a_sphere = Volume(alg_objects=[Sphere(center=np.array([0,0,0]), radius=radius)],
                        generator=a_gen)

        a_sphere.populate_atoms()

        atoms = np.copy(a_sphere.atoms)
        N_atoms = atoms.shape[0]
        ### Shift and scale to unit cube
        s_atoms = atoms - np.min(atoms, axis=0)
        ss_atoms = s_atoms/np.max(atoms, axis=0)

        ### Get a Hilbert index for every point and sort the 
        N_iter = 6 # corresponds to a precision of 3.8e-6 in the unit cube
        h_idx = hilbert_sort_3D_unrolled_fixed_iter(ss_atoms, N_iter, return_hidx=True)

        # sort atoms and the hilbert indices
        h_order = np.argsort(h_idx)
        s_atoms = s_atoms[h_order, :]
        h_idx = h_idx[h_order]

    print(atoms.shape)

    ### Set up a voxel grid for the system
    with ScopedTimer('Setting up voxel grid'):
        bounds = np.ceil(np.max(atoms, axis=0)-np.min(atoms,axis=0))
        N_cells = [int(np.floor(b/r_e)) for b in bounds] # floor, want the boxes to be at least r_e big

        r_c = [bounds[i]/N for i, N in enumerate(N_cells)]

        voxels = get_voxel_grid(N_cells, False, False, False) # no periodic boundaries

        g_z, g_y, g_x = np(np.linspace(0, 1, N_cells[2], endpoint=False) + 1.0/(2*N_cells[0]), # probably better to have different method for odd/even grids
                                    np.linspace(0, 1, N_cells[1], endpoint=False) + 1.0/(2*N_cells[1]),
                                    np.linspace(0, 1, N_cells[0], endpoint=False) + 1.0/(2*N_cells[2]),
                                    )

        voxel_centers = np.hstack([g_x.ravel()[:,None], g_y.ravel()[:,None], g_z.ravel()[:,None]])

    # sort voxels by hilbert order, too
        h_idx_voxels = hilbert_sort_3D_unrolled_fixed_iter(voxel_centers, 3, return_hidx=True)
        h_order_voxels = np.argsort(h_idx_voxels).astype(int)
        new_order = np.argsort(h_order_voxels).astype(int)

        h_voxels = remap_voxels(voxels, new_order, h_order_voxels)
        h_vc = voxel_centers[h_order_voxels,:]
        h_idx_voxels = h_idx_voxels[h_order_voxels]

    # place atoms into voxels
    # should divide number line into equal partitions

    ## sort method 2: modulus based indexing with conversions
    with ScopedTimer('Assigning atoms to voxels'):
        voxel_sets = [set([]) for i in range(len(h_voxels))]
        voxel_indices = np.zeros(N_atoms, dtype=int)
        for i, p in enumerate(s_atoms):
            j_g = N_cells[0]*N_cells[1]*(int(p[2]//r_c[2])) \
                + N_cells[0]*(int(p[1]//r_c[1])) \
                + int(p[0]//r_c[0])
            j_h = new_order[j_g]
            voxel_indices[i] = j_h
            voxel_sets[j_h].update([i])

    """
    # is there a way to do this so that atoms are placed by voxel, i.e., by iterating over the hilbert indices instead?
    # probably only in the scenario that the grid has a multiple of 8 cells in every direction

    ## sort method 1: concurrent iterators
    # assumes that the voxel an atom belongs to would be visited in the same hilbert order
    # I tried getting this to work, but it failed the nearest neighbor searches-- I think I need to know 
    # which neighboring cells are in which direction to be able to debug this further. Not really worth the time.

    voxel_sets_m1 = [set([]) for i in range(len(h_voxels))]
    h_idx_vsort = np.linspace(1,0,len(h_voxels),endpoint=False)[::-1]
    # h_idx_vsort = np.concatenate([h_idx_voxels, 1.0])

    j = 0
    for i, h in enumerate(h_idx):
        if h <= h_idx_vsort[j]:
            voxel_sets_m1[j].update([i])
        else:
            j += 1
            
    ## validate sets
    r_e_eff = float(r_c[0])
    max_r = np.sqrt(3)*r_e_eff

    # check internal
    for s in voxel_sets_m1:
        if len(s) > 1:
            b_atoms = np.concatenate([s_atoms[i,:].reshape(1,3) for i in s], axis=0)
            assert(np.max(batch_pairwise_dist(b_atoms)) < max_r), "Particles in boxes too far from eachother."

    print("M1 cells validated")

    for s in voxel_sets_m2:
        if len(s) > 1:
            b_atoms = np.concatenate([s_atoms[i,:].reshape(1,3) for i in s], axis=0)
            assert(np.max(batch_pairwise_dist(b_atoms)) < max_r), "Particles in boxes too far from eachother."

    print("M2 cells validated")

    # check neigbhors
    max_r_n = 2*np.sqrt(3)*r_e_eff
    for i, s in enumerate(voxel_sets_m1):
        if len(s) > 0:
            b_atoms = np.concatenate([s_atoms[j,:].reshape(1,3) for j in s], axis=0)

            for n in h_voxels[i]:
                if len(voxel_sets_m1[n]) > 0:
                    c_atoms = np.concatenate([s_atoms[j,:].reshape(1,3) for j in voxel_sets_m1[n]], axis=0)
                    assert(np.max(batch_batch_pairwise_dist(b_atoms, c_atoms)) < max_r_n), "Particles in neighboring boxes too far from eachother."

    print("M1 cells validated against neighbors")

    for i, s in enumerate(voxel_sets_m2):
        if len(s) > 0:
            b_atoms = np.concatenate([s_atoms[j,:].reshape(1,3) for j in s], axis=0)

            for n in h_voxels[i]:
                if len(voxel_sets_m2[n]) > 0:
                    c_atoms = np.concatenate([s_atoms[j,:].reshape(1,3) for j in voxel_sets_m2[n]], axis=0)
                    assert(np.max(batch_batch_pairwise_dist(b_atoms, c_atoms)) < max_r_n), "Particles in neighboring boxes too far from eachother."

    print("M2 cells validated against neighbors")

    # Brute force check to make sure we can find all the nearest neighbors for a particle by checking the voxel lists
    # for i in range(len(s_atoms)):
    #     cv = [j for j, v in enumerate(voxel_sets_m1) if i in v] # current_voxel
    #     assert(len(cv) > 0), "Particle not owned by any voxel"
    #     assert(len(cv) == 1), "Particle owned by multiple voxels"
    #     cv = cv[0]

    #     nn_v = set([])
    #     for n in h_voxels[cv]:
    #         if len(voxel_sets_m1[n]) > 0:
    #             c_atoms = np.concatenate([s_atoms[j,:].reshape(1,3) for j in voxel_sets_m1[n]], axis=0)
    #             c_inds = np.array([j for j in voxel_sets_m1[n]])
    #             dists = np.linalg.norm(c_atoms-s_atoms[i,:].reshape(1,3),axis=1)
    #             nn_ind = c_inds[np.flatnonzero(dists < r_e_eff)]
    #             nn_v.update(nn_ind)

    #     dists = np.linalg.norm(s_atoms - s_atoms[i,:].reshape(1,3), axis=1)
    #     nn_b = set(list(np.flatnonzero(dists < r_e_eff)))

    #     assert(len(nn_v.symmetric_difference(nn_b)) == 0), "Nearest neigbhors do not match"

    # print("M1 validated with brute force nearest neighbor search")

    for i in range(len(s_atoms)):
        cv = [j for j, v in enumerate(voxel_sets_m2) if i in v] # current_voxel
        assert(len(cv) > 0), "Particle not owned by any voxel"
        assert(len(cv) == 1), "Particle owned by multiple voxels"
        cv = cv[0]

        nn_v = set([])
        for n in h_voxels[cv]:
            if len(voxel_sets_m2[n]) > 0:
                c_atoms = np.concatenate([s_atoms[j,:].reshape(1,3) for j in voxel_sets_m2[n]], axis=0)
                c_inds = np.array([j for j in voxel_sets_m2[n]])
                dists = np.linalg.norm(c_atoms-s_atoms[i,:].reshape(1,3),axis=1)
                nn_ind = c_inds[np.flatnonzero(dists < r_e_eff)]
                nn_v.update(nn_ind)

        dists = np.linalg.norm(s_atoms - s_atoms[i,:].reshape(1,3), axis=1)
        nn_b = set(list(np.flatnonzero(dists < r_e_eff)))

        assert(len(nn_v.symmetric_difference(nn_b)) == 0), "Nearest neigbhors do not match"

    print("M2 validated with brute force nearest neighbor search")


    for s1, s2 in zip(voxel_sets_m1, voxel_sets_m2):
        assert(len(s1.symmetric_difference(s2)) == 0), "Voxels own different particles"

    """

    ### Form the graphs as matrices
    with ScopedTimer('Creating atomic graph'):
        # A_all = np.zeros((5, N_atoms, N_atoms))
        # A = np.zeros((N_atoms, N_atoms), dtype=int)
        for i in range(N_atoms):
            nn = neighbors_in_radius(i, voxel_indices[i], s_atoms, h_voxels, voxel_sets, r_e)

            # for j in nn:
            #     r_ij = s_atoms[i, :] - s_atoms[j, :]
            #     A_all[:,i,j] = np.array([1.0, np.linalg.norm(r_ij), r_ij[0], r_ij[1], r_ij[2]])
            #     A[i,j] = 1
            #     A[j,i] = 1

        # A_all[:2,:,:] = A_all[:2,:,:] + np.transpose(A_all[:2,:,:], axes=(0, 2, 1))
        # di = np.diag_indices(N_atoms)
        # A_all[:2,di[0], di[1]] = 0

        # A[di] = 0

    # A_0 and A_1 should be real, symnmetric
    # A_2, A_3, A_4 should be real, skew-symmetric
    # All should have no self connections, so diagonals should be zero (implicit for A_2, A_3, and A_4)

    # return A, A_all


if __name__ == "__main__":
    test()