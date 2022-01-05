"""
Script generating particles as described in README
"""

from tqdm import tqdm
from mlr.database.utils import get_new_folder, write_metadata
from mlr.sampling.distribution import sample_normal
from czone.generator import Generator, AmorphousGenerator
from czone.volume import Volume, Sphere, Plane, makeRectPrism
from czone.prefab import fccMixedTwinSF
from czone.transform import Rotation, s2s_alignment, rot_vtv
from czone.scene import Scene

import numpy as np


def get_radius(rng, mean, std, lb, ub):
    radius = sample_normal(rng, mean, std)
    while(not ((radius >= lb) and (radius <= ub))):
        radius = sample_normal(rng, mean, std)
    return radius


def main():
    ## set up generators
    Au_gen = Generator.from_spacegroup(Z=[79],coords=np.array([[0,0,0]]),\
                                      cellDims=4.07825*np.ones(3), cellAngs=[90,90,90], sgn=225)

    substrate_gen = AmorphousGenerator()

    ## set up substrate volume
    sub_rect_prism = makeRectPrism(128,128,50)
    substrate = Volume(points=sub_rect_prism, generator=substrate_gen)
    substrate.priority = 1
    
    ## main loop
    rng = np.random.default_rng()
    N_particles = 512

    metadata = {"radius":0, "orientation":0, "position":0, "N_defects":0, "defect_types":0}

    print("Starting loop for generating %i particles." % N_particles)

    radius_params = [{"mean":25, "std":7.5, "lb":15, "ub":35}, 
                     {"mean":10, "std":2.5, "lb":7.5, "ub":12.5}]

    r_p = radius_params[1]
    for i in tqdm(range(N_particles)):
        r = get_radius(rng=rng, mean=r_p["mean"], std=r_p["std"], lb=r_p["lb"], ub=r_p["ub"])[0]
        sphere = Sphere(center=np.array([0,0,0]), radius=r)
        vol = Volume(alg_objects=[sphere])

        N_defects = rng.integers(low=0, high=1, endpoint=True)

        defected_NP_prefab = fccMixedTwinSF(generator=Au_gen, 
                                            volume=vol, 
                                            ratio=0.5,
                                            N=N_defects)

        current_vol, defect_types = defected_NP_prefab.build_object(return_defect_types=True)
        current_vol.priority = 0

        # apply random rotation
        zone_axis = np.random.randn(3)
        zone_axis /= np.linalg.norm(zone_axis)
        rot = Rotation(matrix=rot_vtv(zone_axis, [0,0,1]))
        current_vol.transform(rot)

        # put on substrate and apply random shift about center of FOV
        moving_plane = Plane(point=[0,0,-0.8*sphere.radius], normal=[0,0,-1]) # not quite the bottom of the NP
        target_plane = Plane(point=[0,0,50], normal=[0,0,1]) # the surface of the substrate

        shift = 54 - r
        final_center = np.array([64,64,0]) + shift*np.random.rand(3)*np.array([1,1,0])
        alignment_transform = s2s_alignment(moving_plane,
                                        target_plane,
                                        sphere.center,
                                        final_center)

        current_vol.transform(alignment_transform)

        ## generate scene and write to output files
        folder_path = get_new_folder(base="structures/particle")
        scene = Scene(bounds=np.array([[0,0,0],[128,128,130]]), objects=[current_vol])
        scene.populate()
        scene.to_file(str(folder_path.joinpath("particle.xyz")), format="prismatic")
        scene.add_object(substrate)
        scene.populate()
        scene.to_file(str(folder_path.joinpath("particle_on_substrate.xyz")), format="prismatic")

        metadata["radius"] = r
        metadata["orientation"] = [x for x in zone_axis]
        metadata["position"] = [x for x in final_center[:-1]]
        metadata["N_defects"] = int(N_defects)
        if N_defects == 0:
            metadata["defect_types"] = [None]
        else:
            metadata["defect_types"] = [x for x in defect_types if defect_types[x]]

        write_metadata(metadata, folder_path.joinpath("metadata.json"))

if __name__ == "__main__":
    main()
