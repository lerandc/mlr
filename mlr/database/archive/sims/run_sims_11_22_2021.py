from tqdm import tqdm
from mlr.database.utils import get_new_folder, write_metadata, read_metadata
from ttt.utils import listfiles

import re
import numpy as np
import pyprismatic
import pandas as pd
import sys

import matplotlib.pyplot as plt

def main(process_info):

    # initialize common simulation metadata
    sim_meta = pyprismatic.Metadata()
    sim_meta.E0 = 300
    sim_meta.realspacePixelSize = 0.1
    sim_meta.potential3D = False
    sim_meta.includeThermalEffects = True
    sim_meta.algorithm = "t"
    sim_meta.numFP = 8
    sim_meta.saveComplexOutputWave = True
    sim_meta.numGPUs = 0
    sim_meta.numThreads = 4
    sim_meta.transferMode = "singlexfer"

    # get all structure folders
    structure_db_root = "/media/files/simulated_databases/defected_Au_np_structures_11_17_2021/structures/"
    folders = [x for x in listfiles(structure_db_root) if x.is_dir()]

    # sort folders by creation ID
    folders.sort(key=lambda x: int(str(x).rsplit("_", 2)[-2]))

    # split the 2048 IDs from the folders amongst the processes
    process_ID = int(process_info[0])
    num_processes = int(process_info[1])

    for f in tqdm(folders[process_ID::num_processes]):

        print(f)

        # set up output metadata
        particle_id = re.search("[0-9]+_[0-9]+", str(f).rsplit("/",1)[-1])[0]
        metadata = read_metadata(f.joinpath("metadata.json"))
        metadata["particle_ID"] = particle_id
        metadata["E0"] = sim_meta.E0
        metadata["numFP"] = sim_meta.numFP
        metadata["potential_parameterization"] = "2D"
        metadata["potential_sampling"] = sim_meta.realspacePixelSize
        metadata["C1"] = 0

        out_folder = get_new_folder(base="simulations/simulation")

        # run simulations
        sim_meta.filenameAtoms = str(f.joinpath("particle_on_substrate.xyz"))
        sim_meta.filenameOutput = str(out_folder.joinpath("particle_on_substrate.h5"))
        sim_meta.go()

        sim_meta.filenameAtoms = str(f.joinpath("particle.xyz"))
        sim_meta.filenameOutput = str(out_folder.joinpath("particle.h5"))
        sim_meta.go()

        # save metadata
        write_metadata(metadata, out_folder.joinpath("metadata.json"))

    return


if __name__ == "__main__":
    main(sys.argv[1:])