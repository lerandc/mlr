from datetime import datetime
import pathlib
import json
from ttt.utils import yield_subfolders
import pandas as pd
import numpy as np


def gen_folder_name(base="particle"):
    today = datetime.today()
    formatToday = today.strftime("_%Y%m%d")
    hTime = int(today.strftime("%H"))
    mTime = int(today.strftime("%M"))
    sTime = int(today.strftime("%S"))
    fTime = today.strftime("%f")

    # convert time into seconds
    time = (hTime * 3600) + (mTime * 60) + sTime
    time = str(time)

    # add together for file name
    folderName = base + "_" + time + fTime + formatToday
    return folderName


def gen_folder_name_by_thread(rank, N_processes, base="particle"):
    today = datetime.today()
    formatToday = today.strftime("_%Y%m%d")
    hTime = int(today.strftime("%H"))
    mTime = int(today.strftime("%M"))
    sTime = int(today.strftime("%S"))

    fTime = today.microsecond

    # separate into blocks of valid microseconds
    # ensures no collision between processes of different ranks
    my_fTime = fTime - (fTime % N_processes) + rank

    # convert time into seconds
    time = (hTime * 3600) + (mTime * 60) + sTime
    time = str(time)

    # add together for file name
    folderName = base + "_" + time + str(my_fTime) + formatToday
    return folderName


def get_new_folder(mkdir=True, base="particle"):
    new_folder = gen_folder_name(base)
    while pathlib.Path(new_folder).is_dir():
        new_folder = gen_folder_name(base)

    if mkdir:
        pathlib.Path(new_folder).mkdir(parents=True)

    return pathlib.Path(new_folder)


def get_new_folder_parallel(rank, N_processes, mkdir=True, base="particle"):
    new_folder = gen_folder_name_by_thread(rank, N_processes, base)

    successful = False
    while not successful:
        while pathlib.Path(new_folder).is_dir():
            new_folder = gen_folder_name_by_thread(rank, N_processes, base)

        try:
            if mkdir:  # TODO when are we not making the directory?
                pathlib.Path(new_folder).mkdir(parents=True)
            successful = True
        except FileExistsError:
            successful = False

    return pathlib.Path(new_folder)

def write_metadata(metadata, path):
    with open(str(path), "w") as f:
        json_string = json.dumps(metadata, sort_keys=True, indent=4)
        f.write(json_string)

    return

def read_metadata(path):
    f = open(path)
    metadata = json.load(f)
    f.close()
    return metadata


def get_metadata(folder):
    return read_metadata(folder.joinpath("metadata.json"))


def get_all_metadata(base_folders, parse_f=lambda x: x):
    for folder in yield_subfolders(base_folders):
        try:
            metadata = get_metadata(folder)
            # TODO: should parse_f raise a ValueError/error of some type to indicate that the job did not complete?
            # e.g., if one needs to grab the history data from the checkpointing dump file
            yield parse_f(metadata)
        except KeyError as e:
            pass 
        except FileNotFoundError:
            pass
        except TypeError as e:
            print(folder)
            raise TypeError(e)


def get_metadataframe(base_folders, parse_f=lambda x: x):
    return pd.DataFrame(get_all_metadata(base_folders, parse_f))


def base_history_parser(metadata):
    history = metadata.pop("history")

    def array_iter(epochkeys, subkey):
        for ek in epochkeys:
            yield history[ek][subkey]

    ## Get arrays
    N_epochs = metadata["N_epochs"]
    skeys = [str(x) for x in sorted(int(x) for x in history.keys())]
    array_dict = {
        k: np.fromiter(array_iter(skeys, k), float, count=N_epochs)
        for k in history[str(skeys[0])]
    }

    return {**metadata, **array_dict}
