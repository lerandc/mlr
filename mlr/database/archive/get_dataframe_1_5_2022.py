import pathlib
import json
import pandas as pd

def write_metadata(metadata, path):
    with open(str(path), 'w') as f:
        json_string = json.dumps(metadata, sort_keys=True, indent=4)
        f.write(json_string)

    return

def read_metadata(path):
    f = open(path)
    metadata = json.load(f)
    f.close()
    return metadata

def is_hidden(path):
    for x in str(path).split("/"):
        if x.startswith(".") and x != "..":
            return True

    return False

def listfiles(folder, include_hidden=False):
    # generator for files in subdirectory

    if include_hidden:
        out = [x for x in pathlib.Path(folder).glob("**/*")]
        return out
    else:
        out = [x for x in pathlib.Path(folder).glob("**/*") if not is_hidden(x)]
        return out

def get_dataframe(root_folder):
    folders = [x for x in listfiles(root_folder) if x.is_dir()]

    metadata_list = []
    for f in folders:
        dataX_path = f.joinpath("train.npy")
        dataY_path = f.joinpath("mask.npy")
        meta_dict = read_metadata(f.joinpath("metadata.json"))
        meta_dict["dataX_path"] = dataX_path
        meta_dict["dataY_path"] = dataY_path
        metadata_list.append(meta_dict)

    return pd.DataFrame(metadata_list)