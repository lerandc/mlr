from datetime import datetime
import pathlib
import json

def gen_folder_name(base="particle"):
    today = datetime.today()
    formatToday = today.strftime("_%Y%m%d")
    hTime = int(today.strftime("%H"))
    mTime = int(today.strftime("%M"))
    sTime = int(today.strftime("%S"))
    fTime = today.strftime("%f")

    #convert time into seconds
    time = (hTime*3600) + (mTime*60) + sTime 
    time = str(time)

    #add together for file name
    folderName = base + "_" + time + fTime+ formatToday
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

    #convert time into seconds
    time = (hTime*3600) + (mTime*60) + sTime 
    time = str(time)

    #add together for file name
    folderName = base + "_" + time + str(my_fTime) + formatToday
    return folderName

def get_new_folder(mkdir=True, base="particle"):
    new_folder = gen_folder_name(base)
    while(pathlib.Path(new_folder).is_dir()):
        new_folder = gen_folder_name(base)

    if mkdir:
        pathlib.Path(new_folder).mkdir(parents=True)

    return pathlib.Path(new_folder)

def get_new_folder_parallel(rank, N_processes, mkdir=True, base="particle"):
    new_folder = gen_folder_name_by_thread(rank, N_processes, base)

    successful = False
    while(not successful):
        while(pathlib.Path(new_folder).is_dir()):
            new_folder = gen_folder_name_by_thread(rank, N_processes, base)

        try:
            if mkdir: #TODO when are we not making the directory?
                pathlib.Path(new_folder).mkdir(parents=True)
            successful = True
        except FileExistsError:
            successful = False

    return pathlib.Path(new_folder)

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