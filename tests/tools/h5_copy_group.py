import h5py

path_s = r'/home/janus/Documents/FS/CHiMaD_composite/Experiment_Data/20210413/rt_180_halfin_1_2_bare.h5' # sourse file
path_d = r'/home/janus/Documents/FS/CHiMaD_composite/Experiment_Data/20210413/rt_180_halfin_1_2_DOTP.h5' # file to append to

group_path_s = r'/raw/ref'
group_path_d = None # the same group

with h5py.File(path_s, 'r') as fs:
    if group_path_d is None:
        group_path_d = group_path_s
    # Get the name of the parent for the group we want to copy
    group_path_partent = fs[group_path_s].parent.name
    with h5py.File(path_d, 'a') as fd:
        # Check that this group exists in the destination file; if it doesn't, create it
        # This will create the parents too, if they don't exist
        group_id = fd.require_group(group_path_partent)

        # Copy fs:group_path_s to fd:group_path_d
        fs.copy(group_path_s, group_id, name=group_path_d.split('/')[-1])
        
        # check the datasets if copied
        print(fd[group_path_d].keys())

