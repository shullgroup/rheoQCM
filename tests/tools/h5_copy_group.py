import os
import h5py
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Copy group (group_s) from source (path_s) to group (group_d) in destination (path_d)')
    
    parser.add_argument('path_s', metavar='path_s', type=str, help='path of source file')
    parser.add_argument('path_d', metavar='path_d', type=str, help='path of destination file')
    parser.add_argument('group_s', metavar='group_s', type=str, help='group path in source file. e.g.: "/raw/ref"')
    parser.add_argument('group_d', metavar='group_d', type=str, default=None, nargs='?', help='group path in destination file')

    args = parser.parse_args()
    # parser.print_help() # print the help information when run the code w/ -h or --help. Can be commonted if don't want to show


    print('\n===============\n')
    # check the args
    if args.path_s: # path is given
        path_s = args.path_s
        if os.path.exists(path_s):
            path_s = os.path.abspath(path_s)
            print(f'source file: {path_s}')
        else:
            print('Source file does not exist.\nExit!')
    else:
        print('Please provide "path_s".\nExit!')
        exit(0)
    if args.path_d: # path is given
        path_d = args.path_d
        if os.path.exists(path_d):
            path_d = os.path.abspath(path_d)
            print(f'source file: {path_d}')
        else:
            print('Source file does not exist.\nExit!')
    else:
        print('Please provide "path_d".\nExit!')
        exit(0)
    if args.group_s: # path is given
        group_s = args.group_s
        print(f'Source group: {group_s}')
    else:
        print('Please provide "group_s".\nExit!')
        exit(0)
    if args.group_d is None: # path is given
        group_d = group_s
        print('Set "group_d" = "group_s"')
    else:
        group_d = args.group_d
    print(f'Destination group: {group_d}')


    # path_s = r'../test_data/BCB_4.h5' # sourse file
    # path_d = r'../test_data/polymer.h5' # file to append to

    # group_s = r'/raw/ref'
    # group_d = None # the same group

    if path_s != path_d:
        with h5py.File(path_s, 'r') as fs:
            # check if group_s exists
            if group_s not in fs:
                print('"group_s" does not exist in source file.\nExit.')
                exit(1)
            if group_d is None:
                group_d = group_s
            # Get the name of the parent for the group we want to copy
            group_path_partent = fs[group_s].parent.name
            with h5py.File(path_d, 'a') as fd:
                # Check that this group exists in the destination file; if it doesn't, create it
                # This will create the parents too, if they don't exist
                parent_group_id = fd.require_group(group_path_partent)

                group_name = group_d.split('/')[-1]
                print(group_name)
                print(list(parent_group_id.keys()))
                if group_name in list(parent_group_id.keys()):
                    print(f'{group_name} already existed in destination file.')
                    del parent_group_id[group_name]
                    fs.copy(group_s, parent_group_id, name=group_name)
                    print('Data was overwriten.')

                else:
                    # Copy fs:group_s to fd:group_d
                    fs.copy(group_s, parent_group_id, name=group_name)
                    print(f'Data group {group_name} is copied to destination file.')
                
                # check the datasets if copied
                # print(fd[group_d].keys())
    else:
        with h5py.File(path_s, 'a') as fs:
            # check if group_s exists
            if group_s not in fs:
                print('"group_s" does not exist in source file.\nExit.')
                exit(1)
            if group_d is None:
                group_d = group_s
            # Get the name of the parent for the group we want to copy
            group_path_partent = fs[group_s].parent.name

            fd = fs
            # Check that this group exists in the destination file; if it doesn't, create it
            # This will create the parents too, if they don't exist
            parent_group_id = fd.require_group(group_path_partent)

            group_name = group_d.split('/')[-1]
            print(group_name)
            print(list(parent_group_id.keys()))
            if group_name in list(parent_group_id.keys()):
                print(f'{group_name} already existed in destination file.')
                del parent_group_id[group_name]
                fs.copy(group_s, parent_group_id, name=group_name)
                print('Data was overwriten.')

            else:
                # Copy fs:group_s to fd:group_d
                fs.copy(group_s, parent_group_id, name=group_name)
                print(f'Data group {group_name} is copied to destination file.')
            
            # check the datasets if copied
            # print(fd[group_d].keys())

